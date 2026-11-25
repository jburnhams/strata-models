"""An analysis in a process of its own (stratamodels.evals.process), and when evals.summaries uses one."""

import json
import sys
import threading
import types

import pytest

from stratamodels.evals import core as C
from stratamodels.evals import process as P
from stratamodels.evals.summaries import Summaries

SPEC = types.SimpleNamespace(name="x")


def runner(code):
    r = P.Runner(argv=[sys.executable, "-c", code])
    return r


def test_progress_and_the_result_come_back_and_stray_lines_are_ignored():
    seen = []
    r = runner(
        "import sys, json; print('a library says hello'); print(json.dumps(dict(progress=['scoring', 1, 4])));"
        "print(json.dumps(dict(result=dict(name=sys.argv[1]))))"
    )
    assert r(SPEC, progress=lambda *a: seen.append(a)) == dict(name="x")
    assert seen == [("scoring", 1, 4)] and r.live == set()


def test_a_child_that_fails_says_why():
    with pytest.raises(C.EvalError, match="no dataset"):
        runner("import json; print(json.dumps(dict(error='no dataset', type='EvalError')))")(SPEC)
    with pytest.raises(P.AnalysisFailed, match="ValueError: x"):
        runner("import json; print(json.dumps(dict(error='ValueError: x', type='ValueError')))")(SPEC)
    with pytest.raises(P.AnalysisFailed, match=r"without an answer \(exit 3\)"):
        runner("import sys; sys.exit(3)")(SPEC)


def test_stop_ends_the_child_and_starts_no_more():
    r = runner('import time; print(\'{"progress": ["s", 0, 1]}\', flush=True); time.sleep(60)')
    going, failed = threading.Event(), []

    def go():
        try:
            r(SPEC, progress=lambda *a: going.set())
        except P.AnalysisFailed as e:
            failed.append(str(e))

    t = threading.Thread(target=go)
    t.start()
    assert going.wait(30)
    r.stop()
    t.join(30)
    assert failed and "without an answer" in failed[0]
    with pytest.raises(P.AnalysisFailed, match="stopping"):
        r(SPEC)


def child(scheduler, monkeypatch, capsys, name):
    from strata_python import oslib

    from stratamodels import settings, store

    monkeypatch.setattr(oslib, "lower_priority", lambda n: None)
    monkeypatch.setattr(settings, "bind", lambda engine: None)
    monkeypatch.setattr(store.Store, "connect", classmethod(lambda cls, migrate=True: scheduler.store))
    code = P.main([name])
    return code, [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_the_child_analyses_the_experiment_as_the_database_has_it(scheduler, setup, monkeypatch, capsys):
    C.list_specs(scheduler.store.engine)  # the service imported it
    code, lines = child(scheduler, monkeypatch, capsys, "fake-asr")
    assert code == 0 and "progress" in lines[0] and lines[-1]["result"]["experiment"]["name"] == "fake-asr"
    code, lines = child(scheduler, monkeypatch, capsys, "nope")
    assert code == 1 and lines == [dict(error="no experiment 'nope'", type="EvalError")]


def test_slow_and_new_analyses_go_to_a_process_and_quick_ones_stay(scheduler, setup):
    engine = scheduler.store.engine
    spec = C.load_spec("fake-asr", engine)
    s = Summaries(lambda: engine)
    calls = []
    s.process = lambda spec, engine, progress=None: calls.append(spec.name) or dict(experiment=spec.to_json())
    s.compute(spec)  # never analysed: it may be a long one
    assert calls == ["fake-asr"]
    s.compute(spec, force=True)  # it was quick
    assert calls == ["fake-asr"]
    with C.session(engine) as db:
        from stratamodels.db.models import EvalSummary

        db.get(EvalSummary, "fake-asr").seconds = 900.0
        db.commit()
    s.compute(spec, force=True)
    assert calls == ["fake-asr"] * 2
    s.process = types.SimpleNamespace(stop=lambda: calls.append("stopped"))
    s.stop()
    s.refresh("fake-asr")  # the pool is closed: left for the next start
    assert calls[-1] == "stopped"
