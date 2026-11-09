"""Experiments end to end on the fake runtime (stratamodels.evals.core and report): datasets, specs, cached runs, analysis, the decision, the write-up."""

import os

import pytest

from stratamodels.evals import core as C
from stratamodels.evals import report as R


@pytest.fixture
def engine(scheduler):
    return scheduler.store.engine


def test_datasets_are_versioned_and_checked(setup, tmp_path):
    ds = C.load_dataset("speech")
    assert ds.version == setup.version and len(ds.items) == 40 and ds.description == "forty fake utterances for tests"
    assert ds.to_json()["tags"] == ["speaker"] and "items" in ds.to_json(items=True)
    assert [d["name"] for d in C.list_datasets()] == ["speech"]
    bad = tmp_path / "bad"
    bad.mkdir()
    for name, text, msg in (
        ("dup", '{"id": 1, "input": {}}\n{"id": 1, "input": {}}', "twice"),
        ("noinput", '{"id": 1}', "needs"),
        ("json", "{x", "line 1"),
        ("empty", "# nothing\n", "no items"),
    ):
        (bad / f"{name}.jsonl").write_text(text)
        with pytest.raises(C.EvalError, match=msg):
            C.load_dataset(name, str(bad))
    assert {d["name"]: "error" in d for d in C.list_datasets(str(bad))} == dict(
        dup=True, empty=True, json=True, noinput=True
    )
    with pytest.raises(C.EvalError, match="no dataset"):
        C.load_dataset("nope")


def test_specs_are_parsed_and_refused_when_wrong(setup):
    s = C.load_spec("fake-asr")
    assert s.baseline == "base" and s.guards == dict(latency_s=1e9) and s.judge and s.candidate("better").model == "big"
    with pytest.raises(C.EvalError, match="no candidate"):
        s.candidate("x")
    with pytest.raises(C.EvalError, match="no experiment"):
        C.load_spec("nope")
    with pytest.raises(C.EvalError, match="unknown metrics"):
        C.parse_spec(
            dict(
                name="x", question="q", role="r", dataset="d", metrics=["bleu"], candidates=[dict(name="a", model="m")]
            )
        )
    with pytest.raises(C.EvalError, match="share a name"):
        C.parse_spec(
            dict(
                name="x",
                question="q",
                role="r",
                dataset="d",
                metrics=["wer"],
                candidates=[dict(name="a", model="m"), dict(name="a", model="m")],
            )
        )
    with pytest.raises(C.EvalError, match="KeyError"):
        C.parse_spec(dict(name="x"))
    p = C.parse_spec(
        dict(
            name="x",
            question="q",
            role="r",
            dataset="d",
            metrics=["cer"],
            decision=dict(primary="wer", guards=dict(failed=1)),
            candidates=[dict(name="a", model="m")],
        )
    )
    assert p.metrics == ["wer", "cer", "failed"] and p.baseline == "a"
    open(os.path.join(os.environ["STRATAMODELS_DATA"], "evals", "specs", "broken.toml"), "w").write("name = ")
    assert isinstance(C.list_specs()["broken"], C.EvalError)
    with pytest.raises(C.EvalError):
        C.load_spec("broken")


def test_run_caches_outputs_and_analysis_decides(setup, scheduler, engine):
    stop = scheduler.start()
    try:
        s = C.load_spec("fake-asr")
        log = []
        r = C.run(s, engine, C.SchedulerExecutor(scheduler), log=log.append)
        assert r == dict(ran=120, cached=0, failed=0) and "to run" in log[0]
        assert C.run(s, engine, C.SchedulerExecutor(scheduler), log=log.append) == dict(ran=0, cached=120, failed=0)
        assert any(
            k.startswith("small~") for k in scheduler.registry.models
        )  # the candidate with options ran on a variant of its own
        res = C.analyse(s, engine)
        assert (
            res["dataset"]["compared"] == 40
            and res["candidates"]["base"]["metrics"]["wer"]["value"]
            > 0.2
            > res["candidates"]["better"]["metrics"]["wer"]["value"]
        )
        assert (
            res["comparisons"]["better"]["wer"]["improves"] is True
            and res["comparisons"]["same"]["wer"]["improves"] is None
            and "p_holm" in res["comparisons"]["same"]["wer"]
        )
        d = res["decision"]
        assert (
            d["recommendation"] == "better"
            and d["change"]
            and d["per_candidate"]["better"]["verdict"] == "adopt"
            and d["per_candidate"]["same"]["verdict"] == "unclear"
        )
        assert res["comparisons"]["better"]["latency_s"]["n"] == 40
        ds, outs, _ = C.outputs_for(s, engine)
        text = R.render(res, R.worst(s, outs, ds.items, "wer"), date="2026-11-09")
        for h in (
            "## Answer",
            "## Setup",
            "## Results",
            "## Compared with the baseline",
            "## Cost",
            "## Failures and worst items",
            "## Method",
            "## Reproduce",
            "**Change to `better`.**",
        ):
            assert h in text
        assert "u0" in text and '"drop": 12' in text
        judged = dict(res, judgments=dict(n=12, strength=dict(base=(0.8, 0.5, 1.1), better=(1.25, 0.9, 1.7))))
        jtext = R.render(judged, R.worst(s, outs, ds.items, "wer"), date="2026-11-09")
        assert "## Human judgments" in jtext and "12 blind pairwise judgments" in jtext
        assert jtext.index("`better`: 1.25 [0.90, 1.70]") < jtext.index("`base`: 0.80 [0.50, 1.10]")  # strongest first
        assert (
            C.forget(s, engine, "same") == 40
            and C.run(s, engine, C.SchedulerExecutor(scheduler), candidates=["same"], limit=5, log=log.append)["ran"]
            == 5
        )
        partial = C.analyse(s, engine)
        assert partial["dataset"]["compared"] == 5
    finally:
        stop.set()


def test_failures_guards_and_keeping_the_baseline(setup, scheduler, engine, monkeypatch):
    s = C.parse_spec(
        dict(
            name="g",
            question="q",
            role="echo",
            dataset="speech",
            metrics=["wer", "failed"],
            decision=dict(primary="wer", baseline="a", min_effect=0.5, guards=dict(failed=1.0)),
            candidates=[
                dict(name="a", model="small", params=dict(drop=4)),
                dict(name="b", model="big", params=dict(drop=12)),
                dict(name="c", model="big", params=dict(fail="broken")),
            ],
        )
    )
    stop = scheduler.start()
    try:
        assert C.run(s, engine, C.SchedulerExecutor(scheduler), log=lambda m: None)["failed"] == 40
    finally:
        stop.set()
    res = C.analyse(s, engine)
    d = res["decision"]["per_candidate"]
    assert (
        d["b"]["verdict"] == "reject"
        and "under the" in d["b"]["why"]
        and d["c"]["verdict"] == "reject"
        and not res["decision"]["change"]
    )
    text = R.render(res, R.worst(s, C.outputs_for(s, engine)[1], C.load_dataset("speech").items, "wer"))
    assert "**Keep `a`.**" in text and "failed: broken" in text
    s2 = C.parse_spec(
        dict(
            name="g2",
            question="q",
            role="echo",
            dataset="speech",
            metrics=["wer"],
            decision=dict(primary="wer", baseline="b"),
            candidates=[
                dict(name="b", model="big", params=dict(drop=12)),
                dict(name="a", model="small", params=dict(drop=4)),
            ],
        )
    )
    assert C.analyse(s2, engine)["decision"]["per_candidate"]["a"]["verdict"] == "reject"
    s3 = C.parse_spec(
        dict(
            name="g3",
            question="q",
            role="echo",
            dataset="speech",
            metrics=["latency_s"],
            decision=dict(primary="wer", baseline="a", guards=dict(latency_s=0.0001)),
            candidates=[
                dict(name="a", model="small", params=dict(drop=4)),
                dict(name="b", model="big", params=dict(drop=12)),
            ],
        )
    )
    assert C.analyse(s3, engine)["decision"]["per_candidate"]["b"]["verdict"] in ("reject", "adopt")


def test_executors_pass_through(scheduler):
    calls = []

    class SM:
        def submit(self, role, params, app, priority, model, options):
            calls.append((role, app, priority, model, options))
            return dict(id=7)

        def wait_job(self, i):
            return dict(id=i, status="done")

    ex = C.ClientExecutor(SM())
    assert (
        ex.submit("r", {}, "m", {}) == 7
        and ex.wait(7)["status"] == "done"
        and calls == [("r", "evals", "batch", "m", None)]
    )
    j = C.SchedulerExecutor(scheduler).submit("echo", dict(x=1), "small", None)
    scheduler.step()
    assert C.SchedulerExecutor(scheduler).wait(j)["status"] == "done"


def test_guards_hold_against_a_zero_baseline_and_duplicates_run_once(setup, scheduler, engine):
    s = C.parse_spec(
        dict(
            name="z",
            question="q",
            role="echo",
            dataset="speech",
            metrics=["wer", "failed"],
            decision=dict(primary="wer", baseline="a", guards=dict(failed=1.0)),
            candidates=[
                dict(name="a", model="small", params=dict(drop=4)),
                dict(name="b", model="big", params=dict(drop=12)),
                dict(name="b2", model="big", params=dict(drop=12)),
            ],
        )
    )
    stop = scheduler.start()
    try:
        assert (
            C.run(s, engine, C.SchedulerExecutor(scheduler), log=lambda m: None)["ran"] == 80
        )  # b2 is the same job as b
    finally:
        stop.set()
    from sqlalchemy.orm import Session

    from stratamodels.db.models import EvalResult

    with Session(engine) as ses:  # b fails on one item: 1 failure against the baseline's 0
        r = ses.query(EvalResult).filter(EvalResult.model == "big").first()
        r.error = "boom"
        r.output = None
        ses.commit()
    d = C.analyse(s, engine)["decision"]["per_candidate"]
    assert d["b"]["verdict"] == "reject" and "against 0" in d["b"]["why"]


def test_a_concurrent_run_that_stored_the_same_output_first_wins(setup, scheduler, engine, monkeypatch):
    from stratamodels.db.models import EvalResult

    s = C.load_spec("fake-asr")
    ds = C.load_dataset("speech")
    c = s.candidate("base")
    it = ds.items[0]
    k = C.result_key(s, c, it)
    mk = lambda out: EvalResult(
        key=k, dataset="speech", item_id=it["id"], role="echo", model="small", created_at=1.0, output=out
    )
    C.save(engine, [mk(dict(first=True))])
    C.save(engine, [mk(dict(second=True))])
    assert C.cached(engine, [k])[k].output == dict(first=True)
    real = C.select
    calls = []

    def blind(*a, **kw):  # the check finds nothing (the other run commits between check and commit)
        calls.append(1)
        q = real(*a, **kw)
        return q.where(EvalResult.id < 0) if len(calls) == 1 else q

    monkeypatch.setattr(C, "select", blind)
    k2 = C.result_key(s, c, ds.items[1])
    other = EvalResult(
        key=k2,
        dataset="speech",
        item_id=ds.items[1]["id"],
        role="echo",
        model="small",
        created_at=1.0,
        output=dict(ok=1),
    )
    C.save(engine, [mk(dict(third=True)), other])
    monkeypatch.undo()
    got = C.cached(engine, [k, k2])
    assert got[k].output == dict(first=True) and got[k2].output == dict(ok=1)


def test_a_higher_is_better_guard_rejects_a_candidate_that_falls_too_far_below_the_baseline():
    s = C.parse_spec(
        dict(
            name="h",
            question="q",
            role="r",
            dataset="d",
            metrics=["wer", "parsed"],
            decision=dict(primary="wer", baseline="a", min_effect=0.02, guards=dict(parsed=2.0)),
            candidates=[dict(name="a", model="m"), dict(name="b", model="m"), dict(name="c", model="m")],
        )
    )
    better = dict(n=40, diff=-0.1, low=-0.2, high=-0.01, improves=True)
    res = dict(
        comparisons=dict(b=dict(wer=better), c=dict(wer=better)),
        candidates=dict(
            a=dict(metrics=dict(parsed=dict(value=1.0))),
            b=dict(metrics=dict(parsed=dict(value=0.3))),  # under 1/2 of the baseline: broken
            c=dict(metrics=dict(parsed=dict(value=0.9))),  # within the guard
        ),
    )
    d = C.decide(s, res, [], lambda _name: [], "a")
    assert (
        d["per_candidate"]["b"]["verdict"] == "reject"
        and "parsed 0.30x the baseline (limit 1/2)" in d["per_candidate"]["b"]["why"]
    )
    assert d["per_candidate"]["c"]["verdict"] == "adopt" and d["recommendation"] == "c" and d["change"]


def test_report_numbers_are_formatted_by_unit():
    from types import SimpleNamespace

    pct, sec, plain = SimpleNamespace(unit="%"), SimpleNamespace(unit="s"), SimpleNamespace(unit="")
    assert [R._v(m, x) for m, x in ((pct, 0.1234), (sec, 1.2345), (plain, 0.5), (sec, None), (pct, float("nan")))] == [
        "12.34%",
        "1.234 s",
        "0.5000",
        "n/a",
        "n/a",
    ]
    assert [R._d(m, x) for m, x in ((pct, -0.05), (sec, 0.25), (plain, 0.25), (sec, None))] == [
        "-5.00 pts",
        "+0.250 s",
        "+0.250",
        "n/a",
    ]


def test_specs_and_datasets_live_in_the_database_once_seen(setup, engine, tmp_path):
    from stratamodels.db.models import EvalCandidate, EvalExperiment, EvalItem

    ds = C.load_dataset("speech", engine=engine)
    assert ds.version == setup.version and ds.items == setup.items and ds.description == setup.description
    assert C.import_dataset(engine, setup) is False  # same version: nothing written
    with C.session(engine) as s:
        assert s.query(EvalItem).count() == 40
    specs = C.list_specs(engine)
    assert specs["fake-asr"].source.endswith("fake-asr.toml")
    with C.session(engine) as s:
        assert [c.name for c in s.query(EvalCandidate).order_by(EvalCandidate.position)] == [
            "base",
            "better",
            "same",
        ] and s.get(EvalExperiment, "fake-asr").edited is False
    # a dataset rewritten with extra fields, and one that only the database has (its file gone)
    items = [dict(setup.items[0], extra_field=1)] + setup.items[1:3]
    ds2 = C.write_dataset("speech", items, "three now", engine=engine)
    assert ds2.items[0]["extra_field"] == 1 and len(C.load_dataset("speech", engine=engine).items) == 3
    C.write_dataset("only-db", items, "x", engine=engine)
    os.remove(os.path.join(C.datasets_dir(), "only-db.jsonl"))
    assert C.load_dataset("only-db", engine=engine).items == items and "only-db" in [
        d["name"] for d in C.list_datasets(engine=engine)
    ]
    with pytest.raises(C.EvalError, match="no dataset"):
        C.load_dataset("never", engine=engine)
    # an experiment edited here is no longer overwritten by its file; one added here exists without a file
    d = C.load_spec("fake-asr", engine).to_json()
    d["question"] = "edited here"
    d["decision"] = dict(primary="wer", baseline="base")
    C.save_spec(engine, d)
    assert C.load_spec("fake-asr", engine).question == "edited here"
    C.save_spec(engine, dict(d, name="db-only", question="no file"))
    assert C.load_spec("db-only", engine).question == "no file" and "db-only" not in C.list_specs()
    with pytest.raises(C.EvalError):
        C.save_spec(engine, dict(d, metrics=["nope"]))
    with C.session(engine) as s:
        s.get(EvalExperiment, "db-only").spec = dict(name="db-only")
        s.commit()
    assert isinstance(C.list_specs(engine)["db-only"], C.EvalError)
