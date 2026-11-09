"""venv-worker against a real child process: this test's own Python runs a tiny echo worker, so the JSON-lines protocol, a worker that dies
and shutdown are exercised over real pipes on every platform (the unit tests use a fake process)."""

import os
import sys
import textwrap

import pytest

from stratamodels.registry import Registry
from stratamodels.runtimes import RunError, worker

SCRIPT = textwrap.dedent("""
    import json, sys
    for line in sys.stdin:
        m = json.loads(line)
        if m['op'] == 'quit': break
        if m['op'] == 'load': print(json.dumps(dict(ok=True, result=dict(engine=m['model']['options'].get('engine')))), flush=True); continue
        p = m['params']
        if p.get('die'): sys.exit(3)
        if p.get('bad'): print(json.dumps(dict(ok=False, params=True, error='bad text')), flush=True); continue
        print(json.dumps(dict(ok=True, result=dict(echo=p))), flush=True)
""")


@pytest.fixture
def env(tmp_path, monkeypatch):
    wd = tmp_path / "workers"
    wd.mkdir()
    (wd / "echo.py").write_text(SCRIPT)
    monkeypatch.setattr(worker, "WORKERS", str(wd))
    monkeypatch.setattr(
        worker, "python_of", lambda model, models_dir: (str(tmp_path / "venv"), sys.executable)
    )  # this Python stands in for the venv's
    monkeypatch.setenv("STRATAMODELS_DATA", str(tmp_path / "data"))
    (tmp_path / "models").mkdir()
    return str(tmp_path / "models")


def model(**options):
    return Registry.from_dict(
        dict(
            roles=dict(speak=dict(model="m")),
            models=dict(m=dict(runtime="venv-worker", roles=["speak"], options=options)),
        )
    ).models["m"]


def test_a_real_worker_process_loads_runs_dies_and_stops(env, tmp_path):
    h = worker.load(model(worker="echo", engine="kokoro"), env)
    assert h["info"] == dict(engine="kokoro")
    assert worker.run(h, dict(text="hi"))["echo"]["text"] == "hi"
    with pytest.raises(RunError, match="bad text"):
        worker.run(h, dict(bad=1))
    p = h["worker"].p
    worker.unload(h)
    assert p.poll() is not None and os.path.isfile(str(tmp_path / "venv") + ".log")
    h = worker.load(model(worker="echo"), env)
    with pytest.raises(RuntimeError, match="has stopped"):
        worker.run(h, dict(die=1))
    worker.unload(h)
