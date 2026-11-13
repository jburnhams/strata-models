"""venv-worker (stratamodels.runtimes.worker): a model in a Python of its own, spoken to in JSON lines. A fake process stands in for the
worker (it answers the protocol in this process), so the protocol, failures and shutdown are exercised without a child process or any model.
The same protocol against a real child process: tests/integration/test_worker_process.py."""

import json
import os
import queue
import subprocess

import pytest

from stratamodels import gpu
from stratamodels.registry import Registry
from stratamodels.runtimes import RunError, worker


class FakeWorker:
    """What subprocess.Popen returns for a worker: stdin takes request lines, stdout gives the answers the echo worker would."""

    def __init__(self, cmd, **kw):
        self.cmd, self.returncode, self.out = cmd, None, queue.Queue()
        proc = self

        class In:
            def write(self, s):
                if proc.returncode is not None:
                    raise BrokenPipeError("worker gone")
                for line in s.splitlines():
                    proc._answer(json.loads(line))

            def flush(self):
                pass

            def close(self):
                pass

        class Out:
            def readline(self):
                return proc.out.get()

            def close(self):
                pass

        self.stdin, self.stdout = In(), Out()

    def _answer(self, m):
        say = lambda **d: self.out.put(json.dumps(d) + "\n")
        if m["op"] == "quit":
            return self._end(0)
        if m["op"] == "load":
            o = m["model"]["options"]
            return say(ok=o.get("engine") != "broken", result=dict(engine=o.get("engine")), error="cannot load")
        p = m["params"]
        if p.get("die"):
            return self._end(3)
        if p.get("hang"):
            return
        if p.get("junk"):
            return self.out.put("not json\n")
        if p.get("bad"):
            return say(ok=False, params=True, error="bad text")
        if p.get("crash"):
            return say(ok=False, error="CUDA out of memory")
        say(ok=True, result=dict(echo=p))

    def _end(self, code):
        self.returncode = code
        self.out.put("")

    def poll(self):
        return self.returncode

    def wait(self, t=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.cmd, t)
        return self.returncode

    def kill(self):
        self._end(-9)


@pytest.fixture
def env(tmp_path, monkeypatch):
    wd = tmp_path / "workers"
    wd.mkdir()
    (wd / "echo.py").write_text("# the fake process answers instead\n")
    monkeypatch.setattr(worker, "WORKERS", str(wd))
    models = str(tmp_path / "models")
    _, exe = worker.python_of(model(worker="echo"), models)  # bin/python, or Scripts/python.exe on Windows
    os.makedirs(os.path.dirname(exe))
    open(exe, "w").close()
    return models


def model(**options):
    return Registry.from_dict(
        dict(
            roles=dict(speak=dict(model="m")),
            models=dict(m=dict(runtime="venv-worker", roles=["speak"], options=options)),
        )
    ).models["m"]


def load(env, **options):
    procs = []
    h = worker.load(model(**options), env, popen=lambda cmd, **kw: procs.append(FakeWorker(cmd, **kw)) or procs[-1])
    return h, procs


def test_loads_runs_and_unloads(env):
    h, procs = load(env, worker="echo", engine="kokoro")
    assert h["info"] == dict(engine="kokoro") and procs[0].cmd[-1].endswith("echo.py")
    r = worker.run(h, dict(text="hi"))
    assert r["echo"]["text"] == "hi" and r["echo"]["out_dir"].endswith(
        os.path.join("generated", r["echo"]["out_dir"][-10:])
    )
    with pytest.raises(RunError, match="bad text"):
        worker.run(h, dict(bad=1))  # the job's fault: the model stays loaded
    assert worker.run(h, dict(text="again"))["echo"]["text"] == "again"
    with pytest.raises(RuntimeError, match="out of memory"):
        worker.run(h, dict(crash=1))
    with pytest.raises(RuntimeError, match="not JSON"):
        worker.run(h, dict(junk=1))
    p = h["worker"].p
    worker.unload(h)
    assert p.poll() == 0 and "worker" not in h
    worker.unload(h)
    assert os.path.isfile(os.path.join(env, "venvs", "echo.log"))


def test_a_worker_that_dies_or_hangs_is_reported(env):
    h, _ = load(env, worker="echo")
    with pytest.raises(RuntimeError, match="has stopped"):
        worker.run(h, dict(die=1))
    with pytest.raises(RuntimeError, match="has stopped"):
        worker.run(h, dict(text="x"))
    worker.unload(h)
    h, _ = load(env, worker="echo")
    w = h["worker"]
    with pytest.raises(RuntimeError, match="more than 0.2 s"):
        w.ask(dict(op="run", params=dict(hang=1)), timeout=0.2)
    assert w.p.poll() is not None  # a hung worker is stopped


def test_a_worker_that_ignores_quit_is_killed(env):
    h, procs = load(env, worker="echo")
    procs[0]._answer = lambda m: None  # deaf from now on
    worker.unload(h)
    assert procs[0].returncode == -9


def test_refuses_what_cannot_load(env, tmp_path):
    with pytest.raises(RunError, match="names no worker"):
        worker.load(model(), env)
    with pytest.raises(RunError, match="no worker script"):
        worker.load(model(worker="nope"), env)
    with pytest.raises(RunError, match="not set up"):
        worker.load(model(worker="echo", venv="other"), env)
    with pytest.raises(RunError, match="could not load: cannot load"):
        load(env, worker="echo", engine="broken")
    assert worker.python_of(model(worker="echo", venv=str(tmp_path / "v")), env)[0] == str(tmp_path / "v")


def test_a_child_process_on_the_card_is_ours(monkeypatch):
    class Out:
        def __init__(self, out, code=0):
            self.stdout, self.returncode = out, code

    def smi(cmd, **k):
        return Out("RTX, 12288, 3000, 10, 50, 40, 170") if "--query-gpu" in cmd[1] else Out("10, 100\n11, 200\n12, 300")

    s = gpu.query(smi, own_pid=10, parent=lambda pid: 10 if int(pid) == 11 else 1)
    assert [p["ours"] for p in s["processes"]] == [True, True, False] and gpu.own_used_mb(s) == 300
    import io

    stats = {
        "/proc/42/stat": "42 (my (odd) name) S 7 42 42 0"
    }  # /proc/<pid>/stat: the parent is the 2nd field after the name

    def fake_open(path, *a, **k):
        if path not in stats:
            raise FileNotFoundError(path)  # no /proc (macOS, Windows) or no such process
        return io.StringIO(stats[path])

    monkeypatch.setattr(gpu, "open", fake_open, raising=False)
    assert gpu.parent_of(42) == 7 and gpu.parent_of(43) is None and gpu.parent_of("x") is None


def test_a_worker_gets_its_own_cuda_libraries_and_none_of_the_services(tmp_path, monkeypatch):
    import sys as _sys

    from stratamodels.registry import Model

    venv = tmp_path / "venvs" / "w"
    lib = venv / "lib" / "python3.12" / "site-packages" / "nvidia" / "cublas" / "lib"
    lib.mkdir(parents=True)
    m = Model(
        key="w", runtime="venv-worker", name="w", roles=("r",), vram_gb=1, options=dict(worker="tts", venv=str(venv))
    )
    exe = worker.python_of(m, str(tmp_path))[1]  # bin/python, or Scripts\python.exe on Windows
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    open(exe, "w").close()
    service_lib = os.path.join(_sys.prefix, "lib", "nvidia", "cu13", "lib")
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join([service_lib, "/opt/other/.venv-like/lib"]))
    seen = {}

    def popen(cmd, **kw):
        seen["env"] = kw["env"]
        return FakeWorker(cmd, **kw)

    m = Model(
        key="w", runtime="venv-worker", name="w", roles=("r",), vram_gb=1, options=dict(worker="tts", venv=str(venv))
    )
    worker.load(m, str(tmp_path), popen=popen)
    path = seen["env"]["LD_LIBRARY_PATH"].split(os.pathsep)
    assert path[0] == str(lib) and service_lib not in path and "/opt/other/.venv-like/lib" in path


def test_a_model_sets_how_long_a_job_may_run(env):
    # SwinIR on an 8 s clip takes ~40 min: the default half hour stopped every one of its jobs (2026-11-13)
    h, _ = load(env, worker="echo")
    assert h["timeout_s"] == worker.DEFAULT_TIMEOUT_S == 1800.0
    h2, _ = load(env, worker="echo", timeout_s=7200)
    seen = []
    real = h2["worker"].ask
    h2["worker"].ask = lambda msg, timeout=None: seen.append(timeout) or real(msg, timeout=timeout)
    assert worker.run(h2, dict(text="hi"))["echo"]["text"] == "hi" and seen == [7200.0]


def test_the_slow_video_models_have_room_to_finish():
    from stratamodels.registry import Registry

    m = Registry.load().models
    assert m["video-swinir-real-x4"].options["timeout_s"] >= 7200
    assert (
        m["video-seedvr2-3b"].options["timeout_s"] >= 7200
        and m["video-flashvsr-plus-base"].options["timeout_s"] >= 7200
    )
    assert (
        m["video-realplksr-nomos"].options["timeout_s"] >= 3600
        and m["video-lanczos-cleanup"].options["timeout_s"] >= 3600
    )
