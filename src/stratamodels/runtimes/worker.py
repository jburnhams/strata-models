"""Models that run in a Python of their own: a long-lived worker process in a separate virtual environment, spoken to in JSON lines.

Some models pin libraries the service cannot share (Chatterbox wants torch 2.6, Qwen3-TTS a newer transformers than the service's): each gets
its own venv, `<models>/venvs/<venv>` (made by `scripts/setup_worker_venv.sh NAME`), and a worker script from `stratamodels/workers/`
that needs nothing from this package. The worker loads the model once and keeps it on the card until the scheduler unloads it (the process
ends, so its memory is really freed; the worker's GPU memory counts as the service's own, see gpu.query).

models.toml options (besides the worker's own, passed through on load):
  worker   the script in stratamodels/workers/ (without .py), e.g. "tts"
  venv     the venv's folder name under <models>/venvs/, or an absolute path (default: the worker's name)

Protocol, one JSON object per line: the service writes {"op": "load", "model": <registry model as JSON>, "models_dir": ...} once, then
{"op": "run", "params": {...}} per job; the worker answers each with {"ok": true, "result": ...} or {"ok": false, "error": "...", "params":
true when the job's params were wrong (the model stays loaded)}. Anything the worker prints goes to stderr, which is kept in
<models>/venvs/<venv>.log, so stdout carries only answers.

Outputs: a job's files go where the runtime says (`out_dir` is added to the params: <data>/generated/<YYYY-MM-DD>/), so the web app can
serve them like generated pictures."""

import contextlib
import datetime
import json
import os
import subprocess
import threading

from stratamodels import settings
from stratamodels.runtimes import RunError

WORKERS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workers")


def python_of(model, models_dir):
    venv = model.options.get("venv") or model.options.get("worker")
    root = venv if os.path.isabs(str(venv)) else os.path.join(models_dir, "venvs", str(venv))
    exe = os.path.join(root, "Scripts", "python.exe") if os.name == "nt" else os.path.join(root, "bin", "python")
    return root, exe


def cuda_libs(root):
    """The venv's own NVIDIA library folders (nvidia/*/lib of the pip wheels), first on the worker's library path: libraries that load CUDA by
    name (CTranslate2, onnxruntime) find the venv's versions, never the service's (which may be another CUDA major version)."""
    import glob

    return sorted(glob.glob(os.path.join(root, "lib", "python3*", "site-packages", "nvidia", "*", "lib")))


class Worker:
    """One worker process. `ask` sends a request and waits for its answer (one at a time: the scheduler runs one job at a time)."""

    def __init__(self, cmd, log_path, env=None, popen=subprocess.Popen):
        self.log = open(log_path, "a", buffering=1)
        self.p = popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env
        )
        self.lock = threading.Lock()

    def ask(self, msg, timeout=1800):
        with self.lock:
            try:
                self.p.stdin.write(json.dumps(msg) + "\n")
                self.p.stdin.flush()
            except (BrokenPipeError, OSError):
                raise RuntimeError(f"the worker has stopped (exit {self.p.poll()}); its log is {self.log.name}")
            box = {}
            t = threading.Thread(target=lambda: box.setdefault("line", self.p.stdout.readline()), daemon=True)
            t.start()
            t.join(timeout)
            if t.is_alive():
                self.close()
                raise RuntimeError(f"the worker took more than {timeout} s; stopped")
            line = box.get("line") or ""
            if not line:
                raise RuntimeError(f"the worker has stopped (exit {self.p.poll()}); its log is {self.log.name}")
            try:
                return json.loads(line)
            except ValueError:
                raise RuntimeError(f"the worker answered something that is not JSON: {line[:200]!r}")

    def close(self):
        try:
            if self.p.poll() is None:
                try:
                    self.p.stdin.write(json.dumps(dict(op="quit")) + "\n")
                    self.p.stdin.flush()
                    self.p.wait(10)
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    self.p.kill()
                    self.p.wait(10)
        finally:
            for f in (self.p.stdin, self.p.stdout):
                with contextlib.suppress(OSError, ValueError):
                    f and f.close()  # a worker that died leaves a broken pipe: nothing to flush
            self.log.close()


def load(model, models_dir, popen=subprocess.Popen):
    name = model.options.get("worker")
    if not name:
        raise RunError(f"model {model.key} names no worker in models.toml (options.worker)")
    script = os.path.join(WORKERS, f"{name}.py")
    if not os.path.isfile(script):
        raise RunError(f"no worker script {script}")
    root, exe = python_of(model, models_dir)
    if not os.path.isfile(exe):
        raise RunError(
            f"the venv for {model.key} is not set up: no {exe} (run scripts/setup_worker_venv.sh {os.path.basename(root)})"
        )
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HOME=os.environ.get("HF_HOME") or os.path.join(models_dir, "hf"))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            *cuda_libs(root),
            *[p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p and ".venv" not in p],
        ]
    )
    w = Worker([exe, "-u", script], root + ".log", env, popen)
    r = w.ask(dict(op="load", model=model.to_json(), models_dir=models_dir))
    if not r.get("ok"):
        w.close()
        raise RunError(f"{model.key} could not load: {r.get('error')}")
    return dict(key=model.key, worker=w, info=r.get("result") or {})


def run(handle, params):
    out_dir = os.path.join(settings.data_dir(), "generated", datetime.date.today().isoformat())
    os.makedirs(out_dir, exist_ok=True)
    r = handle["worker"].ask(dict(op="run", params={**params, "out_dir": out_dir}))
    if r.get("ok"):
        return r.get("result")
    if r.get("params"):
        raise RunError(r.get("error") or "bad params")
    raise RuntimeError(
        r.get("error") or "the worker failed"
    )  # the scheduler unloads the model (ends the worker) and loads it fresh next time


def unload(handle):
    w = handle.pop("worker", None)
    if w:
        w.close()
