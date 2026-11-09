"""Integration-suite fixtures: the CLI in-process, and the real service as a separate process that the client package talks to over HTTP."""
import contextlib, io, os, socket, subprocess, sys, time, urllib.request
from dataclasses import dataclass

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'client'))
FAKE_ROLES = '''
[roles.echo]
description = "echo the params back (fake runtime)"
model = "echo-small"
[models.echo-small]
runtime = "fake"
roles = ["echo"]
vram_gb = 0.5
'''


@dataclass
class Result:
    stdout: str; stderr: str; code: int
    @property
    def out(self): return self.stdout + self.stderr


@pytest.fixture
def cli():
    """`cli('registry', '--json')` -> Result(stdout, stderr, code), in this process."""
    from stratamodels import cli as C

    def run(*args, check=True):
        out, err = io.StringIO(), io.StringIO(); code = 0
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try: code = C.main([str(a) for a in args]) or 0
            except SystemExit as e: code = e.code if isinstance(e.code, int) else (1 if e.code else 0); err.write(str(e.code) + '\n' if isinstance(e.code, str) else '')
        r = Result(out.getvalue(), err.getvalue(), code)
        if check and code: raise AssertionError(f'strata-models {" ".join(map(str, args))} exited {code}:\n{r.out}')
        return r
    return run


def free_port():
    with socket.socket() as s: s.bind(('127.0.0.1', 0)); return s.getsockname()[1]


@pytest.fixture
def service(isolated_env, monkeypatch):
    """`strata-models serve --memory` as a real process on a free port, with an `echo` role on the fake runtime added through the data folder's
    models.toml; STRATA_MODELS_URL points the client at it. Yields the base URL."""
    data = os.environ['STRATAMODELS_DATA']; os.makedirs(data, exist_ok=True)
    with open(os.path.join(data, 'models.toml'), 'w', encoding='utf8') as f: f.write(FAKE_ROLES)
    port = free_port(); base = f'http://127.0.0.1:{port}'
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, 'src'), PYTHONWARNINGS='ignore')
    p = subprocess.Popen([sys.executable, '-m', 'stratamodels', 'serve', '--memory', '--port', str(port)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 60
    while True:
        try: urllib.request.urlopen(base + '/api/health', timeout=1); break
        except OSError:
            if p.poll() is not None or time.time() > deadline: p.kill(); raise AssertionError('the service did not start:\n' + (p.stdout.read() if p.stdout else ''))
            time.sleep(0.1)
    monkeypatch.setenv('STRATA_MODELS_URL', base)
    yield base
    p.terminate()
    try: p.wait(10)
    except subprocess.TimeoutExpired: p.kill()
