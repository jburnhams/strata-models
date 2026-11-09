"""Integration-suite fixtures: the CLI (in-process and as a real process) and the HTTP API over real on-disk files."""
import contextlib, io, os, subprocess, sys
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class Result:
    stdout: str; stderr: str; code: int
    @property
    def out(self): return self.stdout + self.stderr


@pytest.fixture
def cli():
    """`cli('hello', 'World', '--json')` -> Result(stdout, stderr, code), in this process. Raises AssertionError with the output on a non-zero exit unless `check=False`."""
    from app import cli as C

    def run(*args, check=True):
        out, err = io.StringIO(), io.StringIO(); code = 0
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try: code = C.main([str(a) for a in args]) or 0
            except SystemExit as e: code = e.code if isinstance(e.code, int) else (1 if e.code else 0); err.write(str(e.code) + '\n' if isinstance(e.code, str) else '')
        r = Result(out.getvalue(), err.getvalue(), code)
        if check and code: raise AssertionError(f'app {" ".join(map(str, args))} exited {code}:\n{r.out}')
        return r
    return run


@pytest.fixture
def cli_subprocess(isolated_env):
    """The same through a real `python -m app` process: for the entry-point smoke test and anything that must not share this process's state."""
    def run(*args, check=True):
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, 'src'), PYTHONWARNINGS='ignore')
        p = subprocess.run([sys.executable, '-m', 'app', *map(str, args)], capture_output=True, text=True, env=env, timeout=300)
        r = Result(p.stdout, p.stderr, p.returncode)
        if check and p.returncode: raise AssertionError(f'exited {p.returncode}:\n{r.out}')
        return r
    return run


@pytest.fixture
def client():
    """The HTTP API over the real on-disk data and settings files."""
    from app.server.app import create_app
    return TestClient(create_app())
