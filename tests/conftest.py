"""Shared pytest configuration: every test is hermetic (its own home and data folder, no network, no stray override variables). Tests under tests/integration get the `integration` marker automatically."""
import os, socket, sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
TUNING = ('APP_GREETING',)


def pytest_collection_modifyitems(items):
    for item in items:
        if os.sep + 'integration' + os.sep in str(item.fspath): item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path_factory, monkeypatch):
    """`~/.app` and the data folder point into a tmp dir (outside the test's own tmp_path), so nothing reads or writes the developer's real settings or data; override variables are removed."""
    base = tmp_path_factory.mktemp('env')
    monkeypatch.setenv('HOME', str(base / 'home')); monkeypatch.setenv('USERPROFILE', str(base / 'home'))
    monkeypatch.setenv('APP_HOME', str(base / 'app-home')); monkeypatch.setenv('APP_DATA', str(base / 'data'))
    for k in TUNING: monkeypatch.delenv(k, raising=False)
    return base


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """A test that opens an outbound connection fails at once (loopback and unix sockets are fine)."""
    real = socket.socket.connect

    def guarded(self, address, *a, **kw):
        host = address[0] if isinstance(address, tuple) else None
        if self.family in (socket.AF_INET, socket.AF_INET6) and host not in ('127.0.0.1', '::1', 'localhost'): raise AssertionError(f'tests must not use the network (tried {address!r})')
        return real(self, address, *a, **kw)
    monkeypatch.setattr(socket.socket, 'connect', guarded)


@pytest.fixture
def fake_urlopen(monkeypatch):
    """Script `urllib.request.urlopen`: `fake_urlopen.reply({...}, http_error(429))`, then assert on `fake_urlopen.calls` (url, method, headers, parsed JSON body). Unexpected calls fail."""
    import urllib.request
    from fakes import FakeUrlopen
    f = FakeUrlopen(); monkeypatch.setattr(urllib.request, 'urlopen', f); return f


@pytest.fixture
def fake_popen(monkeypatch):
    """Replace `subprocess.Popen` everywhere: nothing starts; `fake_popen.instances` lists the commands (`.cmd`) and kwargs (`.kw`) it was asked to run."""
    import subprocess
    from fakes import FakePopen
    FakePopen.instances = []; monkeypatch.setattr(subprocess, 'Popen', FakePopen); return FakePopen


@pytest.fixture
def fake_run(monkeypatch):
    """Replace `subprocess.run` / `check_output`: record the command lines in `.calls`; set `.returns['cmd'] = b'...'` (substring of the command line) for output, `.code = 1` to fail."""
    import subprocess
    from fakes import FakeRun
    f = FakeRun(); monkeypatch.setattr(subprocess, 'run', f); monkeypatch.setattr(subprocess, 'check_output', f.check_output); return f
