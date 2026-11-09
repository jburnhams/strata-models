"""Shared pytest configuration: every test is hermetic (its own home and data folder, no network, no stray override variables). Tests under tests/integration get the `integration` marker automatically."""

import os, socket, sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
TUNING = (
    "STRATAMODELS_MODELS",
    "STRATAMODELS_VRAM_RESERVE_GB",
    "STRATAMODELS_DATABASE_URL",
    "OMARCHY_DB_URL",
    "STRATA_MODELS_URL",
)


def pytest_collection_modifyitems(items):
    for item in items:
        if os.sep + "integration" + os.sep in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path_factory, monkeypatch):
    """`~/.strata-models`, the data and models folders point into a tmp dir (outside the test's own tmp_path), so nothing reads or writes the developer's real settings or data; override variables are removed."""
    base = tmp_path_factory.mktemp("env")
    monkeypatch.setenv("HOME", str(base / "home"))
    monkeypatch.setenv("USERPROFILE", str(base / "home"))
    monkeypatch.setenv("STRATAMODELS_HOME", str(base / "app-home"))
    monkeypatch.setenv("STRATAMODELS_DATA", str(base / "data"))
    for k in TUNING:
        monkeypatch.delenv(k, raising=False)
    return base


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """A test that opens an outbound connection fails at once (loopback and unix sockets are fine)."""
    real = socket.socket.connect

    def guarded(self, address, *a, **kw):
        host = address[0] if isinstance(address, tuple) else None
        if self.family in (socket.AF_INET, socket.AF_INET6) and host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"tests must not use the network (tried {address!r})")
        return real(self, address, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture
def fake_urlopen(monkeypatch):
    """Script `urllib.request.urlopen`: `fake_urlopen.reply({...}, http_error(429))`, then assert on `fake_urlopen.calls` (url, method, headers, parsed JSON body). Unexpected calls fail."""
    import urllib.request
    from fakes import FakeUrlopen

    f = FakeUrlopen()
    monkeypatch.setattr(urllib.request, "urlopen", f)
    return f


@pytest.fixture
def fake_popen(monkeypatch):
    """Replace `subprocess.Popen` everywhere: nothing starts; `fake_popen.instances` lists the commands (`.cmd`) and kwargs (`.kw`) it was asked to run."""
    import subprocess
    from fakes import FakePopen

    FakePopen.instances = []
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    return FakePopen


@pytest.fixture
def fake_run(monkeypatch):
    """Replace `subprocess.run` / `check_output`: record the command lines in `.calls`; set `.returns['cmd'] = b'...'` (substring of the command line) for output, `.code = 1` to fail."""
    import subprocess
    from fakes import FakeRun

    f = FakeRun()
    monkeypatch.setattr(subprocess, "run", f)
    monkeypatch.setattr(subprocess, "check_output", f.check_output)
    return f


class Clock:
    """A clock the test moves by hand: `clock.t += 5`."""

    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def registry():
    """A registry of fake models: role `echo` (model `small`, 2 GB; also `big`, 6 GB), role `other` (model `other`, 2 GB), and a known-but-unserved `planned`."""
    from stratamodels.registry import Registry

    return Registry.from_dict(
        dict(
            roles=dict(
                echo=dict(model="small", description="echo params"),
                other=dict(model="other"),
                later=dict(model="planned"),
            ),
            models=dict(
                small=dict(runtime="fake", roles=["echo"], vram_gb=2),
                big=dict(runtime="fake", roles=["echo"], vram_gb=6),
                other=dict(runtime="fake", roles=["other"], vram_gb=2),
                planned=dict(runtime="llama-server", roles=["later"], used_by=["strata-360"]),
            ),
        )
    )


class FakeGpu:
    """A 12 GB card whose memory follows the fake models loaded by this process (their registry vram_gb), plus `other_mb` held by other processes."""

    def __init__(self, registry, total_mb=12288.0, other_mb=500.0, pid=4242):
        self.registry, self.total_mb, self.other_mb, self.pid, self.present, self.extra = (
            registry,
            total_mb,
            other_mb,
            pid,
            True,
            [],
        )  # extra: [(pid, MB)] of lease holders

    def __call__(self):
        if not self.present:
            return None
        from stratamodels.runtimes import fake

        own = sum(self.registry.models[h["model"]].vram_gb * 1024 for h in fake.LOADED)
        procs = [dict(pid=self.pid, used_mb=own, command="python -m stratamodels serve", ours=True)] if own else []
        procs.append(dict(pid=1, used_mb=self.other_mb, command="build_arena.py", ours=False))
        procs += [dict(pid=p, used_mb=mb, command="lease holder", ours=False) for p, mb in self.extra]
        used = own + self.other_mb + sum(mb for _, mb in self.extra)
        return dict(
            name="Fake RTX",
            total_mb=self.total_mb,
            used_mb=used,
            free_mb=self.total_mb - used,
            util_pct=10.0,
            temp_c=50.0,
            power_w=40.0,
            power_limit_w=170.0,
            processes=procs,
        )


@pytest.fixture
def scheduler(registry, clock, tmp_path):
    """A Scheduler over an in-memory store, the fake registry and a fake 12 GB GPU (`scheduler.probe`), with a hand-moved clock. Call `.step()` to work."""
    from stratamodels.runtimes import fake
    from stratamodels.scheduler import Scheduler
    from stratamodels.store import memory_store

    fake.LOADED.clear()
    s = Scheduler(
        memory_store(clock), registry, str(tmp_path / "models"), probe=FakeGpu(registry), reserve_gb=1.0, clock=clock
    )
    yield s
    fake.LOADED.clear()


TEXT = "the quick brown fox jumps over the lazy dog again and again"


@pytest.fixture
def setup(scheduler, tmp_path, monkeypatch):
    """A dataset of 40 items, and a spec with a baseline that drops every 4th word, a better candidate (every 12th), and the same model again."""
    from stratamodels.evals import core as C

    monkeypatch.setattr(C, "REPO_SPECS", str(tmp_path / "none"))
    items = [
        dict(
            id=f"u{i}",
            input=dict(say=TEXT + f" item{i}"),
            reference=dict(text=TEXT + f" item{i}"),
            tags=dict(speaker=str(i % 4)),
        )
        for i in range(40)
    ]
    ds = C.write_dataset("speech", items, description="forty fake utterances\nfor tests")
    specs = os.path.join(os.environ["STRATAMODELS_DATA"], "evals", "specs")
    os.makedirs(specs, exist_ok=True)
    open(os.path.join(specs, "fake-asr.toml"), "w").write("""name = "fake-asr"
question = "Which fake transcriber should serve echo?"
role = "echo"
dataset = "speech"
metrics = ["wer", "cer", "latency_s", "failed"]
judge = true
[decision]
primary = "wer"
baseline = "base"
min_effect = 0.02
guards = { latency_s = 1000.0 }
[[candidates]]
name = "base"
model = "small"
params = { drop = 4 }
[[candidates]]
name = "better"
model = "big"
params = { drop = 12 }
[[candidates]]
name = "same"
model = "small"
params = { drop = 4 }
options = { tweak = 1 }
""")
    return ds
