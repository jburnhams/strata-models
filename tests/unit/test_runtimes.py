"""The runtime table and the fake runtime (stratamodels.runtimes). The real ones are tested on the GPU machine in tests/gpu."""

import importlib

import pytest

from stratamodels import runtimes
from stratamodels.registry import Model
from stratamodels.runtimes import fake


def test_get_and_available(monkeypatch):
    assert runtimes.get("fake") is fake
    with pytest.raises(KeyError, match="no runtime"):
        runtimes.get("vllm")
    real = importlib.import_module
    monkeypatch.setattr(
        importlib, "import_module", lambda n: (_ for _ in ()).throw(ImportError(n)) if n == "insightface" else real(n)
    )
    a = runtimes.available()
    assert a["fake"] == [] and "insightface" in a["insightface"]


def test_fake_loads_runs_fails_and_unloads():
    fake.LOADED.clear()
    m = Model(key="m", runtime="fake", name="m", roles=("r",), vram_gb=1, options=dict(a=1))
    h = fake.load(m, "/models")
    assert [h] == fake.LOADED and h["options"] == dict(a=1)
    assert fake.run(h, dict(seconds=0.001, x=1)) == dict(model="m", echo=dict(seconds=0.001, x=1))
    with pytest.raises(runtimes.RunError, match="bad"):
        fake.run(h, dict(fail="bad"))
    fake.unload(h)
    fake.unload(h)
    assert fake.LOADED == []
