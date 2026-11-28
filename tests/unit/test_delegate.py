"""runtimes/delegate.py: a model of the registry asked in this process, or as a job on the running service (the client played by a stand-in
module: unit tests make no requests)."""

import sys
import types

import pytest

from stratamodels.runtimes import RunError, delegate


def test_via_is_service_or_nothing():
    with pytest.raises(RunError, match="via is 'service'"):
        delegate.caller("m", "", "carrier-pigeon")


def test_in_process_loads_the_registry_model_and_runs_it(tmp_path, monkeypatch):
    from PIL import Image

    from stratamodels import settings

    monkeypatch.delenv("STRATAMODELS_DELEGATE_VIA", raising=False)
    monkeypatch.setattr(settings, "engine", lambda: None)  # the registry from models.toml, not a database
    run, unload = delegate.caller("image-kind-rules", str(tmp_path))
    p = tmp_path / "x.png"
    Image.new("RGB", (64, 64), (236, 226, 200)).save(p)
    assert run({"image": str(p)})["kind"] == "blank"
    unload()
    with pytest.raises(RunError, match="not in the registry"):
        delegate.caller("no-such-model", str(tmp_path))


def test_via_service_expands_the_preset_and_queues_a_vision_job(monkeypatch):
    sent = []
    sm = types.ModuleType("strata_models_client")
    sm.run = lambda role, params, app, priority, model: (
        sent.append((role, params, app, priority, model)) or {"text": "ok"}
    )  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "strata_models_client", sm)
    monkeypatch.setenv("STRATAMODELS_DELEGATE_VIA", "service")
    run, unload = delegate.caller("claude-haiku-5-5-api", "")
    assert run({"preset": "image-kind-v1", "context": "Punch, 1828", "image": "/x.png"}) == {"text": "ok"}
    unload()
    role, q, app, prio, model = sent[0]
    assert (role, app, prio, model) == ("vision-language", "evals", "low", "claude-haiku-5-5-api")
    assert "Punch, 1828" in q["prompt"] and "preset" not in q and "context" not in q and q["schema"]["type"] == "object"
    assert q["image"] == "/x.png"
    run({"prompt": "as it is"})
    assert sent[1][1] == {"prompt": "as it is"}


def test_a_gpu_model_is_never_loaded_in_a_runtime(monkeypatch):
    from stratamodels import settings

    monkeypatch.delenv("STRATAMODELS_DELEGATE_VIA", raising=False)
    monkeypatch.setattr(settings, "engine", lambda: None)
    with pytest.raises(RunError, match="needs the GPU .* only as a job on the service"):
        delegate.caller("qwen3.5-9b-vl", "")


def test_a_model_with_a_batch_api_is_always_asked_as_a_job(monkeypatch):
    """Claude is never called from inside a runtime: its job goes to the batcher (half price)."""
    from stratamodels import settings

    sent = []
    sm = types.ModuleType("strata_models_client")
    sm.run = lambda role, params, app, priority, model: sent.append((priority, model)) or {"text": "ok"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "strata_models_client", sm)
    monkeypatch.delenv("STRATAMODELS_DELEGATE_VIA", raising=False)
    monkeypatch.setattr(settings, "engine", lambda: None)
    run, _ = delegate.caller("claude-haiku-5-5-api", "")
    assert run({"prompt": "hi"}) == {"text": "ok"} and sent == [("low", "claude-haiku-5-5-api")]


def test_load_options_make_a_variant_the_service_loads(monkeypatch):
    sent = []
    sm = types.ModuleType("strata_models_client")
    sm.submit_many = lambda role, items, app, priority, model, options: sent.append((model, options, items)) or [9]  # type: ignore[attr-defined]
    sm.wait = lambda jid: {"text": f"job {jid}"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "strata_models_client", sm)
    run, _ = delegate.caller("qwen3.5-9b-vl", "", "service", {"ctx": 32768})
    assert run({"prompt": "draw"}) == {"text": "job 9"} and sent == [
        ("qwen3.5-9b-vl", {"ctx": 32768}, [{"prompt": "draw"}])
    ]
    monkeypatch.delenv("STRATAMODELS_DELEGATE_VIA", raising=False)
    from stratamodels import settings

    monkeypatch.setattr(settings, "engine", lambda: None)
    with pytest.raises(RunError, match="a variant the service loads"):
        delegate.caller("image-kind-rules", "", None, {"engine": "clip"})
