"""A model's `upsample` step (stratamodels.scheduler): a text job expands the prompt first, the picture job waits for it and uses its answer."""

import pytest

from stratamodels.registry import Registry, RegistryError


@pytest.fixture
def registry():
    """Overrides the shared fake registry: `pic` draws after its `upsample` step on `writer`; `plain` has no step; `lost` names a missing one."""
    return Registry.from_dict(
        dict(
            roles=dict(draw=dict(model="pic"), text=dict(model="writer")),
            models=dict(
                pic=dict(
                    runtime="fake",
                    roles=["draw"],
                    vram_gb=4,
                    options=dict(upsample=dict(model="writer", params=dict(upsampler=True))),
                ),
                plain=dict(runtime="fake", roles=["draw"], vram_gb=4),
                writer=dict(runtime="fake", roles=["text"], vram_gb=3),
                lost=dict(
                    runtime="fake", roles=["draw"], vram_gb=4, options=dict(upsample=dict(role="text", model="nope"))
                ),
            ),
        )
    )


@pytest.fixture
def sched(scheduler):
    return scheduler


def run_all(s):
    out = []
    while (r := s.step()) is not None:
        out.append(r)
    return out


def test_the_prompt_is_expanded_first_and_the_picture_uses_it(sched):
    j = sched.submit("draw", dict(prompt="a cat", seed=1), "app")
    from stratamodels.scheduler import AFTER

    step_id = j["params"][AFTER]
    step = sched.store.get(step_id)
    assert (
        step["role"] == "text" and step["model"] == "writer" and step["params"] == dict(upsampler=True, prompt="a cat")
    )
    done = run_all(sched)
    assert [d["id"] for d in done] == [step_id, j["id"]]  # the step first; the picture waits for it
    r = sched.store.get(j["id"])["result"]
    assert r["echo"] == dict(
        prompt="a detailed picture of a cat", seed=1
    )  # the runtime never sees the step's id or `upsample`
    assert (
        r["upsampled"]["prompt_in"] == "a cat"
        and r["upsampled"]["prompt"] == "a detailed picture of a cat"
        and r["upsampled"]["error"] is None
    )
    assert r["upsampled"]["job"] == step_id and r["upsampled"]["model"] == "writer"


def test_upsample_false_edits_and_models_without_a_step_skip_it(sched):
    from stratamodels.scheduler import AFTER

    for params, model in (
        (dict(prompt="a cat", upsample=False), None),
        (dict(prompt="a cat", ref_images=["/x.png"]), None),
        (dict(prompt="  "), None),
        (dict(prompt="a cat"), "plain"),
        (dict(seed=3), None),
    ):
        j = sched.submit("draw", params, "app", model=model)
        assert AFTER not in j["params"] and "upsample" not in j["params"]
    done = run_all(sched)
    assert len(done) == 5 and all("upsampled" not in d["result"] for d in done)
    assert done[0]["result"]["echo"] == dict(prompt="a cat")


def test_a_failed_or_silent_step_leaves_the_prompt_as_typed(sched):
    sched.registry.models["pic"].options["upsample"]["params"] = dict(upsampler=True, silent=True)
    j = sched.submit("draw", dict(prompt="a dog"), "app")
    run_all(sched)
    r = sched.store.get(j["id"])["result"]
    assert r["echo"]["prompt"] == "a dog" and "as typed" in r["upsampled"]["error"]
    sched.registry.models["pic"].options["upsample"]["params"] = dict(fail="no memory")
    j = sched.submit("draw", dict(prompt="a fox"), "app")
    run_all(sched)
    r = sched.store.get(j["id"])["result"]
    assert r["echo"]["prompt"] == "a fox" and "no memory" in r["upsampled"]["error"]


def test_the_picture_waits_while_its_step_is_queued(sched):
    j = sched.submit("draw", dict(prompt="a cat"), "app")
    assert sched._waiting_on_step(sched.store.get(j["id"])) and not sched._waiting_on_step(dict(params=None))
    run_all(sched)
    assert not sched._waiting_on_step(sched.store.get(j["id"]))


def test_a_step_on_a_model_that_is_not_there_is_refused_at_submit(sched):
    with pytest.raises(RegistryError):
        sched.submit("draw", dict(prompt="a cat"), "app", model="lost")


def test_evals_turn_a_models_own_step_off_unless_asked():
    from stratamodels.evals import core as C

    assert C.explicit(dict(prompt="x")) == dict(prompt="x", upsample=False) and C.explicit(
        dict(prompt="x", upsample=True)
    ) == dict(prompt="x", upsample=True)


def test_recovery_after_a_restart_finds_jobs_with_and_without_the_step(sched):
    """An eval run picks up its jobs by their params: `upsample` is dropped from the kept params and a job with its step carries `__after__`."""
    from stratamodels.evals import core as C

    eng = sched.store.engine
    on = sched.submit("draw", dict(prompt="a cat", upsample=True), "evals")
    off = sched.submit("draw", C.explicit(dict(prompt="a cat")), "evals")
    assert C.pending_job(eng, "draw", "pic", dict(prompt="a cat", upsample=True)) == on["id"]
    assert C.pending_job(eng, "draw", "pic", C.explicit(dict(prompt="a cat"))) == off["id"]
    assert C.pending_job(eng, "draw", "pic", dict(prompt="a cat")) == off["id"]  # off unless asked
    run_all(sched)
    assert C.done_job(eng, "draw", "pic", dict(prompt="a cat", upsample=True))["id"] == on["id"]
    assert C.done_job(eng, "draw", "pic", dict(prompt="a cat", upsample=False))["id"] == off["id"]
    assert C.done_job(eng, "draw", "pic", dict(prompt="a dog", upsample=False)) is None
