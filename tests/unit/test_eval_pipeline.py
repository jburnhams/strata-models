"""Pipeline candidates (stages before a candidate's own job, stratamodels.evals.core): each stage cached on its own, the trace kept with the end
result, a failed stage fails the output, judges see only the end result, the Outputs API and the write-up show every step."""

import os

import pytest
from fastapi.testclient import TestClient

from stratamodels.evals import core as C
from stratamodels.evals import report as R
from stratamodels.server.app import create_app

SPEC = """name = "fake-pipe"
question = "Does expanding the prompt first help?"
role = "echo"
dataset = "prompts"
metrics = ["judged", "latency_s", "failed"]
[decision]
primary = "judged"
baseline = "raw"
min_effect = 0.1
[[candidates]]
name = "raw"
model = "small"
[[candidates]]
name = "plain+small"
model = "small"
[[candidates.stages]]
role = "echo"
model = "big"
input = "{prompt}"
send = "expand"
into = "prompt"
[[candidates]]
name = "plain+big"
model = "big"
[[candidates.stages]]
role = "echo"
model = "big"
send = "expand"
[[candidates]]
name = "two-step"
model = "small"
[[candidates.stages]]
role = "echo"
model = "big"
send = "expand"
params = { style = "rich" }
[[candidates.stages]]
role = "echo"
input = "again {prompt} ({seed})"
send = "expand"
params = { style = "short" }
"""


@pytest.fixture
def pipe(setup):
    items = [dict(id=f"p{i}", input=dict(prompt=f"a cat number {i}", seed=i)) for i in range(6)]
    C.write_dataset("prompts", items, description="six prompts")
    open(os.path.join(os.environ["STRATAMODELS_DATA"], "evals", "specs", "fake-pipe.toml"), "w").write(SPEC)
    return C.load_spec("fake-pipe")


def test_stages_are_parsed_and_checked(pipe):
    st = pipe.candidate("two-step").stages
    assert (
        len(st) == 2 and st[1]["model"] is None and st[0]["params"] == dict(style="rich") and st[0]["into"] == "prompt"
    )
    assert (
        "stages" not in pipe.candidate("raw").to_json()
        and pipe.candidate("plain+small").to_json()["stages"][0]["send"] == "expand"
    )
    base = dict(name="x", question="q", role="r", dataset="d", metrics=["failed"])
    with pytest.raises(C.EvalError, match="needs a role"):
        C.parse_spec(dict(base, candidates=[dict(name="a", model="m", stages=[dict(model="m")])]))
    with pytest.raises(C.EvalError, match="unknown fields"):
        C.parse_spec(dict(base, candidates=[dict(name="a", model="m", stages=[dict(role="r", prompt="x")])]))
    with pytest.raises(C.EvalError, match="does not have"):
        C.stage_params(dict(input="{nope}", params={}, send="prompt"), dict(prompt="x"))
    with pytest.raises(C.EvalError, match="stage input"):
        C.stage_params(dict(input="{", params={}, send="prompt"), dict(prompt="x"))


def test_a_stageless_candidate_keeps_its_key(pipe):
    """Adding stages to the spec format must not change the keys of the outputs already made (they would all run again)."""
    it = dict(id="p0", input=dict(prompt="a cat number 0", seed=0))
    raw = pipe.candidate("raw")
    assert C.result_key(pipe, raw, it) == C._h(
        C._canon(dict(role="echo", model="small", options={}, params=dict(prompt="a cat number 0", seed=0), item="p0")),
        32,
    )
    assert C.result_key(pipe, pipe.candidate("plain+small"), it) != C.result_key(pipe, raw, it)


def test_stage_text_reads_text_models_and_transcripts():
    assert C.stage_text(dict(text='  "a detailed cat"  ')) == "a detailed cat" and C.stage_text("'x'") == "x"
    assert C.stage_text(dict(segments=[dict(text=" hello "), dict(text="there")])) == "hello there"
    with pytest.raises(C.EvalError, match="no text"):
        C.stage_text(dict(echo=1))
    with pytest.raises(C.EvalError, match="empty"):
        C.stage_text(dict(text='""'))


def test_pipelines_run_stage_by_stage_cached_and_shared(pipe, scheduler):
    engine = scheduler.store.engine
    stop = scheduler.start()
    log = []
    try:
        r = C.run(pipe, engine, C.SchedulerExecutor(scheduler), log=log.append)
        assert r == dict(ran=24, cached=0, failed=0)
        assert any(
            "stage 1: 0 of 12 cached, 12 to run" in m for m in log
        )  # plain+small and plain+big share their first step: 6 jobs, not 12
        assert any("stage 2" in m for m in log)
        ds, outs, _runs = C.outputs_for(pipe, engine)
        o = outs["plain+small"][0][0]
        trace = o["pipeline"]
        assert (
            o["echo"]["prompt"] == "detailed: a cat number 0" and o["echo"]["seed"] == 0
        )  # the stage's answer replaced the prompt
        assert (
            [t["role"] for t in trace] == ["echo", "echo"]
            and trace[0]["sent"] == "a cat number 0"
            and trace[0]["answer"] == "detailed: a cat number 0"
        )
        assert (
            trace[0]["params"] == {} and trace[-1]["sent"] == "detailed: a cat number 0" and trace[0]["model"] == "big"
        )
        two = outs["two-step"][3][0]
        assert two["echo"]["prompt"] == "short: again rich: a cat number 3 (3)" and len(two["pipeline"]) == 3
        assert "pipeline" not in outs["raw"][0][0]
        assert C.run(pipe, engine, C.SchedulerExecutor(scheduler), log=log.append)["ran"] == 0
        assert C.forget(pipe, engine, "plain+big") == 6  # its own outputs only: the shared stage is kept
        log.clear()
        assert C.run(pipe, engine, C.SchedulerExecutor(scheduler), log=log.append)["ran"] == 6 and not any(
            "stage 1:" in m and "to run" in m for m in log
        )
        assert (
            C.forget(pipe, engine, "plain+big", stages=True) == 12
            and C.run(pipe, engine, C.SchedulerExecutor(scheduler))["ran"] == 6
        )
        res = C.analyse(pipe, engine)
        assert res["dataset"]["compared"] == 6
        text = R.render(res, None, date="2026-11-09", pipelines=R.pipelines(pipe, outs, ds.items))
        assert (
            "## Pipelines" in text
            and "sent: a cat number 0" in text
            and "answered: detailed: a cat number 0" in text
            and "`big` → `small`" in text
            and "step 2: role `echo`" in text
        )
        p = R.pipelines(pipe, outs, ds.items)
        assert (
            p["plain+small"]["words_in"] == 4
            and p["plain+small"]["words_out"] == 5
            and len(p["two-step"]["examples"]) == 2
            and "raw" not in p
        )
    finally:
        stop.set()


def test_a_failed_stage_fails_the_output_without_running_the_job(pipe, scheduler):
    engine = scheduler.store.engine
    stop = scheduler.start()
    try:
        d = C.load_spec("fake-pipe").to_json()
        d["candidates"] = [
            d["candidates"][0],
            dict(
                name="bad", model="small", stages=[dict(role="echo", send="expand", params=dict(fail="no text model"))]
            ),
            dict(name="notext", model="small", stages=[dict(role="echo", send="q")]),
            dict(name="missing", model="small", stages=[dict(role="echo", input="{nope}")]),
        ]
        s = C.parse_spec(d)
        r = C.run(s, engine, C.SchedulerExecutor(scheduler))
        assert r["failed"] == 18
        ds, outs, _ = C.outputs_for(s, engine)
        assert (
            "no text model" in outs["bad"][0][1]
            and outs["bad"][0][0] is None
            and "stage 1" in outs["notext"][0][1]
            and "does not have" in outs["missing"][0][1]
        )
        assert R.pipelines(s, outs, ds.items)["bad"]["failed"] == 6
        assert C.run(s, engine, C.SchedulerExecutor(scheduler))["ran"] == 0  # failures are cached like any output...
        assert (
            C.forget(s, engine, "bad") == 6 and C.run(s, engine, C.SchedulerExecutor(scheduler))["ran"] == 6
        )  # ...forgetting the output alone reuses the failed stage
        assert (
            C.stage_keys(engine, s.candidate("bad"), ds.items[0])
            and C.stage_keys(engine, s.candidate("missing"), ds.items[0]) == []
        )
        log = []
        assert (
            C.forget(s, engine, "bad", stages=True) == 12
        )  # with stages=True the failed stage goes too, and runs again
        C.run(s, engine, C.SchedulerExecutor(scheduler), candidates=["bad"], log=log.append)
        assert any("stage 1: 0 of 6 cached, 6 to run" in m for m in log)
    finally:
        stop.set()


def test_judges_see_the_end_result_only_and_the_outputs_show_every_step(pipe, scheduler):
    stop = scheduler.start()
    client = TestClient(create_app(scheduler))
    try:
        C.run(pipe, scheduler.store.engine, C.SchedulerExecutor(scheduler))
        seen = set()
        for _ in range(12):
            p = client.get("/api/evals/fake-pipe/pair").json()["pair"]
            assert "pipeline" not in p["x"] and "pipeline" not in p["y"] and "detailed" not in p["token"]
            j = client.post("/api/evals/fake-pipe/judgments", json=dict(token=p["token"], winner="x")).json()
            assert j["item_id"] == p["item_id"]
            seen |= {j["x"], j["y"]}
        assert "raw" in seen  # judged head to head with the baseline
        o = client.get("/api/evals/fake-pipe/outputs", params=dict(item_id="p1")).json()
        assert (
            [c["name"] for c in o["candidates"]][:2] == ["raw", "plain+small"]
            and len(o["items"]) == 1
            and o["items"][0]["input"]["prompt"] == "a cat number 1"
        )
        step = o["items"][0]["outputs"]["plain+small"]["output"]["pipeline"][0]
        assert step["answer"] == "detailed: a cat number 1" and o["items"][0]["outputs"]["raw"]["error"] is None
        assert (
            len(client.get("/api/evals/fake-pipe/outputs").json()["items"]) == 6
            and client.get("/api/evals/fake-pipe/outputs", params=dict(item_id="zz")).status_code == 404
        )
        r = client.get("/api/evals/fake-pipe/report").text
        assert "## Pipelines" in r
    finally:
        stop.set()
