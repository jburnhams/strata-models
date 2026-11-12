"""Experiments decided by people (primary metric `judged`) with a [score] pass that reads each output with a second model (text to speech:
transcribe the audio back for tts_wer). Fake runtime: `speak` makes a /fake/ path that spells the words; role `other` hears them back."""

import os

import pytest

from stratamodels.db.models import EvalJudgment
from stratamodels.evals import core as C
from stratamodels.evals import metrics as M

SPEC = """name = "fake-tts"
question = "Which fake voice should speak?"
role = "echo"
dataset = "lines"
metrics = ["judged", "tts_wer", "rtf", "failed"]
[decision]
primary = "judged"
baseline = "base"
min_effect = 0.1
guards = { tts_wer = 2.0 }
[score]
role = "other"
params = { drop = 0 }
[[candidates]]
name = "base"
model = "small"
params = {}
[[candidates]]
name = "liked"
model = "big"
params = {}
[[candidates]]
name = "meh"
model = "small"
params = { seconds = 0 }
"""


@pytest.fixture
def tts(scheduler, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "REPO_SPECS", str(tmp_path / "none"))
    C.write_dataset(
        "lines",
        [
            dict(
                id=f"n{i}",
                input=dict(speak=f"line number {i} of the race"),
                reference=dict(text=f"line number {i} of the race"),
            )
            for i in range(30)
        ],
    )
    specs = os.path.join(os.environ["STRATAMODELS_DATA"], "evals", "specs")
    os.makedirs(specs, exist_ok=True)
    open(os.path.join(specs, "fake-tts.toml"), "w").write(SPEC)
    return C.load_spec("fake-tts")


def judge(engine, spec, a, b, winner, n):
    ds = C.load_dataset(spec.dataset)
    with C.session(engine) as s:
        for it in ds.items[:n]:
            s.add(
                EvalJudgment(
                    experiment=spec.name,
                    item_id=it["id"],
                    a=a,
                    b=b,
                    winner=winner,
                    rater="t",
                    at=0.0,
                    note=None,
                    a_key=C.result_key(spec, spec.candidate(a), it),
                    b_key=C.result_key(spec, spec.candidate(b), it),
                )
            )
        s.commit()


def test_spec_with_score_and_judged(tts):
    assert (
        tts.judge and tts.score == dict(role="other", params=dict(drop=0)) and tts.to_json()["score"]["role"] == "other"
    )
    with pytest.raises(C.EvalError, match="names no role"):
        C.parse_spec(
            dict(
                name="x",
                question="q",
                role="r",
                dataset="d",
                metrics=["wer"],
                score=dict(params={}),
                candidates=[dict(name="a", model="m")],
            )
        )
    assert C.parse_spec(
        dict(
            name="x",
            question="q",
            role="r",
            dataset="d",
            metrics=["wer"],
            decision=dict(primary="judged"),
            candidates=[dict(name="a", model="m")],
        )
    ).judge


def test_score_pass_reads_outputs_once_and_judgments_decide(tts, scheduler):
    engine = scheduler.store.engine
    stop = scheduler.start()
    log = []
    try:
        r = C.run(tts, engine, C.SchedulerExecutor(scheduler), log=log.append)
        assert r == dict(ran=90, cached=0, failed=0, scored=90) and any("scoring 90" in m for m in log)
        assert (
            C.run(tts, engine, C.SchedulerExecutor(scheduler), log=log.append)["scored"] == 0
        )  # read once per scoring config
        first = C.analyse(tts, engine)["candidates"]["base"]["metrics"]["tts_wer"]["value"]
        tts.score = dict(tts.score, params=dict(drop=3))
        assert C.run(tts, engine, C.SchedulerExecutor(scheduler), log=log.append)["scored"] == 90
        tts.score = dict(tts.score, params=dict(drop=0))
        assert (
            C.run(tts, engine, C.SchedulerExecutor(scheduler), log=log.append)["scored"] == 0
        )  # both readings kept apart
        assert (
            C.analyse(tts, engine)["candidates"]["base"]["metrics"]["tts_wer"]["value"] == first
        )  # each config sees its own
        tts.score = dict(tts.score, params=dict(drop=3))
        res = C.analyse(tts, engine)
        assert (
            res["candidates"]["base"]["metrics"]["tts_wer"]["value"] == pytest.approx(10 / 30 * 2 / 2, abs=0.1)
            and res["candidates"]["base"]["metrics"]["rtf"]["value"] > 0
        )
        assert (
            res["decision"]["per_candidate"]["liked"]["verdict"] == "unclear"
            and "judged pairs with the baseline" in res["decision"]["per_candidate"]["liked"]["why"]
        )
        judge(engine, tts, "liked", "base", "liked", 25)
        judge(engine, tts, "base", "liked", "tie", 2)
        judge(engine, tts, "meh", "base", "base", 12)
        judge(engine, tts, "meh", "liked", "meh", 3)
        res = C.analyse(tts, engine)
        c = res["comparisons"]["liked"]["judged"]
        assert (
            (c["wins"], c["losses"], c["ties"], c["n"]) == (25, 0, 2, 27)
            and c["improves"] is True
            and c["p"] < 0.001
            and "p_holm" in c
        )
        assert (
            res["comparisons"]["meh"]["judged"]["improves"] is False
            and res["decision"]["per_candidate"]["meh"]["verdict"] == "reject"
        )
        assert (
            res["decision"]["recommendation"] == "liked"
            and res["decision"]["per_candidate"]["liked"]["verdict"] == "adopt"
        )
        assert (
            res["candidates"]["meh"]["metrics"]["judged"]["n"] == 15
            and res["candidates"]["liked"]["metrics"]["judged"]["value"] < 1
        )  # lost to meh 3 times
    finally:
        stop.set()


def test_tts_metrics_score_failures_and_unread_outputs():
    it = dict(input=dict(text="one two three"))
    assert (
        M.METRICS["tts_wer"].score(it, None, "boom") == (3, 3)
        and M.METRICS["tts_wer"].score(it, dict(path="/x.wav"), None) is None
    )
    assert M.METRICS["tts_wer"].score(it, dict(scored=dict(error="x")), None) == (3, 3)
    assert (
        C.readings(dict(scored=dict(key="k1", error="x"), scores=dict(k2=dict(result=1))))
        == dict(k1=dict(error="x"), k2=dict(result=1))
        and C.readings(None) == {}
    )
    assert C.scored_view(
        C.parse_spec(
            dict(name="x", question="q", role="r", dataset="d", metrics=["wer"], candidates=[dict(name="a", model="m")])
        ),
        dict(a=1),
    ) == dict(a=1)
    assert M.METRICS["tts_wer"].score(
        it, dict(scored=dict(result=dict(segments=[dict(text="One, two tree.")]))), None
    ) == (1, 3)
    assert M.METRICS["tts_wer"].score(dict(input={}), {}, None) is None
    assert (
        M.METRICS["rtf"].score(it, dict(duration_s=2.0, seconds=1.0), None) == 0.5
        and M.METRICS["rtf"].score(it, dict(seconds=1.0), None) is None
        and M.METRICS["rtf"].score(it, None, "x") is None
    )


def test_judged_head_to_head_counts():
    j = [("a", "b", "a"), ("b", "a", "tie"), ("a", "c", "c"), ("b", "c", "b")]
    assert C._tally(j, "a") == (1, 1, 1) and C._tally(j, "a", "b") == (1, 0, 1)
    v = C.judged_vs(j, "a", "b")
    assert v["n"] == 2 and v["diff"] == pytest.approx(0.25) and v["improves"] is None and v["p"] == 1.0
    assert C.judged_vs([], "a", "b")["improves"] is None and C.judged_share([], "a")["n"] == 0


def test_score_takes_params_from_the_item(tts, scheduler):
    """[score] with = {param = "item.field"}: the reading of an output gets a field of its own item (a clean reference recording, here the
    fake listener's `drop`); a field the item lacks is left out, and a with that is not a table of fields is refused."""
    assert C.item_field(dict(reference=dict(audio="/c.flac")), "reference.audio") == "/c.flac"
    assert C.item_field(dict(reference="x"), "reference.audio") is None and C.item_field({}, "a.b") is None
    sp = dict(tts.score, **{"with": dict(drop="tags.drop", reference="reference.audio")})
    tts.score = sp
    assert C.score_params(tts, dict(path="/o.wav"), dict(tags=dict(drop=2))) == dict(drop=2, path="/o.wav")
    assert C.score_params(tts, dict(path="/o.wav"), None) == dict(drop=0, path="/o.wav")

    class Spy(C.SchedulerExecutor):
        sent = []

        def submit(self, role, params, model, options):
            if role == "other":
                Spy.sent.append(params)
            return super().submit(role, params, model, options)

    stop = scheduler.start()
    try:
        r = C.run(tts, scheduler.store.engine, Spy(scheduler), limit=3, log=lambda m: None)
        assert r["scored"] == 9 and all(
            p["drop"] == 0 and "reference" not in p for p in Spy.sent
        )  # the items have no tags.drop
        # a field that changes after the reading (a corrected reference) reads the output again; unchanged, it does not
        tts.score = dict(sp, **{"with": dict(drop="reference.text")})
        assert C.run(tts, scheduler.store.engine, Spy(scheduler), limit=3, log=lambda m: None)["scored"] == 9
        assert C.run(tts, scheduler.store.engine, Spy(scheduler), limit=3, log=lambda m: None)["scored"] == 0
        ds = C.load_dataset("lines")
        C.write_dataset(
            "lines",
            [dict(it, reference=dict(text="changed")) if it["id"] == ds.items[0]["id"] else it for it in ds.items],
        )
        assert C.run(tts, scheduler.store.engine, Spy(scheduler), limit=3, log=lambda m: None)["scored"] == 3
    finally:
        stop.set()
    for bad in ({"with": "reference.audio"}, {"with": dict(reference=3)}):
        with pytest.raises(C.EvalError, match="with is a table"):
            C.parse_spec(
                dict(
                    name="x",
                    question="q",
                    role="r",
                    dataset="d",
                    metrics=["wer"],
                    score=dict(role="o", **bad),
                    candidates=[dict(name="a", model="m")],
                )
            )
