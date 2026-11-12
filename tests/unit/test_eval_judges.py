"""evals.judges: model judges whose verdicts are kept beside people's on the same blind pairs, and their agreement with people. The fake runtime
judges like a model (`favour` a word, or `always` one side); cloud judges get a faked CLI. Never the shared database: the scheduler fixture's
store is in memory."""

import os
import subprocess

import pytest

from stratamodels.db.models import EvalJudgment
from stratamodels.evals import cloud
from stratamodels.evals import core as C
from stratamodels.evals import judges as J
from stratamodels.evals import report as R


def judge_spec(**kw):
    return J.parse_judge(
        dict(dict(name="j", kind="text", role="other", model="other", params=dict(favour="fox")), **kw)
    )


@pytest.fixture
def spec(setup):
    """fake-asr (base drops every 4th word, so it loses "fox"; better every 12th; same = base) with a text judge that favours "fox"."""
    s = C.load_spec("fake-asr")
    s.judges = [judge_spec()]
    return s


@pytest.fixture
def ran(spec, scheduler):
    stop = scheduler.start()
    C.run(spec, scheduler.store.engine, C.SchedulerExecutor(scheduler), limit=10, log=lambda m: None)
    yield spec
    stop.set()


def person(engine, spec, a, b, winner, items, rater="jonathan"):
    ds = C.load_dataset(spec.dataset, engine=engine)
    with C.session(engine) as s:
        for n, it in enumerate(ds.items[:items]):
            s.add(
                EvalJudgment(
                    experiment=spec.name,
                    item_id=it["id"],
                    a=a,
                    b=b,
                    winner=winner(n) if callable(winner) else winner,
                    rater=rater,
                    at=float(n),
                    a_key=C.result_key(spec, spec.candidate(a), it),
                    b_key=C.result_key(spec, spec.candidate(b), it),
                )
            )
        s.commit()


def ask_queue(scheduler):
    return J.QueueAsker(C.SchedulerExecutor(scheduler))


# ---- spec --------------------------------------------------------------------------------------------------------------------------------
def test_judge_defaults_kinds_and_errors():
    j = J.parse_judge(dict(name="v"))
    assert (j["kind"], j["role"], j["model"], j["pairs"], j["orders"], j["auto"], j["limit"]) == (
        "vision",
        "judge-vision",
        None,
        "judged",
        2,
        False,
        None,
    )
    assert J.parse_judge(dict(name="t", kind="text"))["role"] == "judge-text"
    c = J.parse_judge(dict(name="c", cli="agy", model="gemini-3.8-flash-low", effort="low", limit=5))
    assert c["role"] is None and c["cli"] == "agy" and c["limit"] == 5
    for bad, msg in [
        (dict(), "needs a name"),
        (dict(name="c", cli="agy"), "names its model"),
        (dict(name="a", kind="smell"), "text, vision or audio"),
        (dict(name="a", kind="audio", cli="codex", model="gpt"), "cannot hear"),
        (dict(name="a", cli="gemini", model="m"), "codex or agy"),
        (dict(name="a", pairs="some"), "judged or all"),
        (dict(name="a", orders=3), "1 or 2"),
        (dict(name="a", limit=0), "limit"),
        (dict(name="a", limit=True), "limit"),
        (dict(name="a", colour="red"), "unknown fields"),
    ]:
        with pytest.raises(ValueError, match=msg):
            J.parse_judge(bad)


def test_spec_carries_judges_and_rejects_two_of_a_name():
    d = dict(name="x", question="q", role="r", dataset="d", metrics=["wer"], candidates=[dict(name="a", model="m")])
    s = C.parse_spec(dict(d, judges=[dict(name="v", kind="vision")]))
    assert [j["name"] for j in s.judges] == ["v"] and s.to_json()["judges"][0]["role"] == "judge-vision"
    with pytest.raises(C.EvalError, match="two judges share a name"):
        C.parse_spec(dict(d, judges=[dict(name="v"), dict(name="v", kind="text")]))
    with pytest.raises(C.EvalError, match="needs a name"):
        C.parse_spec(dict(d, judges=[dict(kind="text")]))
    with pytest.raises(C.EvalError, match="no judge 'w'"):
        J.find(s, "w")


def test_judge_key_follows_what_decides_a_verdict():
    j = judge_spec()
    k = J.judge_key(j)
    assert J.judge_key(dict(j, limit=3, auto=True, pairs="all", note="n")) == k  # how many and when: the same judge
    assert J.judge_key(dict(j, criteria="shorter is better")) != k and J.judge_key(dict(j, orders=1)) != k
    v = J.parse_judge(dict(name="v"))
    assert J.judge_key(v, "qwen") != J.judge_key(v, "gemma")  # a new model for the kind's role is a new judge


# ---- what a judge sees -------------------------------------------------------------------------------------------------------------------
def test_shown_hides_what_names_the_model():
    assert J.shown(dict(json={"a": 1}, model="m", text="raw")) == '{\n "a": 1\n}'
    assert J.shown(dict(text="hello", model="m")) == "hello"
    assert J.shown(dict(segments=[dict(text=" one "), dict(text="two")], model="m")) == "one two"
    assert (
        J.shown(dict(path="/x.wav", duration_s=2, model="m", seconds=1, pipeline=[1]))
        == '{\n "path": "/x.wav",\n "duration_s": 2\n}'
    )
    assert J.shown("plain") == '"plain"'


def test_prompt_attaches_pictures_for_vision_only():
    item = dict(id="p", input=dict(image="/in/photo.jpg", preset="photo-describe-v1"), reference=dict(text="a dog"))
    sys_, text, imgs = J.prompt(
        C.parse_spec(
            dict(
                name="x",
                question="Describe it",
                role="r",
                dataset="d",
                metrics=["wer"],
                candidates=[dict(name="a", model="m")],
            )
        ),
        J.parse_judge(dict(name="v")),
        item,
        dict(text="a dog", model="secret-a"),
        dict(path="/out/y.png", model="secret-b"),
    )
    assert imgs == ["/in/photo.jpg", "/out/y.png"] and "secret" not in text and "Task: Describe it" in text
    assert (
        "Answer X:\na dog" in text
        and "attached as number 2" in text
        and '"preset"' in text
        and "Reference" in text
        and "blind" in sys_
    )
    _, text, imgs = J.prompt(
        C.parse_spec(
            dict(name="x", question="q", role="r", dataset="d", metrics=["wer"], candidates=[dict(name="a", model="m")])
        ),
        judge_spec(),
        item,
        dict(text="a"),
        dict(text="b"),
    )
    assert imgs == [] and "attached" not in text


def test_an_audio_judge_hears_the_original_then_x_and_y_as_16k_wav(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATAMODELS_DATA", str(tmp_path / "data"))
    src, x, y = (tmp_path / n for n in ("orig.flac", "x.wav", "y.wav"))
    for f in (src, x, y):
        f.write_bytes(b"sound")
    cuts = []

    def cut(path, clip, out):
        cuts.append((path, clip))
        open(out, "wb").write(b"wav16")
        return out

    sp = C.parse_spec(
        dict(
            name="x",
            question="Which is cleaner?",
            role="r",
            dataset="d",
            metrics=["wer"],
            candidates=[dict(name="a", model="m")],
        )
    )
    j = J.parse_judge(dict(name="ear", kind="audio"))
    assert (
        j["role"] == "judge-audio"
        and J.parse_judge(dict(name="g", kind="audio", cli="agy", model="gemini"))["cli"] == "agy"
    )
    item = dict(id="i", input=dict(path=str(src), clip=[1.0, 3.0], note="tape"))
    _, text, files = J.prompt(sp, j, item, dict(path=str(x), model="secret"), dict(path=str(y)), cut=cut)
    assert len(files) == 3 and all(
        f.endswith(".wav") and os.path.dirname(f).endswith(os.path.join("judges", "audio")) for f in files
    )
    assert cuts == [(str(src), [1.0, 3.0]), (str(x), None), (str(y), None)]
    assert "Recording 1 is the input" in text and "Answer X: recording 2" in text and "Answer Y: recording 3" in text
    assert '"note"' in text and "clip" not in text and "secret" not in text
    J.prompt(sp, j, item, dict(path=str(x)), dict(path=str(y)), cut=cut)
    assert len(cuts) == 3  # made once, kept
    _, text, files = J.prompt(sp, j, dict(id="t", input=dict(text="hi")), dict(text="a"), dict(text="b"), cut=cut)
    assert files == [] and "Answer X:\na" in text  # nothing to hear: read as text

    class Ex:
        def __init__(self):
            self.sent = []

        def submit(self, role, params, model, options):
            self.sent.append((role, params))
            return 1

        def wait(self, jid):
            return dict(status="done", result=dict(text='{"winner": "X", "reason": "r"}'))

    ex = Ex()
    J.QueueAsker(ex)(dict(j, resolved=None), "s", "t", ["/a.wav", "/b.wav"], J.SCHEMA)
    assert (
        ex.sent[0][0] == "judge-audio"
        and ex.sent[0][1]["audios"] == ["/a.wav", "/b.wav"]
        and "images" not in ex.sent[0][1]
    )


def test_agy_hears_recordings_through_its_file_viewer(tmp_path):
    a, b = tmp_path / "x.wav", tmp_path / "p.jpg"
    a.write_bytes(b"w")
    b.write_bytes(b"j")
    d = tmp_path / "ws"
    d.mkdir()
    assert cloud.stage_images([str(a), str(a)], str(d)) == ["rec1.wav", "rec2.wav"]
    assert cloud.stage_images([str(b)], str(d)) == ["img0.jpg"]
    p = cloud.prompt_of(None, "judge", ["rec1.wav", "rec2.wav"], None)
    assert "recordings are in your working directory: rec1.wav, rec2.wav" in p and "file viewing tool" in p
    assert "picture" in cloud.prompt_of(None, "judge", ["img0.jpg"], None)


def test_verdict_reads_json_or_text_and_refuses_anything_else():
    assert J.verdict(dict(json=dict(winner="X", reason="r"))) == ("X", "r")
    assert J.verdict(dict(text='```json\n{"reason": "s", "winner": "tie"}\n```')) == ("tie", "s")
    assert J.verdict(dict(json=None, text='{"winner": "y"}')) == ("Y", "")
    for bad in (dict(text="X is better"), dict(json=dict(winner="Z")), None):
        with pytest.raises(J.JudgeError, match="no verdict"):
            J.verdict(bad)


# ---- agreement ---------------------------------------------------------------------------------------------------------------------------
def test_kappa():
    assert J.kappa([("a", "a"), ("b", "b")]) == 1.0
    assert J.kappa([("a", "a"), ("a", "a")]) == 1.0  # one category only: agreement is all there is
    assert J.kappa([("a", "b"), ("b", "a")]) == pytest.approx(-1.0)
    assert J.kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]) == pytest.approx(0.0)
    assert J.kappa([]) != J.kappa([])  # NaN


def test_agreement_counts_the_latest_human_answer_and_ties():
    p = [
        (1, "a", "b", "b", 0.0),
        (1, "a", "b", "a", 1.0),
        (2, "a", "b", "tie", 0.0),
        (3, "b", "a", "b", 0.0),
        (4, "a", "b", "a", 0.0),
    ]
    v = [(1, "b", "a", "a", 5.0), (2, "a", "b", "a", 5.0), (3, "a", "b", "a", 5.0), (9, "a", "b", "a", 5.0)]
    ag = J.agreement(p, v)
    assert ag["n"] == 3 and ag["tally"] == dict(same_winner=1, opposite=1, judge_tie=0, person_tie=1, both_tie=0)
    assert ag["agree"]["k"] == 1 and ag["decided"]["n"] == 2 and ag["decided"]["value"] == pytest.approx(0.5)


# ---- running -----------------------------------------------------------------------------------------------------------------------------
def test_judge_runs_on_pairs_people_judged_and_agreement_is_reported(ran, scheduler):
    engine = scheduler.store.engine
    person(engine, ran, "better", "base", "better", 6)
    person(engine, ran, "better", "same", lambda n: "same" if n == 0 else "better", 3)
    before = C.analyse(ran, engine)["judgments"]["n"]
    log = []
    r = J.run_judge(ran, engine, "j", ask_queue(scheduler), log=log.append)
    assert r == dict(judged=9, failed=0, left=0, stopped=None) and "9 pairs to judge" in log[0], log
    assert J.run_judge(ran, engine, "j", ask_queue(scheduler), log=log.append)["judged"] == 0  # judged once per config
    res = C.analyse(ran, engine)
    assert res["judgments"]["n"] == before == 9  # people's analysis never counts a judge
    s = res["judges"]["j"]
    assert (s["judged"], s["with_people"], s["kind"], s["model"], s["via"]) == (9, 9, "text", "other", "queue")
    assert s["agree"]["k"] == 8 and s["kappa"] < 1 and s["consistent"]["value"] == 1.0 and s["consistent"]["n"] == 9
    assert s["comparisons"]["better"]["wins"] == 6 and s["disagreements"][0]["person"] == "same"
    with C.session(engine) as ses:
        row = ses.query(EvalJudgment).filter(EvalJudgment.judge.isnot(None)).first()
        assert (
            row.rater == "j"
            and [a["said"] for a in row.detail["asks"]] in (["X", "Y"], ["Y", "X"])
            and "fox" in row.note
        )
    text = R.render(res)
    assert "## Model judges" in text and "| `j` | text | `other` | 9 |" in text and "people same" in text
    ran.judges = [judge_spec(criteria="other words")]  # a changed judge is a new judge: its old verdicts stop counting
    assert C.analyse(ran, engine)["judges"]["j"]["judged"] == 0


def test_a_judge_that_follows_the_position_gives_ties(ran, scheduler):
    engine = scheduler.store.engine
    person(engine, ran, "better", "base", "better", 4)
    ran.judges = [judge_spec(params=dict(always="X"))]
    assert J.run_judge(ran, engine, "j", ask_queue(scheduler))["judged"] == 4
    s = C.analyse(ran, engine)["judges"]["j"]
    assert s["consistent"]["value"] == 0.0 and s["tally"]["judge_tie"] == 4 and s["agree"]["k"] == 0
    ran.judges = [judge_spec(params=dict(always="X"), orders=1)]  # asked once, it just picks a side
    assert J.run_judge(ran, engine, "j", ask_queue(scheduler))["judged"] == 4
    assert C.analyse(ran, engine)["judges"]["j"]["consistent"]["n"] == 0


def test_every_pair_mode_skips_identical_outputs_and_keeps_to_its_limit(ran, scheduler):
    engine = scheduler.store.engine
    ran.judges = [judge_spec(pairs="all", limit=7)]
    r = J.run_judge(ran, engine, "j", ask_queue(scheduler))
    assert r == dict(
        judged=7, failed=0, left=13, stopped=None
    )  # 10 items x (better v base, better v same); base v same are identical
    assert J.run_judge(ran, engine, "j", ask_queue(scheduler), limit=100)["judged"] == 13
    ran.primary = "judged"
    assert J.todo(ran, engine, ran.judges[0])[3] == []  # every pair with the baseline is judged


def test_a_failed_judging_job_or_bad_answer_is_left_for_the_next_run(ran, scheduler):
    engine = scheduler.store.engine
    person(engine, ran, "better", "base", "better", 2)
    ran.judges = [judge_spec(params=dict(fail="out of memory"))]
    log = []
    assert J.run_judge(ran, engine, "j", ask_queue(scheduler), log=log.append) == dict(
        judged=0, failed=2, left=2, stopped=None
    )
    assert any("out of memory" in m for m in log)
    ran.judges = [judge_spec(params={})]  # the fake echoes: no verdict in its answer
    assert J.run_judge(ran, engine, "j", ask_queue(scheduler))["failed"] == 2
    with pytest.raises(C.EvalError, match="need the service"):
        J.asker_for(ran.judges[0])


def test_cloud_judge_calls_the_cli_and_stops_on_quota(ran, scheduler, monkeypatch):
    engine = scheduler.store.engine
    person(engine, ran, "better", "base", "better", 3)
    ran.judges = [J.parse_judge(dict(name="luna", cli="codex", model="gpt-6-luna", kind="text", orders=1))]
    calls = []

    def fake_run(cmd, input=None, cwd=None, **kw):
        calls.append(cmd)
        if len(calls) == 3:
            return subprocess.CompletedProcess(cmd, 1, "", "You've hit your usage limit")
        # the answer goes to the -o file, as codex writes it
        open(os.path.join(cwd, "answer.txt"), "w").write('{"reason": "has fox", "winner": "X"}')
        return subprocess.CompletedProcess(cmd, 0, "", "")

    assert isinstance(J.asker_for(ran.judges[0]), J.CliAsker)
    r = J.run_judge(ran, engine, "luna", J.CliAsker(run=fake_run))
    assert r["judged"] == 2 and r["stopped"].startswith("quota") and r["left"] == 1
    assert (
        calls[0][:2] == ["codex", "exec"]
        and "gpt-6-luna" in calls[0]
        and any(c.startswith("--output-schema") or c == "--output-schema" for c in calls[0])
    )
    s = C.analyse(ran, engine)["judges"]["luna"]
    assert s["via"] == "codex" and s["judged"] == 2

    def broken(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 2, "", "boom")

    assert J.run_judge(ran, engine, "luna", J.CliAsker(run=broken))["failed"] == 1
    assert cloud.CLIS["codex"] is cloud.call_codex


def test_verdicts_on_lists_people_and_judges(ran, scheduler):
    engine = scheduler.store.engine
    person(engine, ran, "better", "base", "better", 1)
    J.run_judge(ran, engine, "j", ask_queue(scheduler))
    v = J.verdicts_on(engine, ran, "u0")
    assert (
        [x["rater"] for x in v] == ["j", "jonathan"]
        and v[0]["judge"]
        and v[1]["judge"] is None
        and v[0]["detail"]["consistent"]
        and v[0]["current"]
        and v[1]["current"]
    )
    ran.judges = [
        judge_spec(criteria="new rules")
    ]  # another judge config: the old verdict is history, the person's still stands
    assert [x["current"] for x in J.verdicts_on(engine, ran, "u0")] == [False, True]
    ran.candidates[1].params = dict(drop=6)  # other outputs: the person's judgment of the old ones is history too
    assert [x["current"] for x in J.verdicts_on(engine, ran, "u0")] == [False, False]
    assert J.verdicts_on(engine, ran, "nope") == []


def test_a_new_model_for_a_judges_role_changes_the_analysis_signature(ran, scheduler, monkeypatch):
    from stratamodels.evals import summaries

    engine = scheduler.store.engine
    ran.judges = [J.parse_judge(dict(name="v"))]
    monkeypatch.setattr(J, "resolve", lambda engine, j: "qwen")
    a = summaries.signature(ran, engine)
    monkeypatch.setattr(J, "resolve", lambda engine, j: "gemma")
    assert summaries.signature(ran, engine) != a


def test_a_vision_judge_sees_videos_as_region_stills_beside_the_tape():
    spec = C.parse_spec(
        dict(
            name="v",
            question="Which upscale is better?",
            role="video-enhance",
            dataset="d",
            metrics=["judged"],
            candidates=[dict(name="a", model="m")],
        )
    )
    j = J.parse_judge(dict(name="v", kind="vision", criteria="detail without invented texture"))
    crop = lambda k, i: dict(id=i, label=i.title(), path=f"/g/{k}.{i}.mp4", stills=[f"/g/{k}.{i}.3s.png"])
    item = dict(
        id="c",
        input=dict(path="/c.mkv", crop=[14, 0, 692, 568]),
        reference=dict(preview="/m/c.mp4", crops=[dict(id="face", stills=["/m/c.face.3s.png"])]),
    )
    x = dict(path="/g/x.mp4", model="secret", crops=[crop("x", "face"), crop("x", "text")], stills=["/g/x.1s.png"])
    y = dict(path="/g/y.mp4", crops=[crop("y", "face"), crop("y", "text")], stills=["/g/y.1s.png"])
    _, text, imgs = J.prompt(spec, j, item, x, y)
    assert imgs == ["/m/c.face.3s.png", "/g/x.face.3s.png", "/g/y.face.3s.png", "/g/x.text.3s.png", "/g/y.text.3s.png"]
    assert (
        "Region Face, moment 1: picture 1 is the tape (source, enlarged without enhancement), picture 2 is X, picture 3 is Y."
        in text
    )
    assert "Region Text, moment 1: picture 4 is X, picture 5 is Y." in text  # no tape crop for this region
    assert "Reference" not in text and "secret" not in text and "/m/c.mp4" not in text
    bare = J.prompt(
        spec,
        j,
        dict(id="c", input={}),
        dict(path="/g/x.mp4", stills=["/g/x.1s.png"]),
        dict(path="/g/y.mp4", stills=["/g/y.1s.png"]),
    )
    assert bare[2] == ["/g/x.1s.png", "/g/y.1s.png"] and "Whole frame, moment 1" in bare[1]
    text_judge = J.parse_judge(dict(name="t", kind="text"))
    assert J.prompt(spec, text_judge, item, x, y)[2] == []  # a text judge sees no pictures
