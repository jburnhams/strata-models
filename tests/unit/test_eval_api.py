"""The Evals API (stratamodels.evals.api): listing, running in the background, the analysis, the write-up, blind judging, the media file."""

import time

import pytest
from fastapi.testclient import TestClient

from stratamodels.server.app import create_app


@pytest.fixture
def client(scheduler, setup):
    stop = scheduler.start()
    app = create_app(scheduler)
    yield TestClient(app)
    app.state.eval_summaries.pool.shutdown(
        wait=True
    )  # a refresh still queued would call C.analyse after this test, inside the next one's monkeypatched count
    stop.set()


def wait_run(client, name):
    for _ in range(1200):  # up to a minute: a loaded machine runs the fake jobs slowly
        st = client.get(f"/api/evals/{name}/status").json()
        if st["state"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("the run did not finish")


def test_list_run_analyse_and_report(client):
    d = client.get("/api/evals").json()
    e = d["experiments"][0]
    assert e["name"] == "fake-asr" and e["outputs"] == 0 and e["total"] == 120 and d["datasets"][0]["n"] == 40
    assert client.get("/api/evals/fake-asr/status").json()["state"] == "idle"
    assert client.post("/api/evals/fake-asr/run", params=dict(limit=10)).json()["state"] == "running"
    st = wait_run(client, "fake-asr")
    assert st["state"] == "done" and st["result"]["ran"] == 30
    assert (
        client.post("/api/evals/fake-asr/run").status_code == 202
        and wait_run(client, "fake-asr")["result"]["ran"] == 90
    )
    a = client.get("/api/evals/fake-asr").json()
    assert a["decision"]["recommendation"] == "better" and a["dataset"]["compared"] == 40
    r = client.get("/api/evals/fake-asr/report")
    assert r.status_code == 200 and r.text.startswith("# fake-asr") and "## Method" in r.text
    assert client.get("/api/evals/nope").status_code == 404 and client.post("/api/evals/nope/run").status_code == 404


def test_blind_judging(client, scheduler):
    assert client.get("/api/evals/fake-asr/pair").json() == dict(pair=None)
    client.post("/api/evals/fake-asr/run")
    wait_run(client, "fake-asr")
    p = client.get("/api/evals/fake-asr/pair").json()["pair"]
    assert p["x"] and p["y"] and "path" not in p["input"] and not p["has_media"] and "better" not in p["token"]
    j = client.post("/api/evals/fake-asr/judgments", json=dict(token=p["token"], winner="x")).json()
    assert j["winner"] == j["x"]
    for _ in range(5):
        p = client.get("/api/evals/fake-asr/pair").json()["pair"]
        client.post("/api/evals/fake-asr/judgments", json=dict(token=p["token"], winner="tie"))
    wait_idle(
        client
    )  # each judgment queues a refresh; on a slow runner the analysis takes over QUICK_S and a GET before it lands is served the kept one, marked stale
    assert client.get("/api/evals/fake-asr").json()["judgments"]["n"] == 6
    assert client.post("/api/evals/fake-asr/judgments", json=dict(token=p["token"], winner="z")).status_code == 400
    assert (
        client.post("/api/evals/fake-asr/judgments", json=dict(token=p["token"][:-3] + "abc", winner="x")).status_code
        == 400
    )
    assert client.post("/api/evals/other/judgments", json=dict(token=p["token"], winner="x")).status_code == 400


def test_media_only_for_files_a_dataset_names(client, tmp_path, setup):
    from stratamodels.evals import core as C

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF0000WAVE")
    C.write_dataset(
        "speech",
        [
            dict(id="a", input=dict(path=str(wav), say="hello there"), reference=dict(text="hello there")),
            dict(id="b", input=dict(say="x"), reference=dict(text="x")),
        ],
    )
    r = client.get("/api/evals/fake-asr/media/a")
    assert r.status_code == 200 and r.content == b"RIFF0000WAVE"
    assert (
        client.get("/api/evals/fake-asr/media/b").status_code == 404
        and client.get("/api/evals/fake-asr/media/zz").status_code == 404
    )


def test_broken_specs_and_datasets_are_reported(client, setup):
    import os

    from stratamodels.evals import core as C

    folder = os.path.join(os.environ["STRATAMODELS_DATA"], "evals", "specs")
    open(os.path.join(folder, "bad.toml"), "w").write("name = ")
    open(os.path.join(folder, "nodata.toml"), "w").write(
        'name = "nodata"\nquestion = "q"\nrole = "echo"\ndataset = "missing"\nmetrics = ["wer"]\n[[candidates]]\nname = "a"\nmodel = "small"\n'
    )
    ex = {e["name"]: e for e in client.get("/api/evals").json()["experiments"]}
    assert (
        "error" in ex["bad"]
        and "no dataset" in ex["nodata"]["error"]
        and client.get("/api/evals/bad").status_code == 400
    )
    assert client.get("/api/evals/nodata").status_code == 400
    one = os.path.join(folder, "one.toml")
    open(one, "w").write(
        'name = "one"\nquestion = "q"\nrole = "echo"\ndataset = "speech"\nmetrics = ["wer"]\n[[candidates]]\nname = "a"\nmodel = "small"\n'
    )
    assert client.get("/api/evals/one/pair").status_code == 400

    def boom(*a, **k):
        raise RuntimeError("db away")

    C_run = C.run
    C.run = boom
    try:
        client.post("/api/evals/one/run")
        st = wait_run(client, "one")
        assert st["state"] == "failed" and "db away" in st["error"]
    finally:
        C.run = C_run


def test_no_observations_come_back_as_null_not_nan(client):
    from stratamodels.evals.api import finite

    assert finite(dict(a=float("nan"), b=[1.0, float("inf")], c=(2, "x"))) == dict(a=None, b=[1.0, None], c=[2, "x"])
    a = client.get("/api/evals/fake-asr").json()  # nothing run yet: every value is null, and it is valid JSON
    assert a["dataset"]["compared"] == 0


def test_judgments_count_only_while_their_outputs_are_current(client, scheduler):
    from stratamodels.evals import core as C

    client.post("/api/evals/fake-asr/run")
    wait_run(client, "fake-asr")
    for _ in range(4):
        p = client.get("/api/evals/fake-asr/pair").json()["pair"]
        client.post("/api/evals/fake-asr/judgments", json=dict(token=p["token"], winner="x"))
    assert client.get("/api/evals/fake-asr").json()["judgments"]["n"] == 4
    s = C.load_spec("fake-asr")
    C.forget(s, scheduler.store.engine)  # outputs regenerated: new keys? same keys (same params), so still counted
    assert len(C.judgments(scheduler.store.engine, "fake-asr")) == 4
    keys = {(c.name, it["id"]): "changed" for c in s.candidates for it in C.load_dataset("speech").items}
    assert C.judgments(scheduler.store.engine, "fake-asr", keys) == []


def test_pairs_with_the_same_output_are_not_offered(client, scheduler):
    from stratamodels.evals.api import same

    assert same(
        dict(segments=[dict(text=" Anders' face grew red.")]), dict(segments=[dict(text="anders' face grew red")])
    )
    assert not same(dict(segments=[dict(text="a b")]), dict(segments=[dict(text="a c")]))
    assert same(dict(path="/a.png", seconds=1.0), dict(path="/a.png", seconds=2.0)) and not same(
        dict(path="/a.png"), dict(path="/b.png")
    )
    client.post("/api/evals/fake-asr/run")
    wait_run(client, "fake-asr")
    seen = set()
    for _ in range(30):
        p = client.get("/api/evals/fake-asr/pair").json()["pair"]
        assert not same(p["x"], p["y"])
        seen.add(p["item_id"])
        client.post("/api/evals/fake-asr/judgments", json=dict(token=p["token"], winner="tie"))
    assert len(seen) > 20  # unjudged pairs come first, so it moves through the items


def test_media_of_a_phrase_is_cut_to_its_clip(client, tmp_path, setup, fake_run):
    from stratamodels.evals import core as C

    wav = tmp_path / "long.flac"
    wav.write_bytes(b"RIFF" + b"0" * 1000)
    C.write_dataset(
        "speech", [dict(id="p", input=dict(path=str(wav), clip=[2.0, 3.0], say="x"), reference=dict(text="x"))]
    )
    fake_run.returns["ffmpeg"] = b"RIFFcut"
    whole = client.get("/api/evals/fake-asr/media/p")  # the whole file, typed: the page cues it to the phrase
    assert whole.headers["content-type"] == "audio/flac" and whole.content.startswith(b"RIFF0") and not fake_run.calls
    r = client.get("/api/evals/fake-asr/media/p", params=dict(cut=1))
    assert r.headers["content-type"] == "audio/wav" and r.content == b"RIFFcut"
    cmd = fake_run.calls[-1]
    assert (
        cmd[cmd.index("-ss") + 1] == "0.00" and cmd[cmd.index("-to") + 1] == "5.00"
    )  # 2 s either side, never before 0
    fake_run.code = 1
    assert client.get("/api/evals/fake-asr/media/p", params=dict(cut=1)).content.startswith(
        b"RIFF0"
    )  # ffmpeg failed: the whole file


def test_the_most_different_pairs_come_first(client, scheduler, setup):
    from stratamodels.evals import core as C

    client.post("/api/evals/fake-asr/run")
    wait_run(client, "fake-asr")
    first = client.get("/api/evals/fake-asr/pair").json()
    assert 0 < first["pair"]["similarity"] < 1 and first["remaining"] > 0
    sims = []
    for _ in range(10):
        p = client.get("/api/evals/fake-asr/pair").json()
        sims.append(p["pair"]["similarity"])
        client.post("/api/evals/fake-asr/judgments", json=dict(token=p["pair"]["token"], winner="tie"))
    assert (
        sims == sorted(sims) or max(sims) - min(sims) < 0.05
    )  # least alike first (ties within 0.02 are picked at random)
    import pytest

    with pytest.raises(C.EvalError, match="unknown similarity"):
        C.parse_spec(
            dict(
                name="x",
                question="q",
                role="r",
                dataset="d",
                metrics=["wer"],
                similarity="video",
                candidates=[dict(name="a", model="m")],
            )
        )


def test_judging_skips_only_judgments_of_current_outputs_and_rescores_new_outputs(client, scheduler):
    from stratamodels.db.models import EvalJudgment
    from stratamodels.evals import core as C

    client.post("/api/evals/fake-asr/run")
    wait_run(client, "fake-asr")
    first = client.get("/api/evals/fake-asr/pair").json()
    n0 = first["remaining"]
    with C.session(
        scheduler.store.engine
    ) as ses:  # a stale judgment (old outputs) of every pair: must not count as judged
        for it in C.load_dataset("speech").items:
            ses.add(
                EvalJudgment(
                    experiment="fake-asr",
                    item_id=it["id"],
                    a="base",
                    b="better",
                    winner="tie",
                    rater="x",
                    at=1.0,
                    a_key="old",
                    b_key="old",
                )
            )
        ses.commit()
    assert client.get("/api/evals/fake-asr/pair").json()["remaining"] == n0
    s = C.load_spec("fake-asr")
    C.forget(s, scheduler.store.engine, "same")
    client.post("/api/evals/fake-asr/run")
    wait_run(client, "fake-asr")
    assert client.get("/api/evals/fake-asr/pair").json()["pair"] is not None


def wait_idle(client):
    pool = client.app.state.eval_summaries
    pool.pool.submit(lambda: None).result(timeout=60)  # the one worker has done what was queued before


def test_the_analysis_is_kept_and_redone_only_when_asked_or_changed(client, monkeypatch):
    from stratamodels.evals import core as C

    wait_idle(
        client
    )  # the service's start-up warm() works the analysis out in the background: let it finish before counting
    calls = []
    real = C.analyse
    monkeypatch.setattr(C, "analyse", lambda *a, **k: calls.append(1) or real(*a, **k))
    a = client.get("/api/evals/fake-asr").json()
    client.get("/api/evals/fake-asr")
    assert len(calls) == 1 and a["summary"]["stale"] is False  # kept
    client.app.state.eval_summaries.memo.clear()  # a restarted service reads it back from the database
    client.get("/api/evals/fake-asr")
    assert len(calls) == 1
    assert (
        client.post("/api/evals/fake-asr/refresh").status_code == 202
    )  # a score pass changes outputs in place: asked for
    wait_idle(client)
    assert len(calls) == 2
    client.post("/api/evals/fake-asr/run", params=dict(limit=5))
    wait_run(client, "fake-asr")
    wait_idle(client)
    assert len(calls) == 3  # a finished run works it out again
    assert client.get("/api/evals/fake-asr").json()["dataset"]["compared"] == 5 and len(calls) == 3


def test_a_slow_analysis_is_served_stale_while_redone_in_the_background(client, monkeypatch):
    import threading

    from stratamodels.evals import summaries as SU

    sm = client.app.state.eval_summaries
    first = client.get("/api/evals/fake-asr").json()
    assert first["dataset"]["compared"] == 0
    monkeypatch.setattr(SU, "QUICK_S", -1.0)  # every analysis counts as slow
    gate = threading.Event()
    from stratamodels.evals import core as C

    real = C.analyse
    monkeypatch.setattr(C, "analyse", lambda *a, **k: gate.wait(30) and real(*a, **k))
    client.post("/api/evals/fake-asr/run", params=dict(limit=4))
    wait_run(client, "fake-asr")
    a = client.get("/api/evals/fake-asr").json()
    assert a["summary"]["stale"] is True and a["dataset"]["compared"] == 0  # the old one at once, not a wait
    gate.set()
    wait_idle(client)
    b = client.get("/api/evals/fake-asr").json()
    assert b["summary"]["stale"] is False and b["dataset"]["compared"] == 4
    assert sm.pending == {}


def test_a_judgment_refreshes_the_kept_analysis(client):
    client.post("/api/evals/fake-asr/run", params=dict(limit=10))
    wait_run(client, "fake-asr")
    wait_idle(client)
    p = client.get("/api/evals/fake-asr/pair").json()["pair"]
    assert client.post("/api/evals/fake-asr/judgments", json=dict(token=p["token"], winner="x")).status_code == 201
    wait_idle(client)
    from stratamodels.evals import core as C

    sig = client.app.state.eval_summaries.memo["fake-asr"][0]
    from stratamodels.evals.summaries import signature

    assert sig == signature(C.load_spec("fake-asr"), client.app.state.eval_summaries.engine())


def test_warm_works_out_every_experiment(client):
    from stratamodels.db.models import EvalSummary
    from stratamodels.evals import core as C

    sm = client.app.state.eval_summaries
    sm.warm()
    wait_idle(client)
    with C.session(sm.engine()) as s:
        assert {r.experiment for r in s.scalars(C.select(EvalSummary))} >= {"fake-asr"}


def test_a_forced_refresh_joining_a_queued_one_keeps_its_force(client, monkeypatch):
    import threading

    from stratamodels.evals import core as C

    sm = client.app.state.eval_summaries
    client.get("/api/evals/fake-asr")  # kept, current
    calls = []
    real = C.analyse
    monkeypatch.setattr(C, "analyse", lambda *a, **k: calls.append(1) or real(*a, **k))
    gate = threading.Event()
    sm.pool.submit(gate.wait, 30)  # the worker busy (warming another experiment)
    sm.refresh("fake-asr")  # ordinary: would find the signature unchanged
    sm.refresh("fake-asr", force=True)  # a score pass finished meanwhile
    gate.set()
    wait_idle(client)
    assert len(calls) == 1


def test_a_run_refreshes_every_experiment_sharing_its_dataset(client, monkeypatch):
    from stratamodels.evals import core as C

    spec = C.load_spec("fake-asr").to_json()
    assert client.put("/api/evals/twin/spec", json=spec).status_code == 200
    wait_idle(client)
    client.get("/api/evals/twin")
    names = []
    real = C.analyse
    monkeypatch.setattr(C, "analyse", lambda s, *a, **k: names.append(s.name) or real(s, *a, **k))
    client.post("/api/evals/fake-asr/run", params=dict(limit=2))
    wait_run(client, "fake-asr")
    wait_idle(client)
    assert sorted(names) == [
        "fake-asr",
        "twin",
    ]  # twin's outputs may have been rescored in place: worked out again though unchanged
