"""The real service process, driven through the client package as an app would: jobs, failures, leases, cancellation, history."""

import json
import threading
import time
import urllib.request
import warnings

import pytest
import strata_models_client as sm


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.load(r)


def test_run_submit_wait_and_failures(service):
    assert sm.available()
    assert sm.run("echo", dict(a=1), app="it") == dict(model="echo-small", echo=dict(a=1))
    j = sm.submit("echo", dict(seconds=0.2), app="it")
    with pytest.raises(TimeoutError):
        sm.wait(j["id"], timeout=0.01)
    assert sm.wait(j["id"], timeout=10)["echo"] == dict(seconds=0.2)
    with pytest.raises(sm.JobFailed, match="not a real file"):
        sm.run("echo", dict(fail="not a real file"), app="it")
    with pytest.raises(sm.JobFailed, match="no role"):
        sm.run("nope", {}, app="it")
    with pytest.raises(sm.JobFailed, match="does not serve"):
        sm.faces("/x.jpg", app="it", model="echo-small")
    with pytest.raises(sm.JobFailed, match="does not serve"):
        sm.transcribe("/a.wav", app="it", model="buffalo_l", clip=(0, 1))
    with pytest.raises(sm.JobFailed, match="does not serve"):
        sm.speak("hello", app="it", model="buffalo_l", voice=__file__)
    assert sm.role_model("echo") == "echo-small" and sm.role_model("nope") is None
    assert sm.role_settings("nope") is None
    hist = get(service, "/api/jobs?app=it")["items"]
    assert [x["status"] for x in hist[:2]] == ["failed", "done"]
    assert get(service, "/api/status")["resident"][0]["model"] == "echo-small"


def test_many_jobs_in_one_call(service):
    ids = sm.submit_many("echo", [dict(a=1), dict(a=2)], app="it")
    assert [sm.wait(i, timeout=10)["echo"] for i in ids] == [dict(a=1), dict(a=2)]
    with pytest.raises(sm.JobFailed):
        sm.submit_many("no-such-role", [dict(a=1)], app="it")


def test_a_sized_lease_lets_jobs_run_beside_it(service):
    with sm.gpu_lease("it-stills", vram_gb=1, ttl_s=30, note="shared") as lease:
        assert lease.job["status"] == "running" and sm.run("echo", dict(z=1), app="it", timeout=20) == dict(
            model="echo-small", echo=dict(z=1)
        )


def test_a_whole_gpu_lease_holds_it_and_jobs_wait_for_it(service):
    order = []
    with sm.gpu_lease("it-stills", vram_gb=None, ttl_s=6, note="integration") as lease:
        assert lease.job["status"] == "running"
        t = threading.Thread(target=lambda: order.append(("job", sm.run("echo", {}, app="it"))))
        t.start()
        time.sleep(3)  # longer than ttl/3: the heartbeat thread must keep it
        assert get(service, f"/api/leases/{lease.id}")["status"] == "running"
        order.append(("lease-end", None))
    t.join(10)
    assert [o[0] for o in order] == ["lease-end", "job"]
    assert get(service, f"/api/leases/{lease.id}")["status"] == "done"


def test_a_second_lease_waits_and_can_time_out(service):
    first = sm.acquire("it-a", ttl_s=30)
    try:
        with pytest.raises(TimeoutError):
            sm.acquire("it-b", ttl_s=30, timeout=0.5)
        cancelled = get(service, "/api/jobs?kind=lease&app=it-b")["items"][0]
        assert cancelled["status"] == "cancelled"
    finally:
        first.release()


def test_without_the_service_the_client_says_so(monkeypatch):
    monkeypatch.setenv("STRATA_MODELS_URL", "http://127.0.0.1:9")
    assert not sm.available(timeout=0.5)
    with pytest.raises(sm.Unavailable):
        sm.run("echo", {}, app="x")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with sm.gpu_lease("x") as lease:
            assert lease is None
    assert "not running" in str(w[0].message)
    with pytest.raises(sm.Unavailable), sm.gpu_lease("x", required=True):
        pass


def test_the_judge_plays_only_the_phrase(service, tmp_path):
    import os
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    wav = tmp_path / "long.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=60", str(wav)], check=True)
    data = os.environ["STRATAMODELS_DATA"]
    os.makedirs(os.path.join(data, "evals", "datasets"), exist_ok=True)
    os.makedirs(os.path.join(data, "evals", "specs"), exist_ok=True)
    open(os.path.join(data, "evals", "datasets", "phr.jsonl"), "w").write(
        json.dumps(dict(id="p", input=dict(path=str(wav), clip=[20.0, 21.0]), reference=dict(text="x"))) + "\n"
    )
    open(os.path.join(data, "evals", "specs", "phr.toml"), "w").write(
        'name = "phr"\nquestion = "q"\nrole = "echo"\ndataset = "phr"\nmetrics = ["failed"]\n[[candidates]]\nname = "a"\nmodel = "echo-small"\n'
    )
    with urllib.request.urlopen(service + "/api/evals/phr/media/p?cut=1", timeout=30) as r:
        body = r.read()
        ctype = r.headers["content-type"]
    assert (
        ctype == "audio/wav" and len(body) < wav.stat().st_size / 5
    )  # 5 s (the phrase and 2 s either side) of a 60 s recording


def test_ocr_reads_a_page_through_the_service(service, tmp_path):
    """The shipped `ocr` role on the real Tesseract, through the client as strata-scan calls it: a CPU job, so it never takes a GPU slot."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", (1200, 300), "white")
    ImageDraw.Draw(im).text((80, 110), "Punch and Judy", fill="black", font=ImageFont.load_default(size=64))
    im.save(tmp_path / "page.png", dpi=(300, 300))
    r = sm.ocr(str(tmp_path / "page.png"), app="it", pdf=str(tmp_path / "page.pdf"))
    assert r["text"].lower().split() == ["punch", "and", "judy"] and r["model"] == "tesseract-eng"
    assert (tmp_path / "page.pdf").read_bytes()[:5] == b"%PDF-"
    assert sm.role_model("ocr") == "tesseract-eng"


def test_shots_finds_the_cuts_through_the_service(service, tmp_path):
    """The shipped `shot-boundaries` role on the real ffmpeg, through the client as strata-reels calls it: a CPU job, like ocr."""
    import subprocess

    video = tmp_path / "two-shots.mp4"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=s=160x120:r=25:d=2",
        "-f",
        "lavfi",
        "-i",
        "smptebars=s=160x120:r=25:d=2",
    ]
    subprocess.run(
        cmd + ["-filter_complex", "[0][1]concat=n=2:v=1[v]", "-map", "[v]", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    r = sm.shots(str(video), app="it")
    assert [c["time"] for c in r["cuts"]] == [2.0] and len(r["shots"]) == 2 and r["model"] == "ffmpeg-scene-score"
    assert sm.role_model("shot-boundaries") == "ffmpeg-scene-score"
    result, p = sm.run_with_provenance("shot-boundaries", dict(path=str(video)), app="it")
    now = sm.role_settings("shot-boundaries")
    assert result["cuts"] == r["cuts"] and p["model"] == now["model"] == "ffmpeg-scene-score"
    assert p["options"]["threshold"] == 0.2 and p["settings_hash"] == now["settings_hash"] and p["job_id"] > 0
    assert sm.provenance(get(service, f"/api/jobs/{p['job_id']}")) == p
