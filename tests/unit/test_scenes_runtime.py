"""The shot-boundaries runtime (stratamodels/runtimes/scenes.py) without ffmpeg: reading its metadata listing, choosing cuts, pairing black
marks, and refusing bad params. tests/integration/test_scenes_runtime.py runs the real command on generated video."""

from types import SimpleNamespace

import pytest

from stratamodels.runtimes import RunError, scenes

LISTING = """frame:0    pts:0       pts_time:0
lavfi.scene_score=0.000000
frame:1    pts:512     pts_time:0.04
lavfi.scene_score=0.010000
frame:2    pts:1024    pts_time:0.08
lavfi.black_start=0.08
lavfi.scene_score=0.900000
frame:3    pts:1536    pts_time:0.12
lavfi.black_end=0.12
lavfi.scene_score=0.700000
lavfi.other=1
"""


def test_the_listing_gives_each_frame_its_time_score_and_black_marks():
    frames = scenes.parse_metadata(LISTING)
    assert [f["time"] for f in frames] == [0, 0.04, 0.08, 0.12]
    assert frames[2] == {"time": 0.08, "black_start": 0.08, "score": 0.9}
    assert frames[3] == {"time": 0.12, "black_end": 0.12, "score": 0.7}  # an unknown key is left out
    assert scenes.parse_metadata("lavfi.scene_score=1\n") == []  # a value before any frame line belongs to nothing


def test_cuts_are_the_strongest_frames_at_least_min_shot_apart_in_time_order():
    times = [i / 25 for i in range(100)]
    scores = [0.0] * 100
    scores[0] = 1.0  # the first frame has nothing before it: never a cut
    scores[3] = 0.5  # close after the start: kept (a clip may start just before a real cut)
    scores[40], scores[42] = 0.6, 0.95  # a noisy restart: the stronger of two frames 0.08 s apart
    scores[80] = 0.29  # under the threshold
    assert scenes.pick_cuts(scores, times, 0.3, 0.6) == [3, 42]
    assert scenes.pick_cuts(scores, times, 0.25, 0.6) == [3, 42, 80]
    assert scenes.pick_cuts(scores, times, 0.3, 0.0) == [3, 40, 42]


def test_black_marks_pair_into_stretches_and_one_still_open_ends_with_the_video():
    frames = [
        {"time": 0, "black_start": 1.0},
        {"time": 1, "black_end": 2.0},
        {"time": 2, "black_end": 9.0},
        {"time": 3, "black_start": 5.0},
    ]
    assert scenes.black_stretches(frames, 7.0) == [{"start": 1.0, "end": 2.0}, {"start": 5.0, "end": 7.0}]
    # blackdetect marks a flash shorter than its d= too, so the minimum is applied here; one still open at the end may go on, so it stays
    assert scenes.black_stretches(frames, 5.5, 1.5) == [{"start": 5.0, "end": 5.5}]


def test_load_needs_ffmpeg_and_reads_the_options(monkeypatch, tmp_path):
    model = SimpleNamespace(key="ffmpeg-scene-score", options={"threshold": 0.25, "width": 200})
    monkeypatch.setattr(scenes.shutil, "which", lambda name: None)
    with pytest.raises(RunError, match="ffmpeg is not installed"):
        scenes.load(model, str(tmp_path))
    assert scenes.binary(str(tmp_path / "nope")) is None
    exe = tmp_path / "ffmpeg"
    exe.write_text("")
    assert scenes.binary(str(exe)) == str(exe)

    def fake_run(cmd, **kw):
        return SimpleNamespace(stdout="ffmpeg version 8.0 Copyright\nmore\n", returncode=0)

    monkeypatch.setattr(scenes.subprocess, "run", fake_run)
    h = scenes.load(SimpleNamespace(key=model.key, options={**model.options, "binary": str(exe)}), str(tmp_path))
    assert h["version"] == "ffmpeg version 8.0 Copyright" and h["threshold"] == 0.25 and h["width"] == 200
    assert h["min_shot_s"] == 0.6 and h["deinterlace"] is True
    scenes.unload(h)
    assert h == {}


def test_load_reports_a_command_that_will_not_start(monkeypatch, tmp_path):
    exe = tmp_path / "ffmpeg"
    exe.write_text("")

    def boom(cmd, **kw):
        raise OSError("exec format error")

    monkeypatch.setattr(scenes.subprocess, "run", boom)
    with pytest.raises(RunError, match="cannot run"):
        scenes.load(SimpleNamespace(key="k", options={"binary": str(exe)}), str(tmp_path))


HANDLE = dict(exe="ffmpeg", version="v", threshold=0.3, min_shot_s=0.6, width=160, deinterlace=True, key="k")


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"threshold": "high"}, "threshold must be a number"),
        ({"threshold": 2}, "threshold must be from 0.0 to 1.0"),
        ({"width": 8}, "width must be from 32"),
        ({"clip": [5]}, r"clip must be \[start, end\] in seconds"),
        ({"clip": [5, 2]}, "0 <= start < end"),
    ],
)
def test_bad_params_are_refused_before_ffmpeg_runs(tmp_path, params, message):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    with pytest.raises(RunError, match=message):
        scenes.run(dict(HANDLE), {"path": str(video), **params})
    with pytest.raises(RunError, match="no such video"):
        scenes.run(dict(HANDLE), {"path": str(tmp_path / "missing.mp4")})


def test_a_failed_or_hung_ffmpeg_is_a_run_error(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(
        scenes.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=1, stderr="moov atom not found\n")
    )
    with pytest.raises(RunError, match="moov atom not found"):
        scenes.run(dict(HANDLE), {"path": str(video)})

    def hang(cmd, **kw):
        raise scenes.subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(scenes.subprocess, "run", hang)
    with pytest.raises(RunError, match="ffmpeg took over"):
        scenes.run(dict(HANDLE), {"path": str(video)})


def test_the_filter_and_answer_from_a_listing(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    seen = {}

    def fake_run(cmd, cwd, **kw):
        seen["cmd"] = cmd
        with open(f"{cwd}/frames.txt", "w", encoding="utf8") as f:
            f.write(LISTING.replace("lavfi.black_end=0.12\n", ""))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(scenes.subprocess, "run", fake_run)
    r = scenes.run(dict(HANDLE), {"path": str(video), "clip": [10, 20], "scores": True, "min_shot_s": 0, "width": 201})
    vf = seen["cmd"][seen["cmd"].index("-vf") + 1]
    assert vf.startswith("yadif=mode=0,scale=200:-2,blackdetect=") and vf.endswith(
        "metadata=mode=print:file=frames.txt"
    )
    assert seen["cmd"][seen["cmd"].index("-ss") + 1] == "10.000000"
    assert r["cuts"] == [{"time": 10.08, "frame": 2, "score": 0.9}, {"time": 10.12, "frame": 3, "score": 0.7}]
    assert r["shots"][0] == {"start": 10.0, "end": 10.08} and r["shots"][-1]["end"] == 10.16
    assert r["black"] == [{"start": 10.08, "end": 10.16}]  # still black when the clip ends
    assert r["fps"] == 25.0 and r["frames"] == 4 and r["duration"] == 0.16 and r["scores"] == [0, 0.01, 0.9, 0.7]
    r = scenes.run(dict(HANDLE), {"path": str(video), "black": False, "deinterlace": False})
    vf = seen["cmd"][seen["cmd"].index("-vf") + 1]
    assert vf.startswith("scale=160:-2,select=") and r["black"] == [] and "scores" not in r


def test_no_frames_is_a_run_error(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")

    def fake_run(cmd, cwd, **kw):
        open(f"{cwd}/frames.txt", "w").close()
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(scenes.subprocess, "run", fake_run)
    with pytest.raises(RunError, match="no video frames"):
        scenes.run(dict(HANDLE), {"path": str(video)})
