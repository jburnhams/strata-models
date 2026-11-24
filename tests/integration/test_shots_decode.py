"""The shot-boundary worker's decode with the real ffmpeg, on the four-shot video of test_scenes_runtime (cuts at 2, 3 and 5 s, black from 2
to 3 s). TransNet V2 itself is not in this suite's venv (it lives in the shots venv; tests/gpu runs it): a stand-in that scores each frame by
how unlike the next one it is (TransNet V2 puts a hard cut's peak on the old shot's last frame, as this does) checks everything around it:
frames in step with ffmpeg's times, the cut on the new shot's first frame, black stretches, a clip in the file's time."""

import itertools
import os
import sys

import numpy as np
import pytest
from test_scenes_runtime import video  # noqa: F401  (the fixture)

WORKERS = os.path.join(os.path.dirname(__file__), "..", "..", "src", "stratamodels", "workers")
sys.path.insert(0, os.path.abspath(WORKERS))
import shots as S  # noqa: E402
from protocol import ParamError  # noqa: E402


def unlike_the_next(frames):
    fs = [f.astype(np.int16) for f in frames]
    assert all(f.shape == (27, 48, 3) for f in fs)  # TransNet V2's input size
    d = [float(np.abs(a - b).mean()) / 255 for a, b in itertools.pairwise(fs)] + [0.0]
    return [min(1.0, 5 * x) for x in d]


@pytest.fixture
def transnet(monkeypatch):
    monkeypatch.setattr(S, "transnet_scores", unlike_the_next)
    S.M.clear()
    S.M.update(key="transnetv2", engine="transnetv2", opts={"threshold": 0.5}, version="stand-in", device="cpu")


def test_cuts_shots_and_black_in_the_baselines_shape(video, transnet):  # noqa: F811
    r = S.run({"path": video, "scores": True})
    assert [c["time"] for c in r["cuts"]] == [2.0, 3.0, 5.0] and [c["frame"] for c in r["cuts"]] == [50, 75, 125]
    assert [(s["start"], s["end"]) for s in r["shots"]] == [(0.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 7.0)]
    assert r["black"] == [{"start": 2.0, "end": 3.0}]
    assert r["fps"] == 25.0 and r["frames"] == 175 == len(r["scores"]) and r["duration"] == 7.0
    assert r["model"] == "transnetv2" and r["settings"] == dict(threshold=0.5, min_shot_s=0.6, deinterlace=True)


def test_a_clip_answers_in_the_files_time(video, transnet):  # noqa: F811
    r = S.run({"path": video, "clip": [1.5, 5.5], "deinterlace": False})
    assert [c["time"] for c in r["cuts"]] == [2.0, 3.0, 5.0]
    assert abs(r["shots"][0]["start"] - 1.5) <= 0.04 and abs(r["shots"][-1]["end"] - 5.5) <= 0.04


def test_pyscenedetect_engines_get_frames_at_their_width(video, tmp_path):  # noqa: F811
    s = S.settings_of("content", {}, {})
    p, err, frames, listing = S.decode(video, s, "content", None, str(tmp_path), True, 0.4, 0.1)
    got = list(frames)
    assert p.wait() == 0 and len(got) == 175 and got[0].shape == (192, 256, 3)  # 320 x 240 at 256 across
    err.close()
    assert len(S.parse_metadata(open(listing).read())) == 175


def test_a_file_that_is_not_video_fails_with_ffmpegs_reason(tmp_path, transnet):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    with pytest.raises(ParamError, match="ffmpeg failed|ffprobe could not"):
        S.run({"path": str(bad)})
