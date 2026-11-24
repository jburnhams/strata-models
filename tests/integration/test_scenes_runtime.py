"""The shot-boundaries runtime running the real ffmpeg on video it generates: four shots (a test pattern, black, a fractal, colour bars) with
cuts at 2, 3 and 5 s. ffmpeg is installed in CI for this (apt, brew and choco in integration.yml)."""

import subprocess

import pytest

from stratamodels.registry import Model
from stratamodels.runtimes import RunError, scenes


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("scenes") / "four-shots.mp4"
    sources = [
        "testsrc=s=320x240:r=25:d=2",
        "color=black:s=320x240:r=25:d=1",
        "mandelbrot=s=320x240:r=25",
        "smptebars=s=320x240:r=25:d=2",
    ]
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    for s in sources:
        cmd += ["-f", "lavfi", "-i", s]
    graph = "[2]trim=duration=2,setpts=PTS-STARTPTS[m];[0][1][m][3]concat=n=4:v=1[v]"
    subprocess.run(
        cmd + ["-filter_complex", graph, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True
    )
    return str(path)


def handle(tmp_path):
    return scenes.load(
        Model(
            key="ffmpeg-scene-score",
            runtime="ffmpeg-scenes",
            name="scene score",
            roles=("shot-boundaries",),
            vram_gb=0,
            options={},
        ),
        str(tmp_path),
    )


def test_finds_the_cuts_shots_and_black_stretch(video, tmp_path):
    h = handle(tmp_path)
    assert h["version"].startswith("ffmpeg")
    r = scenes.run(h, {"path": video})
    assert [c["time"] for c in r["cuts"]] == [2.0, 3.0, 5.0]
    assert [c["frame"] for c in r["cuts"]] == [50, 75, 125]
    assert [(s["start"], s["end"]) for s in r["shots"]] == [(0.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 7.0)]
    assert r["black"] == [{"start": 2.0, "end": 3.0}]
    assert r["fps"] == 25.0 and r["frames"] == 175 and r["duration"] == 7.0


def test_a_clip_answers_in_the_files_time(video, tmp_path):
    r = scenes.run(handle(tmp_path), {"path": video, "clip": [1.5, 5.5], "deinterlace": False, "scores": True})
    assert [c["time"] for c in r["cuts"]] == [2.0, 3.0, 5.0]  # 0.5 s into the clip: still a cut
    # 1.5 s falls between two frames (1.48, 1.52): the first frame looked at is the one after it
    assert (
        abs(r["shots"][0]["start"] - 1.5) <= 0.04
        and abs(r["shots"][-1]["end"] - 5.5) <= 0.04
        and abs(len(r["scores"]) - 100) <= 1
    )


def test_a_file_that_is_not_video_fails_with_ffmpegs_reason(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    with pytest.raises(RunError, match="ffmpeg failed"):
        scenes.run(handle(tmp_path), {"path": str(bad)})


def test_a_flashing_stretch_with_no_picture_is_one_black_shot(tmp_path):
    """A test pattern, then 2 s of flat grey that jumps between two levels every few frames and 2 s of snow: one gap, no cut in it."""
    path = tmp_path / "lost.mp4"
    flash = "geq=lum='if(lt(mod(N,8),4),30,190)':cb=128:cr=128"
    snow = "geq=lum='random(1)*255':cb=128:cr=128"
    grey = "color=gray:s=720x576:r=25:d=2"
    graph = f"[1]{flash}[f];[2]{snow}[n];[0][f][n]concat=n=3:v=1,format=yuv420p[v]"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=720x576:r=25:d=2"]
    cmd += ["-f", "lavfi", "-i", grey, "-f", "lavfi", "-i", grey, "-filter_complex", graph, "-map", "[v]"]
    subprocess.run([*cmd, "-c:v", "libx264", "-crf", "10", str(path)], check=True)
    r = scenes.run(handle(tmp_path), {"path": str(path)})
    assert [c["frame"] for c in r["cuts"]] == [50] and r["black"] == [{"start": 2.0, "end": 6.0}]
    assert r["frames"] == 150
    # without it, the flashes are cuts
    assert len(scenes.run(handle(tmp_path), {"path": str(path), "blank": False})["cuts"]) > 1
