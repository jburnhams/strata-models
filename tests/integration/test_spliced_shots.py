"""scripts/evals/spliced_shots.py with the real ffmpeg: two generated camera files spliced into videos whose cuts are known, then found
again by the shot-boundaries runtime. ffmpeg is installed in CI for this (integration.yml)."""

import importlib.util
import json
import os
import subprocess

import pytest

from stratamodels.registry import Model
from stratamodels.runtimes import scenes

spec = importlib.util.spec_from_file_location(
    "spliced_shots", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "evals", "spliced_shots.py")
)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


@pytest.fixture(scope="module")
def cameras(tmp_path_factory):
    """Two 'uncut' files of different shape and rate, one of them flagged interlaced."""
    d = tmp_path_factory.mktemp("cameras")
    for name, src, extra in (
        ("a.mp4", "testsrc2=s=480x270:r=30:d=30", []),
        ("b.mkv", "mandelbrot=s=320x240:r=25,trim=duration=30", ["-flags", "+ildct+ilme", "-field_order", "tt"]),
    ):
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                src,
                *extra,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(d / name),
            ],
            check=True,
        )
    return d


def frames_of(path):
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,width,height,r_frame_rate",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(r.stdout)["streams"][0]


def test_spliced_videos_have_their_frames_and_their_cuts_are_found(cameras):
    ds = B.build([str(cameras)], "spliced-it", items=1, pieces=4, size=(320, 180), far_s=10.0)
    assert {i["tags"]["kind"] for i in ds.items} == {"files", "far", "near"}
    for it in ds.items:
        ref = it["reference"]
        s = frames_of(it["input"]["path"])
        assert (s["width"], s["height"], s["r_frame_rate"]) == (320, 180, "25/1")
        assert int(s["nb_read_frames"]) == round(ref["pieces"][-1]["at"] * 25) + round(
            (ref["pieces"][-1]["end"] - ref["pieces"][-1]["start"]) * 25
        )
    files = next(i for i in ds.items if i["tags"]["kind"] == "files")
    h = scenes.load(
        Model(
            key="ffmpeg-scene-score",
            runtime="ffmpeg-scenes",
            name="scene score",
            roles=("shot-boundaries",),
            vram_gb=0,
            options={},
        ),
        "",
    )
    found = [c["time"] for c in scenes.run(h, dict(path=files["input"]["path"]))["cuts"]]
    assert len(found) == len(files["reference"]["cuts"])
    for want, got in zip(files["reference"]["cuts"], found, strict=True):
        assert abs(want - got) < 0.001  # to the frame: the spliced cut is exact
