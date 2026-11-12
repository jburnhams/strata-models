"""scripts/evals/video_clips.py: a video-enhance dataset from a folder of clips and its clips.toml, with fake ffmpeg (no video is decoded)."""

import importlib.util
import json
import os
import subprocess

import pytest

spec = importlib.util.spec_from_file_location(
    "video_clips", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "evals", "video_clips.py")
)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

MANIFEST = """description = "two clips"
crop = [14, 0, 692, 568]
field_order = "tff"
[[clips]]
id = "a"
file = "a.mkv"
note = "a face"
tags = { content = "face" }
regions = [{ id = "face", label = "Face", box = [200, 90, 260, 195] }]
[[clips]]
id = "b"
file = "b.mkv"
deinterlace = "none"
crop = [0, 0, 720, 576]
"""


def fake_run(calls):
    def run(cmd, capture_output=True, text=False, **kw):
        calls.append(cmd)
        if cmd[0] == "ffprobe":
            out = cmd[-1].endswith(".mp4")
            v = dict(
                codec_type="video",
                width=1440 if out else 720,
                height=1080 if out else 576,
                r_frame_rate="50/1" if out else "25/1",
                nb_frames="400",
            )
            return subprocess.CompletedProcess(cmd, 0, json.dumps(dict(streams=[v], format=dict(duration="8.0"))), "")
        open(cmd[-1], "wb").close()
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    return run


@pytest.fixture
def folder(tmp_path):
    for n in ("a.mkv", "b.mkv"):
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "clips.toml").write_text(MANIFEST)
    return tmp_path


def test_build_writes_items_with_the_candidates_choice_of_deinterlacer_and_source_previews(folder):
    calls = []
    ds = B.build(str(folder), "vhs-test", run=fake_run(calls))
    a, b = ds.items
    assert a["input"] == dict(
        path=str(folder / "a.mkv"),
        field_order="tff",
        crop=[14, 0, 692, 568],
        regions=[dict(id="face", label="Face", box=[200, 90, 260, 195])],
    )
    assert (
        "deinterlace" not in a["input"]
        and b["input"]["deinterlace"] == "none"
        and b["input"]["crop"] == [0, 0, 720, 576]
    )
    media = os.path.join(os.environ["STRATAMODELS_DATA"], "evals", "media", "vhs-test")
    assert a["reference"]["preview"] == os.path.join(media, "a.source.mp4") and a["reference"]["crops"][0][
        "path"
    ] == os.path.join(media, "a.source.face.mp4")
    assert a["tags"] == dict(content="face") and b["reference"]["crops"] == []
    enc = next(c for c in calls if c[0] == "ffmpeg" and c[-1].endswith("a.source.mp4"))
    assert "scale=1440:1080:flags=neighbor" in enc[enc.index("-vf") + 1]  # the tape enlarged, not enhanced
    n = len(calls)
    B.build(str(folder), "vhs-test", run=fake_run(calls))
    assert not any(
        c[0] == "ffmpeg" and c[-1].endswith(".source.mp4") for c in calls[n:]
    )  # the previews are kept, not made again (the crops are cheap and made again)


def test_build_without_media_and_errors(folder, tmp_path):
    ds = B.build(str(folder), "vhs-items", make=False)
    assert "preview" not in ds.items[0]["reference"]
    (folder / "b.mkv").unlink()
    with pytest.raises(SystemExit, match="no clip"):
        B.build(str(folder), "vhs-items", make=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "clips.toml").write_text('description = "none"\n')
    with pytest.raises(SystemExit, match="lists no clips"):
        B.build(str(empty), "x", make=False)


def test_main_prints_the_version(folder, capsys):
    B.main([str(folder), "--name", "vhs-cli", "--no-media"])
    assert capsys.readouterr().out.startswith("2 clips, version ")
