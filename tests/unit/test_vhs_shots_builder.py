"""scripts/evals/vhs_shots.py: the shot-boundaries dataset from a folder of tapes and its shots.toml (no video is decoded)."""

import importlib.util
import json
import os

import pytest

spec = importlib.util.spec_from_file_location(
    "vhs_shots", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "evals", "vhs_shots.py")
)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

MANIFEST = """tolerance_s = 0.5
[[stretches]]
id = "t1-party"
file = "t1.mkv"
clip = [600, 900]
cuts = [640.0, 612.36]
note = "party"
tags = { kind = "indoor" }
[[stretches]]
id = "t1-garden"
file = "t1.mkv"
clip = [0, 60]
tolerance_s = 1.0
"""


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "t1.mkv").write_bytes(b"x")
    (tmp_path / "shots.toml").write_text(MANIFEST)
    return tmp_path


def test_build_writes_items_with_clip_and_marked_cuts(folder):
    ds = B.build(str(folder), "vhs-shots-test")
    a, b = ds.items
    assert a["input"] == {"path": str(folder / "t1.mkv"), "clip": [600.0, 900.0]}
    assert a["reference"] == {"cuts": [612.36, 640.0], "tolerance_s": 0.5, "note": "party"} and a["tags"] == {
        "kind": "indoor"
    }
    assert b["reference"]["cuts"] == [] and b["reference"]["tolerance_s"] == 1.0
    assert "2 stretches of tape, 2 cuts marked by hand" in ds.description
    B.main([str(folder), "--name", "vhs-shots-test"])
    with open(os.path.join(B.C.datasets_dir(), "vhs-shots-test.jsonl"), encoding="utf8") as f:
        assert len([json.loads(line) for line in f if line.strip() and not line.startswith("#")]) == 2


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ('[[stretches]]\nid = "x"\nfile = "missing.mkv"\nclip = [0, 1]\n', "no tape"),
        ('[[stretches]]\nid = "x"\nfile = "t1.mkv"\nclip = [5, 1]\n', "0 <= start < end"),
        ('[[stretches]]\nid = "x"\nfile = "t1.mkv"\nclip = [0, 10]\ncuts = [12.0]\n', "cuts outside the clip"),
        ("description = 'empty'\n", "lists no stretches"),
    ],
)
def test_a_bad_manifest_is_refused(folder, manifest, message):
    (folder / "shots.toml").write_text(manifest)
    with pytest.raises(SystemExit, match=message):
        B.build(str(folder), "vhs-shots-test")
