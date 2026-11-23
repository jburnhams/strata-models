"""scripts/evals/spliced_shots.py: planning spliced videos with known cuts (no video is decoded here; tests/integration splices real ones)."""

import importlib.util
import itertools
import json
import os
import random
import subprocess
from fractions import Fraction

import pytest

spec = importlib.util.spec_from_file_location(
    "spliced_shots", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "evals", "spliced_shots.py")
)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

LENGTHS = {"/v/a.mp4": 300.0, "/v/b.mp4": 200.0, "/v/c.mp4": 40.0}
ARGS = dict(pieces=6, fps=25, min_s=2.0, max_s=6.0, far_s=60.0, near_s=(2.0, 12.0))


def ends(p):
    return p.start + Fraction(p.frames) / 25


@pytest.mark.parametrize("kind", B.KINDS)
def test_every_stretch_lies_inside_its_file_and_lasts_whole_frames(kind):
    for seed in range(30):
        ps = B.plan(kind, LENGTHS, random.Random(seed), **ARGS)
        assert len(ps) == 6
        for p in ps:
            assert p.start >= 0
            assert ends(p) <= LENGTHS[p.file] - 1
            assert 50 <= p.frames <= 150


def test_files_cuts_always_change_file():
    ps = B.plan("files", LENGTHS, random.Random(3), **ARGS)
    assert all(a.file != b.file for a, b in itertools.pairwise(ps))


@pytest.mark.parametrize(("kind", "lo", "hi"), [("far", 60.0, 1e9), ("near", 2.0, 12.0)])
def test_same_file_cuts_jump_the_asked_distance(kind, lo, hi):
    for seed in range(30):
        ps = B.plan(kind, LENGTHS, random.Random(seed), **ARGS)
        assert len({p.file for p in ps}) == 1
        if kind == "far":
            assert ps[0].file != "/v/c.mp4"  # 40 s is too short to jump 60 s
        for a, b in itertools.pairwise(ps):
            gap = abs(b.start - ends(a))
            assert lo - Fraction(1, 10) <= gap <= hi + Fraction(1, 10)


def test_a_kind_the_sources_cannot_give_is_none():
    assert B.plan("files", {"/v/a.mp4": 300.0}, random.Random(1), **ARGS) is None
    assert B.plan("far", {"/v/c.mp4": 40.0}, random.Random(1), **ARGS) is None


def test_cuts_are_exact_frame_counts():
    ps = [B.Piece("a", Fraction(0), 50), B.Piece("a", Fraction(9), 63), B.Piece("b", Fraction(1), 75)]
    assert B.cuts(ps, 25) == [Fraction(2), Fraction(113, 25)]


def test_sources_walks_folders_in_order_and_refuses_a_missing_path(tmp_path):
    (tmp_path / "x").mkdir()
    for n in ("x/b.MP4", "x/a.mov", "notes.txt", "z.mkv"):
        (tmp_path / n).write_bytes(b"")
    assert [os.path.relpath(f, tmp_path) for f in B.sources([str(tmp_path)])] == ["z.mkv", "x/a.mov", "x/b.MP4"]
    with pytest.raises(SystemExit, match="no file or folder"):
        B.sources([str(tmp_path / "gone")])


def ffprobe_says(lengths):
    """ffprobe, answered from a table (the real one runs in tests/integration/test_spliced_shots.py)."""

    def run(cmd, **_):
        d = lengths.get(cmd[-1])
        body = {"streams": [{"codec_type": "video"}], "format": {"duration": str(d)}} if d else {"streams": []}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(body), "")

    return run


def test_build_writes_items_by_kind_with_their_known_cuts(tmp_path):
    lengths = {}
    for n, d in (("a.mp4", 300.0), ("b.mp4", 200.0), ("audio.mp4", None)):
        (tmp_path / n).write_bytes(b"")
        lengths[str(tmp_path / n)] = d
    ds = B.build([str(tmp_path)], "spliced-test", items=2, pieces=5, run=ffprobe_says(lengths), make=False)
    assert [i["id"] for i in ds.items] == ["files-01", "files-02", "far-01", "far-02", "near-01", "near-02"]
    for it in ds.items:
        ref = it["reference"]
        assert len(ref["cuts"]) == 4 and ref["tolerance_s"] == 0.5
        assert [p["at"] for p in ref["pieces"][1:]] == ref["cuts"]
        assert it["tags"] == {"kind": it["id"][:-3]}
        assert it["input"]["path"].endswith(f"spliced-test/{it['id']}.mp4")
    assert "6 spliced videos, 24 known cuts, from 2 uncut files (seed 1)" in ds.description
    again = B.build([str(tmp_path)], "spliced-test", items=2, pieces=5, run=ffprobe_says(lengths), make=False)
    assert again.items == ds.items  # the same seed gives the same dataset


def test_build_says_which_kinds_it_could_not_make_and_refuses_no_sources(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    ds = B.build(
        [str(tmp_path)], "spliced-test", items=1, run=ffprobe_says({str(tmp_path / "a.mp4"): 30.0}), make=False
    )
    assert [i["id"] for i in ds.items] == ["near-01"]
    assert "no files, far items" in ds.description
    with pytest.raises(SystemExit, match="no video of at least 8 s"):
        B.build([str(tmp_path)], "spliced-test", run=ffprobe_says({}), make=False)
