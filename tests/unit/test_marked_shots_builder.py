"""scripts/evals/marked_shots.py: windows of one real tape whose cuts a person marked (no video is read)."""

import importlib.util
import itertools
import os
from fractions import Fraction

import pytest

spec = importlib.util.spec_from_file_location(
    "marked_shots", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "evals", "marked_shots.py")
)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def test_marks_are_frames_in_order_with_their_comments():
    assert B.read_marks("# a tape\n\n50\n 25 # signal: rolls in\n") == [(25, "signal: rolls in"), (50, "")]
    with pytest.raises(SystemExit, match="line 2: not a frame number: '1.5'"):
        B.read_marks("3\n1.5\n")


def test_windows_cover_the_tape_and_keep_clear_of_cuts():
    cuts = [Fraction(79), Fraction(81), Fraction(200)]
    w = B.windows(cuts, Fraction(300), Fraction(80))
    assert w[0][0] == 0 and w[-1][1] == 300 and all(a[1] == b[0] for a, b in itertools.pairwise(w))
    assert all(abs(edge - c) >= B.CLEAR_S for edge, _ in w[1:] for c in cuts)
    assert B.windows([Fraction(5)], Fraction(30), Fraction(80)) == [(0, 30)]
    assert B.windows([Fraction(98)], Fraction(100), Fraction(50)) == [(0, 50), (50, 100)]
    assert B.windows([Fraction(40), Fraction(43), Fraction(46)], Fraction(48), Fraction(30)) == [(0, 30), (30, 48)]
    assert B.windows([Fraction(b) for b in range(50, 79, 3)], Fraction(80), Fraction(50)) == [(0, 80)]


def test_the_items_are_clips_with_the_cuts_inside_them(monkeypatch, tmp_path, capsys):
    wrote = {}
    monkeypatch.setattr(
        B.C, "write_dataset", lambda name, items, description: wrote.update({name: items}) or description
    )
    video, marks = tmp_path / "t.mp4", tmp_path / "m.txt"
    video.write_bytes(b"x")
    marks.write_text("250\n3000\n4900 # signal: ends\n")
    B.main([str(video), str(marks), "--frames", "5000", "--window", "80"])
    items = wrote["shots-vhs"]
    assert [i["input"]["clip"] for i in items] == [[0.0, 80.0], [80.0, 200.0]]
    assert [i["reference"]["cuts"] for i in items] == [[10.0], [120.0, 196.0]]
    assert [i["tags"]["kind"] for i in items] == ["shots", "signal"] and items[0]["input"]["path"] == str(video)
    assert "2 windows of one real tape (200 s), 3 cuts marked by eye" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="past the video's end"):
        B.build(str(video), [(5000, "")], "n", 5000)
    with pytest.raises(SystemExit, match="no marked cuts"):
        B.build(str(video), [], "n", 5000)
    with pytest.raises(SystemExit, match="no video"):
        B.main([str(tmp_path / "gone.mp4"), str(marks), "--frames", "5000"])
