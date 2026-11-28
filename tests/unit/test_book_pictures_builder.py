"""scripts/evals/book_pictures.py: the image-kind and vectorise-plates datasets from a folder of strata-scan's figures, its labels, and
pictures from anywhere else (`--other`), on small drawn files."""

import importlib.util
import os
import sys

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "book_pictures", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "evals", "book_pictures.py")
)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def pic(path, size):
    Image.new("RGB", size, (200, 190, 170)).save(path)
    return str(path)


def lines(name):
    from stratamodels.evals import core as C

    return C.load_dataset(name).items


def test_files_take_their_pages_label_by_shape_and_others_join_by_path(tmp_path, monkeypatch, capsys):
    d = tmp_path / "export"
    d.mkdir()
    pic(d / "c1828-p010-f1.png", (200, 100))  # wide: the frieze
    pic(d / "c1828-p010-f2.png", (100, 200))  # tall: the figure
    pic(d / "c1828-p010-fpage.png", (150, 150))  # square: neither, on a page of two labels
    pic(d / "c1828-p011-f3.png", (40, 300))  # a sliver: never a picture
    pic(d / "notes.txt.png", (100, 100))  # not a strata-scan name
    extra = tmp_path / "extra.toml"
    extra.write_text(
        '[[pictures]]\nbook = "c1828"\npage = 10\nshape = 2.0\nkind = "engraving"\nvectorise = true\n'
        '[[pictures]]\nbook = "c1828"\npage = 10\nshape = 0.5\nkind = "woodcut"\nnote = "faint"\n'
    )
    photo = pic(tmp_path / "IMG_1.jpg", (120, 90))
    other = tmp_path / "others.toml"
    other.write_text(f'[[pictures]]\npath = "{photo}"\nkind = "photograph"\nsource = "Takeout photos"\n')
    (tmp_path / "here").mkdir()
    (tmp_path / "here" / "labels.toml").write_text(
        "pictures = []\n"
    )  # the repo's own labels left out: only this test's
    monkeypatch.setattr(B, "HERE", str(tmp_path / "here"))
    monkeypatch.setattr(sys, "argv", ["b", "--pictures", str(d), "--extra", str(extra), "--other", str(other)])
    B.main()
    kinds = {it["id"]: it for it in lines("image-kind")}
    assert kinds["c1828-p010-f1"]["reference"] == {"kind": "engraving"}
    assert kinds["c1828-p010-f2"]["tags"]["hard"] is True and kinds["c1828-p010-f2"]["tags"]["line_art"] is True
    (taken,) = [k for k in kinds if k.startswith("takeout-photos-")]
    assert kinds[taken]["tags"]["group"] == "photo" and os.path.isfile(kinds[taken]["input"]["image"])
    assert [it["id"] for it in lines("vectorise-plates")] == ["c1828-p010-f1"]
    out = capsys.readouterr().out
    assert "c1828-p010-fpage.png" in out and "p011" not in out and len(kinds) == 3


def test_unknown_kinds_and_missing_pictures_are_refused(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('[[pictures]]\nbook = "hist"\npage = 1\nshape = 1.0\nkind = "doodle"\n')
    with pytest.raises(SystemExit, match="unknown kind 'doodle'"):
        B.labels([str(bad)])
    gone = tmp_path / "gone.toml"
    gone.write_text('[[pictures]]\npath = "/no/such.jpg"\nkind = "photograph"\n')
    with pytest.raises(SystemExit, match="no such picture"):
        B.others([str(gone)])
    gone.write_text('[[pictures]]\npath = "/no/such.jpg"\nkind = "selfie"\n')
    with pytest.raises(SystemExit, match="unknown kind 'selfie'"):
        B.others([str(gone)])
