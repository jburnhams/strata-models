"""The Tesseract runtime (role `ocr`): TSV parsing on Tesseract's own output shape, and the real command on a rendered page (Tesseract is
installed in CI for this: apt, brew and choco in unit.yml). No mock of the command: a page drawn with Pillow is read back."""

import os

import pytest
from PIL import Image, ImageDraw, ImageFont

from stratamodels import runtimes
from stratamodels.registry import Model
from stratamodels.runtimes import RunError, tesseract

HEAD = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"


def row(level, block, par, line, word, box, conf, text=""):
    x, y, w, h = box
    return f"{level}\t1\t{block}\t{par}\t{line}\t{word}\t{x}\t{y}\t{w}\t{h}\t{conf}\t{text}\n"


def test_parse_tsv_groups_words_into_lines_and_drops_empty_ones():
    tsv = (
        HEAD
        + row(1, 0, 0, 0, 0, (0, 0, 800, 1200), -1)
        + row(4, 1, 1, 1, 0, (10, 10, 200, 30), -1)
        + row(5, 1, 1, 1, 1, (10, 10, 90, 30), 96.5, "Punch")
        + row(5, 1, 1, 1, 2, (110, 12, 100, 28), 90.5, "and")
        + row(5, 1, 2, 1, 1, (10, 60, 60, 30), 80, "Judy")  # paragraph 2 restarts line numbers: still its own line
        + row(5, 2, 1, 1, 1, (10, 300, 400, 400), 95, " ")  # a picture block: an empty word, no line
        + row(5, 3, 1, 1, 1, (5, 900, 20, 20), -1, "~")  # conf -1 counts as 0
    )
    r = tesseract.parse_tsv(tsv)
    assert (r["width"], r["height"]) == (800, 1200)
    assert [ln["text"] for ln in r["lines"]] == ["Punch and", "Judy", "~"]
    assert r["lines"][0]["box"] == [10, 10, 200, 30] and r["lines"][0]["conf"] == 93.5 and r["lines"][0]["block"] == 1
    assert r["lines"][2]["conf"] == 0.0
    assert r["text"] == "Punch and\nJudy\n~"
    assert tesseract.parse_tsv(HEAD) == dict(width=0, height=0, lines=[], text="")


def page(path, lines):
    im = Image.new("RGB", (1400, 220 + 110 * len(lines)), "white")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=64)
    for i, t in enumerate(lines):
        d.text((90, 100 + 110 * i), t, fill="black", font=font)
    im.save(path, dpi=(300, 300))
    return path


def model(**options):
    return Model(key="tesseract-eng", runtime="tesseract", name="eng", roles=("ocr",), vram_gb=0, options=options)


def test_reads_a_page_and_writes_a_searchable_pdf(tmp_path):
    h = tesseract.load(model(psm=3), str(tmp_path))
    assert h["version"].lower().startswith("tesseract")
    r = tesseract.run(
        h,
        dict(
            path=page(str(tmp_path / "p.png"), ["The tragical comedy", "of Punch and Judy"]),
            pdf=str(tmp_path / "p.pdf"),
        ),
    )
    words = r["text"].lower().split()
    assert "punch" in words and "judy" in words and "comedy" in words
    assert len(r["lines"]) == 2 and r["lines"][1]["box"][1] > r["lines"][0]["box"][1]
    assert all(0 <= w["conf"] <= 100 for ln in r["lines"] for w in ln["words"])
    assert (r["width"], r["height"]) == (1400, 440) and r["lang"] == "eng" and r["model"] == "tesseract-eng"
    with open(r["pdf"], "rb") as f:
        assert f.read(5) == b"%PDF-"
    blank = tesseract.run(h, dict(path=page(str(tmp_path / "b.png"), []), psm=6, dpi=300))
    assert blank["lines"] == [] and blank["text"] == "" and blank["psm"] == 6
    tesseract.unload(h)
    assert h == {}


def test_bad_params_and_a_missing_command(tmp_path, monkeypatch):
    h = tesseract.load(model(), str(tmp_path))
    with pytest.raises(RunError, match="no such picture"):
        tesseract.run(h, dict(path=str(tmp_path / "none.png")))
    p = page(str(tmp_path / "p.png"), ["x"])
    with pytest.raises(RunError, match="pdf must be"):
        tesseract.run(h, dict(path=p, pdf=str(tmp_path / "nowhere" / "x.pdf")))
    with pytest.raises(RunError, match="tesseract failed"):
        tesseract.run(h, dict(path=p, lang="zz-not-a-language"))
    with open(tmp_path / "junk.png", "w") as f:
        f.write("not a picture")
    with pytest.raises(RunError, match="tesseract failed"):
        tesseract.run(h, dict(path=str(tmp_path / "junk.png")))
    assert tesseract.binary(h["exe"]) == h["exe"] and tesseract.binary("no-such-tesseract") is None
    with pytest.raises(RunError, match="cannot run"):
        tesseract.load(model(binary=str(tmp_path / "junk.png")), str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path))
    assert tesseract.binary() is None or os.name == "nt"
    if os.name != "nt":
        with pytest.raises(RunError, match="not installed"):
            tesseract.load(model(), str(tmp_path))
        assert "the tesseract command (Tesseract 5)" in runtimes.available()["tesseract"]
