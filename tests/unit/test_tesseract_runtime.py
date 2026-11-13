"""The Tesseract runtime (role `ocr`): TSV parsing on Tesseract's own output shape, without the command. The real command on a rendered
page is in tests/integration/test_tesseract_runtime.py (unit tests run no external processes)."""

import pytest

from stratamodels.runtimes import RunError, tesseract

HEAD = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"


def row(level, block, par, line, word, box, conf, text="", page=1):
    x, y, w, h = box
    return f"{level}\t{page}\t{block}\t{par}\t{line}\t{word}\t{x}\t{y}\t{w}\t{h}\t{conf}\t{text}\n"


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


def test_parse_tsv_refuses_more_than_one_page():
    # a multipage TIFF restarts block, paragraph and line numbers on each page: one key would merge lines from different pages
    tsv = HEAD + row(1, 0, 0, 0, 0, (0, 0, 800, 1200), -1) + row(1, 0, 0, 0, 0, (0, 0, 800, 1200), -1, page=2)
    with pytest.raises(RunError, match="one page at a time"):
        tesseract.parse_tsv(tsv)
