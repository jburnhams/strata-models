"""Rebuilding a page as a PDF (runtimes/pagebuild.py, role `page-build`): the real renderer (strata_python.pages, the pdf extra) on small
synthetic descriptions, checked by what a reader of the PDF gets back (its text in order, its size, the raster beside it)."""

import json
import os
from types import SimpleNamespace

import pytest
from PIL import Image

from stratamodels import settings
from stratamodels.runtimes import RunError, pagebuild

OCR = dict(
    width=600,
    height=400,
    lines=[
        dict(text="Punch and Judy", box=[50, 40, 400, 40], block=1),
        dict(text="  ", box=[50, 90, 10, 10], block=1),  # an empty line: dropped
        dict(text="A Comedy", box=[60, 100, 200, 30], block=1),
        dict(text="Folio 12", box=[250, 350, 100, 20], block=4),
    ],
)


def handle():
    return pagebuild.load(SimpleNamespace(key="pagebuild", options={}), "")


def test_from_ocr_makes_one_block_per_tesseract_block_in_order():
    page = pagebuild.from_ocr(OCR, 150)
    assert (page.width, page.height, page.dpi) == (600, 400, 150)
    assert [b.order for b in page.blocks] == [0, 1]
    assert [[ln.text for ln in b.lines] for b in page.blocks] == [["Punch and Judy", "A Comedy"], ["Folio 12"]]
    assert page.blocks[0].box == (50, 40, 400, 90)  # the union of its lines' boxes
    assert page.blocks[0].lines[0].spans[0].style == "roman"
    assert page.text == "Punch and Judy\nA Comedy\n\nFolio 12"


def test_from_ocr_needs_a_page_size():
    with pytest.raises(RunError, match="no page size"):
        pagebuild.from_ocr(dict(lines=[]), 300)


def test_run_draws_the_ocr_baseline_and_reads_its_text_back(tmp_path):
    out = tmp_path / "p.pdf"
    r = pagebuild.run(handle(), dict(ocr=OCR, dpi=150, out=str(out)))
    assert r["pdf"] == str(out)
    assert out.read_bytes().startswith(b"%PDF")
    assert (r["blocks"], r["lines"], r["dpi"], r["model"]) == (2, 3, 150, "pagebuild")
    assert r["text"].split() == ["Punch", "and", "Judy", "A", "Comedy", "Folio", "12"]
    with Image.open(r["png"]) as im:  # rasterised at the scan's dpi: the scan's own size, for the ink metrics
        assert im.size == (600, 400)


def test_run_caches_by_what_decides_the_rebuild(tmp_path):
    a = pagebuild.run(handle(), dict(ocr=OCR, dpi=150, png=False))
    b = pagebuild.run(handle(), dict(ocr=OCR, dpi=150, png=False))
    c = pagebuild.run(handle(), dict(ocr=OCR, dpi=150, png=False, fit=False))
    assert a["pdf"] == b["pdf"] != c["pdf"]
    assert a["pdf"].startswith(str(settings.data_dir()))
    assert a["png"] is None


def test_page_description_as_json_or_a_path(tmp_path):
    page = dict(
        width=300,
        height=200,
        dpi=100,
        blocks=[
            dict(
                box=[10, 10, 200, 30],
                role="heading",
                lines=[dict(box=[10, 10, 200, 30], spans=[dict(text="TITLE", style="caps")])],
            )
        ],
    )
    path = tmp_path / "page.json"
    path.write_text(json.dumps(page))
    assert pagebuild.page_of(dict(page=page)).to_dict() == pagebuild.page_of(dict(page=str(path))).to_dict()
    r = pagebuild.run(handle(), dict(page=str(path), out=str(tmp_path / "t.pdf"), png=False))
    assert r["text"].strip() == "TITLE"
    assert r["page"]["blocks"][0]["role"] == "heading"


def test_dpi_and_size_come_from_the_picture(tmp_path):
    pic = tmp_path / "scan.png"
    Image.new("L", (600, 400), 255).save(pic, dpi=(150, 150))
    page = pagebuild.page_of(dict(ocr=dict(OCR, width=0, height=0), image=str(pic)))
    assert (page.width, page.height, page.dpi) == (600, 400, 150)
    assert pagebuild.page_of(dict(ocr=OCR, image=str(pic), dpi=300)).dpi == 300  # a given dpi wins
    bare = tmp_path / "bare.png"
    Image.new("L", (60, 40), 255).save(bare)
    with pytest.raises(RunError, match="no dpi"):
        pagebuild.page_of(dict(ocr=OCR, image=str(bare)))
    with pytest.raises(RunError, match="no such picture"):
        pagebuild.page_of(dict(ocr=OCR, image=str(tmp_path / "missing.png")))


@pytest.mark.parametrize(
    ("params", "match"),
    [
        (dict(), "give a page description"),
        (dict(page="/nowhere/page.json"), "no such page description"),
        (dict(page=dict(width=1, height=1)), "bad page description"),
        (
            dict(page=dict(width=10, height=10, dpi=72, blocks=[dict(box=[0, 0, 1, 1], role="sidebar")])),
            "bad page description",
        ),
        (dict(ocr=OCR), "or its dpi"),
    ],
)
def test_page_of_refuses_what_it_cannot_draw(params, match):
    with pytest.raises(RunError, match=match):
        pagebuild.page_of(params)


def test_fonts_from_a_folder_and_by_name(tmp_path):
    (tmp_path / "OldStandardTT-Regular.ttf").write_bytes(b"")
    (tmp_path / "Other.otf").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("")
    fonts = pagebuild._fonts(dict(font_dir=str(tmp_path), fonts={"Other": "/x/other.ttf"}))
    assert fonts == {"OldStandardTT-Regular": str(tmp_path / "OldStandardTT-Regular.ttf"), "Other": "/x/other.ttf"}
    with pytest.raises(RunError, match="not a folder"):
        pagebuild._fonts(dict(font_dir=str(tmp_path / "nope")))


def test_run_errors(tmp_path):
    with pytest.raises(RunError, match="out must be a .pdf"):
        pagebuild.run(handle(), dict(ocr=OCR, dpi=150, out=str(tmp_path / "p.png")))
    page = dict(
        width=100,
        height=100,
        dpi=72,
        blocks=[dict(box=[0, 0, 50, 20], lines=[dict(box=[0, 0, 50, 20], face="f", spans=[dict(text="x")])])],
    )
    with pytest.raises(RunError, match="no such file"):  # a face named in the description whose font file is gone
        pagebuild.run(handle(), dict(page=page, fonts={"f": str(tmp_path / "f.ttf")}, png=False))


def test_unload_drops_the_handle():
    h = handle()
    pagebuild.unload(h)
    assert h == {}


def test_an_upscaled_ocr_answer_keeps_its_pixels_and_scales_the_dpi(tmp_path):
    # clean = "upscale" doubles the picture Tesseract reads: its boxes are in those pixels, so the page is drawn in them at twice the dpi
    pic = tmp_path / "scan.png"
    Image.new("L", (300, 200), 255).save(pic, dpi=(75, 75))
    page = pagebuild.page_of(dict(ocr=OCR, image=str(pic)))
    assert (page.width, page.height, page.dpi) == (600, 400, 150)
    with pytest.raises(RunError, match="not the same page"):
        pagebuild.page_of(dict(ocr=dict(OCR, height=600), image=str(pic)))


# ---- a vision model's layout (preset page-layout-v1): boxes in thousandths, blocks in reading order, lines without boxes -------------------
LAYOUT = dict(
    blocks=[
        dict(box=[100, 50, 700, 100], role="heading", lines=["PUNCH AND JUDY"]),
        dict(
            box=[100, 150, 900, 500],
            role="body",
            lines=["The Punch Puppet, has a big belly", "and a haunch on his back", " "],
        ),
        dict(box=[900, 950, 860, 980], role="folio", lines=[]),  # corners given the wrong way round: sorted
    ],
    figures=[dict(box=[400, 600, 600, 640], kind="ornament")],
)


def picture(tmp_path, size=(600, 400), dpi=150):
    pic = tmp_path / "scan.png"
    Image.new("L", size, 255).save(pic, dpi=(dpi, dpi))
    return str(pic)


def test_a_layout_is_drawn_in_the_pictures_pixels(tmp_path):
    page = pagebuild.page_of(dict(layout=LAYOUT, image=picture(tmp_path)))
    assert (page.width, page.height, page.dpi) == (600, 400, 150)
    assert [b.role for b in page.blocks] == ["heading", "body", "folio"]
    assert [b.order for b in page.blocks] == [0, 1, 2]
    assert page.blocks[0].box == pytest.approx((60, 20, 360, 20))
    assert page.blocks[2].box == pytest.approx((516, 380, 24, 12))
    body = page.blocks[1].lines  # the empty line dropped, the block's height shared by the rest
    assert [ln.text for ln in body] == ["The Punch Puppet, has a big belly", "and a haunch on his back"]
    assert [ln.box[1] for ln in body] == pytest.approx([60, 130])
    assert page.figures[0].kind == "ornament"
    assert page.figures[0].box == pytest.approx((240, 240, 120, 16))


def test_a_layout_as_a_models_answer_text(tmp_path):
    pic = picture(tmp_path)
    fenced = "```json\n" + json.dumps(LAYOUT) + "\n```"
    want = pagebuild.page_of(dict(layout=LAYOUT, image=pic)).to_dict()
    assert pagebuild.page_of(dict(layout=json.dumps(LAYOUT), image=pic)).to_dict() == want
    assert pagebuild.page_of(dict(layout=fenced, image=pic)).to_dict() == want
    # the item's own ocr answer is given too in the eval: the model's layout is what is drawn
    assert pagebuild.page_of(dict(layout=fenced, image=pic, ocr=OCR, dpi=150)).blocks[0].role == "heading"
    r = pagebuild.run(handle(), dict(layout=fenced, image=pic, out=str(tmp_path / "l.pdf")))
    assert r["text"].split()[:3] == ["PUNCH", "AND", "JUDY"]
    assert r["path"] == r["png"]


@pytest.mark.parametrize(
    ("layout", "match"),
    [
        ("{not json", "not valid JSON"),
        (dict(blocks=[dict(box=[0, 0, 10], role="body")]), "thousandths"),
        (dict(blocks=[dict(box=[0, 0, 10, 10], role="sidebar")]), "bad layout"),
        (dict(figures=[dict(box=[0, 0, 10, 10], kind="photo")]), "bad layout"),
        ("no such answer", "no such layout"),
    ],
)
def test_a_bad_layout_is_refused(tmp_path, layout, match):
    with pytest.raises(RunError, match=match):
        pagebuild.page_of(dict(layout=layout, image=picture(tmp_path)))


def test_an_ocr_answer_from_a_file(tmp_path):
    path = tmp_path / "scan.ocr.json"
    path.write_text(json.dumps(OCR))
    assert pagebuild.page_of(dict(ocr=str(path), dpi=150)).text == pagebuild.page_of(dict(ocr=OCR, dpi=150)).text


def test_the_raster_has_the_scans_own_size(tmp_path):
    # pdfium rounds a 2660 px page at 500 dpi (383.04 pt) up to 2661 px: the raster is put back on the scan's pixels for the ink metrics
    page = dict(
        width=2660,
        height=4056,
        dpi=500,
        blocks=[
            dict(box=[100, 100, 800, 80], lines=[dict(box=[100, 100, 800, 80], spans=[dict(text="WOODEN ACTORS")])])
        ],
    )
    r = pagebuild.run(handle(), dict(page=page, out=str(tmp_path / "w.pdf")))
    with Image.open(r["png"]) as im:
        assert im.size == (2660, 4056)
        assert im.getpixel((2659, 4055)) == 255  # padded with paper, not ink


def test_the_layout_preset_carries_its_schema_and_the_spec_sends_it():
    # `preset` (not `system_preset`) is what makes llama and the cloud runtimes send a preset's schema, so every vision stage must use it
    import tomllib

    from stratamodels.runtimes import llama

    text, schema = llama.preset("page-layout-v1")
    assert "thousandths" in text
    assert schema["required"] == ["blocks", "figures"]
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "evals", "specs", "page-layout.toml"), "rb") as f:
        spec = tomllib.load(f)
    stages = [s for c in spec["candidates"] for s in c.get("stages", [])]
    assert len(stages) == 4
    assert all(s["params"].get("preset") == "page-layout-v1" and "prompt" not in s["params"] for s in stages)
