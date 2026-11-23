"""Rebuilding a page as a PDF (runtimes/pagebuild.py, role `page-build`): the real renderer (strata_python.pages, the pdf extra) on small
synthetic descriptions, checked by what a reader of the PDF gets back (its text in order, its size, the raster beside it)."""

import json
import os
import re
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFont

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
    stages = [s for c in spec["candidates"] for s in c.get("stages", []) if s["role"] == "vision-language"]
    assert len(stages) == 12
    assert all(
        s["params"].get("preset") in ("page-layout-v1", "page-layout-fix-v1", "page-layout-fix-v2")
        and "prompt" not in s["params"]
        for s in stages
    )
    fix = [s for s in stages if s["params"]["preset"].startswith("page-layout-fix-")]
    assert len(fix) == 7 and all(
        s["send"] == "context" and s["also"] == ["image"] and s["into"] == "groups" for s in fix
    )
    for name in ("page-layout-fix-v1", "page-layout-fix-v2"):
        text, schema = llama.preset(name)
        assert "{context}" in text and schema["properties"]["blocks"]["items"]["required"] == ["lines", "role"]


# ---- Tesseract's lines regrouped by a vision model (preset page-layout-fix-v1): list_lines numbers them, groups names them by number ------
SPLIT = dict(
    width=1000,
    height=500,
    lines=[
        dict(
            text="THE HISTORY OF PUNCH. 17",  # a running head and its folio, read as one line: split at the gap
            box=[100, 20, 800, 20],
            block=1,
            words=[
                dict(text="THE", box=[100, 20, 60, 20]),
                dict(text="HISTORY", box=[170, 20, 140, 20]),
                dict(text="OF", box=[320, 20, 40, 20]),
                dict(text="PUNCH.", box=[370, 20, 120, 20]),
                dict(text="17", box=[860, 20, 40, 20]),
            ],
        ),
        dict(
            text="First line of text",
            box=[100, 100, 800, 20],
            block=2,
            words=[dict(text="First", box=[100, 100, 90, 20])],
        ),
        dict(text="second line", box=[100, 130, 500, 20], block=3),
        dict(text="%%", box=[950, 480, 20, 10], block=4),  # show-through: in no block
    ],
)


def test_list_lines_numbers_tesseract_lines_in_thousandths_split_at_wide_gaps():
    assert pagebuild.list_lines(SPLIT).splitlines() == [
        "1 b1 [100,40,490,80] THE HISTORY OF PUNCH.",
        "2 b1 [860,40,900,80] 17",
        "3 b2 [100,200,900,240] First line of text",
        "4 b3 [100,260,600,300] second line",
        "5 b4 [950,960,970,980] %%",
    ]
    out = pagebuild.run(handle(), dict(list_lines=True, ocr=json.dumps(SPLIT)))
    assert out["lines"] == 5 and out["text"].startswith("1 b1")
    with pytest.raises(RunError, match="needs an ocr"):
        pagebuild.run(handle(), dict(list_lines=True))
    with pytest.raises(RunError, match="no page size"):
        pagebuild.list_lines(dict(lines=[]))


def test_groups_keep_tesseracts_lines_in_the_models_blocks(tmp_path):
    groups = dict(
        blocks=[
            dict(lines=[1], role="running-head"),
            dict(lines=[2, "x", 99], role="folio"),  # not a number, out of range: skipped
            dict(lines=[4, 3, 1], role="body"),  # Tesseract's order within a block; 1 is taken already
            dict(lines=[], role="other"),  # nothing: dropped
        ],
        figures=[dict(box=[100, 500, 900, 900], kind="engraving")],
    )
    pic = str(tmp_path / "p.png")
    Image.new("L", (1000, 500), 255).save(pic, dpi=(200, 200))
    out = pagebuild.run(
        handle(),
        dict(
            ocr=json.dumps(SPLIT),
            groups="```json\n" + json.dumps(groups) + "\n```",
            image=pic,
            out=str(tmp_path / "g.pdf"),
        ),
    )
    page = out["page"]
    assert [(b.get("role", "body"), b["box"], b.get("order")) for b in page["blocks"]] == [
        ("running-head", [100, 20, 390, 20], 0),
        ("folio", [860, 20, 40, 20], 1),
        ("body", [100, 100, 800, 50], 2),
    ]
    assert [ln["spans"][0]["text"] for ln in page["blocks"][2]["lines"]] == ["First line of text", "second line"]
    assert page["figures"] == [dict(box=[100, 250, 800, 200], kind="engraving")] and page["dpi"] == 200
    with pytest.raises(RunError, match="bad groups"):
        pagebuild.page_of(dict(ocr=SPLIT, groups=dict(blocks=[dict(lines=[1], role="nonsense")]), dpi=200))


# ---- the full rebuild: corrected text (texts), fitted type (font_fit), sizes snapped, pictures and rules cut from the scan --------------
@pytest.fixture
def ttf(tmp_path):
    p = tmp_path / "aileron.ttf"
    p.write_bytes(
        ImageFont.load_default(20).path.getvalue()
    )  # Pillow's bundled TrueType: a real font file, no download
    return str(p)


def _word(text, box, style="roman", size=10.0, x=None):
    return dict(
        text=text,
        box=box,
        style=style,
        size_pt=size,
        x=x if x is not None else box[0],
        baseline_y=box[1] + box[3] - 2,
        hscale=1.05,
        tracking_em=0.01,
        stroke_em=0.0,
    )


TWO = dict(
    width=1000,
    height=500,
    lines=[
        dict(
            text="Tbe Punch",
            box=[100, 100, 300, 30],
            block=1,
            words=[dict(text="Tbe", box=[100, 100, 90, 30]), dict(text="Punch", box=[200, 100, 200, 30])],
        ),
        dict(
            text="and Jndy",
            box=[100, 140, 300, 30],
            block=1,
            words=[dict(text="and", box=[100, 140, 90, 30]), dict(text="Jndy", box=[200, 140, 200, 30])],
        ),
        dict(text="%%", box=[900, 450, 30, 20], block=2, words=[dict(text="%%", box=[900, 450, 30, 20])]),
    ],
)


def _fit(ttf, styles=("roman", "italic", "small-caps")):
    return dict(
        face="caslon",
        fonts={"caslon": dict(family="Caslon", files=dict.fromkeys(styles, ttf))},
        blocks=[
            dict(
                lines=[
                    dict(
                        words=[
                            _word("Tbe", [100, 100, 90, 30], size=10.0),
                            _word("Punch", [200, 100, 200, 30], "bold", 10.6),
                        ]
                    ),
                    dict(
                        words=[
                            _word("and", [100, 140, 90, 30], "italic", 13.0),
                            _word("Jndy", [200, 140, 200, 30], "small-caps", 9.8),
                        ]
                    ),
                ]
            )
        ],
    )


GROUPS = dict(blocks=[dict(lines=[1, 2, 3], role="body")], figures=[])


def test_texts_correct_lines_by_number_and_an_empty_text_drops_one(tmp_path):
    texts = dict(
        lines=[
            dict(n=1, text="The  Punch"),
            dict(n=3, text=""),
            dict(n="x", text="?"),
            dict(n=1, text="later answers lose"),
        ]
    )
    r = pagebuild.run(handle(), dict(ocr=TWO, groups=GROUPS, texts=json.dumps(texts), dpi=200, png=False))
    assert [ln["spans"][0]["text"] for ln in r["page"]["blocks"][0]["lines"]] == ["The Punch", "and Jndy"]
    with pytest.raises(RunError, match="give groups too"):
        pagebuild.page_of(dict(ocr=TWO, texts=texts, dpi=200))


def test_font_fit_sets_each_word_where_it_was_fitted_in_its_own_cut(tmp_path, ttf):
    r = pagebuild.run(handle(), dict(ocr=TWO, groups=GROUPS, font_fit=json.dumps(_fit(ttf)), dpi=200, png=False))
    a, b, junk = r["page"]["blocks"][0]["lines"]
    assert [(s["text"], s.get("style", "roman"), s["face"], s["size_pt"], s["x"]) for s in a["spans"]] == [
        ("Tbe ", "roman", "caslon:roman", 10.0, 100),
        ("Punch", "roman", "caslon:roman", 10.6, 200),  # the face has no bold cut: roman
    ]
    assert [(s.get("style", "roman"), s["face"]) for s in b["spans"]] == [
        ("italic", "caslon:italic"),
        ("small-caps", "caslon:small-caps"),
    ]
    assert a["spans"][0]["hscale"] == 1.05 and a["spans"][0]["baseline_y"] == 128
    assert junk["spans"] == [dict(text="%%")]  # no fitted word: drawn as it was
    assert r["text"].split() == ["Tbe", "Punch", "and", "Jndy", "%%"]


def test_a_corrected_line_with_other_words_keeps_the_fits_type_and_is_spaced_to_its_box(ttf):
    texts = dict(lines=[dict(n=1, text="The Punch show")])
    page = pagebuild.page_of(dict(ocr=TWO, groups=GROUPS, texts=texts, font_fit=_fit(ttf), dpi=200))
    ln = page.blocks[0].lines[0]
    assert [s.text for s in ln.spans] == ["The Punch show"] and ln.spans[0].x is None
    assert (ln.face, ln.size_pt, ln.baseline_y) == ("caslon:roman", 10.3, 128)
    page = pagebuild.page_of(
        dict(ocr=TWO, groups=GROUPS, texts=dict(lines=[dict(n=1, text="The Punch")]), font_fit=_fit(ttf), dpi=200)
    )
    assert [s.text for s in page.blocks[0].lines[0].spans] == ["The ", "Punch"]  # same words: still placed one by one


def test_styles_keeps_only_the_named_fitted_styles(ttf):
    page = pagebuild.page_of(dict(ocr=TWO, groups=GROUPS, font_fit=_fit(ttf), styles=["roman", "italic"], dpi=200))
    assert [s.style for s in page.blocks[0].lines[1].spans] == ["italic", "roman"]
    with pytest.raises(RunError, match="styles is a list"):
        pagebuild.page_of(dict(ocr=TWO, groups=GROUPS, font_fit=_fit(ttf), styles=["fancy"], dpi=200))


def test_a_fit_without_a_roman_cut_is_refused(ttf):
    with pytest.raises(RunError, match="no face with a roman file"):
        pagebuild.page_of(dict(ocr=TWO, groups=GROUPS, font_fit=_fit(ttf, styles=("italic",)), dpi=200))


def test_snap_sets_near_sizes_to_the_blocks_median_and_keeps_each_words_width(ttf):
    r = pagebuild.run(handle(), dict(ocr=TWO, groups=GROUPS, font_fit=_fit(ttf), snap=0.1, dpi=200, png=False))
    spans = [s for ln in r["page"]["blocks"][0]["lines"][:2] for s in ln["spans"]]
    # median of 10, 10.6, 13, 9.8 is 10.3: all but the 13 pt word are within 10 %
    assert [s["size_pt"] for s in spans] == [10.3, 10.3, 13.0, 10.3]
    assert spans[1]["hscale"] == pytest.approx(1.05 * 10.6 / 10.3)
    assert spans[1]["tracking_em"] == pytest.approx(0.01 * 10.6 / 10.3)


def test_crop_cuts_each_figure_from_the_scan_with_its_paper_whitened(tmp_path):
    pic = tmp_path / "scan.png"
    im = Image.new("L", (1000, 500), 200)  # yellowed paper
    im.paste(50, (300, 300, 400, 400))  # a picture
    im.save(pic, dpi=(200, 200))
    groups = dict(blocks=GROUPS["blocks"], figures=[dict(box=[300, 600, 400, 800], kind="engraving")])
    r = pagebuild.run(handle(), dict(ocr=TWO, groups=groups, image=str(pic), crop=True, png=False))
    f = r["page"]["figures"][0]
    assert f["box"] == [300, 300, 100, 100] and os.path.dirname(f["image"]) == os.path.join(
        str(settings.data_dir()), "pages", "crops"
    )
    with Image.open(f["image"]) as c:
        assert c.size == (100, 100) and abs(c.getpixel((50, 50)) - 50 * 255 / 200) <= 1
    with pytest.raises(RunError, match="give its picture"):
        pagebuild.run(handle(), dict(ocr=TWO, groups=groups, dpi=200, crop=True, png=False))


def test_rules_finds_the_border_and_a_rule_but_not_type_or_the_books_edge(tmp_path):
    pic = tmp_path / "scan.png"
    im = Image.new("L", (1000, 1400), 230)
    d = ImageDraw.Draw(im)
    d.rectangle((160, 220, 840, 900), outline=20, width=4)  # the border round the print
    d.line((300, 702, 700, 705), fill=20, width=3)  # a rule under a heading, a little skewed
    d.rectangle((420, 300, 432, 600), fill=20)  # the stem of a tall condensed letter, inside a line of type
    d.line((20, 20, 20, 1380), fill=20, width=4)  # the book's edge beside the leaf, far from the print
    im.save(pic, dpi=(300, 300))
    ocr = dict(
        width=1000,
        height=1400,
        lines=[
            dict(text="BIG", box=[400, 280, 200, 340], block=1, words=[dict(text="BIG", box=[400, 280, 200, 340])]),
            dict(text="text", box=[200, 800, 600, 40], block=2, words=[dict(text="text", box=[200, 800, 600, 40])]),
        ],
    )
    groups = dict(blocks=[dict(lines=[1], role="heading"), dict(lines=[2], role="body")], figures=[])
    r = pagebuild.run(handle(), dict(ocr=ocr, groups=groups, image=str(pic), rules=True, png=True))
    rules = sorted([round(v) for v in f["box"]] for f in r["page"]["figures"] if f["kind"] == "rule")
    assert len(rules) == 5  # the border's four sides and the rule; not the stem, not the edge
    assert any(abs(b[1] - 700) < 10 and b[2] > 380 for b in rules)
    with Image.open(r["png"]) as out:  # drawn: the border's ink is on the rebuilt page
        assert out.getpixel((162, 600)) < 128 and out.getpixel((20, 700)) > 128


def test_the_rebuild_spec_climbs_one_step_a_candidate_and_its_presets_carry_their_schemas():
    import tomllib

    from stratamodels.evals import core as C
    from stratamodels.runtimes import llama

    text, schema = llama.preset("page-text-fix-v1")
    assert "{context}" in text and schema["properties"]["lines"]["items"]["required"] == ["n", "text"]
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "evals", "specs", "page-rebuild.toml"), "rb") as f:
        spec = C.parse_spec(tomllib.load(f))
    roles = {c.name: [s["params"].get("preset") or s["role"] for s in c.stages] for c in spec.candidates}
    assert roles == dict(
        tesseract=[],
        layout=["page-build", "page-layout-fix-v2"],
        **{"layout-text": ["page-build", "page-layout-fix-v2", "page-text-fix-v1"]},
        **{"layout-text-type": ["page-build", "page-layout-fix-v2", "page-text-fix-v1", "font-match"]},
        full=["page-build", "page-layout-fix-v2", "page-text-fix-v1", "font-match"],
        **{"full-opus": ["page-build", "page-layout-fix-v2", "page-text-fix-v1", "font-match"]},
        **{"full-every-face": ["page-build", "page-layout-fix-v2", "page-text-fix-v1", "font-match"]},
    )
    assert [
        s["params"].get("propose", True) for s in spec.candidate("full-every-face").stages if s["role"] == "font-match"
    ] == [False]
    fonts = [s for c in spec.candidates for s in c.stages if s["role"] == "font-match"]
    assert all(s["answer"] == "json" and s["into"] == "font_fit" and s["also"] == ["image"] for s in fonts)
    assert {c.name for c in spec.candidates if not c.judge} == {"layout", "layout-text"} and spec.pairs == "all"


# ---- the role's default: a layout model regroups a plain ocr answer's lines (group) -------------------------------------------------------
SPLIT_GROUPS = dict(blocks=[dict(lines=[1, 2], role="running-head"), dict(lines=[3, 4], role="body")], figures=[])


def grouping(answer, asked):
    """A handle whose layout model is a stand-in answering `answer` (or raising it), recording what it was asked."""
    h = pagebuild.load(SimpleNamespace(key="pagebuild-opus-layout", options=dict(group="claude-opus-5-5-api")), "")

    def call(params):
        asked.append(params)
        if isinstance(answer, Exception):
            raise answer
        return answer

    h["grouper"] = call
    return h


def scan(tmp_path):
    pic = str(tmp_path / "p.png")
    Image.new("L", (1000, 500), 255).save(pic, dpi=(200, 200))
    return pic


def test_a_plain_ocr_answer_is_regrouped_by_the_models_layout_model(tmp_path):
    asked = []
    h = grouping(dict(text="```json\n" + json.dumps(SPLIT_GROUPS) + "\n```", cost_usd=0.03), asked)
    out = pagebuild.run(h, dict(ocr=SPLIT, image=scan(tmp_path), out=str(tmp_path / "g.pdf")))
    assert [b.get("role", "body") for b in out["page"]["blocks"]] == ["running-head", "body"]
    assert out["grouper"]["model"] == "claude-opus-5-5-api" and out["cost_usd"] == 0.03
    (p,) = asked
    assert (
        p["preset"] == "page-layout-fix-v2"
        and p["context"] == pagebuild.list_lines(SPLIT)
        and p["image"].endswith("p.png")
    )
    # a structured answer is read from its json; groups given, group = false, list_lines: no call
    h = grouping(dict(text="", json=SPLIT_GROUPS), asked)
    assert pagebuild.run(h, dict(ocr=SPLIT, image=scan(tmp_path), out=str(tmp_path / "j.pdf")))["blocks"] == 2
    for extra in (dict(groups=SPLIT_GROUPS), dict(group=False), dict(list_lines=True)):
        out = pagebuild.run(h, dict(ocr=SPLIT, image=scan(tmp_path), out=str(tmp_path / "n.pdf"), **extra))
        assert out.get("grouper") is None
    assert len(asked) == 2


@pytest.mark.parametrize(
    ("answer", "why"),
    [
        (RuntimeError("budget refused"), "budget refused"),
        (dict(text="I cannot read this page."), "not valid JSON|no such"),
        (dict(text=json.dumps(dict(blocks=[]))), "no blocks"),
    ],
)
def test_a_failed_layout_call_draws_tesseracts_blocks_and_says_why(tmp_path, answer, why):
    out = pagebuild.run(grouping(answer, []), dict(ocr=SPLIT, image=scan(tmp_path), out=str(tmp_path / "f.pdf")))
    assert out["blocks"] == 4 and "cost_usd" not in out  # Tesseract's four blocks
    assert out["grouper"]["model"] == "claude-opus-5-5-api" and re.search(why, out["grouper"]["error"])


def test_without_the_picture_the_layout_model_is_not_asked(tmp_path):
    asked = []
    out = pagebuild.run(grouping(dict(json=SPLIT_GROUPS), asked), dict(ocr=SPLIT, dpi=200, out=str(tmp_path / "d.pdf")))
    assert asked == [] and "picture" in out["grouper"]["error"] and out["blocks"] == 4


def test_the_layout_model_is_loaded_from_the_registry_once_and_unloaded(monkeypatch):
    from stratamodels import runtimes

    calls = []

    class RT:
        @staticmethod
        def load(m, d):
            calls.append(("load", m.key))
            return {"m": m.key}

        @staticmethod
        def run(h, params):
            calls.append(("run", h["m"]))
            return {"json": SPLIT_GROUPS}

        @staticmethod
        def unload(h):
            calls.append(("unload", h["m"]))

    monkeypatch.setattr(settings, "engine", lambda: None)
    monkeypatch.setattr(runtimes, "get", lambda name: RT)
    h = pagebuild.load(SimpleNamespace(key="pagebuild-opus-layout", options=dict(group="claude-opus-5-5-api")), "")
    f = pagebuild._grouper(h)
    assert f({}) == {"json": SPLIT_GROUPS} and pagebuild._grouper(h) is f
    pagebuild.unload(h)
    assert calls == [("load", "claude-opus-5-5-api"), ("run", "claude-opus-5-5-api"), ("unload", "claude-opus-5-5-api")]
    groups, why = pagebuild.group_lines(dict(options=dict(group="no-such-model")), SPLIT, "p.png")
    assert groups is None and "not in the registry" in why["error"]


def test_the_page_build_role_defaults_to_opus_regrouping_and_the_evals_draw_only():
    from stratamodels import registry

    reg = registry.Registry.load()
    m = reg.resolve("page-build")
    assert m.key == "pagebuild-opus-layout" and m.options["group"] == "claude-opus-5-5-api"
    assert "vision-language" in reg.models["claude-opus-5-5-api"].roles
    assert (
        reg.models["pagebuild"].options.get("group") is None
    )  # what the eval candidates name: their baselines stay Tesseract's blocks
