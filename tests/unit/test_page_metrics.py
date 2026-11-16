"""Layout metrics of a rebuilt page (evals/metrics.py: block_f1, figure_f1, role_acc, order_tau) against a cleaned reference page, both as
strata_python.pages descriptions (output["page"] is what page-build drew, or what a layout reader returned)."""

import pytest

from stratamodels.evals import metrics as M


def block(box, role="body", order=None):
    b = dict(box=box, role=role, lines=[])
    if order is not None:
        b["order"] = order
    return b


REF = dict(
    width=1000,
    height=1400,
    dpi=300,
    blocks=[
        block([100, 50, 800, 40], "running-head", 0),
        block([100, 150, 800, 300], "body", 1),
        block([100, 500, 800, 300], "body", 2),
        block([450, 1300, 100, 30], "folio", 3),
    ],
    figures=[dict(box=[200, 850, 600, 400])],
)
ITEM = dict(reference=dict(page=REF))


def score(name, output, error=None, item=ITEM):
    return M.METRICS[name].score(item, output, error)


def test_a_perfect_rebuild_scores_full_marks():
    out = dict(page=REF)
    assert score("block_f1", out) == 1.0
    assert score("figure_f1", out) == 1.0
    assert score("role_acc", out) == 1.0
    assert score("order_tau", out) == 1.0


def test_merged_blocks_match_nothing_and_wrong_roles_and_order_count():
    got = dict(
        blocks=[
            block([100, 150, 800, 650], "body", 0),  # the two body blocks run together: matches neither
            block([105, 52, 790, 38], "heading", 2),  # the running head, called a heading
            block([450, 1300, 100, 30], "folio", 1),  # read before the running head: order reversed for this pair
        ]
    )
    out = dict(page=got)
    assert score("block_f1", out) == pytest.approx(2 * 2 / (4 + 3))
    assert score("role_acc", out) == 0.5
    assert score("order_tau", out) == -1.0
    assert score("figure_f1", out) == 0.0  # the figure was missed


def test_reading_order_falls_back_to_list_order_like_the_renderer():
    # no order numbers: as listed, which is the reference's order
    got = dict(blocks=[block(b["box"]) for b in REF["blocks"]])
    assert score("order_tau", dict(page=got)) == 1.0
    assert M.match_boxes(REF["blocks"], list(reversed(REF["blocks"]))) == [(0, 3), (1, 2), (2, 1), (3, 0)]


def test_failures_and_missing_parts():
    assert score("block_f1", None, error="boom") == 0.0
    assert score("role_acc", None, error="boom") is None
    assert score("block_f1", dict(page=REF), item={}) is None  # no reference page
    assert score("block_f1", dict(text="no description")) == 0.0
    no_figs = dict(reference=dict(page=dict(REF, figures=[])))
    assert score("figure_f1", dict(page=dict(REF, figures=[])), item=no_figs) is None  # nothing to find, nothing found
    one = dict(page=dict(REF, blocks=REF["blocks"][:1]))
    assert score("order_tau", one) is None  # one block found: no order to compare
    assert score("role_acc", dict(page=dict(blocks=[]))) is None


def test_summarise_runs_over_pages():
    s = M.summarise("block_f1", [ITEM, ITEM], [(dict(page=REF), None), (None, "boom")])
    assert s["value"] == 0.5
    assert s["n"] == 2


@pytest.mark.parametrize("cut", [50, 60, 75])
def test_a_split_or_merged_block_matches_nothing_even_at_iou_one_half_or_more(cut):
    whole = [block([0, 0, 100, 100])]
    halves = [block([0, 0, 100, cut]), block([0, cut, 100, 100 - cut])]
    assert M.match_boxes(whole, halves) == []  # split: the larger piece has IoU cut/100 with the block
    assert M.match_boxes(halves, whole) == []  # merged
    speck = [block([0, 0, 100, 98]), block([10, 98, 5, 2])]  # a sliver beside the block is not a split
    assert M.match_boxes(whole, speck) == [(0, 0)]


# ---- page_ink_f1: the rebuilt page's raster against the scan, within the reference text blocks ---------------------------------------------
def _scan(tmp_path, name, bars, size=(400, 300), dpi=200):
    from PIL import Image, ImageDraw

    im = Image.new("L", size, 235)  # paper
    d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in bars:
        d.rectangle([x0, y0, x1, y1], fill=20)
    p = str(tmp_path / name)
    im.save(p, dpi=(dpi, dpi))
    return p


def test_page_ink_f1_scores_the_raster_against_the_print_in_the_text_blocks(tmp_path):
    text = [(50, 50, 350, 60), (50, 80, 350, 90)]
    stamp = (300, 200, 380, 280)  # ink outside every text block (a library stamp): no candidate is held to it
    scan = _scan(tmp_path, "scan.png", [*text, stamp])
    item = dict(
        input=dict(image=scan),
        reference=dict(page=dict(width=400, height=300, dpi=200, blocks=[dict(box=[50, 50, 301, 41])])),
    )
    m = M.METRICS["page_ink_f1"]
    assert m.score(item, dict(png=_scan(tmp_path, "same.png", text)), None) == pytest.approx(1.0)
    assert m.score(item, dict(png=_scan(tmp_path, "low.png", [(50, 64, 350, 74), (50, 94, 350, 104)])), None) < 0.5
    assert m.score(item, dict(png=_scan(tmp_path, "other.png", text, size=(401, 300))), None) == 0.0
    assert m.score(item, None, "boom") == 0.0
    assert m.score(item, dict(pdf="x.pdf"), None) is None  # drawn without a raster
    assert m.score(item, dict(png=str(tmp_path / "gone.png")), None) is None
    assert m.score(dict(input=dict(image=scan), reference={}), dict(png=scan), None) is None


def test_the_labelled_pages_are_drawable_and_consistent():
    """scripts/evals/page_layout/pages.toml: every page a valid description whose blocks and figures each match only themselves (no label
    is split or merged by its neighbours, so a perfect reader scores 1)."""
    import os
    import sys
    import tomllib

    from strata_python.pages import Page

    here = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "evals")
    with open(os.path.join(here, "page_layout", "pages.toml"), "rb") as f:
        pages = tomllib.load(f)
    sys.path.insert(0, here)
    try:
        import page_layout as PL
    finally:
        sys.path.remove(here)
    assert len(pages) == 20 and sum(p.get("set") == "check" for p in pages.values()) == 10
    for pid, p in pages.items():
        ref = PL.reference(p, 2700, 4300, 500)
        Page.from_dict(ref)  # roles and kinds are the renderer's
        for part in ("blocks", "figures"):
            parts = ref[part]
            assert M.match_boxes(parts, parts) == [(i, i) for i in range(len(parts))], pid
        assert M.METRICS["block_f1"].score(dict(reference=dict(page=ref)), dict(page=ref), None) == 1.0
