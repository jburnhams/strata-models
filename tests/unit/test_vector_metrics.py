"""Scores of a vectorised picture against the picture (evals/vector.py) and the image-kind metrics (evals/metrics.py), on drawn pictures and
hand-written SVGs: no tracer runs here."""

import numpy as np
from PIL import Image, ImageDraw

from stratamodels.evals import metrics as M
from stratamodels.evals import vector


def picture(tmp_path):
    im = Image.new("RGB", (200, 120), (236, 226, 200))  # yellowed paper: made white before scoring
    ImageDraw.Draw(im).rectangle((40, 30, 100, 90), fill=(10, 10, 10))
    p = tmp_path / "p.png"
    im.save(p)
    return {"input": {"image": str(p)}}


RIGHT = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 120"><rect x="40" y="30" width="61" height="61" fill="#000"/></svg>'
WRONG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 120"><rect x="130" y="30" width="61" height="61" fill="#c00"/></svg>'


def test_a_faithful_drawing_scores_better_on_every_reading(tmp_path):
    it = picture(tmp_path)
    good = vector.readings(it, {"svg": RIGHT, "out": "right"}, None)
    bad = vector.readings(it, {"svg": WRONG, "out": "wrong"}, None)
    assert good["ink_f1"] > 0.95 and bad["ink_f1"] < 0.1
    assert good["ssim"] > bad["ssim"] and good["tone"] < 0.02 < bad["tone"] and good["colour"] < 3 < bad["colour"]
    assert M.METRICS["vec_ink_f1"].score(it, {"svg": RIGHT, "out": "right"}, None) == good["ink_f1"]


def test_svg_read_from_its_file_and_missing_ones_score_nothing(tmp_path):
    it = picture(tmp_path)
    f = tmp_path / "d.svg"
    f.write_text(RIGHT)
    assert vector.readings(it, {"out": str(f)}, None)["ink_f1"] > 0.95
    assert vector.readings(it, {"out": str(tmp_path / "gone.svg")}, None) is None
    assert vector.readings(it, {"svg": RIGHT}, "the job failed") is None
    assert vector.readings({"input": {}}, {"svg": RIGHT}, None) is None
    assert vector.readings(it, {"svg": "<svg"}, None) is None  # one that does not draw scores nothing, not zero
    assert vector.size_kb(it, {"bytes": 2048}, None) == 2.0 and vector.size_kb(it, {}, None) is None
    assert vector.nodes(it, {"nodes": 7}, None) == 7.0 and vector.nodes(it, None, "x") is None


def test_white_paper_lifts_the_paper_and_keeps_the_ink():
    a = np.zeros((10, 10, 3), np.uint8) + np.array([236, 226, 200], np.uint8)
    a[0, 0] = 10
    w = vector.white_paper(a)
    assert min(w[5, 5].tolist()) >= 254 and w[0, 0].max() < 15


def item(kind, also=None):
    return {"reference": {"kind": kind, **({"also": also} if also else {})}}


def test_kind_metrics():
    eng = {"kind": "engraving", "group": "illustration", "line_art": True}
    assert M._kind_right(item("engraving"), eng, None) and not M._kind_right(item("etching-ish"), eng, None)
    assert M._kind_right(item("wood-engraving", ["engraving"]), eng, None)
    assert not M._kind_right(item("engraving"), eng, "failed")
    assert M._group_right(item("woodcut"), eng, None) and not M._group_right(item("halftone"), eng, None)
    assert not M._group_right({"reference": {}}, eng, None) and not M._group_right(item("woodcut"), eng, "x")
    assert M._line_right(item("woodcut"), eng, None) and not M._line_right(item("painting"), eng, None)
    assert not M._line_right(item("woodcut"), None, None) and not M._line_right({"reference": {}}, eng, None)
    items = [item("engraving"), item("engraving"), item("halftone"), item("pattern")]
    outs = [(eng, None), (eng, None), ({"kind": "engraving", "group": "illustration"}, None), (None, "failed")]
    # kinds: engraving F1 2*2/(2*2+1+0) = 0.8, halftone 0, pattern 0 -> 0.267; groups: illustration 0.8, photo 0, other 0
    assert round(M.METRICS["kind_f1"].dataset(items, outs), 3) == 0.267
    assert round(M.METRICS["group_f1"].dataset(items, outs), 3) == 0.267
    assert M._macro_f1([], [], "kind") is None
