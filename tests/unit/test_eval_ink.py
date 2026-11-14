"""Ink masks and ink agreement (evals/ink.py): what the font-match role fits by and the page-rebuild evals score by."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from stratamodels.evals import ink as INK


def page(angle=0.0, size=(600, 400)):
    """Grey paper with dark bars for lines of type, optionally tilted, plus a dust speck."""
    im = Image.new("L", size, 215)
    d = ImageDraw.Draw(im)
    for y in range(60, 360, 40):
        for x in range(40, 560, 60):
            d.rectangle([x, y, x + 40, y + 14], fill=40)
    im = im.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=215)
    im.putpixel((590, 5), 30)
    return im


def test_binarise_finds_the_type_drops_specks_and_leaves_paper_white():
    ink, angle = INK.binarise(page(), dpi=300, deskew=False)
    assert angle == 0.0
    assert ink.shape == (400, 600)
    assert ink[67, 60] and not ink[30, 60]  # a bar is ink, the margin above is paper
    assert not ink[5, 590]  # the lone speck is gone
    assert 0.1 < ink.mean() < 0.3


def test_skew_is_found_and_undone():
    tilted = page(angle=1.5)
    ink, angle = INK.binarise(tilted, dpi=300)
    assert angle == pytest.approx(-1.5, abs=0.3)
    level, _ = INK.binarise(page(), dpi=300, deskew=False)
    assert INK.ink_f1(level, ink, tol_px=3)["f1"] > 0.85
    again, a2 = INK.binarise(tilted, dpi=300, angle=angle)  # a known angle is applied, not searched for
    assert a2 == angle and (again == ink).all()


def test_gray_takes_paths_images_and_arrays(tmp_path):
    p = str(tmp_path / "p.png")
    page().save(p)
    assert INK.gray(p).shape == (400, 600)
    assert INK.gray(np.array([[True, False]])).tolist() == [[0, 255]]
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    assert INK.gray(rgb).shape == (2, 2)


def test_ink_f1_forgives_a_pixel_and_counts_the_rest():
    a = np.zeros((50, 50), dtype=bool)
    a[10:40, 20:24] = True
    shifted = np.roll(a, 1, axis=1)
    r = INK.ink_f1(a, shifted, tol_px=1)
    assert r["f1"] == 1.0 and r["iou"] < 1.0
    assert INK.ink_f1(a, shifted, tol_px=0)["f1"] == pytest.approx(0.75)
    extra = shifted.copy()
    extra[0:5, 0:5] = True  # ink where the scan has none lowers precision only
    r = INK.ink_f1(a, extra, tol_px=1)
    assert r["recall"] == 1.0 and r["precision"] < 1.0
    assert INK.ink_f1(a, extra, tol_px=1, box=[10, 5, 30, 40])["f1"] == 1.0  # outside the box does not count


def test_ink_f1_edges():
    z = np.zeros((5, 5), dtype=bool)
    o = np.ones((5, 5), dtype=bool)
    assert INK.ink_f1(z, z)["f1"] == 1.0
    assert INK.ink_f1(o, z)["f1"] == 0.0
    assert INK.ink_f1(z, o)["f1"] == 0.0
    with pytest.raises(ValueError, match="differ in size"):
        INK.ink_f1(z, np.zeros((4, 5), dtype=bool))
    m = np.zeros((5, 5), dtype=bool)
    m[0, 0] = True
    assert INK.ink_f1(o, o, mask=m)["ink_ref"] == 1


def test_dilate_and_erode():
    a = np.zeros((9, 9), dtype=bool)
    a[4, 4] = True
    assert INK.dilate(a, 1).sum() == 9
    assert (INK.dilate(a, 0) == a).all()
    assert INK.erode(INK.dilate(a, 1), 1).sum() == 1
    assert (INK.erode(a, 0) == a).all()
