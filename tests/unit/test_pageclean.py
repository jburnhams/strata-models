"""Cleaning a scanned page before OCR (runtimes/pageclean.py, the `clean` param of the `ocr` role): each step on a synthetic page of
grey paper, a dark gutter and black strokes, checked by what it is for (the paper goes white, the ink stays black)."""

import numpy as np
import pytest
from PIL import Image

from stratamodels.runtimes import pageclean


def scan(w=480, h=320):
    """Yellowed paper (200) darkening to 70 at the left edge, as at a gutter, with black bars of 'type' across it."""
    a = np.tile(np.linspace(70, 200, w), (h, 1))
    for y in range(40, h - 40, 40):
        a[y : y + 6, 40 : w - 40] = 5
    return Image.fromarray(a.astype(np.uint8)).convert("RGB")


def test_steps_of_reads_lists_and_strings_and_refuses_unknown_steps():
    assert pageclean.steps_of(None) == [] and pageclean.steps_of("") == []
    assert pageclean.steps_of("flatten, upscale") == ["flatten", "upscale"]
    assert pageclean.steps_of(["otsu", ""]) == ["otsu"]
    with pytest.raises(ValueError, match="unknown clean steps"):
        pageclean.steps_of("flatten,blur")
    with pytest.raises(ValueError, match="list of step names"):
        pageclean.steps_of(3)


def test_no_steps_is_the_page_in_lightness():
    out, dpi = pageclean.clean(scan(), [], 300)
    assert out.mode == "L" and out.size == (480, 320) and dpi == 300


def test_flatten_whitens_the_paper_and_its_gutter_and_keeps_the_ink():
    out, _ = pageclean.clean(scan(), ["flatten"], 300)
    a = np.asarray(out)
    assert a[20, 10] > 225 and a[20, 470] > 240  # the dark edge and the light middle are both paper now
    assert a[42, 240] < 60  # a stroke is still dark


def test_contrast_sharpen_and_denoise_keep_the_page_readable():
    for step in ("contrast", "sharpen", "denoise"):
        a = np.asarray(pageclean.clean(scan(), [step], 300)[0])
        assert a[42, 240] < 60 and a[20, 470] > 150, step
    a = np.asarray(pageclean.clean(scan(), ["contrast"], 300)[0])
    assert a.min() == 0 and a.max() == 255


def test_denoise_removes_a_lone_speck():
    im = Image.new("L", (60, 60), 230)
    im.putpixel((30, 30), 0)
    assert pageclean.clean(im, ["denoise"], None)[0].getpixel((30, 30)) == 230


def test_otsu_and_adaptive_give_black_and_white_and_adaptive_keeps_the_gutter_as_paper():
    for step in ("otsu", "adaptive"):
        a = np.asarray(pageclean.clean(scan(), [step], 300)[0])
        assert set(np.unique(a)) <= {0, 255} and a[42, 240] == 0, step
    # one threshold for the page turns the dark gutter's paper black; a local one does not
    assert np.asarray(pageclean.clean(scan(), ["otsu"], 300)[0])[20, 5] == 0
    assert np.asarray(pageclean.clean(scan(), ["adaptive"], 300)[0])[20, 5] == 255


def test_otsu_threshold_splits_two_levels():
    a = np.array([[50] * 10 + [200] * 10], dtype=np.uint8)
    assert 50 <= pageclean._otsu_threshold(a) < 200


def test_upscale_doubles_the_size_and_the_dpi_and_steps_run_in_order():
    out, dpi = pageclean.clean(scan(), ["flatten", "upscale"], 300)
    assert out.size == (960, 640) and dpi == 600
    assert pageclean.clean(scan(), ["upscale"], None)[1] is None
