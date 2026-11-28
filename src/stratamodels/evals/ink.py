"""Ink: the black of a scanned page as a mask, and how well a rendering's ink matches it. Shared by the `font-match` role (runtimes/fontfit.py
fits a face by it and its eval scores by it) and the page-rebuild evals (page_ink_iou: a rebuilt PDF rendered at the scan's dpi against the
cleaned scan). numpy and Pillow only, like runtimes/pageclean.py, whose flatten and Sauvola it reuses.

- `binarise(img, dpi)`: lightness, flatten (paper tone, gutter shadow), deskew (the angle within +-3 deg whose row profile is sharpest), Sauvola
  with a window of about two lines of body type (from the dpi), then specks removed (ink pixels with too few ink neighbours). True = ink.
- `ink_f1(ref, got, tol_px)`: precision (the rendered ink that lies within tol_px of the scan's ink), recall (the scan's ink within tol_px
  of the rendered), their F1, and plain IoU beside them. The tolerance forgives the half-pixel jitter of a scan and a stroke's ragged edge,
  so a face scores by its shapes, not by sub-pixel placement. `mask` or `box` limits both to a part of the page (a block; figures left out)."""

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from stratamodels.runtimes import pageclean

Mask = NDArray[np.bool_]


def gray(img):
    """A picture (path, PIL image or array) as lightness, uint8."""
    if isinstance(img, np.ndarray):
        if img.dtype == np.bool_:
            return np.where(img, 0, 255).astype(np.uint8)
        return img.astype(np.uint8) if img.ndim == 2 else np.asarray(Image.fromarray(img).convert("L"))
    im = Image.open(img) if isinstance(img, str) else img
    return np.asarray(ImageOps.exif_transpose(im).convert("L"))


def _sharpness(ink, angle):
    im = Image.fromarray(np.where(ink, 255, 0).astype(np.uint8)).rotate(
        angle, resample=Image.Resampling.NEAREST, fillcolor=0
    )
    rows = np.asarray(im, dtype=np.float64).sum(1)
    return float(np.square(np.diff(rows)).sum())


def skew_angle(ink, limit=3.0, step=0.1):
    """The rotation (degrees, PIL's sense: counter-clockwise) that makes the text lines level: the angle whose row profile changes most
    sharply, searched on a reduced copy, coarse then fine."""
    h, w = ink.shape
    f = max(1, int(max(h, w) / 1200))
    small = ink[::f, ::f]
    if small.sum() < 50:
        return 0.0
    best = max(np.arange(-limit, limit + 1e-9, 0.5), key=lambda a: _sharpness(small, float(a)))
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, step)
    return round(float(max(fine, key=lambda a: _sharpness(small, float(a)))), 2)


def despeckle(ink, min_neighbours=2, r=1):
    """Ink pixels with fewer than `min_neighbours` other ink pixels in their (2r+1)-square: dust and scan noise, not type."""
    a = ink.astype(np.float64)
    n = pageclean._window_sums(a, r) - a  # pyright: ignore[reportPrivateUsage]
    return ink & (n >= min_neighbours)


def sauvola(g, r, k=0.2):
    a = g.astype(np.float64)
    h, w = a.shape
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    n = np.outer(y1 - y0, x1 - x0).astype(np.float64)
    ws = pageclean._window_sums  # pyright: ignore[reportPrivateUsage]
    mean = ws(a, r) / n
    sd = np.sqrt(np.maximum(ws(a * a, r) / n - mean * mean, 0.0))
    t = mean * (1 + k * (sd / 128.0 - 1))
    return a <= t


def binarise(img, dpi=None, deskew=True, flatten=True, angle=None):
    """(ink mask, angle applied): the page's ink after flattening, deskewing and Sauvola. Pass `angle` to apply a known rotation (another
    picture of the same page) instead of searching for one; deskew=False keeps the page as it is (crops of a page already level)."""
    g = gray(img)
    im = Image.fromarray(g)
    if flatten and min(g.shape) >= 64:
        im = pageclean._flatten(im)  # pyright: ignore[reportPrivateUsage]
    g = np.asarray(im)
    dpi = float(dpi or 300)
    r = max(7, int(dpi / 10))  # ~2.5 mm: about two lines of body type
    ink = sauvola(g, r)
    # Sauvola marks faint paper texture in empty areas as ink when the window holds no type: keep only what is also darker than the paper.
    ink &= g < 200
    if angle is None:
        angle = skew_angle(ink) if deskew else 0.0
    if angle:
        rot = Image.fromarray(np.where(ink, 255, 0).astype(np.uint8)).rotate(
            angle, resample=Image.Resampling.BILINEAR, fillcolor=0
        )
        ink = np.asarray(rot) > 127
    return despeckle(ink, min_neighbours=2, r=max(1, int(dpi / 300))), float(angle)


def dilate(mask, r):
    """`mask` grown by `r` pixels each way (a square of 2r+1). OpenCV's, a twentieth of the time of PIL's MaxFilter, which was a third
    of a font fit (2026-11-28); the same result but at the picture's edge, where PIL's filter takes the edge pixel as repeated."""
    if r <= 0:
        return mask
    import cv2

    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k, borderType=cv2.BORDER_REPLICATE) > 0


def erode(mask, r):
    if r <= 0:
        return mask
    import cv2

    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k, borderType=cv2.BORDER_REPLICATE) > 0


def box_mask(shape, box):
    m = np.zeros(shape, dtype=bool)
    x, y, w, h = (int(round(v)) for v in box)
    m[max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)] = True
    return m


def ink_f1(ref, got, tol_px=2, mask=None, box=None):
    """{precision, recall, f1, iou, ink_ref, ink_got} of `got` (a rendering's ink) against `ref` (the scan's), both boolean masks of one size.
    Empty on both sides is a perfect 1; ink on one side only is 0."""
    if ref.shape != got.shape:
        raise ValueError(f"ink masks differ in size: {ref.shape} and {got.shape}")
    if box is not None:
        bm = box_mask(ref.shape, box)
        mask = bm if mask is None else (mask & bm)
    if mask is not None:
        ref, got = ref & mask, got & mask
    nr, ng = int(ref.sum()), int(got.sum())
    if nr == 0 and ng == 0:
        return dict(precision=1.0, recall=1.0, f1=1.0, iou=1.0, ink_ref=0, ink_got=0)
    p = float((got & dilate(ref, tol_px)).sum()) / ng if ng else 0.0
    r = float((ref & dilate(got, tol_px)).sum()) / nr if nr else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = float((ref & got).sum()) / float((ref | got).sum())
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4), iou=round(iou, 4), ink_ref=nr, ink_got=ng)
