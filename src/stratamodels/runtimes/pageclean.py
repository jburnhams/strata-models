"""Cleaning a scanned page before OCR: the `clean` param of the `ocr` role (runtimes/tesseract.py). Each step is one named change, applied in
the order given, so an eval can try them alone and together (evals/specs/ocr-pages.toml) and a winner is a list an app passes.

Pillow and numpy only (both already dependencies): no OpenCV in the service's own environment for this.

- `flatten`: divide the page by an estimate of its paper (a dilation then a wide median of the lightness, at an eighth of the size), so
  yellowed paper, a shadow at the gutter or uneven light become white and the ink keeps its weight. From diary-study's
  `remove_background_and_lighten_fold` (image_enhancer.py), which did this for a handwritten diary's folds.
- `contrast`: stretch the lightness so the darkest and lightest 1 % reach black and white.
- `sharpen`: an unsharp mask (radius 2, 150 %), for soft scans.
- `denoise`: a 3x3 median, for speckle and grain; thin strokes survive it at book resolutions (300 dpi and up).
- `otsu`: black and white at one threshold for the page (Otsu's).
- `adaptive`: black and white by Sauvola's local threshold (a window of 1/30 of the width, k 0.2), which keeps faint type next to dark.
- `upscale`: twice the size (Lanczos), and twice the dpi, for small type and small capitals Tesseract reads better larger.

Every step works on the lightness (L): colour is not needed for reading."""

# numpy 2.5's stubs leave clip, cumsum, where and argmax partially unknown in strict mode; everything else here is checked strictly
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter, ImageOps

STEPS = ("flatten", "contrast", "sharpen", "denoise", "otsu", "adaptive", "upscale")


def steps_of(clean: object) -> list[str]:
    """The steps a `clean` param names: a list, or a comma-separated string ("flatten,upscale"); "" or None is no cleaning."""
    if clean is None or clean == "":
        return []
    names = [s.strip() for s in clean.split(",")] if isinstance(clean, str) else clean
    if not isinstance(names, list) or not all(isinstance(s, str) for s in names):
        raise ValueError(f"clean is a list of step names or a comma-separated string, not {clean!r}")
    out = [str(s) for s in names if s]
    unknown = [s for s in out if s not in STEPS]
    if unknown:
        raise ValueError(f"unknown clean steps {unknown} (known: {', '.join(STEPS)})")
    return out


def _flatten(im: Image.Image) -> Image.Image:
    small = im.reduce(8)
    bg = (
        small.filter(ImageFilter.MaxFilter(5))
        .filter(ImageFilter.MedianFilter(13))
        .resize(im.size, Image.Resampling.BILINEAR)
    )
    a = np.asarray(im, dtype=np.float32)
    b = np.maximum(np.asarray(bg, dtype=np.float32), 1.0)
    return Image.fromarray(np.clip(a / b * 255.0, 0, 255).astype(np.uint8))


def _otsu_threshold(a: NDArray[np.uint8]) -> int:
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    w0 = np.cumsum(hist)
    m = np.cumsum(hist * np.arange(256))
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)
    between = np.zeros(256)
    between[valid] = (m[-1] * w0[valid] / total - m[valid]) ** 2 / (w0[valid] * w1[valid])
    return int(np.argmax(between))


def _otsu(im: Image.Image) -> Image.Image:
    a = np.asarray(im, dtype=np.uint8)
    return Image.fromarray(np.where(a > _otsu_threshold(a), 255, 0).astype(np.uint8))


def _window_sums(a: NDArray[np.float64], r: int) -> NDArray[np.float64]:
    """The sum of `a` over a (2r+1)-square window around each pixel (edges clipped), from one integral image."""
    h, w = a.shape
    ii = np.zeros((h + 1, w + 1))
    ii[1:, 1:] = a.cumsum(0).cumsum(1)
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    return ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]


def _adaptive(im: Image.Image, k: float = 0.2) -> Image.Image:
    a = np.asarray(im, dtype=np.float64)
    h, w = a.shape
    r = max(7, w // 60)  # a window of about 1/30 of the page's width: a few lines of type
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    n = np.outer(y1 - y0, x1 - x0).astype(np.float64)
    mean = _window_sums(a, r) / n
    sd = np.sqrt(np.maximum(_window_sums(a * a, r) / n - mean * mean, 0.0))
    t = mean * (1 + k * (sd / 128.0 - 1))
    return Image.fromarray(np.where(a > t, 255, 0).astype(np.uint8))


def clean(im: Image.Image, steps: list[str], dpi: float | None) -> tuple[Image.Image, float | None]:
    """The page after `steps`, in lightness, and its dpi (doubled by `upscale`)."""
    out = ImageOps.exif_transpose(im).convert("L")
    for s in steps:
        if s == "flatten":
            out = _flatten(out)
        elif s == "contrast":
            out = ImageOps.autocontrast(out, cutoff=1)
        elif s == "sharpen":
            out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))
        elif s == "denoise":
            out = out.filter(ImageFilter.MedianFilter(3))
        elif s == "otsu":
            out = _otsu(out)
        elif s == "adaptive":
            out = _adaptive(out)
        else:  # upscale (steps_of has refused anything else)
            out = out.resize((out.width * 2, out.height * 2), Image.Resampling.LANCZOS)
            dpi = dpi * 2 if dpi else dpi
    return out, dpi
