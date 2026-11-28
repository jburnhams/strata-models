"""How close a vectorised picture (role `vectorise`, runtimes/vectorise.py) is to the picture it came from, with no reference but the picture
itself: the SVG is drawn on white at the picture's size (resvg) and compared three ways, since a drawing can be right in one and wrong in
another (a trace that fills the hatching solid keeps the tone and loses the lines; a model's tidy redrawing keeps neither exactly).

- `vec_ink_f1`: the picture's ink (flattened paper, Otsu's cut) against the drawing's (darker than mid-grey), F1 with 1 px tolerance at
  INK_SIDE on the long side (evals/ink.ink_f1, as the page-rebuild evals score type): the lines themselves.
- `vec_ssim`: structural similarity of the two in grey at SSIM_SIDE, both slightly blurred: how alike they look at reading size.
The picture's paper is made white first (`white_paper`): a drawing on white paper is what is wanted.

- `vec_tone`: the mean grey difference (0..1) at an eighth of the size: whether the light and dark areas are where they were, which is what a
  tracer that cuts at the wrong threshold gets wrong.
- `vec_colour`: the mean colour difference (CIE76 delta E, Lab) at an eighth of the size; 0 for a grey picture drawn grey.
- `vec_kb`, `vec_nodes`: size of the SVG and its path commands (what an editor moves); smaller is better only when the rest holds.

Every reading of one output is computed once and kept (by the output's file and the picture's), since the four share one render."""

import functools
import os
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from stratamodels.evals import ink as INK

INK_SIDE = 1600
SSIM_SIDE = 1024


def _svg_of(output: Any) -> str | None:
    if not isinstance(output, dict):
        return None
    if isinstance(output.get("svg"), str):
        return output["svg"]
    p = output.get("out")
    if isinstance(p, str) and os.path.isfile(p):
        return open(p, encoding="utf8").read()
    return None


def _size(w: int, h: int, side: int) -> tuple[int, int]:
    s = min(1.0, side / max(w, h))
    return max(1, round(w * s)), max(1, round(h * s))


def _ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Mean SSIM (Wang et al. 2004) over 7x7 windows, by box filters."""
    from scipy.ndimage import uniform_filter

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    x, y = x.astype(np.float64), y.astype(np.float64)
    mx, my = uniform_filter(x, 7), uniform_filter(y, 7)
    vx = uniform_filter(x * x, 7) - mx * mx
    vy = uniform_filter(y * y, 7) - my * my
    cxy = uniform_filter(x * y, 7) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(s.mean())


def _lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def white_paper(rgb: np.ndarray) -> np.ndarray:
    """The picture with its paper made white (each channel divided by its 95th percentile): a redrawing on white paper is what is
    asked for, so the yellowed page must not count against it as tone or colour."""
    f = rgb.astype(np.float64)
    paper = np.maximum(np.percentile(f.reshape(-1, 3), 95, axis=0), 1.0)
    return np.clip(f * (255.0 / paper), 0, 255).astype(np.uint8)


@functools.lru_cache(maxsize=64)
def _readings(picture: str, svg_key: str, svg: str) -> dict[str, float]:
    from stratamodels.runtimes import vectorise as V

    src = white_paper(V.read(picture))
    h, w = src.shape[:2]
    drawn = V.render(svg, w, h)
    out: dict[str, float] = {}
    # ink: the picture's flattened and cut as the tracer's own default; the drawing's darker than mid-grey
    iw, ih = _size(w, h, INK_SIDE)
    g_src = np.asarray(Image.fromarray(V.flat_grey(src)).resize((iw, ih), Image.Resampling.LANCZOS))
    g_drw = np.asarray(Image.fromarray(drawn).convert("L").resize((iw, ih), Image.Resampling.LANCZOS))
    out["ink_f1"] = INK.ink_f1(g_src <= V.otsu(g_src), g_drw < 128, tol_px=1)["f1"]
    sw, sh = _size(w, h, SSIM_SIDE)
    a = np.asarray(
        Image.fromarray(src).convert("L").resize((sw, sh), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(1))
    )
    b = np.asarray(
        Image.fromarray(drawn)
        .convert("L")
        .resize((sw, sh), Image.Resampling.LANCZOS)
        .filter(ImageFilter.GaussianBlur(1))
    )
    out["ssim"] = round(_ssim(a, b), 4)
    tw, th = max(1, w // 8), max(1, h // 8)
    sa = np.asarray(Image.fromarray(src).resize((tw, th), Image.Resampling.BOX))
    sb = np.asarray(Image.fromarray(drawn).resize((tw, th), Image.Resampling.BOX))
    out["tone"] = round(float(np.abs(sa.mean(-1) - sb.mean(-1)).mean()) / 255.0, 4)
    out["colour"] = round(float(np.linalg.norm(_lab(sa) - _lab(sb), axis=-1).mean()), 3)
    return out


def readings(item: dict[str, Any], output: Any, error: Any) -> dict[str, float] | None:
    """{ink_f1, ssim, tone, colour} of one output, or None for a failed job or one with no SVG."""
    if error:
        return None
    svg = _svg_of(output)
    pic = (item.get("input") or {}).get("image")
    if not svg or not isinstance(pic, str) or not os.path.isfile(pic):
        return None
    key = str((output or {}).get("out") or hash(svg))
    try:
        return _readings(pic, key, svg)
    except Exception:
        return None


def reader(name: str):
    def score(item: dict[str, Any], output: Any, error: Any) -> float | None:
        r = readings(item, output, error)
        return None if r is None else r[name]

    return score


def size_kb(item: dict[str, Any], output: Any, error: Any) -> float | None:
    if error or not isinstance(output, dict) or output.get("bytes") is None:
        return None
    return float(output["bytes"]) / 1024.0


def nodes(item: dict[str, Any], output: Any, error: Any) -> float | None:
    if error or not isinstance(output, dict) or output.get("nodes") is None:
        return None
    return float(output["nodes"])
