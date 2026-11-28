"""A picture redrawn as an SVG (runtime `vectorise`, role `vectorise`): above all the line art of old books (engravings, woodcuts, line
drawings, ornaments, hand-coloured plates), so a rebuilt book can print it sharp at any size and small. Which kinds are line art is the
`image-kind` role's answer (runtimes/imagekind.py).

Engines (model options `engine`, each overridable per job by the params below), compared by eval vectorise-plates:
  potrace       black ink on clear paper: the paper flattened (flat_grey), the picture enlarged `scale` times, cut at one
                threshold (Otsu's, or `threshold` 0..1 of the way from ink to paper), traced by the potrace command (GPL, a separate
                program: never linked) into Bezier outlines. The classic tracer; what Inkscape's Trace Bitmap runs.
  potrace-tones the same at `levels` thresholds, each traced and stacked darkest on top in its own grey: tone as flat steps
  vtracer       VTracer (visioncortex, MIT, a Python wheel): colour or black-and-white clustering into stacked spline shapes
  lines-colour  a hand-coloured plate as two layers: its colours by VTracer on a smoothed, few-colour copy, then potrace's black ink on top
  llm           a vision-language model of the registry (options `llm`) writes the SVG itself from the picture (preset vectorise-llm-v1);
                `refine` > 0 shows it the picture beside its own drawing that many times and takes its corrected SVG
  llm-plan      the model looks at the picture and chooses an engine and its settings (preset vectorise-plan-v1); the tracer draws
  llm-colour    potrace's ink, and under it the colour washes the model writes as large flat shapes (preset vectorise-colour-v1): lines
                exact, colour by eye
  llm-clean     potrace's trace given to the model with the picture to tidy (preset vectorise-clean-v1): specks, paper marks, broken
                lines. Only a trace under `clean_kb` fits a model's answer, so a detailed engraving fails here by design
params: image (path, required); out (an .svg path to write; by default <data>/vectorise/<hash of the SVG>.svg), and any of
  engine, scale, threshold, levels, turdsize, alphamax, opttolerance, colours, speckle, refine, max_tokens, background ("white" or none).
Answer: {out (the SVG file), path (it drawn as a PNG at the picture's size), svg (the SVG itself, when under 32 KB), width, height, bytes, paths, nodes, engine, seconds, cost_usd? (models), plan? (llm-plan), steps? (what each part did)}.
The SVG's viewBox is the picture's own pixels, so it lies exactly over the picture it came from."""

import io
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from stratamodels.runtimes import RunError, pageclean

ENGINES = ("potrace", "potrace-tones", "vtracer", "lines-colour", "llm", "llm-plan", "llm-colour", "llm-clean")
PARAMS = {
    "image",
    "out",
    "engine",
    "scale",
    "threshold",
    "levels",
    "turdsize",
    "alphamax",
    "opttolerance",
    "colours",
    "speckle",
    "refine",
    "max_tokens",
    "background",
    "context",
}
LONG_SIDE = (
    2400  # the most a picture is traced at (before `scale`): book plates at 400 dpi; more only adds the paper's grain
)
INLINE = 32 * 1024  # an SVG this small also comes back in the answer itself
SVG_RE = re.compile(r"<svg\b.*?</svg>", re.S | re.I)


def binary() -> str | None:
    return shutil.which("potrace")


def load(model: Any, models_dir: str) -> dict[str, Any]:
    o = dict(model.options or {})
    engine = str(o.get("engine", "potrace"))
    if engine not in ENGINES:
        raise RunError(f"model {model.key}: engine is one of {', '.join(ENGINES)}, not {engine!r}")
    if engine.startswith("llm") and not o.get("llm"):
        raise RunError(f"model {model.key}: engine {engine} names no vision-language model (options.llm)")
    return dict(key=model.key, options=o, models_dir=models_dir)


def unload(handle: dict[str, Any]) -> None:
    undo = handle.get("llm_unload")
    if callable(undo):
        undo()
    handle.clear()


def run(handle: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(params) - PARAMS
    if unknown:
        raise RunError(f"unknown params {sorted(unknown)}")
    path = params.get("image")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise RunError(f"no such image: {path!r}")
    o: dict[str, Any] = {**handle["options"], **{k: v for k, v in params.items() if k not in ("image", "out")}}
    engine = str(o.get("engine", "potrace"))
    if engine not in ENGINES:
        raise RunError(f"engine is one of {', '.join(ENGINES)}, not {engine!r}")
    t0 = time.monotonic()
    rgb = read(path)
    extra: dict[str, Any] = {}
    if engine == "potrace":
        svg = potrace_svg(rgb, o)
    elif engine == "potrace-tones":
        svg = tones_svg(rgb, o)
    elif engine == "vtracer":
        svg = vtracer_svg(rgb, o)
    elif engine == "lines-colour":
        svg = lines_colour_svg(rgb, o)
    else:
        svg, extra = _llm_engine(handle, engine, path, rgb, o)
    if o.get("background") == "white":
        svg = _with_background(svg)
    out = dict(
        stats(svg),
        width=rgb.shape[1],
        height=rgb.shape[0],
        engine=engine,
        seconds=round(time.monotonic() - t0, 3),
        **extra,
    )
    dest = params.get("out")
    if dest and (
        not isinstance(dest, str)
        or not dest.endswith(".svg")
        or not os.path.isdir(os.path.dirname(os.path.abspath(dest)))
    ):
        raise RunError("out is an .svg path in an existing folder")
    if not dest:  # kept by its content in the data folder: an eval's answers would be megabytes in the job row
        import hashlib

        from stratamodels import settings

        d = os.path.join(settings.data_dir(), "vectorise")
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, hashlib.sha256(svg.encode()).hexdigest()[:20] + ".svg")
    with open(
        str(dest), "w", encoding="utf8", newline=""
    ) as f:  # as written: on Windows text mode would make each \n a \r\n, and `bytes` would not be the file's size
        f.write(svg)
    # the SVG drawn at the picture's size beside it: what people and model judges compare (evals.similarity reads `path` as a picture)
    preview = str(dest)[:-4] + ".png"
    Image.fromarray(render(svg, rgb.shape[1], rgb.shape[0])).save(preview, optimize=False, compress_level=3)
    out.update(out=str(dest), path=preview)
    if len(svg) <= INLINE:
        out["svg"] = svg
    return out


# ---- pictures and SVGs --------------------------------------------------------------------------------------------------------------------
def read(path: str) -> np.ndarray:
    """The picture as RGB uint8, its long side at most LONG_SIDE (the SVG's viewBox is in these pixels)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > LONG_SIDE:
            im.thumbnail((LONG_SIDE, LONG_SIDE), Image.Resampling.LANCZOS)
        return np.asarray(im)


def stats(svg: str) -> dict[str, int]:
    """bytes, paths (shapes drawn) and nodes (path commands: what an editor would have to move) of an SVG."""
    shapes = len(re.findall(r"<(path|polygon|polyline|rect|circle|ellipse|line)\b", svg))
    nodes = len(re.findall(r"[MLCQSTAHVZmlcqstahvz]", " ".join(re.findall(r'\sd="([^"]*)"', svg))))
    return dict(bytes=len(svg.encode()), paths=shapes, nodes=nodes)


def render(svg: str, width: int, height: int, background: str = "#ffffff") -> np.ndarray:
    """An SVG drawn at width x height on white (resvg), as RGB uint8: how the metrics and a refining model see it."""
    import resvg_py  # pyright: ignore[reportMissingImports]

    try:
        png = resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height, background=background)
    except (ValueError, RuntimeError) as e:
        raise RunError(f"the SVG does not draw: {e}")
    with Image.open(io.BytesIO(bytes(png))) as im:
        out = im.convert("RGB")
        if out.size != (width, height):
            out = out.resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(out)


def _with_background(svg: str) -> str:
    return re.sub(r"(<svg\b[^>]*>)", r'\1<rect width="100%" height="100%" fill="#ffffff"/>', svg, count=1)


def _body(svg: str) -> str:
    """What is inside an SVG's root element."""
    return re.sub(r"^.*?<svg\b[^>]*>|</svg>\s*$", "", svg, flags=re.S)


def _root(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        f"{body}</svg>"
    )


def flat_grey(rgb: np.ndarray) -> np.ndarray:
    """Lightness with the paper made white: yellowed paper, a gutter shadow and uneven light go, the ink stays. Like pageclean's flatten
    (a local maximum of the lightness as the paper), except that the paper is never taken darker than 85 % of the picture's usual paper: a
    woodcut's solid black, wider than any window, is ink, where flatten (sized for a page of type) would whiten it as paper."""
    im = Image.fromarray(rgb).convert("L")
    a = np.asarray(im, dtype=np.float32)
    small = im.reduce(8) if min(im.size) >= 64 else im
    bg = (
        small.filter(ImageFilter.MaxFilter(5))
        .filter(ImageFilter.MedianFilter(13))
        .resize(im.size, Image.Resampling.BILINEAR)
    )
    paper = float(np.percentile(a, 95))
    b = np.maximum(np.asarray(bg, dtype=np.float32), max(0.85 * paper, 1.0))
    return np.clip(a / b * 255.0, 0, 255).astype(np.uint8)


def otsu(grey: np.ndarray) -> int:
    return pageclean._otsu_threshold(grey.astype(np.uint8))  # pyright: ignore[reportPrivateUsage]


def _cut(grey: np.ndarray, threshold: Any) -> int:
    """The grey level at or below which a pixel is ink: Otsu's, or `threshold` (0..1) of the way from the ink's level to the paper's."""
    if threshold in (None, "otsu"):
        return otsu(grey)
    t = float(threshold)
    if not 0 < t < 1:
        raise RunError("threshold is 'otsu' or between 0 and 1")
    ink, paper = np.percentile(grey, 2), np.percentile(grey, 98)
    return int(ink + t * (paper - ink))


def _trace(mask: np.ndarray, o: Mapping[str, Any], fill: str = "#000000") -> str:
    """potrace's outlines of a boolean mask (True = ink), as one <path> in the mask's pixels."""
    exe = binary()
    if not exe:
        raise RunError("the potrace command is not installed (potrace 1.16: the distribution's potrace package)")
    try:
        turd, alpha, opt = int(o.get("turdsize", 2)), float(o.get("alphamax", 1.0)), float(o.get("opttolerance", 0.2))
    except (TypeError, ValueError):
        raise RunError("turdsize, alphamax and opttolerance are numbers")
    with tempfile.TemporaryDirectory(prefix="vectorise-") as d:
        pbm, svgf = os.path.join(d, "in.pbm"), os.path.join(d, "out.svg")
        Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).convert("1").save(pbm)
        cmd = [
            exe,
            pbm,
            "-b",
            "svg",
            "--flat",  # one path: a trace is drawn whole, and a file of 10 000 <path>s is no easier to edit than one
            "-u",
            "1",  # coordinates in whole pixels of the traced bitmap: smaller files, below what a print can show
            "-t",
            str(turd),
            "-a",
            str(alpha),
            "-O",
            str(opt),
            "-C",
            fill,
            "-o",
            svgf,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if p.returncode:
            raise RunError(f"potrace failed: {p.stderr.strip()[:300]}")
        text = open(svgf, encoding="utf8").read()
    g = re.search(r"<g\b.*</g>", text, re.S)
    return g.group(0) if g else ""


def potrace_svg(rgb: np.ndarray, o: Mapping[str, Any]) -> str:
    h, w = rgb.shape[:2]
    s = float(o.get("scale", 2))
    grey = flat_grey(rgb)
    big = (
        np.asarray(Image.fromarray(grey).resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS))
        if s != 1
        else grey
    )
    g = _trace(big <= _cut(big, o.get("threshold")), o)
    return _root(w, h, f'<g transform="scale({1 / s:.6g})">{g}</g>')


def tones_svg(rgb: np.ndarray, o: Mapping[str, Any]) -> str:
    """`levels` greys (default 3) as stacked traces: each mask is everything at least that dark, drawn in the mean grey of its band, the
    darkest last, so the steps nest like a print's tints."""
    h, w = rgb.shape[:2]
    n = int(o.get("levels", 3))
    if not 2 <= n <= 8:
        raise RunError("levels is 2 to 8")
    s = float(o.get("scale", 2))
    grey = flat_grey(rgb)
    big = (
        np.asarray(Image.fromarray(grey).resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS))
        if s != 1
        else grey
    )
    ink, paper = float(np.percentile(big, 1)), float(np.percentile(big, 99))
    cuts = [ink + (paper - ink) * (i + 1) / (n + 1) for i in range(n)][::-1]  # lightest first
    parts = []
    for i, c in enumerate(cuts):
        lo = cuts[i + 1] if i + 1 < len(cuts) else 0
        band = big[(big < c) & (big >= lo)]
        v = int(band.mean()) if band.size else int(lo)
        v = 0 if i == len(cuts) - 1 else v
        parts.append(_trace(big < c, o, fill=f"#{v:02x}{v:02x}{v:02x}"))
    return _root(w, h, f'<g transform="scale({1 / s:.6g})">{"".join(parts)}</g>')


def vtracer_svg(rgb: np.ndarray, o: Mapping[str, Any], *, colour: bool | None = None) -> str:
    import vtracer  # pyright: ignore[reportMissingImports]

    h, w = rgb.shape[:2]
    # defaults for a book's plates: VTracer's own (speckle 4, 64 levels, layers 16 apart) made a 28 MB SVG of one coloured page in 68 s;
    # 8 colours, specks under 8 px and layers 32 apart, 1.4 MB in 4 s with the washes still there (2026-11-28, Speaight's plate 29)
    mode = o.get("colours", 8)
    colour = colour if colour is not None else mode not in (2, "binary")
    im = Image.fromarray(
        rgb
        if colour
        else np.stack([np.where(flat_grey(rgb) <= _cut(flat_grey(rgb), o.get("threshold")), 0, 255)] * 3, -1).astype(
            np.uint8
        )
    )
    buf = io.BytesIO()
    im.save(buf, "PNG")
    # color_precision: bits kept per channel (6 ~ 64 levels); layer_difference: how far apart two stacked colours must be
    prec = 6 if not isinstance(mode, int) or mode >= 16 else max(2, min(6, int(np.log2(max(mode, 2))) + 1))
    svg = vtracer.convert_raw_image_to_svg(
        buf.getvalue(),
        img_format="png",
        colormode="color" if colour else "binary",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=int(o.get("speckle", 8)),
        color_precision=prec,
        layer_difference=int(o.get("layer_difference", 32)),
        corner_threshold=60,
        length_threshold=4.0,
        max_iterations=10,
        splice_threshold=45,
        path_precision=2,
    )
    return _root(w, h, _body(svg))


def _washes(rgb: np.ndarray, n: int) -> np.ndarray:
    """The picture's colour without its lines: a median filter wide enough to close hatching, quantised to n colours plus paper."""
    im = Image.fromarray(rgb).filter(ImageFilter.MedianFilter(9)).filter(ImageFilter.GaussianBlur(2))
    q = im.quantize(colors=n, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")
    return np.asarray(q)


def lines_colour_svg(rgb: np.ndarray, o: Mapping[str, Any]) -> str:
    h, w = rgb.shape[:2]
    under = vtracer_svg(
        _washes(rgb, int(o.get("colours", 8))), {**o, "speckle": int(o.get("speckle", 16))}, colour=True
    )
    ink = potrace_svg(rgb, o)
    return _root(w, h, f'<g>{_body(under)}</g><g style="mix-blend-mode:multiply">{_body(ink)}</g>')


# ---- the engines that ask a model ---------------------------------------------------------------------------------------------------------
def _llm(h: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    """The vision-language model of options `llm` (runtimes/delegate.py: in this process, or `via = "service"`)."""
    if "llm" not in h:
        from stratamodels.runtimes import delegate

        h["llm"], h["llm_unload"] = delegate.caller(
            str(h["options"]["llm"]), h.get("models_dir", ""), h["options"].get("via")
        )
    return cast("Callable[[dict[str, Any]], Any]", h["llm"])


def _ask(
    h: dict[str, Any], preset: str, context: str, images: list[str], max_tokens: int, cost: list[float]
) -> Mapping[str, Any]:
    p: dict[str, Any] = dict(preset=preset, context=context, max_tokens=max_tokens)
    if len(images) == 1:
        p["image"] = images[0]
    else:
        p["images"] = images
    r = _llm(h)(p)
    r = cast("Mapping[str, Any]", r if isinstance(r, Mapping) else {"text": r})
    if r.get("cost_usd") is not None:
        cost.append(float(r["cost_usd"]))
    return r


def svg_of(text: str) -> str:
    """The SVG in a model's answer (it may wrap it in a fence or a sentence); one that does not draw fails the job."""
    m = SVG_RE.search(text or "")
    if not m:
        raise RunError(f"the model's answer has no <svg>: {(text or '')[:200]!r}")
    svg = m.group(0)
    if "xmlns" not in svg[:300]:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return svg


def _fit(svg: str, w: int, h: int) -> str:
    """The model's SVG set to the picture's own pixels: width and height replaced (its viewBox kept), so it lies over the picture."""
    head = re.match(r"<svg\b[^>]*>", svg, re.S)
    if not head:
        return svg
    # every sizing attribute the model wrote goes, in either quote style, so the root never holds one twice (and a fitted SVG fits again)
    tag = re.sub(r"""\s(width|height|preserveAspectRatio)\s*=\s*("[^"]*"|'[^']*')""", "", head.group(0))
    if "viewBox" not in tag:
        tag = tag.replace("<svg", f'<svg viewBox="0 0 {w} {h}"', 1)
    tag = tag.replace("<svg", f'<svg width="{w}" height="{h}" preserveAspectRatio="none"', 1)
    return tag + svg[head.end() :]


def _side_by_side(rgb: np.ndarray, svg: str, d: str) -> str:
    """The picture (left) and the model's drawing of it (right) in one PNG, for a refining round."""
    h, w = rgb.shape[:2]
    s = min(1.0, 1200 / max(w, h))
    sw, sh = max(1, round(w * s)), max(1, round(h * s))
    a = Image.fromarray(rgb).resize((sw, sh), Image.Resampling.LANCZOS)
    b = Image.fromarray(render(svg, sw, sh))
    both = Image.new("RGB", (sw * 2 + 16, sh), "#808080")
    both.paste(a, (0, 0))
    both.paste(b, (sw + 16, 0))
    p = os.path.join(d, "compare.png")
    both.save(p)
    return p


def _llm_engine(
    h: dict[str, Any], engine: str, path: str, rgb: np.ndarray, o: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    H, W = rgb.shape[:2]
    cost: list[float] = []
    tokens = int(o.get("max_tokens", 24000))
    size = f'The picture is {W} x {H} pixels: use viewBox="0 0 {W} {H}".'
    extra: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="vectorise-") as d:
        # the model reads a copy no bigger than the APIs read anyway, as PNG: JPEG noise reads as texture to draw
        small = os.path.join(d, "picture.png")
        im = Image.fromarray(rgb)
        im.thumbnail((1568, 1568), Image.Resampling.LANCZOS)
        im.save(small)
        if engine == "llm":
            r = _ask(h, "vectorise-llm-v1", size + " " + str(o.get("context") or ""), [small], tokens, cost)
            svg = _fit(svg_of(str(r.get("text") or "")), W, H)
            steps = [dict(step="draw", bytes=len(svg))]
            for i in range(int(o.get("refine", 0))):
                cmp = _side_by_side(rgb, svg, d)
                r = _ask(h, "vectorise-refine-v1", size + "\n\nYour SVG:\n" + svg, [cmp], tokens, cost)
                try:  # drawn before it replaces the drawing: a revision that does not render must not end the job
                    revised = _fit(svg_of(str(r.get("text") or "")), W, H)
                    render(revised, 64, 64)
                    svg = revised
                    steps.append(dict(step=f"refine {i + 1}", bytes=len(svg)))
                except RunError as e:  # a broken revision keeps the drawing before it
                    steps.append(dict(step=f"refine {i + 1}", error=str(e)[:200]))
            extra["steps"] = steps
        elif engine == "llm-plan":
            r = _ask(h, "vectorise-plan-v1", size, [small], 1024, cost)
            plan = cast("Mapping[str, Any]", r.get("json") if isinstance(r.get("json"), Mapping) else {})
            if not plan:
                import json

                try:
                    plan = json.loads(str(r.get("text") or "{}"))
                except ValueError:
                    raise RunError(f"the plan is not JSON: {str(r.get('text'))[:200]!r}")
            chosen = {
                k: plan[k]
                for k in ("engine", "threshold", "levels", "turdsize", "alphamax", "colours")
                if plan.get(k) is not None
            }
            if chosen.get("engine") not in ("potrace", "potrace-tones", "vtracer", "lines-colour"):
                chosen["engine"] = "potrace"
            if chosen.get("threshold") is not None and not 0 < float(chosen["threshold"]) < 1:
                chosen.pop("threshold")
            oo = {**o, **chosen}
            e = chosen["engine"]
            fn = {
                "potrace": potrace_svg,
                "potrace-tones": tones_svg,
                "vtracer": vtracer_svg,
                "lines-colour": lines_colour_svg,
            }[e]
            svg = fn(rgb, oo)
            extra["plan"] = dict(chosen, why=str(plan.get("why", ""))[:300])
        elif engine == "llm-colour":
            ink = potrace_svg(rgb, o)
            from stratamodels.runtimes.imagekind import features

            if (
                features(rgb)["colour"] <= 0.25
            ):  # nothing for the model to colour: the trace alone, and no call paid for
                under, extra["colour"] = _root(W, H, ""), "none: a picture without colour"
            else:
                r = _ask(h, "vectorise-colour-v1", size, [small], tokens, cost)
                try:
                    under = _fit(svg_of(str(r.get("text") or "")), W, H)
                    render(under, 64, 64)
                except RunError as e:
                    under, extra["colour_error"] = _root(W, H, ""), str(e)[:200]
            # the model's washes keep their own viewBox, nested as an <svg> the picture's size (_fit), so its coordinates mean what it meant
            svg = _root(W, H, f"{under}{_body(ink)}")
        else:  # llm-clean
            traced = potrace_svg(rgb, o)
            kb = len(traced.encode()) / 1024
            limit = float(o.get("clean_kb", 60))
            if kb > limit:
                raise RunError(
                    f"the trace is {kb:.0f} KB, more than a model can read and write back ({limit:.0f} KB): llm-clean is for simple line art"
                )
            r = _ask(h, "vectorise-clean-v1", size + "\n\nThe traced SVG:\n" + traced, [small], tokens, cost)
            svg = _fit(svg_of(str(r.get("text") or "")), W, H)
    render(svg, 32, 32)  # an SVG that does not draw fails here, not in the app that receives it
    if cost:
        extra["cost_usd"] = round(sum(cost), 5)
    return svg, extra
