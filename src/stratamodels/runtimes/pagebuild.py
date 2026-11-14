"""A scanned page rebuilt as a PDF with real text, role `page-build` (CPU lane, vram_gb = 0): a page description in, the PDF out, drawn by
strata-python's renderer (`strata_python.pages`, the `[pdf]` extra), the same code strata-scan prints books with, so an eval scores what the
app prints. The readers that fill the description (Tesseract's boxes, a layout model, a vision model's JSON, `font-match`'s fitted faces) are
other roles; this one only draws, and every candidate of a page-rebuild eval differs only in what it hands here.

params (one of `page`, `ocr` or `layout`; each as JSON, as JSON text (a model's answer, code fences allowed) or as a path to a JSON file):
  page        a page description (strata_python.pages.Page: width, height, dpi in scan pixels, blocks of lines of styled spans, figures)
  ocr         an `ocr` role answer (Tesseract: lines with boxes and their block number), turned into one body block per Tesseract block, each
              line one roman span in the standard face, sized from its box: today's text overlay set as visible text (the baseline)
  layout      a vision model's reading of the page (preset page-layout-v1): blocks in reading order, each a box [x0, y0, x1, y1] in
              thousandths of the page's width and height, a role and its lines of text; figures as boxes and kinds. Each block's lines
              share its height evenly (the model gives no line boxes), so the drawing is rough; the layout metrics read the boxes
  image       the page picture: its size and dpi for `layout`, its dpi for `ocr` (the boxes stay in the ocr answer's pixels; an upscaled read
              scales the dpi)
  dpi         the scan's resolution when the picture does not say
  fonts       {face id: TTF/OTF path}; `font_dir` a folder whose fonts are named by file stem (`OldStandardTT-Regular`)
  small_caps_faces   face ids that are small-caps cuts (drawn as they are, not synthesised)
  fit         fit each line's spacing to its box (default true)
  out         the .pdf path to write (default: <data>/pages/<hash of the description>.pdf, so a rebuild is cached by what decides it)
  png         also write the PDF rasterised at the scan's dpi beside it (default true): what the blind judge and the ink metrics look at
Answer: {pdf, png, path (the png again: the picture the Judge and Outputs tabs show), width, height, dpi, text (read back from the PDF, in reading order), page (the description drawn), blocks, lines, seconds}."""

import glob
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

from PIL import Image
from strata_python.pages import Block, Figure, Line, Page, PageError, Span
from strata_python.pages.render import rasterise, read_text, render

from stratamodels import settings  # pyright: ignore[reportMissingTypeStubs]  (settings.py is not in the strict list yet)
from stratamodels.runtimes import RunError  # pyright: ignore[reportMissingTypeStubs]  (the package __init__ is not in the strict list yet)


def load(model: Any, models_dir: str) -> dict[str, Any]:
    return dict(key=model.key, options=dict(model.options))


def unload(handle: dict[str, Any]) -> None:
    handle.clear()


def _union(boxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def from_ocr(ocr: Mapping[str, Any], dpi: float, width: float | None = None, height: float | None = None) -> Page:
    """Tesseract's answer as a page description: one body block per Tesseract block (its box the union of its lines'), in Tesseract's reading
    order, each line one roman span in the standard face. Sizes come from the renderer's rule for a line box, so this is the plain overlay a
    reader with no idea of the type would make."""
    w = float(width or ocr.get("width") or 0)
    h = float(height or ocr.get("height") or 0)
    order: list[int] = []
    lines: dict[int, list[Line]] = {}
    for ln in cast("list[Mapping[str, Any]]", ocr.get("lines") or []):
        text = str(ln.get("text") or "")
        if not text.strip():
            continue
        b = int(ln.get("block") or 0)
        if b not in lines:
            order.append(b)
            lines[b] = []
        x, y, bw, bh = (float(v) for v in cast("Sequence[Any]", ln["box"]))
        lines[b].append(Line(box=(x, y, bw, bh), spans=[Span(text)]))
    blocks = [Block(box=_union([ln.box for ln in lines[b]]), lines=lines[b], order=i) for i, b in enumerate(order)]
    try:
        return Page(width=w, height=h, dpi=dpi, blocks=blocks)
    except PageError as e:
        raise RunError(f"ocr answer has no page size: {e}") from e


def _picture(path: object, dpi: object) -> tuple[int, int, float]:
    if not isinstance(path, str) or not os.path.isfile(path):
        raise RunError(f"no such picture: {path!r}")
    with Image.open(path) as im:
        found = im.info.get("dpi")
        # PNG keeps pixels a metre, so a 150 dpi scan reads back as 150.0124: round what the file says (a given dpi is taken as it is)
        d = float(cast("Any", dpi)) if dpi else (round(float(found[0]), 1) if found and found[0] > 1 else 0.0)
        if not d:
            raise RunError(f"{path}: no dpi in the picture; give dpi")
        return im.width, im.height, d


def _fonts(params: Mapping[str, Any]) -> dict[str, str]:
    fonts: dict[str, str] = {}
    folder = params.get("font_dir")
    if folder:
        if not os.path.isdir(str(folder)):
            raise RunError(f"font_dir is not a folder: {folder!r}")
        for p in sorted(glob.glob(os.path.join(str(folder), "*.[ot]tf"))):
            fonts[os.path.splitext(os.path.basename(p))[0]] = p
    fonts.update({str(k): str(v) for k, v in cast("Mapping[str, Any]", params.get("fonts") or {}).items()})
    return fonts


def _permille(box: object, w: float, h: float, where: str) -> tuple[float, float, float, float]:
    try:
        x0, y0, x1, y1 = (min(1000.0, max(0.0, float(v))) for v in cast("Sequence[Any]", box))
    except (TypeError, ValueError) as e:
        raise RunError(f"{where}: a layout box is [x0, y0, x1, y1] in thousandths, not {box!r}") from e
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return (x0 * w / 1000, y0 * h / 1000, (x1 - x0) * w / 1000, (y1 - y0) * h / 1000)


def from_layout(layout: Mapping[str, Any], w: float, h: float, dpi: float) -> Page:
    """A vision model's layout (boxes in thousandths of the page, blocks in reading order) as a page description in the picture's pixels."""
    blocks: list[Block] = []
    for i, b in enumerate(cast("list[Mapping[str, Any]]", layout.get("blocks") or [])):
        box = _permille(b.get("box"), w, h, f"block {i}")
        texts = [str(t) for t in cast("list[Any]", b.get("lines") or []) if str(t).strip()]
        step = box[3] / len(texts) if texts else 0.0
        lines = [Line(box=(box[0], box[1] + k * step, box[2], step), spans=[Span(t)]) for k, t in enumerate(texts)]
        role = str(b.get("role") or "body")
        try:
            blocks.append(Block(box=box, lines=lines, role=role, order=i))
        except PageError as e:
            raise RunError(f"bad layout: {e}") from e
    figures: list[Figure] = []
    for i, f in enumerate(cast("list[Mapping[str, Any]]", layout.get("figures") or [])):
        try:
            figures.append(
                Figure(box=_permille(f.get("box"), w, h, f"figure {i}"), kind=str(f.get("kind") or "figure"))
            )
        except PageError as e:
            raise RunError(f"bad layout: {e}") from e
    return Page(width=w, height=h, dpi=dpi, blocks=blocks, figures=figures)


def _json(v: object, what: str) -> object:
    """A param given as JSON, as JSON text (a model's answer, perhaps in a code fence) or as the path of a JSON file."""
    if not isinstance(v, str):
        return v
    t = v.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    if t[:1] in ("{", "["):
        try:
            return json.loads(t)
        except ValueError as e:
            raise RunError(f"{what} is not valid JSON: {e}") from e
    if not os.path.isfile(v):
        raise RunError(f"no such {what}: {v[:200]!r}")
    with open(v, encoding="utf8") as f:
        return json.load(f)


def page_of(params: Mapping[str, Any]) -> Page:
    """The description a job asks to draw, from `page`, from `ocr` and the picture, or from `layout` and the picture."""
    page = _json(params.get("page"), "page description")
    if isinstance(page, Mapping):
        try:
            return Page.from_dict(cast("Mapping[str, Any]", page))
        except (PageError, KeyError, TypeError, ValueError) as e:
            raise RunError(f"bad page description: {e}") from e
    layout = _json(params.get("layout"), "layout")
    if isinstance(layout, Mapping):
        w, h, dpi = _picture(params.get("image"), params.get("dpi"))
        return from_layout(cast("Mapping[str, Any]", layout), w, h, dpi)
    ocr = _json(params.get("ocr"), "ocr answer")
    if isinstance(ocr, Mapping):
        o = cast("Mapping[str, Any]", ocr)
        if params.get("image"):
            w, h, dpi = _picture(params.get("image"), params.get("dpi"))
            ow = float(o.get("width") or 0)
            if ow and ow != w:
                # Tesseract's boxes are in the picture it read, which `clean = "upscale"` doubled: keep its pixels and scale the dpi
                # with them, so the page is the picture's size in inches and every box stays on it
                oh = float(o.get("height") or 0)
                if not oh or abs(ow / w - oh / h) > 0.01:
                    raise RunError(f"the ocr answer is {ow:g}x{oh:g} but the picture is {w}x{h}: not the same page")
                dpi, (w, h) = dpi * ow / w, (ow, oh)
        else:
            w, h = o.get("width"), o.get("height")
            dpi = float(params.get("dpi") or 0)
            if not dpi:
                raise RunError("give the page picture (image) or its dpi with an ocr answer")
        return from_ocr(o, dpi, w, h)
    raise RunError("give a page description (page), an ocr answer (ocr) or a layout (layout)")


def run(handle: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    t0 = time.monotonic()
    page = page_of(params)
    fonts = _fonts(params)
    fit = params.get("fit", True) is not False
    small_caps = [str(s) for s in cast("Sequence[Any]", params.get("small_caps_faces") or [])]
    try:
        pdf = render(page, fonts, small_caps_faces=small_caps, fit=fit)
    except PageError as e:
        raise RunError(str(e)) from e
    out = params.get("out")
    if out is None:
        key = hashlib.sha256(
            json.dumps([page.to_dict(), fonts, small_caps, fit], sort_keys=True).encode() + pdf
        ).hexdigest()[:16]
        folder = os.path.join(str(cast("object", settings.data_dir())), "pages")
        os.makedirs(folder, exist_ok=True)
        out = os.path.join(folder, f"{key}.pdf")
    out = str(out)
    if not out.lower().endswith(".pdf") or not os.path.isdir(os.path.dirname(os.path.abspath(out))):
        raise RunError(f"out must be a .pdf path in an existing folder: {out!r}")
    with open(out, "wb") as f:
        f.write(pdf)
    png = None
    if params.get("png", True) is not False:
        png = out[:-4] + ".png"
        img = rasterise(pdf, dpi=page.dpi)
        size = (round(page.width), round(page.height))
        if img.size != size:
            # pdfium rounds the page's size in points up to whole pixels, so a 2660 px scan at 500 dpi comes back 2661 wide: put it back on
            # the scan's own pixels (white where it was short), or no ink metric could lay it over the scan
            fixed = Image.new("L", size, 255)
            fixed.paste(img.crop((0, 0, min(size[0], img.width), min(size[1], img.height))), (0, 0))
            img = fixed
        img.save(png, dpi=(page.dpi, page.dpi))
    return dict(
        pdf=os.path.abspath(out),
        png=os.path.abspath(png) if png else None,
        path=os.path.abspath(png) if png else None,
        width=page.width,
        height=page.height,
        dpi=page.dpi,
        text=read_text(pdf),
        page=page.to_dict(),
        blocks=len(page.blocks),
        lines=sum(len(b.lines) for b in page.blocks),
        seconds=round(time.monotonic() - t0, 3),
        model=handle.get("key"),
    )
