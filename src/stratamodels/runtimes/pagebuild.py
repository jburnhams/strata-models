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
  groups      with `ocr`: a vision model's regrouping of Tesseract's lines (preset page-layout-fix-v1, which is shown them numbered by
              `list_lines`): blocks in reading order, each the numbers of its lines and a role, and figures as boxes in thousandths. The
              lines keep Tesseract's measured boxes and text; a line in no block is left out (a stain, a stamp, the next page showing through)
  texts       with `ocr` and `groups`: a model's corrections of Tesseract's text (preset page-text-fix-v1, shown the same numbered lines):
              {lines: [{n, text}]}, each replacing line n's text (an empty text drops the line: not print); lines it leaves out keep
              Tesseract's text
  font_fit    with `ocr`: a `font-match` answer for the page (its words are Tesseract's, matched here by their boxes): each line set in the
              fitted face, each word in its own style, size, scale, tracking and weight where it was measured. A line whose corrected text
              has another number of words keeps the face, style and size of its words and is spaced to its box. Its `fonts` files are
              the faces drawn with (`fonts` and `font_dir` add to them)
  styles      with `font_fit`: the fitted word styles to keep (default all: roman, italic, bold, small-caps); a word fitted in another is
              set in roman. A fit word by word to worn print calls a heavy word bold and a line's last word small capitals now and then
  snap        with `font_fit`: sizes within this fraction of their block's median size take the median (default 0: as fitted), so a block of
              one type size is drawn in one size, not the word-by-word scatter of a fit to worn print
  crop        cut each figure's picture out of the scan (`image`), its paper whitened, rather than leave its box empty
  rules       find the ruled lines the scan's ink holds (the border round the page, the rule under a running head) and draw each as a
              figure of kind rule cut from the scan: no reader is asked for them (page-layout-fix-v2 leaves the border out)
  group       with `ocr` and no `groups`, on a model whose options name a layout model (`group`, as `pagebuild-opus-layout` does: the
              page-build role's default since 2026-11-23): that vision model is shown the picture and Tesseract's numbered lines
              (preset `group_preset`, default page-layout-fix-v2) and its answer is the `groups`, so an app sending a plain OCR answer
              gets the page-layout eval's pick, not Tesseract's blocks. `group = false` draws Tesseract's blocks (the eval baselines name
              model `pagebuild`, which has no layout model). A failed call (budget, network, no blocks) draws Tesseract's blocks and the
              answer's `grouper` says why
  list_lines  with `ocr`: draw nothing, answer {text: Tesseract's lines numbered for page-layout-fix-v1, one a line: number, Tesseract's
              block, box [x0, y0, x1, y1] in thousandths, text; lines}
  image       the page picture: its size and dpi for `layout`, its dpi for `ocr` (the boxes stay in the ocr answer's pixels; an upscaled read
              scales the dpi)
  dpi         the scan's resolution when the picture does not say
  fonts       {face id: TTF/OTF path}; `font_dir` a folder whose fonts are named by file stem (`OldStandardTT-Regular`)
  small_caps_faces   face ids that are small-caps cuts (drawn as they are, not synthesised)
  fit         fit each line's spacing to its box (default true)
  out         the .pdf path to write (default: <data>/pages/<hash of the description>.pdf, so a rebuild is cached by what decides it)
  png         also write the PDF rasterised at the scan's dpi beside it (default true): what the blind judge and the ink metrics look at
Answer: {pdf, png, path (the png again: the picture the Judge and Outputs tabs show), width, height, dpi, text (read back from the PDF, in reading order), page (the description drawn), blocks, lines, seconds,
grouper ({model, seconds, cost_usd} or {model, seconds, error}; null when no layout model was asked), cost_usd (the layout model's, when it cost any)}."""

import glob
import hashlib
import itertools
import json
import os
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
from PIL import Image
from strata_python.pages import Block, Figure, Line, Page, PageError, Span
from strata_python.pages.render import rasterise, read_text, render

from stratamodels import settings  # pyright: ignore[reportMissingTypeStubs]  (settings.py is not in the strict list yet)
from stratamodels.runtimes import RunError  # pyright: ignore[reportMissingTypeStubs]  (the package __init__ is not in the strict list yet)

GROUP_PRESET = "page-layout-fix-v2"  # page-layout's winner (2026-11-15): Opus regrouping Tesseract's lines, block_f1 96.5 % on 20 pages


def load(model: Any, models_dir: str) -> dict[str, Any]:
    return dict(key=model.key, options=dict(model.options), models_dir=models_dir)


def unload(handle: dict[str, Any]) -> None:
    undo = handle.get("grouper_unload")
    if callable(undo):
        undo()
    handle.clear()


def _grouper(handle: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    """A callable that runs the model's layout model (options `group`, a vision-language model of the registry) on params: loaded once, in
    this process, as font-match loads its proposer (a cloud model holds no GPU; its spend is counted by its own runtime)."""
    if "grouper" not in handle:
        from stratamodels import registry, runtimes  # pyright: ignore[reportMissingTypeStubs]  (not in the strict list yet)

        key = str(handle["options"]["group"])
        eng = cast(
            "Any", settings
        ).engine()  # the service's registry is the database's (models edited in the app); the files only without one
        reg = cast("Any", registry).Registry.from_db(eng) if eng is not None else cast("Any", registry).Registry.load()
        m = reg.models.get(key)
        if m is None:
            raise RunError(f"the layout model {key!r} is not in the registry")
        rt = cast("Any", runtimes).get(m.runtime)
        h = rt.load(m, handle.get("models_dir", ""))

        def call(params: dict[str, Any]) -> Any:
            return rt.run(h, params)

        handle["grouper"] = call
        handle["grouper_unload"] = lambda: rt.unload(h)
    return cast("Callable[[dict[str, Any]], Any]", handle["grouper"])


def group_lines(
    handle: dict[str, Any], ocr: Mapping[str, Any], image: str
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    """The layout model's `groups` for Tesseract's lines (shown numbered, with the picture) and what the call cost; or (None, why) when it
    fails, and the page is drawn in Tesseract's blocks: a refused or failed call costs the page its layout, never the job."""
    opts = cast("Mapping[str, Any]", handle.get("options") or {})
    model, t0 = str(opts.get("group")), time.monotonic()
    try:
        r = _grouper(handle)(
            dict(
                preset=str(opts.get("group_preset") or GROUP_PRESET),
                context=list_lines(ocr),
                image=image,
                max_tokens=int(opts.get("group_tokens", 8192)),
            )
        )
        r = cast("Mapping[str, Any]", r if isinstance(r, Mapping) else {"text": r})
        groups = (
            r.get("json")
            if isinstance(r.get("json"), Mapping)
            else _json(str(r.get("text") or ""), "layout model's answer")
        )
        if not isinstance(groups, Mapping) or not cast("Mapping[str, Any]", groups).get("blocks"):
            raise RunError("the layout model gave no blocks")
        return cast("Mapping[str, Any]", groups), dict(
            model=model, seconds=round(time.monotonic() - t0, 2), cost_usd=r.get("cost_usd")
        )
    except Exception as e:  # budget, network, a bad answer: Tesseract's blocks instead
        return None, dict(model=model, seconds=round(time.monotonic() - t0, 2), error=f"{type(e).__name__}: {e}"[:500])


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


GAP = 1.5  # a space between words wider than this many line heights splits a line: a running head and its folio, side-by-side columns


def _pieces(ln: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """One Tesseract line, split where its words stand far apart. Tesseract reads a running head and the page number at the end of it as one
    line, which no regrouping could then part (the commonest block it gets wrong on these pages); a justified line's spaces are well under a
    line height, so only a real gap splits."""
    words = [w for w in cast("list[Mapping[str, Any]]", ln.get("words") or []) if str(w.get("text") or "").strip()]
    if len(words) < 2:
        return [ln]
    h = float(cast("Sequence[Any]", ln["box"])[3])
    runs: list[list[Mapping[str, Any]]] = [[words[0]]]
    for a, b in itertools.pairwise(words):
        ab, bb = cast("Sequence[Any]", a["box"]), cast("Sequence[Any]", b["box"])
        if float(bb[0]) - (float(ab[0]) + float(ab[2])) > GAP * h:
            runs.append([])
        runs[-1].append(b)
    if len(runs) == 1:
        return [ln]
    return [
        {
            **ln,
            "box": list(_union([cast("Sequence[float]", w["box"]) for w in r])),
            "text": " ".join(str(w["text"]) for w in r),
            "words": r,
        }
        for r in runs
    ]


def ocr_lines(ocr: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The lines of an ocr answer that hold text, in its order, each split at wide gaps between its words (`_pieces`): what `list_lines`
    numbers from 1 and `groups` refers to by those numbers."""
    return [
        p
        for ln in cast("list[Mapping[str, Any]]", ocr.get("lines") or [])
        if str(ln.get("text") or "").strip()
        for p in _pieces(ln)
    ]


def list_lines(ocr: Mapping[str, Any]) -> str:
    w, h = float(ocr.get("width") or 0), float(ocr.get("height") or 0)
    if not w or not h:
        raise RunError("the ocr answer has no page size")
    out: list[str] = []
    for n, ln in enumerate(ocr_lines(ocr), 1):
        x, y, bw, bh = (float(v) for v in cast("Sequence[Any]", ln["box"]))
        box = ",".join(str(round(v)) for v in (x * 1000 / w, y * 1000 / h, (x + bw) * 1000 / w, (y + bh) * 1000 / h))
        out.append(f"{n} b{int(ln.get('block') or 0)} [{box}] {str(ln['text']).strip()}")
    return "\n".join(out)


MakeLine = Callable[[int, Mapping[str, Any]], "Line | None"]


def _plain(n: int, ln: Mapping[str, Any]) -> Line | None:
    return Line(box=cast("Any", tuple(float(v) for v in ln["box"])), spans=[Span(str(ln["text"]))])


def _texts(texts: Mapping[str, Any]) -> dict[int, str]:
    """{line number: corrected text} from a page-text-fix answer; numbers that are not whole numbers are skipped (the first answer wins)."""
    out: dict[int, str] = {}
    for c in cast("list[Any]", texts.get("lines") or []):
        if not isinstance(c, Mapping):
            continue
        c = cast("Mapping[str, Any]", c)
        try:
            n = int(cast("Any", c.get("n")))
        except (TypeError, ValueError):
            continue
        out.setdefault(n, " ".join(str(c.get("text") or "").split()))
    return out


def _wkey(box: object) -> tuple[int, ...]:
    return tuple(round(float(v)) for v in cast("Sequence[Any]", box))


def _fitted(fit: Mapping[str, Any]) -> tuple[str, dict[tuple[int, ...], Mapping[str, Any]], dict[str, str], list[str]]:
    """(face, {Tesseract word box: fitted word}, {face key: font file}, small-caps cuts) from a font-match answer. A face key is the face and
    a style (`caslon:italic`): the renderer picks a file by face only, so each cut is a face of its own."""
    face = str(fit.get("face") or "")
    words: dict[tuple[int, ...], Mapping[str, Any]] = {}
    for b in cast("list[Mapping[str, Any]]", fit.get("blocks") or []):
        for ln in cast("list[Mapping[str, Any]]", b.get("lines") or []):
            for wd in cast("list[Mapping[str, Any]]", ln.get("words") or []):
                if wd.get("box") is not None:
                    words.setdefault(_wkey(wd["box"]), wd)
    fonts: dict[str, str] = {}
    small_caps: list[str] = []
    for fid, entry in cast("Mapping[str, Mapping[str, Any]]", fit.get("fonts") or {}).items():
        for style, path in cast("Mapping[str, Any]", entry.get("files") or {}).items():
            fonts[f"{fid}:{style}"] = str(path)
            if style == "small-caps":
                small_caps.append(f"{fid}:{style}")
    if not face or f"{face}:roman" not in fonts:
        raise RunError(f"the font-match answer names no face with a roman file: {face!r}")
    return face, words, fonts, small_caps


FIT_STYLES = ("roman", "italic", "bold", "small-caps")


def _styled(
    face: str, fonts: Mapping[str, str], wd: Mapping[str, Any], keep: Sequence[str] = FIT_STYLES
) -> tuple[str, str]:
    """(span style, face key) of a fitted word: its own cut when the face has one; a small-caps word without a cut is synthesised from the
    roman by the renderer. A style not in `keep` is set in roman."""
    style = str(wd.get("style") or "roman")
    if style not in FIT_STYLES or style not in keep:
        style = "roman"
    if f"{face}:{style}" in fonts:
        return style, f"{face}:{style}"
    return (style if style == "small-caps" else "roman"), f"{face}:roman"


def line_maker(
    texts: Mapping[int, str] | None = None,
    fit: tuple[str, Mapping[tuple[int, ...], Mapping[str, Any]], Mapping[str, str]] | None = None,
    keep: Sequence[str] = FIT_STYLES,
) -> MakeLine:
    """How each Tesseract line (numbered as list_lines numbers it) becomes a drawn line: its text corrected (`texts`), its words set as the
    font fit placed them (`fit`), in the fitted styles named in `keep` (others in roman). A corrected line whose words no longer pair one for
    one with Tesseract's keeps the fit's face, style and size and is spaced to its box: the words it changed were never measured."""

    def make(n: int, ln: Mapping[str, Any]) -> Line | None:
        box = cast("Any", tuple(float(v) for v in ln["box"]))
        text = texts.get(n, str(ln["text"])) if texts is not None else str(ln["text"])
        if not text.strip():
            return None
        if fit is None:
            return Line(box=box, spans=[Span(text)])
        face, fwords, fonts = fit
        own = [w for w in cast("list[Mapping[str, Any]]", ln.get("words") or []) if str(w.get("text") or "").strip()]
        got = [fwords.get(_wkey(w["box"])) for w in own]
        known = [g for g in got if g is not None]
        if not known:
            return Line(box=box, spans=[Span(text)])
        parts = text.split()
        if len(parts) == len(own) and len(known) == len(got):
            spans: list[Span] = []
            for i, (t, wd) in enumerate(zip(parts, known, strict=True)):
                style, key = _styled(face, fonts, wd, keep)
                spans.append(
                    Span(
                        t + (" " if i < len(parts) - 1 else ""),
                        style=style,
                        face=key,
                        size_pt=float(wd["size_pt"]),
                        tracking_em=float(wd.get("tracking_em") or 0.0),
                        hscale=float(wd.get("hscale") or 1.0),
                        stroke_em=float(wd.get("stroke_em") or 0.0),
                        x=float(wd["x"]),
                        baseline_y=float(wd["baseline_y"]),
                    )
                )
            return Line(box=box, spans=spans)
        styles = [str(k.get("style") or "roman") for k in known]
        style, key = _styled(face, fonts, {"style": max(set(styles), key=styles.count)}, keep)
        return Line(
            box=box,
            spans=[
                Span(
                    text,
                    style=style,
                    hscale=statistics.median(float(k.get("hscale") or 1.0) for k in known),
                    stroke_em=statistics.median(float(k.get("stroke_em") or 0.0) for k in known),
                )
            ],
            face=key,
            size_pt=statistics.median(float(k["size_pt"]) for k in known),
            baseline_y=statistics.median(float(k["baseline_y"]) for k in known),
        )

    return make


def snap_sizes(page: Page, within: float) -> None:
    """Sizes within `within` of their block's median (over its words) set to that median: a block is mostly one type size, and a fit to worn
    print scatters it word by word. The width each word was fitted to is kept (its scale and tracking scaled by the change), so only the
    letters' height changes. Headings, small capitals and other sizes further off keep their own."""
    for b in page.blocks:
        sizes = [s.size_pt for ln in b.lines for s in ln.spans if s.size_pt] + [
            ln.size_pt for ln in b.lines if ln.size_pt and not any(s.size_pt for s in ln.spans)
        ]
        if not sizes:
            continue
        m = statistics.median(sizes)
        for ln in b.lines:
            if ln.size_pt and abs(ln.size_pt - m) <= within * m:
                for s in ln.spans:
                    if not s.size_pt:
                        s.hscale *= ln.size_pt / m
                ln.size_pt = m
            for s in ln.spans:
                if s.size_pt and abs(s.size_pt - m) <= within * m:
                    # the fit trades size against width (a larger size, narrower): keep the word's width and its spacing, change its height
                    k = s.size_pt / m
                    s.hscale *= k
                    s.tracking_em *= k
                    s.size_pt = m


def from_groups(
    ocr: Mapping[str, Any], groups: Mapping[str, Any], dpi: float, w: float, h: float, make: MakeLine = _plain
) -> Page:
    """Tesseract's lines regrouped by a vision model: each block the lines it names (in Tesseract's order within the block, which reads them
    top to bottom; a line named twice stays in its first block), its box their union, its role the model's; figures in thousandths of the
    page. A block naming no line there is dropped, so a model that numbers wrongly loses blocks rather than inventing them. `make` draws each
    line (line_maker: corrected text, fitted type)."""
    lines = ocr_lines(ocr)
    taken: set[int] = set()
    blocks: list[Block] = []
    for i, g in enumerate(cast("list[Mapping[str, Any]]", groups.get("blocks") or [])):
        nums: list[int] = []
        for v in cast("list[Any]", g.get("lines") or []):
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= len(lines) and n not in taken:
                taken.add(n)
                nums.append(n)
        if not nums:
            continue
        mine = [ln for ln in (make(n, lines[n - 1]) for n in sorted(nums)) if ln is not None]
        if not mine:
            continue
        try:
            blocks.append(
                Block(
                    box=_union([ln.box for ln in mine]),
                    lines=mine,
                    role=str(g.get("role") or "body"),
                    order=len(blocks),
                )
            )
        except PageError as e:
            raise RunError(f"bad groups (block {i}): {e}") from e
    figures: list[Figure] = []
    for i, f in enumerate(cast("list[Mapping[str, Any]]", groups.get("figures") or [])):
        try:
            figures.append(
                Figure(box=_permille(f.get("box"), w, h, f"figure {i}"), kind=str(f.get("kind") or "figure"))
            )
        except PageError as e:
            raise RunError(f"bad groups: {e}") from e
    try:
        return Page(width=w, height=h, dpi=dpi, blocks=blocks, figures=figures)
    except PageError as e:
        raise RunError(f"ocr answer has no page size: {e}") from e


def _paper(path: str) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """The scan in grey with its paper made white (each pixel over the paper's brightness, the 90th percentile): a crop of a yellowed page
    drawn on a white PDF page would show as a tinted patch."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L"), dtype=np.float32)
    npa: Any = np  # numpy's percentile and clip overloads are partially unknown to pyright
    paper = max(float(npa.percentile(g, 90)), 1.0)
    return npa.clip(g * (255.0 / paper), 0, 255).astype(np.uint8)


def _crop_folder() -> str:
    folder = os.path.join(str(cast("object", settings.data_dir())), "pages", "crops")
    os.makedirs(folder, exist_ok=True)
    return folder


def _save_crop(arr: np.ndarray[Any, np.dtype[np.uint8]], what: object) -> str:
    """A crop written once under <data>/pages/crops, named by what it is cut from (so a rebuild is cached by it too)."""
    p = os.path.join(
        _crop_folder(), hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()[:16] + ".png"
    )
    if not os.path.isfile(p):
        part = f"{p}.{os.getpid()}.part.png"
        Image.fromarray(arr).save(part)
        os.replace(part, p)
    return p


def _stamp(path: str) -> list[object]:
    st = os.stat(path)
    return [os.path.abspath(path), st.st_size, int(st.st_mtime)]


def crop_figures(page: Page, image: str) -> None:
    """Each figure with no picture gets its box cut from the scan (paper whitened). The page may be in the pixels of an upscaled reading:
    boxes are scaled to the picture's."""
    todo = [f for f in page.figures if f.image is None]
    if not todo:
        return
    g = _paper(image)
    kx, ky = g.shape[1] / page.width, g.shape[0] / page.height
    for f in todo:
        x, y, w, h = f.box
        x0, y0 = max(0, int(x * kx)), max(0, int(y * ky))
        x1, y1 = min(g.shape[1], int(round((x + w) * kx))), min(g.shape[0], int(round((y + h) * ky)))
        if x1 > x0 and y1 > y0:
            f.image = _save_crop(g[y0:y1, x0:x1], ["figure", _stamp(image), [x0, y0, x1, y1]])


RULE_MIN = 0.12  # a ruled line runs at least this fraction of the page's shorter side: a border or a running head's rule, not a dash
RULE_MAX_MM = 1.2  # and is at most this thick on average (thicker is a black bar, the binding's shadow, a picture)
RULE_REACH = 0.06  # and lies within this fraction of the page's size around the text: a border, not the book's edge beside the leaf


def _overlap(a: tuple[float, float, float, float], b: Sequence[float]) -> float:
    """The area two boxes share: `a` as [x0, y0, x1, y1], `b` as [x, y, w, h]."""
    w = min(a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def find_rules(image: str, dpi: float) -> list[tuple[int, int, int, int, np.ndarray[Any, np.dtype[np.bool_]]]]:
    """The ruled lines in a scan: [(x0, y0, x1, y1, its ink inside that box)] in the picture's pixels. Ink that stays after an opening by a
    long line (a minimum then a maximum filter, linear in the page's size) along rows, and along columns, after a short closing that
    bridges worn gaps; each connected piece whose mean thickness is a rule's, away from the picture's edge (the dark edge of the leaf)."""
    from scipy import ndimage  # pyright: ignore[reportMissingTypeStubs]  (scipy ships no stubs)

    from stratamodels.evals import ink as INK  # pyright: ignore[reportMissingTypeStubs]

    with Image.open(image) as im:
        ink = cast("np.ndarray[Any, np.dtype[np.bool_]]", cast("Any", INK).binarise(im, dpi, deskew=False)[0])
    hgt, wid = ink.shape
    length = max(20, int(RULE_MIN * min(hgt, wid)))
    thick = max(2.0, RULE_MAX_MM * dpi / 25.4)
    edge = max(4, int(0.01 * min(hgt, wid)))
    across = max(3, round(dpi / 50))
    nd: Any = ndimage
    out: list[tuple[int, int, int, int, np.ndarray[Any, np.dtype[np.bool_]]]] = []
    for axis in (1, 0):
        m = ink.astype(np.uint8)
        # a printed rule is rarely level on the scan: thickened across (a few pixels) a slightly skewed rule's rows overlap into long runs,
        # then gaps of a few pixels along it (worn print) are closed
        m = nd.maximum_filter1d(m, across, axis=1 - axis)
        m = nd.minimum_filter1d(nd.maximum_filter1d(m, 5, axis=axis), 5, axis=axis)
        m = nd.maximum_filter1d(nd.minimum_filter1d(m, length, axis=axis), length, axis=axis).astype(bool) & ink
        lab, _n = nd.label(m, structure=np.ones((3, 3), dtype=bool))
        for i, sl in enumerate(cast("list[Any]", nd.find_objects(lab)), 1):
            if sl is None:
                continue
            ys, xs = cast("slice", sl[0]), cast("slice", sl[1])
            y0, y1, x0, x1 = int(ys.start), int(ys.stop), int(xs.start), int(xs.stop)
            piece = lab[ys, xs] == i
            run = (x1 - x0) if axis == 1 else (y1 - y0)
            if run < length or float(piece.sum()) / run > thick:
                continue
            if x0 < edge or y0 < edge or x1 > wid - edge or y1 > hgt - edge:
                continue
            out.append((x0, y0, x1, y1, piece))
    return out


def add_rules(page: Page, image: str) -> None:
    """The scan's ruled lines (find_rules) as figures of kind rule, each its own ink cut from the scan (paper whitened, everything else in its
    box white, so no letter near a skewed border comes with it)."""
    if not page.blocks:
        return
    g = _paper(image)
    kx, ky = page.width / g.shape[1], page.height / g.shape[0]
    pad = 2
    # only near the print: a scan often holds the book's edge and cover round the leaf, whose edges are long straight lines too
    tx, ty, tw, th = _union([b.box for b in page.blocks])
    rx, ry = RULE_REACH * page.width, RULE_REACH * page.height
    lines = [ln.box for b in page.blocks for ln in b.lines]
    for x0, y0, x1, y1, piece in find_rules(image, page.dpi / kx):
        bx0, by0, bx1, by1 = x0 * kx, y0 * ky, x1 * kx, y1 * ky
        if bx0 < tx - rx or by0 < ty - ry or bx1 > tx + tw + rx or by1 > ty + th + ry:
            continue
        area = max((bx1 - bx0) * (by1 - by0), 1e-9)
        if sum(_overlap((bx0, by0, bx1, by1), b) for b in lines) > 0.5 * area:
            continue  # a stroke of display type (the stems of a condensed letter are long and thin too), not a rule
        a0, b0 = max(0, x0 - pad), max(0, y0 - pad)
        a1, b1 = min(g.shape[1], x1 + pad), min(g.shape[0], y1 + pad)
        keep = np.zeros((b1 - b0, a1 - a0), dtype=bool)
        keep[y0 - b0 : y1 - b0, x0 - a0 : x1 - a0] = piece
        from scipy import ndimage  # pyright: ignore[reportMissingTypeStubs]

        keep = cast("np.ndarray[Any, np.dtype[np.bool_]]", cast("Any", ndimage).binary_dilation(keep, iterations=pad))
        crop = cast(
            "np.ndarray[Any, np.dtype[np.uint8]]", cast("Any", np).where(keep, g[b0:b1, a0:a1], 255).astype(np.uint8)
        )
        path = _save_crop(crop, ["rule", _stamp(image), [a0, b0, a1, b1]])
        page.figures.append(Figure(box=(a0 * kx, b0 * ky, (a1 - a0) * kx, (b1 - b0) * ky), image=path, kind="rule"))


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
        groups = _json(params.get("groups"), "groups")
        texts = _json(params.get("texts"), "texts") if params.get("texts") else None
        fit = _json(params.get("font_fit"), "font fit") if params.get("font_fit") else None
        if isinstance(groups, Mapping):
            keep = params.get("styles") or FIT_STYLES
            if not isinstance(keep, (list, tuple)) or not all(st in FIT_STYLES for st in cast("Sequence[Any]", keep)):
                raise RunError(f"styles is a list of {', '.join(FIT_STYLES)}, not {keep!r}")
            make = line_maker(
                _texts(cast("Mapping[str, Any]", texts)) if isinstance(texts, Mapping) else None,
                _fitted(cast("Mapping[str, Any]", fit))[:3] if isinstance(fit, Mapping) else None,
                [str(st) for st in cast("Sequence[Any]", keep)],
            )
            return from_groups(o, cast("Mapping[str, Any]", groups), dpi, float(w or 0), float(h or 0), make)
        if texts is not None or fit is not None:
            raise RunError("texts and font_fit name Tesseract's lines by number: give groups too")
        return from_ocr(o, dpi, w, h)
    raise RunError("give a page description (page), an ocr answer (ocr) or a layout (layout)")


def run(handle: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    t0 = time.monotonic()
    if params.get("list_lines"):
        ocr = _json(params.get("ocr"), "ocr answer")
        if not isinstance(ocr, Mapping):
            raise RunError("list_lines needs an ocr answer (ocr)")
        text = list_lines(cast("Mapping[str, Any]", ocr))
        return dict(text=text, lines=text.count("\n") + 1 if text else 0, model=handle.get("key"))
    grouper: dict[str, Any] | None = None
    opts = cast("Mapping[str, Any]", handle.get("options") or {})
    if (
        opts.get("group")
        and params.get("group", True) is not False
        and params.get("ocr")
        and not any(params.get(k) for k in ("groups", "page", "layout"))
    ):
        ocr = _json(params.get("ocr"), "ocr answer")
        image = params.get("image")
        if isinstance(ocr, Mapping) and isinstance(image, str) and os.path.isfile(image):
            groups, grouper = group_lines(handle, cast("Mapping[str, Any]", ocr), image)
            if groups is not None:
                params = {**params, "groups": groups}
        elif isinstance(ocr, Mapping):
            grouper = dict(
                model=str(opts["group"]), seconds=0.0, error="the layout model reads the page: give its picture (image)"
            )
    page = page_of(params)
    fonts: dict[str, str] = {}
    small_caps = [str(s) for s in cast("Sequence[Any]", params.get("small_caps_faces") or [])]
    ff = _json(params.get("font_fit"), "font fit") if params.get("font_fit") and params.get("groups") else None
    if isinstance(ff, Mapping):
        _face, _words, ffonts, fsc = _fitted(cast("Mapping[str, Any]", ff))
        fonts.update(ffonts)
        small_caps += fsc
        if params.get("snap"):
            snap_sizes(page, float(params["snap"]))
    fonts.update(_fonts(params))
    image = params.get("image")
    if params.get("crop") or params.get("rules"):
        if not isinstance(image, str) or not os.path.isfile(image):
            raise RunError("crop and rules cut from the scan: give its picture (image)")
        if params.get("crop"):
            crop_figures(page, image)
        if params.get("rules"):
            add_rules(page, image)
    fit = params.get("fit", True) is not False
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
        grouper=grouper,
        **({"cost_usd": grouper["cost_usd"]} if grouper and isinstance(grouper.get("cost_usd"), (int, float)) else {}),
    )
