"""Which typeface a scanned page was printed in, and at what size, weight, width and spacing: role `font-match`. Asked for by the page
rebuild (strata-scan turns a scan into a PDF with real text set in a matching face), which sets each word in the face, style and size given
here. On the CPU (vram_gb = 0): a block takes seconds to a minute, the GPU never waits for it.

How: render and compare. For each candidate face, each line of the block is set in it from Tesseract's words and boxes and compared with the
scan's ink (evals/ink.py: F1 of ink within a pixel or two), and the size, width (hscale), tracking and weight (the threshold of the
anti-aliased rendering: heavier or lighter strokes) are searched to make the two agree best; each word then keeps the style that fits it
(roman, italic, bold, small capitals). The face whose best fit agrees most wins; the others come back ranked, since period type rarely exists
as a font file and the nearest few are what a person or the eval reads.

Candidates: `candidates` names faces of the catalogue (fonts.toml), as a list or as a vision model's answer (any text: the ids and family names
in it are picked out in order, and a `class` with no names picks that class's faces). Without candidates every face is screened (a quick fit
of each, then the full fit of the best `top`): the exhaustive search a proposer should match at a fraction of the time. Unless the job
gives candidates (or `propose = false`), the model's `propose` option (models.toml; Sonnet since 2026-11-23, the font-match eval's pick)
names a vision-language model that is asked for the shortlist first (preset font-propose-v1, the block cut out when `box` is given); if
that call fails, every face is screened and the answer's `proposer` says why.

params:
  image       the page or block picture (PNG with its dpi, else `dpi`); `path` is the same
  ocr         a JSON file holding the `ocr` role's answer for this picture (lines with words and boxes); or `lines` (that answer's lines)
              given directly. Neither: Tesseract reads the picture first (runtimes/tesseract.py)
  box         [x, y, w, h]: fit only the lines inside it (the page's coordinates are kept in the answer)
  candidates  as above; `face` fixes one face (the naive mode's default: tinos)
  mode        fit (default) | measure (the first candidate, sized from the measured x-height and widths, no search: what a proposer alone
              gives) | naive (one face, the line's box height as the size and each word stretched to its box: Tesseract's own PDF overlay)
  propose     false: no proposer, even when the model has one (the eval's `exhaustive`)
  top         faces given the full fit after the screen (default 6)
  min_conf    the Tesseract confidence below which a word is set but not fitted on (default 70)
  out         where to write the rendering (PNG, the picture's size: black type on white), else <data>/generated/fonts/
Answer: {face, class, style, size_pt, ink_f1, precision, recall, path (the rendering), fonts: {id: {family, class, like, files: {style: ttf}}},
faces: [{face, class, ink_f1}] (ranked), blocks: [{box, face, class, ink_f1, lines: [{text, box, baseline_y, size_px, size_pt, hscale,
tracking_em, weight_delta, style, words: [{text, box, style, x, baseline_y, size_px, size_pt, hscale, tracking_em, weight_delta, stroke_em}]}]}],
proposed, proposer ({model, seconds, cost_usd} or {model, seconds, error}; null when none was asked), skipped_lines (the text of lines left
unfitted and undrawn as no type: a word of three letters taller than wide, or an x-height over BIG times the page's median), mode, dpi, seconds}. Boxes are [x, y, w, h] in the picture's pixels, as Tesseract's. The line's values are its first word's, for a
summary; a renderer sets each word with the word's own: `size_pt` (or size_px at the scan's dpi), stretched by `hscale`, `tracking_em` times
the size added after every glyph, its baseline at `baseline_y` and its left ink edge at `x` (that is the rendering's geometry exactly, so a
rebuild reproduces what was scored); weight_delta > 0 means the
print is heavier than the font (ink spread; it is the fit's own unit, 0.5 less the coverage threshold), and `stroke_em` is the same as a
stroke round the glyphs in ems of the size, what a renderer draws: measured on the word as rendered (the ink the lower threshold added, over
the outline's length, both sides), 0 when the print is lighter, since a stroke cannot thin."""

import collections
import difflib
import json
import os
import re
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stratamodels.runtimes import RunError

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = (
    "old-style",
    "transitional",
    "modern",
    "fat-face",
    "clarendon",
    "slab",
    "grotesque",
    "decorated",
    "blackletter",
    "script",
)
# Neighbouring classes count half in the eval's closeness (a Scotch modern taken for a transitional is near; for a slab, far).
NEAR = {
    frozenset(p)
    for p in [
        ("old-style", "transitional"),
        ("transitional", "modern"),
        ("modern", "fat-face"),
        ("clarendon", "slab"),
        ("fat-face", "clarendon"),
        ("decorated", "clarendon"),
        ("decorated", "slab"),
    ]
}
STYLES = ("roman", "italic", "bold", "bold-italic", "small-caps")
STROKE_PER_WEIGHT = (
    20.0  # weight_delta per em of stroke: a coverage threshold 0.1 lower thickens a stem by about a 200th of the size
)
TALL = 3.0  # a word of three or more letters whose ink is this many times taller than wide is no word (Scan)
BIG = 8.0  # a line whose x-height is this many times the page's median is no line of type (Scan)
WORK_XH = 26.0  # the fit runs with the x-height scaled to about this many pixels: enough for serifs, fast enough for every face


def closeness(a, b):
    """1 for the same class, 0.5 for neighbouring classes, else 0."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return 0.5 if frozenset((a, b)) in NEAR else 0.0


@lru_cache(maxsize=1)
def catalogue():
    with open(os.path.join(HERE, "fonts.toml"), "rb") as f:
        faces = tomllib.load(f)["faces"]
    for k, v in faces.items():
        if v["class"] not in CLASSES:
            raise ValueError(f"fonts.toml: {k} has an unknown class {v['class']!r}")
    return faces


# ---- fetching the files --------------------------------------------------------------------------------------------------------------------
# Fontsource (fontsource.org) is the one source: every Google Fonts family and the other open fonts it hosts, each weight and style a TrueType
# file at a versioned URL (its JSON API names them), so a fit is reproducible and nothing parses Google's CSS. The latin subset keeps the
# kerning and the ligatures. A face that Fontsource lacks (the TeX Gyre and Latin Modern faces from CTAN) gives its files' URLs in fonts.toml.
FONTSOURCE = "https://api.fontsource.org/v1/fonts/"
BOLD = 700


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "strata-models"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fontsource_id(family):
    return re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")


def _nearest(weights, w):
    return min(weights, key=lambda x: (abs(x - w), -x)) if weights else None


def fontsource_urls(meta, weight=400, bold=True):
    """{style: ttf url} from a Fontsource font's metadata: the roman and italic at `weight` (or the nearest it has), and at 700 as bold when
    the face is not itself heavy and the family has a bold."""
    variants = meta.get("variants") or {}
    have = sorted(int(w) for w in variants)
    w = _nearest(have, weight)
    if w is None:
        return {}
    picks = {"roman": (w, "normal"), "italic": (w, "italic")}
    if bold and weight < 600 and BOLD in have and w != BOLD:
        picks.update({"bold": (BOLD, "normal"), "bold-italic": (BOLD, "italic")})
    out = {}
    for style, (wt, st) in picks.items():
        subsets = variants.get(str(wt), {}).get(st) or {}
        sub = subsets.get("latin") or subsets.get(meta.get("defSubset", "")) or next(iter(subsets.values()), None)
        url = (sub or {}).get("url", {}).get("ttf")
        if url:
            out[style] = url
    if "roman" not in out and "italic" in out:  # an italic-only family: its one cut is what it prints
        out["roman"] = out.pop("italic")
    return out


def _fontsource(family, get, **kw):
    try:
        meta = json.loads(get(FONTSOURCE + fontsource_id(family)).decode())
    except Exception:
        return {}
    return fontsource_urls(meta, **kw) if isinstance(meta, dict) else {}


def fetch(face_id, models_dir, get=_get):
    """{style: font path} of a catalogue face, downloaded once into <models_dir>/fonts/<id>/ (a manifest.json records what the family has)."""
    entry = catalogue().get(face_id)
    if entry is None:
        raise RunError(f"no face {face_id!r} in fonts.toml")
    folder = os.path.join(models_dir, "fonts", face_id)
    manifest = os.path.join(folder, "manifest.json")
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf8") as f:
            files = json.load(f)
        if all(os.path.isfile(p) for p in files.values()):
            return files
    if entry.get("files"):
        urls = dict(entry["files"])
    else:
        urls = _fontsource(entry["family"], get, weight=int(entry.get("weight", 400)))
        if not urls:
            raise RunError(f"Fontsource has no {entry['family']!r} (face {face_id})")
        if entry.get("sc"):
            sc = _fontsource(entry["sc"], get, bold=False)
            if sc.get("roman"):
                urls["small-caps"] = sc["roman"]
    os.makedirs(folder, exist_ok=True)
    files = {}
    for style, url in urls.items():
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
        p = os.path.join(folder, f"{style}{ext if ext in ('.otf', '.ttf') else '.ttf'}")
        if not os.path.isfile(p):
            try:
                data = get(url)
            except Exception as e:
                # Fontsource's metadata lists every weight with every style, cuts the family lacks too: those answer 404 and are left out.
                # Anything else (a timeout, a 5xx) fails the fetch, so no manifest records a style as missing that a retry would find.
                if style != "roman" and isinstance(e, urllib.error.HTTPError) and e.code == 404:
                    continue
                raise RunError(f"cannot download {face_id} {style} from {url}: {e}") from e
            part = f"{p}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part"  # a name of its own: parallel fits fetch the same face at once
            with open(part, "wb") as f:
                f.write(data)
            os.replace(part, p)
        files[style] = p
    part = f"{manifest}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part"
    with open(part, "w", encoding="utf8") as f:
        json.dump(files, f, indent=1)
    os.replace(part, manifest)
    return files


# ---- rendering ----------------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def _font(path, size):
    return ImageFont.truetype(path, size=size)


@lru_cache(maxsize=512)
def x_ratio(path):
    """(x-height, cap height, descent) of a font file as fractions of its size, measured from the ink of 'x', 'H' and 'p'."""
    f = _font(path, 200.0)

    def ink(ch):
        b = f.getbbox(ch, anchor="ls")
        return b

    xb, hb, pb = ink("x"), ink("H"), ink("p")
    xh = max(1.0, -xb[1]) / 200.0
    ch = max(1.0, -hb[1]) / 200.0
    return xh, ch, max(0.0, pb[3]) / 200.0


# the masks render_word keeps, by their size: a count let one fit hold 33 GB (2026-11-24, see Scan)
RENDER_CACHE_BYTES = 512 * 2**20
_renders: "collections.OrderedDict[tuple[Any, ...], tuple[Any, int, int]]" = collections.OrderedDict()
_render_bytes = [0]


def render_word(path, text, size, hscale, tracking, threshold, synth_sc=False):
    """_render_word's answer, kept least recently used first within RENDER_CACHE_BYTES of masks (a fit renders each word in every face, size
    and weight it tries, and a candidate's words again for the next line, so the cache is what makes a fit fast)."""
    key = (path, text, float(size), float(hscale), float(tracking), float(threshold), bool(synth_sc))
    hit = _renders.get(key)
    if hit is not None:
        _renders.move_to_end(key)
        return hit
    out = _render_word(*key)
    n = int(out[0].nbytes)
    if n <= RENDER_CACHE_BYTES // 8:  # one giant mask is not kept at the cost of everything else
        _renders[key] = out
        _render_bytes[0] += n
        while _render_bytes[0] > RENDER_CACHE_BYTES and _renders:
            _, old = _renders.popitem(last=False)
            _render_bytes[0] -= int(old[0].nbytes)
    return out


def _render_word(path, text, size, hscale, tracking, threshold, synth_sc=False):
    """(ink mask, ink left x, baseline row) of a word set in one font: kerned (Raqm or basic layout), `tracking` ems added after each glyph,
    stretched by `hscale`, and black where the anti-aliased coverage passes `threshold` (lower = heavier). synth_sc: lowercase letters drawn as
    capitals at x-height (small capitals for a face without them)."""
    size = max(2.0, float(size))
    f = _font(path, size)
    asc, desc = f.getmetrics()
    pad = int(size * 0.6) + 4
    if synth_sc:
        xh, ch, _ = x_ratio(path)
        small = _font(path, max(2.0, size * xh / ch * 1.06))
        runs = [(c.upper(), small if c.islower() else f) for c in text]
    else:
        runs = None
    if runs is None and not tracking:
        width = f.getlength(text)
    else:
        width = sum((ff.getlength(c) for c, ff in runs)) if runs else f.getlength(text)
        width += tracking * size * len(text)
    W = int(width + 2 * pad)
    H = int(asc + desc + 2 * pad)
    im = Image.new("L", (max(W, 1), max(H, 1)), 0)
    d = ImageDraw.Draw(im)
    base = pad + asc
    if runs is None and not tracking:
        d.text((pad, base), text, font=f, fill=255, anchor="ls")
    else:
        x = float(pad)
        for i, c in enumerate(text):
            ff = runs[i][1] if runs else f
            ch = runs[i][0] if runs else c
            # kerning between this glyph and the last survives tracking: advance by the pair's length less the last glyph's
            if i and not runs:
                x = pad + f.getlength(text[: i + 1]) - f.getlength(c) + tracking * size * i
            d.text((x, base), ch, font=ff, fill=255, anchor="ls")
            if runs:
                x += ff.getlength(ch) + tracking * size
    if abs(hscale - 1.0) > 1e-3:
        im = im.resize((max(1, int(round(im.width * hscale))), im.height), Image.Resampling.BILINEAR)
    a = np.asarray(im) >= int(255 * threshold)
    cols = np.flatnonzero(a.any(0))
    if not len(cols):
        return a[:, :1], 0, base
    left = int(cols[0])
    return a, left, base


# ---- the scan -----------------------------------------------------------------------------------------------------------------------------
def _lines_of(params, image_path, dpi):
    if isinstance(params.get("lines"), list):
        return params["lines"]
    if params.get("ocr"):
        try:
            with open(params["ocr"], encoding="utf8") as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            raise RunError(f"cannot read ocr {params['ocr']!r}: {e}") from e
        return d.get("lines") or []
    from stratamodels.runtimes import tesseract

    exe = tesseract.binary()
    if not exe:
        raise RunError("no lines given and tesseract is not installed to read them")
    h = dict(key="tesseract-eng", exe=exe, version="tesseract", lang="eng", psm=3, oem=1, tessdata=None)
    return tesseract.run(h, {"image": image_path, "dpi": dpi})["lines"]


def _inside(box, region):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    rx, ry, rw, rh = region
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


class Scan:
    """The page's ink at working scale, with the lines to fit: each line's words with their ink boxes and the line's measured baseline and
    x-height (rows where the ink profile passes 40 % of its peak: the band between baseline and x-line, ascenders and descenders being
    sparse)."""

    def __init__(self, ink_full, lines, dpi, min_conf=70.0):
        self.full = ink_full
        self.dpi = dpi
        H, W = ink_full.shape
        self.lines: list[dict[str, Any]] = []
        self.skipped: list[str] = []
        for ln in lines:
            words = []
            for w in ln.get("words") or [dict(text=ln["text"], box=ln["box"], conf=ln.get("conf", 100))]:
                x, y, bw, bh = (int(v) for v in w["box"])
                x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + bw), min(H, y + bh)
                if x1 <= x0 or y1 <= y0 or not str(w.get("text", "")).strip():
                    continue
                sub = ink_full[y0:y1, x0:x1]
                cols, rows = np.flatnonzero(sub.any(0)), np.flatnonzero(sub.any(1))
                if not len(cols):
                    continue
                ib = [x0 + int(cols[0]), y0 + int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)]
                if len(str(w["text"]).strip()) >= 3 and ib[3] > TALL * ib[2]:
                    continue  # no word of three letters is set taller than wide: a rule, the gutter's shadow or a speck Tesseract read as text
                words.append(dict(text=str(w["text"]), box=[x, y, bw, bh], ink=ib, conf=float(w.get("conf", 100))))
            if not words:
                continue
            x0 = min(w["ink"][0] for w in words)
            y0 = min(w["ink"][1] for w in words)
            x1 = max(w["ink"][0] + w["ink"][2] for w in words)
            y1 = max(w["ink"][1] + w["ink"][3] for w in words)
            prof = ink_full[y0:y1, x0:x1].sum(1).astype(float)
            band = np.flatnonzero(prof >= 0.4 * prof.max()) if prof.max() > 0 else np.array([0, y1 - y0 - 1])
            base = y0 + int(band[-1]) + 1
            xh = max(2, int(band[-1] - band[0] + 1))
            self.lines.append(
                dict(
                    text=" ".join(w["text"] for w in words),
                    box=[x0, y0, x1 - x0, y1 - y0],
                    baseline=base,
                    xh=xh,
                    words=words,
                    block=ln.get("block"),
                )
            )
        # A line far bigger than the page's type is a misreading (2026-11-24: Tesseract read a 1,863 px streak as "Seater"; fitted, its words
        # were drawn at 4,140 px, 143 MB a mask, and one fit took 33 GB before the kernel stopped the service). Display type on these pages
        # is at most about 5 times the body's x-height.
        if self.lines:
            med = float(np.median([ln["xh"] for ln in self.lines]))
            keep = [
                ln for ln in self.lines if ln["xh"] <= BIG * med
            ]  # relative only: a cropped block of one line is all line
            self.skipped = [ln["text"] for ln in self.lines if ln not in keep]
            self.lines = keep
        if not self.lines:
            raise RunError("no words with ink to fit")
        med = float(np.median([ln["xh"] for ln in self.lines]))
        self.med_xh = med
        self.k = min(1.0, WORK_XH / med)  # working scale
        if self.k < 1.0:
            g = Image.fromarray(np.where(ink_full, 255, 0).astype(np.uint8))
            g = g.resize((max(1, int(W * self.k)), max(1, int(H * self.k))), Image.Resampling.BOX)
            self.ink = np.asarray(g) >= 90
        else:
            self.ink = ink_full
        self.min_conf = min_conf

    def fit_words(self, ln):
        """The words a line is fitted on: confident, with letters; all of them when that leaves fewer than two."""
        good = [w for w in ln["words"] if w["conf"] >= self.min_conf and re.search(r"[A-Za-z]{2}", w["text"])]
        return good if len(good) >= 2 else ln["words"]


def _f1(ref, got, tol=1):
    from stratamodels.evals.ink import dilate

    nr, ng = int(ref.sum()), int(got.sum())
    if not nr or not ng:
        return 0.0
    p = float((got & dilate(ref, tol)).sum()) / ng
    r = float((ref & dilate(got, tol)).sum()) / nr
    return 2 * p * r / (p + r) if p + r else 0.0


def _paste(canvas, mask, left, base, x, y):
    """OR a word's mask into the canvas with its ink left at x and its baseline at row y."""
    ox, oy = int(round(x)) - left, int(round(y)) - base
    H, W = canvas.shape
    h, w = mask.shape
    x0, y0, x1, y1 = max(0, ox), max(0, oy), min(W, ox + w), min(H, oy + h)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] |= mask[y0 - oy : y1 - oy, x0 - ox : x1 - ox]


def _word_text(text, style):
    return text.upper() if style == "small-caps-synth-upper" else text


class Face:
    def __init__(self, fid, files):
        self.id = fid
        self.files = files
        self.entry = catalogue()[fid]

    def styles(self):
        out = [s for s in ("roman", "italic", "bold") if s in self.files]
        out.append("small-caps")
        return out

    def file(self, style):
        if style == "small-caps":
            return self.files.get("small-caps") or self.files["roman"]
        return self.files.get(style) or self.files["roman"]

    def word(self, style, text, size, hscale, tracking, thr):
        synth = style == "small-caps" and "small-caps" not in self.files
        return render_word(
            self.file(style), text, round(size, 2), round(hscale, 3), round(tracking, 3), round(thr, 2), synth
        )


def _line_canvas(scan, ln, words, face, styles, p, k, pad=6, shifts=True):
    """(scan ink, rendered ink) of a line's box at scale k with the words set by params p = (size, hscale, tracking, thr) in work pixels."""
    x0, y0, w, h = ln["box"]
    X0, Y0 = int(x0 * k) - pad, int(y0 * k) - pad
    X1, Y1 = int((x0 + w) * k) + pad, int((y0 + h) * k) + pad
    ink = scan.ink if k == scan.k else scan.full
    H, W = ink.shape
    cx0, cy0, cx1, cy1 = max(0, X0), max(0, Y0), min(W, X1), min(H, Y1)
    ref = ink[cy0:cy1, cx0:cx1]
    got = np.zeros_like(ref)
    size, hs, trk, thr = p
    for wd, st in zip(words, styles):
        m, left, base = face.word(st, wd["text"], size, hs, trk, thr)
        x, y = wd["ink"][0] * k - cx0, ln["baseline"] * k - cy0
        if (
            shifts
        ):  # a word's own placement (the scan's spacing, a baseline that wanders): a pixel either way, kept if it helps
            best = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    tmp = np.zeros_like(ref)
                    _paste(tmp, m, left, base, x + dx, y + dy)
                    sx0 = max(0, int(x) - 2)
                    sx1 = min(ref.shape[1], int(x + m.shape[1]) + 2)
                    s = float((tmp[:, sx0:sx1] & ref[:, sx0:sx1]).sum()) - 0.5 * float(
                        (tmp[:, sx0:sx1] & ~ref[:, sx0:sx1]).sum()
                    )
                    if best is None or s > best[0]:
                        best = (s, dx, dy)
            x, y = x + best[1], y + best[2]
            wd["_d"] = (best[1] / k, best[2] / k)
        _paste(got, m, left, base, x, y)
    return ref, got


def _hscale_for(face, ln, words, styles, size, trk, thr, k):
    """The width that makes the set words as wide as the printed ones (median over words)."""
    rs = []
    for wd, st in zip(words, styles):
        m, _left, _ = face.word(st, wd["text"], size, 1.0, trk, thr)
        cols = np.flatnonzero(m.any(0))
        if len(cols) and len(wd["text"]) >= 2:
            rs.append(wd["ink"][2] * k / max(1, cols[-1] - cols[0] + 1))
    return float(np.clip(np.median(rs), 0.7, 1.4)) if rs else 1.0


def _size0(face, ln, k, style="roman"):
    xh, ch, _ = x_ratio(face.file(style))
    letters = re.sub(r"[^A-Za-z]", "", ln["text"])
    caps = letters and sum(c.isupper() for c in letters) / len(letters) > 0.7
    return ln["xh"] * k / (ch if caps else xh)


TALLER = 1.2  # times the page's median x-height: a line measured taller may be a heading, or its capitals measured for its x-height


def _sizes0(scan, face, ln, k, style):
    """The sizes a line's fit starts from: its measured x-height's (`_size0`) and, for a line `TALLER` than the page's that is not in capitals,
    the size at which that height is its capitals' and the page's own size (the best fit wins). A short line of tall letters ("Polly.—Whiskey!") has
    ink over 40 % of its profile's peak up to its capitals' tops, so its x-height comes out half as tall again (2026-11-25: 46 px against the
    page's 31 on four pages of the 1874 book, each such line set 1.4 to 1.6 times too big, over its neighbours), and the search reaches only
    18 % either way. The page's own size is for a line measured over two rows of type (90 px, set at 230 for 67)."""
    s = _size0(face, ln, k, style)
    xh, ch, _ = x_ratio(face.file(style))
    capitals = s * xh < ln["xh"] * k * 0.999  # _size0 took the height for the capitals'
    if capitals or ln["xh"] <= TALLER * scan.med_xh:
        return [s]
    return [s, ln["xh"] * k / ch, scan.med_xh * k / xh]


def fit_line(scan, ln, face, style, quick=False, p0=None):
    """((size, hscale, tracking, thr), F1) of one line in one face and style at working scale."""
    k = scan.k
    words = scan.fit_words(ln)
    styles = [style] * len(words)
    s0 = p0[0] if p0 else _size0(face, ln, k, style)

    def score(size, trk=0.0, thr=0.5):
        hs = _hscale_for(face, ln, words, styles, size, trk, thr, k)
        ref, got = _line_canvas(scan, ln, words, face, styles, (size, hs, trk, thr), k, shifts=not quick)
        return _f1(ref, got), (size, hs, trk, thr)

    if quick:  # the screen: three sizes at two weights (a scan is often heavier than the font, and a light face must not lose for that alone)
        starts = [s0] if p0 else _sizes0(scan, face, ln, k, style)
        best = max(
            (score(s * g, 0.0, t) for s in starts for g in (0.9, 1.0, 1.1) for t in (0.3, 0.5)), key=lambda r: r[0]
        )
        return best[1], best[0]
    best = max(
        (score(s0 * g, 0.0, p0[3] if p0 else 0.5) for g in (0.82, 0.88, 0.94, 1.0, 1.06, 1.12, 1.18)),
        key=lambda r: r[0],
    )
    step = 0.03
    for _ in range(4):  # local refinement of the size
        s = best[1][0]
        cand = [score(s * (1 - step)), score(s * (1 + step))]
        top = max(cand, key=lambda r: r[0])
        if top[0] > best[0]:
            best = top
        else:
            step /= 2
    for trk in (-0.03, 0.03, 0.06):
        r = score(best[1][0], trk, best[1][3])
        if r[0] > best[0] + 0.002:
            best = r
    for thr in (0.2, 0.3, 0.4, 0.6, 0.7):
        r = score(best[1][0], best[1][2], thr)
        if r[0] > best[0] + 0.002:
            best = r
    return best[1], best[0]


def _line_style(face, scan, ln, quick):
    """The line's main style: roman or italic (small capitals and bold are decided word by word after)."""
    fits = [
        (st, *fit_line(scan, ln, face, st, quick=True))
        for st in (("roman", "italic") if "italic" in face.files else ("roman",))
    ]
    return max(fits, key=lambda r: r[2])


def score_line(scan, ln, face, style, size, trk=0.0, thr=0.5, quick=False):
    """(F1, (size, hscale, tracking, thr)) of one line set with these params, the width solved from the words."""
    words = scan.fit_words(ln)
    styles = [style] * len(words)
    hs = _hscale_for(face, ln, words, styles, size, trk, thr, scan.k)
    ref, got = _line_canvas(scan, ln, words, face, styles, (size, hs, trk, thr), scan.k, shifts=not quick)
    return _f1(ref, got), (size, hs, trk, thr)


def fit_face(scan, face, quick=False, keep=0.015):
    """{face, f1, lines: [...]} of one face over the block. quick: the screen (three sizes, roman or italic per line, no word shifts).
    After each line's own fit, the block's lines take its median size, tracking and weight unless their own are better by `keep` (F1): body
    type is one size, and a line fitted alone trades size against width."""
    lines = []
    for ln in scan.lines:
        st, p, f = _line_style(face, scan, ln, True)
        if not quick:
            p, f = fit_line(scan, ln, face, st, p0=p)
        lines.append(dict(style=st, p=p, f1=f))
    if not quick and len(lines) > 1:
        rel = [
            lf["p"][0] / _size0(face, ln, scan.k, lf["style"]) for ln, lf in zip(scan.lines, lines)
        ]  # size against the measured x-height
        med = float(np.median(rel))
        trk = float(np.median([lf["p"][2] for lf in lines]))
        thr = float(np.median([lf["p"][3] for lf in lines]))
        for ln, lf, r in zip(scan.lines, lines, rel):
            if abs(r / med - 1) > 0.12:  # a line of another size (a heading, a footnote): keep its own
                continue
            f, p = score_line(scan, ln, face, lf["style"], med * _size0(face, ln, scan.k, lf["style"]), trk, thr)
            if f >= lf["f1"] - keep:
                lf.update(p=p, f1=f)
    total = sum(lf["f1"] * float(sum(w["ink"][2] for w in ln["words"])) for ln, lf in zip(scan.lines, lines))
    wsum = sum(float(sum(w["ink"][2] for w in ln["words"])) for ln in scan.lines)
    return dict(face=face.id, f1=total / max(wsum, 1.0), lines=lines)


MARGIN = {
    "bold": 0.07,
    "bold-italic": 0.07,
}  # bold must win clearly: a heavy impression otherwise reads as bold words in roman lines


def word_styles(scan, ln, face, lfit, margin=0.04):
    """Each word's style: the line's own unless another of the face's styles fits that word better by `margin` (F1; more for bold), with the
    line's size, width and weight (small capitals made from capitals get their own width)."""
    k = scan.k
    out = []
    p = lfit["p"]
    for wd in ln["words"]:
        best = None
        for st in face.styles():
            pp = p
            if st == "small-caps" and "small-caps" not in face.files:
                hs = _hscale_for(face, ln, [wd], [st], p[0], p[2], p[3], k)
                pp = (p[0], min(hs, p[1] * 1.1), p[2], p[3])
            ref, got = _line_canvas(scan, dict(ln, box=wd["ink"]), [wd], face, [st], pp, k)
            f = _f1(ref, got)
            if st != lfit["style"]:
                f -= MARGIN.get(st, margin)
            if best is None or f > best[0]:
                best = (f, st, pp, wd.get("_d", (0.0, 0.0)))  # each trial moves the word: keep the winner's own shift
        out.append((best[1], best[2], best[3]))
    return out


def stroke_em(face, style, text, size, hscale, tracking, thr):
    """The stroke (ems of the size, centred on the outline) that thickens the word as much as the threshold `thr` does: the anti-aliased
    edge is about a pixel wide, so this depends on the size in pixels, and is measured, not converted. 0 for thr >= 0.5."""
    if thr >= 0.5:
        return 0.0
    a = face.word(style, text, size, hscale, tracking, 0.5)[0]
    b = face.word(style, text, size, hscale, tracking, thr)[0]
    if a.shape != b.shape:
        return 0.0
    edge = int((a[:, 1:] != a[:, :-1]).sum() + (a[1:, :] != a[:-1, :]).sum())
    if not edge:
        return 0.0
    return round(2.0 * float(b.sum() - a.sum()) / edge / float(size), 5)


# ---- the whole job ------------------------------------------------------------------------------------------------------------------------
def parse_candidates(value):
    """Face ids from a list or any text (a vision model's answer): ids and family names in the order they appear; a class named and no face:
    that class's faces. [] when nothing is recognised."""
    cat = catalogue()
    if value is None or value == "":
        return []
    text = " , ".join(str(v) for v in value) if isinstance(value, list) else str(value)
    low = text.lower()
    hits = []
    for fid, e in cat.items():
        for name in (fid, e["family"].lower() + (" black" if fid.endswith("-black") else "")):
            for m in re.finditer(r"(?<![a-z0-9-])" + re.escape(name) + r"(?![a-z0-9-])", low):
                hits.append((m.start(), -len(name), fid))
    # a near miss of an id (gemma wrote "libra-bodoni" for libre-bodoni in 26 of 65 answers): the closest id, if it is close
    ids = list(cat)
    for m in re.finditer(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)+", low):
        if m.group(0) not in cat and not any(h[0] <= m.start() < h[0] - h[1] for h in hits):
            near = difflib.get_close_matches(m.group(0), ids, n=1, cutoff=0.85)
            if near:
                hits.append((m.start(), -len(m.group(0)), near[0]))
    seen, out = set(), []
    for _, _, fid in sorted(hits):
        if fid not in seen:
            seen.add(fid)
            out.append(fid)
    if not out:
        m = re.search(r'"?class"?\s*[:=]\s*"?([a-z-]+)', low)
        named = [m.group(1)] if m else [c for c in CLASSES if c in low]
        for c in named:
            out += [fid for fid, e in cat.items() if e["class"] == c and fid not in out]
    return out


PROPOSE_PROMPT = "Identify the typeface of the printed text in this picture."


def load(model, models_dir):
    opts = dict(model.options)
    return dict(
        key=model.key,
        models_dir=models_dir,
        top=int(opts.get("top", 6)),
        get=_get,
        propose=opts.get("propose") or None,
        propose_tokens=int(opts.get("propose_tokens", 1500)),
    )


def _proposer(handle):
    """A callable that runs the `propose` model (a vision-language model of the registry, e.g. claude-sonnet-5-5-api) on params: loaded once,
    in this process (a cloud model holds no GPU; its spend is counted by its own runtime)."""
    if "proposer" not in handle:
        from stratamodels import registry, runtimes, settings

        eng = (
            settings.engine()
        )  # the service's registry is the database's (models edited in the app or API); the files only without one
        reg = registry.Registry.from_db(eng) if eng is not None else registry.Registry.load()
        m = reg.models.get(handle["propose"])
        if m is None:
            raise RunError(f"the propose model {handle['propose']!r} is not in the registry")
        rt = runtimes.get(m.runtime)
        h = rt.load(m, handle["models_dir"])
        handle["proposer"] = lambda params: rt.run(h, params)
        handle["proposer_unload"] = lambda: rt.unload(h)
    return handle["proposer"]


def _propose(handle, path, box):
    """The proposer's shortlist for the picture (the block when `box` is given), as its answer's text; or (None, error) when it fails, and
    the fit then screens every face."""
    t0 = time.monotonic()
    crop = None
    try:
        image = path
        if box:
            x, y, w, h = (int(round(v)) for v in box)
            fd, crop = tempfile.mkstemp(suffix=".png", prefix="fontfit-propose-")
            os.close(fd)
            with Image.open(path) as im:
                im.crop((x, y, x + w, y + h)).save(crop, dpi=im.info.get("dpi") or (300, 300))
            image = crop
        r = _proposer(handle)(
            dict(
                system_preset="font-propose-v1",
                prompt=PROPOSE_PROMPT,
                image=image,
                max_tokens=handle.get("propose_tokens", 1500),
            )
        )
        r = r if isinstance(r, dict) else {"text": r}
        if not isinstance(r.get("text"), str) or not r["text"].strip():
            raise RunError("the proposer answered no text")
        if not parse_candidates(
            r["text"]
        ):  # an answer naming no face or class (an unreadable block): screen every face instead
            raise RunError(f"no catalogue face in the proposer's answer: {r['text'][:200]}")
        return r["text"], dict(
            model=handle["propose"], seconds=round(time.monotonic() - t0, 2), cost_usd=r.get("cost_usd")
        )
    except Exception as e:  # a refused or failed call (budget, network) costs the fit its shortlist, never the job
        return None, dict(
            model=handle["propose"], seconds=round(time.monotonic() - t0, 2), error=f"{type(e).__name__}: {e}"[:500]
        )
    finally:
        if crop:
            os.remove(crop)


def _out_path(params):
    out = params.get("out")
    if out:
        return str(out)
    from stratamodels import settings

    folder = os.path.join(settings.data_dir(), "generated", "fonts", time.strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{uuid.uuid4().hex[:12]}.png")


def _naive(scan, face):
    """Tesseract's own overlay: the line box's height is the font's ascent plus descent, the baseline sits a descent above the box's
    bottom, and each word is stretched to its box's width."""
    lines = []
    for ln in scan.lines:
        boxes = [w["box"] for w in ln["words"]]
        y0 = min(b[1] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        f = _font(face.file("roman"), 100.0)
        a, d = f.getmetrics()
        size = (y1 - y0) * 100.0 / (a + d)
        base = y1 - d * size / 100.0
        lines.append(dict(style="roman", size=size, base=base, words=[]))
        for w in ln["words"]:
            m, _left, _ = face.word("roman", w["text"], size, 1.0, 0.0, 0.5)
            cols = np.flatnonzero(m.any(0))
            hs = w["box"][2] / max(1, cols[-1] - cols[0] + 1) if len(cols) else 1.0
            lines[-1]["words"].append(
                dict(style="roman", p=(size, float(np.clip(hs, 0.3, 3.0)), 0.0, 0.5), d=(0.0, 0.0))
            )
    return lines


def _measured(scan, face, style="roman"):
    """A face as a proposer gives it, without the search: size from the measured x-height, width from the words, normal weight."""
    lines = []
    for ln in scan.lines:
        words = scan.fit_words(ln)
        size = _size0(face, ln, scan.k, style)
        hs = _hscale_for(face, ln, words, [style] * len(words), size, 0.0, 0.5, scan.k)
        p = (size / scan.k, hs, 0.0, 0.5)
        lines.append(
            dict(
                style=style,
                size=p[0],
                base=ln["baseline"],
                words=[dict(style=style, p=p, d=(0.0, 0.0)) for _ in ln["words"]],
            )
        )
    return lines


def _final(scan, face, fit):
    """The fitted face's lines at full scale, with word styles."""
    lines = []
    for ln, lf in zip(scan.lines, fit["lines"]):
        ws = word_styles(scan, ln, face, lf)
        wl = []
        for st, p, d in ws:
            wl.append(dict(style=st, p=(p[0] / scan.k, p[1], p[2], p[3]), d=d))
        lines.append(dict(style=lf["style"], size=lf["p"][0] / scan.k, base=ln["baseline"], words=wl))
    return lines


def _compose(scan, face, lines):
    canvas = np.zeros_like(scan.full)
    for ln, lf in zip(scan.lines, lines):
        for wd, wf in zip(ln["words"], lf["words"]):
            m, left, base = face.word(wf["style"], wd["text"], *wf["p"])
            _paste(canvas, m, left, base, wd["ink"][0] + wf["d"][0], lf["base"] + wf["d"][1])
    return canvas


def run(handle, params):
    from stratamodels.evals import ink as INK

    t0 = time.monotonic()
    path = params.get("image") or params.get("path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise RunError(f"no such picture: {path!r}")
    mode = str(params.get("mode") or "fit")
    if mode not in ("fit", "measure", "naive"):
        raise RunError(f"mode is fit, measure or naive, not {mode!r}")
    try:
        im = Image.open(path)
        im.load()
    except OSError as e:
        raise RunError(f"not a picture: {path} ({e})") from e
    found = im.info.get("dpi")
    dpi = float(params.get("dpi") or (found[0] if found and found[0] > 1 else 300))
    lines = _lines_of(params, path, dpi)
    if params.get("box"):
        lines = [ln for ln in lines if _inside(ln["box"], params["box"])]
    full, _ = INK.binarise(im, dpi, deskew=False)
    scan = Scan(full, lines, dpi, float(params.get("min_conf", 70)))
    proposer = None
    if (
        mode == "fit"
        and "candidates" not in params
        and not params.get("face")
        and handle.get("propose")
        and params.get("propose", True) is not False
    ):  # no shortlist given: the role's proposer makes one (models.toml: options.propose), unless the job says propose = false
        text, proposer = _propose(handle, path, params.get("box"))
        if text is not None:
            params = dict(params, candidates=text)
    proposed = parse_candidates(params.get("candidates"))
    if params.get("face"):
        proposed = parse_candidates([params["face"]]) or proposed
    if mode == "naive" and not proposed:
        proposed = ["tinos"]
    if params.get("candidates") not in (None, "", []) and not proposed and mode != "naive":
        raise RunError(f"no catalogue face in the candidates: {str(params.get('candidates'))[:200]}")
    cat = catalogue()
    pool = proposed or list(cat)
    md = handle["models_dir"]
    get = handle.get("get", _get)
    faces = {}
    for fid in pool if mode == "fit" else pool[:1]:
        try:
            faces[fid] = Face(fid, fetch(fid, md, get))
        except RunError:
            if mode != "fit":
                raise
    if not faces:
        raise RunError("none of the candidate faces could be fetched")
    ranked, screen, top = [], [], 0
    if mode == "fit":
        top = int(params.get("top") or handle.get("top", 6))
        screen = (
            sorted((fit_face(scan, f, quick=True) for f in faces.values()), key=lambda r: -r["f1"])
            if len(faces) > top
            else []
        )
        keep = [r["face"] for r in screen[:top]] if screen else list(faces)
        fits = sorted((fit_face(scan, faces[fid]) for fid in keep), key=lambda r: -r["f1"])
        ranked = fits + screen[top:]
        best = faces[fits[0]["face"]]
        flines = _final(scan, best, fits[0])
    else:
        best = next(iter(faces.values()))
        style = "roman"
        m = re.search(
            r'"?style"?\s*[:=]\s*"?(roman|italic|bold|small-caps)', str(params.get("candidates") or "").lower()
        )
        if m and mode == "measure" and m.group(1) in best.styles():
            style = m.group(1)
        flines = _naive(scan, best) if mode == "naive" else _measured(scan, best, style)
    canvas = _compose(scan, best, flines)
    out = _out_path(params)
    Image.fromarray(np.where(canvas, 0, 255).astype(np.uint8)).save(out, dpi=(dpi, dpi))
    words_box = np.zeros_like(scan.full)
    for ln in scan.lines:
        for wd in ln["words"]:
            words_box |= INK.box_mask(
                scan.full.shape, [wd["ink"][0] - 3, wd["ink"][1] - 3, wd["ink"][2] + 6, wd["ink"][3] + 6]
            )
    score = INK.ink_f1(scan.full, canvas, tol_px=max(1, round(dpi / 250)), mask=words_box)
    out_lines, styles = [], []
    for ln, lf in zip(scan.lines, flines):
        size = lf["size"]
        p0 = lf["words"][0]["p"] if lf["words"] else (size, 1.0, 0.0, 0.5)
        ws = []
        for wd, wf in zip(ln["words"], lf["words"]):
            sz, hs, trk, thr = wf["p"]
            ws.append(
                dict(
                    text=wd["text"],
                    box=wd["box"],
                    style=wf["style"],
                    x=round(
                        float(wd["ink"][0] + wf["d"][0]), 1
                    ),  # where the word's left ink edge goes, after its own shift
                    baseline_y=round(float(lf["base"] + wf["d"][1]), 1),
                    size_px=round(float(sz), 2),
                    size_pt=round(float(sz) * 72.0 / dpi, 2),
                    hscale=round(float(hs), 3),
                    tracking_em=round(float(trk), 3),
                    weight_delta=round(0.5 - float(thr), 2),
                    stroke_em=stroke_em(best, wf["style"], wd["text"], sz, hs, trk, thr),
                )
            )
            styles.append((wf["style"], wd["ink"][2]))
        out_lines.append(
            dict(
                text=ln["text"],
                box=ln["box"],
                baseline_y=round(float(lf["base"]), 1),
                size_px=round(size, 2),
                size_pt=round(size * 72.0 / dpi, 2),
                hscale=round(float(p0[1]), 3),
                tracking_em=round(float(p0[2]), 3),
                weight_delta=round(0.5 - float(p0[3]), 2),
                style=lf["style"],
                words=ws,
            )
        )
    tally = {}
    for st, w in styles:
        tally[st] = tally.get(st, 0) + w
    main_style = max(tally, key=lambda s: tally[s]) if tally else "roman"
    sizes = [ln["size_pt"] for ln in out_lines]
    entry = cat[best.id]
    fonts = {
        fid: dict(
            family=cat[fid]["family"], cls=cat[fid]["class"], like=cat[fid].get("like", ""), files=faces[fid].files
        )
        for fid in faces
        if fid == best.id or mode == "fit"
    }
    for v in fonts.values():
        v["class"] = v.pop("cls")
    bx0 = min(ln["box"][0] for ln in scan.lines)
    by0 = min(ln["box"][1] for ln in scan.lines)
    bx1 = max(ln["box"][0] + ln["box"][2] for ln in scan.lines)
    by1 = max(ln["box"][1] + ln["box"][3] for ln in scan.lines)
    nfull = len(ranked) - max(0, len(screen) - top) if mode == "fit" else 0
    faces_ranked = [
        dict(face=r["face"], ink_f1=round(r["f1"], 4), screened=i >= nfull, **{"class": cat[r["face"]]["class"]})
        for i, r in enumerate(ranked)
    ]
    return dict(
        face=best.id,
        family=entry["family"],
        **{"class": entry["class"]},
        style=main_style,
        size_pt=round(float(np.median(sizes)), 2) if sizes else None,
        ink_f1=score["f1"],
        precision=score["precision"],
        recall=score["recall"],
        iou=score["iou"],
        path=out,
        fonts=fonts,
        faces=faces_ranked,
        blocks=[
            dict(
                box=[bx0, by0, bx1 - bx0, by1 - by0],
                face=best.id,
                **{"class": entry["class"]},
                ink_f1=score["f1"],
                lines=out_lines,
            )
        ],
        proposed=proposed,
        skipped_lines=scan.skipped,
        proposer=proposer,
        mode=mode,
        dpi=dpi,
        seconds=round(time.monotonic() - t0, 3),
        model=handle.get("key"),
    )


def unload(handle):
    if "proposer_unload" in handle:
        handle["proposer_unload"]()
    handle.clear()
    _renders.clear()  # up to half a GB of drawn words, and the fonts they were drawn with
    _render_bytes[0] = 0
    _font.cache_clear()
    x_ratio.cache_clear()
