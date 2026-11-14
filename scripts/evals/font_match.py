"""Build the eval dataset `font-match`: blocks of type for the `font-match` role (runtimes/fontfit.py) to identify.

Two kinds of item:
- real: text blocks of the two public-domain books (`font_match/regions.toml`: Ward 1874's title page, advertisements in a dozen display
  faces and its dialogue; Collier 1828's chapter openings, text and footnotes), cut from the Internet Archive scans at their own resolution
  and labelled by eye with the kind of type (class) and main style. Their faces mostly exist in no font file, so they are scored by ink
  (how closely the fitted face reproduces the print) and by class.
- synthetic: lines of the same books' text (the ocr-hard readings) set in a known catalogue face, style and size and printed badly on
  purpose (ink spread or starved, paper tone, noise, blur, a slight tilt): the face is known, so these score exact hits, top-3 and size.

Each item's input is `image` (a PNG with its dpi) and `ocr` (Tesseract's reading of it, JSON: the words and boxes every candidate is given,
so all of them set the same words). Pictures go to `<data>/evals/font-match/`.

    .venv/bin/python scripts/evals/font_match.py                      # fetches the scans from archive.org (cached in the output folder)
    .venv/bin/python scripts/evals/font_match.py --synthetic 40 --seed 7
"""

import argparse
import glob
import io
import json
import os
import random
import sys
import tomllib
import urllib.request

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from stratamodels import settings
from stratamodels.evals import core as C
from stratamodels.runtimes import fontfit as F
from stratamodels.runtimes import tesseract as T

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "font_match")
TEXTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_hard")
# book -> (Internet Archive item, its scans' resolution from scandata.xml)
BOOKS = {"ward1874": ("punchjudywithins00war", 500), "collier1828": ("bib_fict_4117090_1", 360)}
DISPLAY = {"fat-face", "clarendon", "slab", "grotesque", "decorated", "blackletter", "script"}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "strata-models eval builder"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def leaf(book, n, cache, get=get):
    """The scan of one leaf of a book as a lightness picture (cached as PNG)."""
    ia, dpi = BOOKS[book]
    path = os.path.join(cache, f"{book}-{n:04d}.png")
    if not os.path.isfile(path):
        url = f"https://archive.org/download/{ia}/{ia}_jp2.zip/{ia}_jp2%2F{ia}_{n:04d}.jp2"
        im = Image.open(io.BytesIO(get(url))).convert("L")
        im.save(path, dpi=(dpi, dpi))
    return Image.open(path), dpi


def ocr(png, dpi):
    exe = T.binary()
    if not exe:
        raise SystemExit("tesseract is needed to read the blocks (pacman -S tesseract tesseract-data-eng)")
    h = dict(key="tesseract-eng", exe=exe, version="tesseract", lang="eng", psm=6, oem=1, tessdata=None)
    r = T.run(h, {"image": png, "dpi": dpi})
    path = png[:-4] + ".ocr.json"
    with open(path, "w", encoding="utf8") as f:
        json.dump(dict(lines=r["lines"], text=r["text"], engine=r["engine"], psm=6), f)
    return path, r


def real_items(out, cache, get=get):
    with open(os.path.join(HERE, "regions.toml"), "rb") as f:
        regions = tomllib.load(f)
    items = []
    for rid, r in regions.items():
        page, dpi = leaf(r["book"], int(r["leaf"]), cache, get)
        x, y, w, h = r["box"]
        png = os.path.join(out, f"{rid}.png")
        page.crop((x, y, x + w, y + h)).save(png, dpi=(dpi, dpi))
        path, got = ocr(png, dpi)
        if not got["lines"]:
            print(f"{rid}: Tesseract found no words; left out", file=sys.stderr)
            continue
        items.append(
            dict(
                id=rid,
                input=dict(image=png, ocr=path),
                reference=dict(classes=r["classes"], style=r["style"], text=got["text"]),
                tags=dict(book=r["book"], kind=r["kind"], cls=r["classes"][0], synthetic="no"),
            )
        )
    return items


def _lines_of_text():
    out = []
    for p in sorted(glob.glob(os.path.join(TEXTS, "*.txt"))):
        with open(p, encoding="utf8") as f:
            for ln in f.read().splitlines():
                ln = ln.strip()
                if len(ln) > 25 and not ln[:3].strip().isdigit():
                    out.append(ln)
    return out


def set_text(face, style, lines, size, scale=2):
    """Black type on white at `scale` times the size asked (rendered anti-aliased, for the degradation to work on)."""
    path = face.file(style)
    sz = size * scale
    f = F._font(path, sz)
    space = f.getlength(" ")
    rendered = []
    for ln in lines:
        x, words = 0.0, []
        for w in ln.split():
            m, left, base = face.word(style, w, sz, 1.0, 0.0, 0.5)
            words.append((m, left, base, x))
            cols = np.flatnonzero(m.any(0))
            x += (cols[-1] - cols[0] + 1 if len(cols) else 0) + space
        rendered.append((words, x))
    W = int(max(x for _, x in rendered) + sz)
    lead = sz * 1.3
    H = int(lead * len(lines) + sz)
    canvas = np.zeros((H, W), dtype=bool)
    for i, (words, _) in enumerate(rendered):
        for m, left, base, x in words:
            F._paste(canvas, m, left, base, sz * 0.5 + x, sz * 1.0 + i * lead)
    return canvas


def degrade(canvas, rng, scale=2):
    """A print and a scan of it: ink spread or starved, uneven paper, noise, blur, a slight tilt; back to the asked size."""
    im = Image.fromarray(np.where(canvas, 255, 0).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.8)))
    a = np.asarray(im, dtype=np.float32) / 255
    a = (a > rng.uniform(0.3, 0.65)).astype(np.float32)  # ink spread (low threshold) or starved (high)
    im = Image.fromarray((a * 255).astype(np.uint8)).rotate(
        rng.uniform(-0.4, 0.4), resample=Image.Resampling.BILINEAR, fillcolor=0
    )
    im = im.resize((im.width // scale, im.height // scale), Image.Resampling.LANCZOS).filter(
        ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9))
    )
    cov = np.asarray(im, dtype=np.float32) / 255
    h, w = cov.shape
    paper = (
        rng.uniform(195, 235)
        + rng.uniform(-12, 12) * np.linspace(0, 1, w)[None, :]
        + rng.uniform(-8, 8) * np.linspace(0, 1, h)[:, None]
    )
    ink = rng.uniform(25, 70)
    g = (
        paper * (1 - cov)
        + ink * cov
        + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(4, 11), cov.shape)
    )
    return Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))


def synthetic_items(n, out, models_dir, seed=7, dpi=400):
    rng = random.Random(seed)
    cat = F.catalogue()
    by_class = {c: [k for k, v in cat.items() if v["class"] == c] for c in F.CLASSES}
    texts = _lines_of_text()
    items = []
    for i in range(n):
        cls = F.CLASSES[i % len(F.CLASSES)]
        fid = rng.choice(by_class[cls])
        face = F.Face(fid, F.fetch(fid, models_dir))
        style = "italic" if "italic" in face.files and rng.random() < 0.3 else "roman"
        display = cls in DISPLAY or rng.random() < 0.2
        size_pt = rng.uniform(16, 36) if display else rng.uniform(8.5, 12.5)
        nlines = rng.randint(1, 2) if display else rng.randint(2, 4)
        start = rng.randrange(len(texts) - nlines)
        lines = texts[start : start + nlines]
        if display:
            lines = [" ".join(ln.split()[: rng.randint(2, 4)]) for ln in lines]
            if rng.random() < 0.4:
                lines = [ln.upper() for ln in lines]
        canvas = set_text(face, style, lines, size_pt * dpi / 72)
        im = degrade(canvas, rng)
        sid = f"synthetic-{i:02d}-{fid}"
        png = os.path.join(out, f"{sid}.png")
        im.save(png, dpi=(dpi, dpi))
        path, got = ocr(png, dpi)
        if not got["lines"]:
            print(f"{sid}: Tesseract found no words; left out", file=sys.stderr)
            continue
        items.append(
            dict(
                id=sid,
                input=dict(image=png, ocr=path),
                reference=dict(face=fid, classes=[cls], style=style, size_pt=round(size_pt, 2), text="\n".join(lines)),
                tags=dict(book="synthetic", kind="display" if display else "body", cls=cls, synthetic="yes"),
            )
        )
    return items


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="font-match")
    ap.add_argument("--synthetic", type=int, default=30, help="synthetic items (a known face each)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-real", action="store_true", help="synthetic items only (no download)")
    a = ap.parse_args(argv)
    out = os.path.join(settings.data_dir(), "evals", a.name)
    cache = os.path.join(out, "scans")
    os.makedirs(cache, exist_ok=True)
    items = [] if a.no_real else real_items(out, cache)
    items += synthetic_items(a.synthetic, out, settings.models_dir(), a.seed)
    rng = random.Random(a.seed)
    rng.shuffle(items)  # samples (`items = N`) take the first N: mixed real and synthetic
    ds = C.write_dataset(
        a.name,
        items,
        description="Blocks of type to identify: real blocks of Ward 1874 and Collier 1828 (class and style labelled) and synthetic blocks in a known face",
    )
    print(f"{a.name}: {len(items)} items, version {ds.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
