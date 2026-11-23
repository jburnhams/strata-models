"""Build the eval dataset `page-layout`: ten whole pages of the two public-domain books (Ward 1874's title page, dialogue and two pages of
advertisements in a dozen display faces; Collier 1828's chapter openings, prose, footnotes and a sonnet) with their layout labelled by eye
(`page_layout/pages.toml`: text blocks in reading order with their roles, ornaments and rules as figures).

Each item's input is `image` (the leaf as scanned, a PNG with its dpi) and `ocr` (Tesseract's reading of it, JSON, psm 3 as strata-scan
reads pages: what the baseline draws and what a correcting candidate is given), and its reference is `page`, the labelled layout as a
strata_python.pages description, which the layout metrics (block_f1, figure_f1, role_acc, order_tau) and page_ink_f1 read, and `text` when
`<data>/evals/page-layout/text/<id>.txt` holds a reading of the page (scripts/evals/ocr_reference.py --pages: the page-rebuild eval's cer). The scans come
from archive.org (the same leaves and cache as scripts/evals/font_match.py); pictures go to `<data>/evals/page-layout/`.

    .venv/bin/python scripts/evals/page_layout.py
"""

import argparse
import json
import os
import random
import sys
import tomllib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import font_match as FM  # the archive.org leaf fetcher and its cache

from stratamodels import settings
from stratamodels.evals import core as C
from stratamodels.runtimes import tesseract as T

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "page_layout")


def ocr(png, dpi):
    exe = T.binary()
    if not exe:
        raise SystemExit("tesseract is needed to read the pages (pacman -S tesseract tesseract-data-eng)")
    h = dict(key="tesseract-eng", exe=exe, version="tesseract", lang="eng", psm=3, oem=1, tessdata=None)
    r = T.run(h, {"image": png, "dpi": dpi})
    path = png[:-4] + ".ocr.json"
    with open(path, "w", encoding="utf8") as f:
        json.dump(
            dict(width=r["width"], height=r["height"], lines=r["lines"], text=r["text"], engine=r["engine"], psm=3), f
        )
    return path


def reference(p, width, height, dpi):
    """The labelled page as a strata_python.pages description (no lines: the layout metrics read blocks and figures only)."""
    return dict(
        width=width,
        height=height,
        dpi=dpi,
        blocks=[dict(box=b["box"], role=b["role"], order=i) for i, b in enumerate(p["blocks"])],
        figures=[dict(box=f["box"], kind=f["kind"]) for f in p["figures"]],
    )


def items(out, cache, get=FM.get):
    with open(os.path.join(HERE, "pages.toml"), "rb") as f:
        pages = tomllib.load(f)
    got = []
    for pid, p in pages.items():
        page, dpi = FM.leaf(p["book"], int(p["leaf"]), cache, get)
        png = os.path.join(out, f"{pid}.png")
        page.save(png, dpi=(dpi, dpi))
        ref = dict(page=reference(p, page.width, page.height, dpi))
        txt = os.path.join(out, "text", f"{pid}.txt")
        if os.path.isfile(
            txt
        ):  # a strong model's reading of the page (scripts/evals/ocr_reference.py --pages), for the rebuild's text metrics
            with open(txt, encoding="utf8") as f:
                ref["text"] = f.read().strip()
        got.append(
            dict(
                id=pid,
                input=dict(image=png, ocr=ocr(png, dpi)),
                reference=ref,
                tags=dict(book=p["book"], kind=p["kind"], set=p.get("set", "tune")),
            )
        )
    return got


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="page-layout")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    out = os.path.join(settings.data_dir(), "evals", a.name)
    cache = os.path.join(out, "scans")
    os.makedirs(cache, exist_ok=True)
    got = items(out, cache)
    random.Random(a.seed).shuffle(got)  # samples (`items = N`) take the first N: both books, every kind of page
    ds = C.write_dataset(
        a.name,
        got,
        description="Whole pages of Ward 1874 and Collier 1828 with their layout labelled: blocks in reading order with roles, ornaments and rules",
    )
    print(f"{a.name}: {len(got)} items, version {ds.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
