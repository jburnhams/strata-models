"""Build the eval dataset `ocr-pages`: hard pages of two public-domain books (title pages, advertisements in display type, dialogue set in
small capitals and italics, footnotes, a grey 1828 scan), each with its text typed out by hand from the page (`ocr_pages/<item>.txt`, kept
as printed: old spellings, line-end hyphens). The pages themselves come from strata-scan, which cut them out of the Internet Archive scans:

- ward1874: "Punch and Judy, with instructions how to manage the little wooden actors" (T. A. M. Ward, 1874), archive.org punchjudywithins00war
- collier1828: "Punch and Judy, with illustrations drawn and engraved by George Cruikshank" (J. P. Collier, second edition, 1828)

Either from a running strata-scan (each book named by its id there; the page at that position is fetched at full size with its scan's dpi):

    .venv/bin/python scripts/evals/ocr_pages.py --strata-scan http://127.0.0.1:8368 --book ward1874=1 --book collier1828=2

or from a folder of `<item>.png` files (the dpi from each file, else --dpi). Items without a page are left out and named. The pictures are copied
to `<data>/evals/ocr-pages/` with their dpi written in, so the item's only input is `image`, which both Tesseract and a vision-language model
read (`evals/specs/ocr-pages.toml`)."""

import argparse
import glob
import json
import os
import random
import re
import sys
import urllib.request

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from stratamodels.evals import core as C

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_pages")
# what makes each page hard, for the report's breakdown by tag
HARD = {
    "ward1874-p011": "title",
    "ward1874-p014": "small-caps",
    "ward1874-p019": "small-caps",
    "ward1874-p022": "italic",
    "ward1874-p031": "display",
    "ward1874-p036": "display",
    "collier1828-p013": "title",
    "collier1828-p047": "prose",
    "collier1828-p053": "footnotes",
    "collier1828-p112": "footnotes",
}


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def from_strata_scan(base, books, out):
    """{item: (png path, dpi)} for the items of the named books, fetched from strata-scan."""
    got = {}
    for name, book_id in books.items():
        pages = json.loads(get(f"{base}/api/books/{book_id}/pages"))
        by_pos = {p["position"]: p["id"] for p in pages}
        for ref in sorted(glob.glob(os.path.join(HERE, f"{name}-p*.txt"))):
            item = os.path.basename(ref)[:-4]
            pos = int(re.sub(r"\D", "", item.rsplit("-p", 1)[1]))
            if pos not in by_pos:
                continue
            detail = json.loads(get(f"{base}/api/pages/{by_pos[pos]}"))
            path = os.path.join(out, f"{item}.fetched.png")
            with open(path, "wb") as f:
                f.write(get(f"{base}/api/pages/{by_pos[pos]}/image?size=full"))
            got[item] = (path, (detail.get("scan") or {}).get("dpi"))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata-scan", help="a strata-scan server's address, to fetch the pages from")
    ap.add_argument(
        "--book", action="append", default=[], help="NAME=ID: which strata-scan book is ward1874 / collier1828"
    )
    ap.add_argument("--pages", help="or a folder of <item>.png")
    ap.add_argument("--dpi", type=float, help="the dpi of pictures in --pages that do not say")
    ap.add_argument("--name", default="ocr-pages")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = os.path.join(os.path.dirname(C.datasets_dir()), a.name)
    os.makedirs(out, exist_ok=True)
    if a.strata_scan:
        found = from_strata_scan(a.strata_scan.rstrip("/"), dict(b.split("=", 1) for b in a.book), out)
    elif a.pages:
        found = {os.path.basename(p)[:-4]: (p, None) for p in glob.glob(os.path.join(a.pages, "*.png"))}
    else:
        sys.exit("give --strata-scan URL --book NAME=ID ... or --pages DIR")
    items, missing = [], []
    for ref in sorted(glob.glob(os.path.join(HERE, "*.txt"))):
        item = os.path.basename(ref)[:-4]
        if item not in found:
            missing.append(item)
            continue
        src, dpi = found[item]
        with Image.open(src) as im:
            dpi = dpi or (im.info.get("dpi") or (None,))[0] or a.dpi
            if not dpi:
                sys.exit(f"{src}: no dpi in the file; give --dpi")
            dest = os.path.join(out, f"{item}.png")
            im.save(dest, dpi=(dpi, dpi))
        if src.endswith(".fetched.png"):
            os.remove(src)
        with open(ref, encoding="utf8") as f:
            text = f.read().strip()
        book = item.split("-")[0]
        items.append(
            dict(
                id=item,
                input=dict(image=dest),
                reference=dict(text=text),
                tags=dict(book=book, hard=HARD.get(item, ""), dpi=round(dpi)),
            )
        )
    if not items:
        sys.exit("no pages found for any reference")
    random.Random(a.seed).shuffle(items)
    ds = C.write_dataset(
        a.name,
        items,
        description=f"Hard pages of two public-domain books (Ward 1874, Collier 1828) with hand-typed text; seed {a.seed}. Missing: {', '.join(missing) or 'none'}",
    )
    print(f"{len(ds.items)} items, version {ds.version}" + (f"; no page for {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
