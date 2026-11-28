"""Build the eval datasets `image-kind` and `vectorise-plates` from the pictures strata-scan cut from three scanned books (Ward 1874, Collier
and Cruikshank 1828, Speaight's history of 1970), labelled by eye in `image_kind/labels.toml`.

The pictures come from a folder of files named `<book>-p<page>-f<figure id or "page">.<ext>` (book w1874, c1828 or hist; page its position
in strata-scan, which is its page in the finished PDF): strata-scan's figures and its picture and cover pages, exported from its data
folder (books/<id>/figures/<figure>.png and pages/<page>.png). Each file takes the label of its page whose shape (width / height) is
nearest; a page with one label gives it to every file (a picture page and the figure cut from it are the same picture). Files on pages with
no label are listed, so they can be labelled (`--extra`, the same format) or left out. They are copied to <data>/evals/book-pictures/.

  image-kind        every labelled file: input {image}, reference {kind}, tags {book, kind, group, line_art, hard (a note says why)}
  vectorise-plates  the labels marked `vectorise = true`, one file each (the figure when there is one): input {image}, tags {book, kind}

    uv run python scripts/evals/book_pictures.py --pictures ~/code/_plates-export [--extra more-labels.toml]
"""

import argparse
import math
import os
import random
import re
import shutil
import sys
import tomllib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from PIL import Image

from stratamodels import settings
from stratamodels.evals import core as C
from stratamodels.runtimes.imagekind import KINDS, kind_of

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_kind")
NAME = re.compile(r"^(w1874|c1828|hist)-p(\d+)-f(\w+)\.(jpe?g|png)$", re.I)
MIN_SIDE = 64
SHAPE_TOL = (
    0.25  # |log(shape ratio)| within which a file on a page of several labels is taken to be that label's picture
)


def labels(paths):
    out = []
    for p in paths:
        with open(p, "rb") as f:
            for lab in tomllib.load(f)["pictures"]:
                if lab["kind"] not in KINDS:
                    raise SystemExit(f"{p}: unknown kind {lab['kind']!r}")
                out.append(lab)
    return out


def match(files, labs):
    """[(file, label)] and the files no label fits."""
    by_page = {}
    for lab in labs:
        by_page.setdefault((lab["book"], int(lab["page"])), []).append(lab)
    got, unmatched = [], []
    for f in files:
        m = NAME.match(os.path.basename(f))
        if not m:
            continue
        cands = by_page.get((m.group(1).lower(), int(m.group(2))), [])
        with Image.open(f) as im:
            shape = im.size[0] / im.size[1]
            if min(im.size) < MIN_SIDE:  # a sliver cut along a rule or the gutter: not a picture of anything
                continue
        best = min(cands, key=lambda lab: abs(math.log(shape / lab["shape"])), default=None)
        if best is not None and (len(cands) == 1 or abs(math.log(shape / best["shape"])) < SHAPE_TOL):
            got.append((f, best))
        else:
            unmatched.append(f)
    return got, unmatched


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pictures", required=True, help="folder of <book>-p<page>-f<id>.jpg files")
    ap.add_argument(
        "--extra", nargs="*", default=[], help="more label files, the same format as image_kind/labels.toml"
    )
    a = ap.parse_args()
    files = sorted(os.path.join(a.pictures, f) for f in os.listdir(a.pictures))
    got, unmatched = match(files, labels([os.path.join(HERE, "labels.toml"), *a.extra]))
    dest = os.path.join(settings.data_dir(), "evals", "book-pictures")
    os.makedirs(dest, exist_ok=True)
    kinds, plates, chosen = [], [], {}
    for f, lab in got:
        to = os.path.join(dest, os.path.basename(f))
        if not os.path.isfile(to) or os.path.getsize(to) != os.path.getsize(f):
            shutil.copyfile(f, to)
        stem = os.path.splitext(os.path.basename(f))[0]
        k = kind_of(lab["kind"])
        kinds.append(
            dict(
                id=stem,
                input=dict(image=to),
                reference=dict(kind=lab["kind"], **({"also": lab["also"]} if lab.get("also") else {})),
                tags=dict(
                    book=lab["book"],
                    kind=lab["kind"],
                    group=k["group"],
                    line_art=k["line_art"],
                    hard=bool(lab.get("note")),
                ),
            )
        )
        if lab.get("vectorise"):
            key = (lab["book"], lab["page"], lab["shape"])
            if key not in chosen or chosen[key][0].endswith(
                "-fpage"
            ):  # the figure, not the whole page, when strata-scan cut one
                chosen[key] = (stem, to, lab)
    for stem, to, lab in chosen.values():
        plates.append(dict(id=stem, input=dict(image=to), reference={}, tags=dict(book=lab["book"], kind=lab["kind"])))
    rnd = random.Random(20261128)  # a fixed shuffle, so a candidate on the first N items has a fair sample
    rnd.shuffle(kinds)
    plates.sort(key=lambda it: it["id"])
    rnd.shuffle(plates)
    C.write_dataset(
        "image-kind",
        kinds,
        description=f"{len(kinds)} pictures of three scanned books, kinds labelled by eye (scripts/evals/image_kind/labels.toml)",
    )
    C.write_dataset(
        "vectorise-plates",
        plates,
        description=f"{len(plates)} pictures of three scanned books to redraw as SVG: a spread of kinds (labels marked vectorise)",
    )
    print(
        f"image-kind: {len(kinds)} items; vectorise-plates: {len(plates)} items; {len(unmatched)} files with no label:"
    )
    for f in unmatched:
        print("  ", os.path.basename(f))


if __name__ == "__main__":
    main()
