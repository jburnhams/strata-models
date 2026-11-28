"""Build the dataset `image-kind-hard`: the pictures of `image-kind` that one candidate (Haiku) got wrong, plus a few it got right, so a dearer
model (Sonnet, Opus: `items = "image-kind-hard"` in the spec) runs only where it could change the answer (Jonathan, 2026-11-28: "use haiku for
all and only doing a few in sonnet and opus to get a sense of if any dramatic improvement, ie on the ones that haiku struggles with"). The
few it got right show whether the dearer model breaks what the cheap one had.

    uv run python scripts/evals/image_kind_hard.py [--from haiku] [--right 10]

Run it after the candidate has run (`eval run image-kind --candidate haiku`); it reads the cached outputs and writes nothing else."""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from stratamodels import settings
from stratamodels.evals import core as C
from stratamodels.evals import metrics as M


def split(spec, engine, name):
    """(wrong, right) items of the dataset for candidate `name`; an item it has no output for is in neither."""
    ds = C.load_dataset(spec.dataset, engine=engine)
    c = spec.candidate(name)
    keys = {C.result_key(spec, c, it): it for it in ds.items}
    got = C.cached(engine, keys)
    wrong, right = [], []
    for k, it in keys.items():
        r = got.get(k)
        if r is None:
            continue
        ok = M._kind_right(it, r.output, r.error) and M._group_right(it, r.output, r.error)  # pyright: ignore[reportPrivateUsage]
        (right if ok else wrong).append(it)
    return wrong, right


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="frm", default="haiku", help="the candidate whose misses are kept")
    ap.add_argument("--right", type=int, default=10, help="how many of the pictures it got right to keep too")
    a = ap.parse_args()
    engine = settings.engine()
    spec = C.load_spec("image-kind", engine)
    wrong, right = split(spec, engine, a.frm)
    keep = wrong + random.Random(20261128).sample(right, min(a.right, len(right)))
    C.write_dataset(
        "image-kind-hard",
        keep,
        description=f"{len(wrong)} pictures of image-kind that {a.frm} got wrong and {len(keep) - len(wrong)} it got right (scripts/evals/image_kind_hard.py)",
        engine=engine,
    )
    print(f"image-kind-hard: {len(keep)} items ({len(wrong)} {a.frm} got wrong)")


if __name__ == "__main__":
    main()
