"""Build the shot-boundaries dataset (eval `shots-vhs`) from stretches of digitised tapes with their cuts marked by hand, listed in
`shots.toml` beside the tapes (private: it describes private footage):

    tolerance_s = 0.5                  # how far an answered cut may lie from the marked one (seconds); a stretch may set its own
    [[stretches]]
    id = "tape03-birthday"
    file = "tape03.mkv"
    clip = [600.0, 900.0]              # seconds of the tape to look at: a few minutes with several kinds of cut
    cuts = [612.36, 640.0, 702.52]     # every cut inside the clip, in the tape's seconds: the first frame of the new shot
    note = "party indoors, camcorder stopped and started"
    tags = { kind = "indoor", light = "low" }

Each item's input is the tape and the clip (what runtimes/scenes.py takes); its reference holds the cuts and the tolerance.

    .venv/bin/python scripts/evals/vhs_shots.py /path/to/tapes --name vhs-shots"""

import argparse
import os
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from stratamodels.evals import core as C  # noqa: E402


def item(stretch, folder, defaults):
    """One dataset item from a manifest entry; refuses a missing tape, a bad clip or a cut outside its clip."""
    path = os.path.join(folder, stretch["file"])
    if not os.path.isfile(path):
        raise SystemExit(f"no tape {path}")
    start, end = (float(t) for t in stretch["clip"])
    if not 0 <= start < end:
        raise SystemExit(f"{stretch['id']}: clip must be [start, end] with 0 <= start < end")
    cuts = sorted(float(t) for t in stretch.get("cuts") or [])
    outside = [t for t in cuts if not start < t < end]
    if outside:
        raise SystemExit(f"{stretch['id']}: cuts outside the clip {start}-{end}: {outside}")
    tol = float(stretch.get("tolerance_s", defaults.get("tolerance_s", 0.5)))
    return dict(
        id=str(stretch["id"]),
        input=dict(path=os.path.abspath(path), clip=[start, end]),
        reference=dict(cuts=cuts, tolerance_s=tol, note=stretch.get("note", "")),
        tags=dict(stretch.get("tags") or {}),
    )


def build(folder, name):
    with open(os.path.join(folder, "shots.toml"), "rb") as f:
        m = tomllib.load(f)
    items = [item(s, folder, m) for s in m.get("stretches") or []]
    if not items:
        raise SystemExit(f"{folder}/shots.toml lists no stretches")
    n = sum(len(i["reference"]["cuts"]) for i in items)
    return C.write_dataset(
        name, items, description=m.get("description") or f"{len(items)} stretches of tape, {n} cuts marked by hand"
    )


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("folder", help="the tapes and their shots.toml")
    a.add_argument("--name", required=True, help="the dataset's name, e.g. vhs-shots")
    o = a.parse_args(argv)
    ds = build(o.folder, o.name)
    print(f"{len(ds.items)} stretches, version {ds.version}")


if __name__ == "__main__":
    main()
