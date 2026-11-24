"""Build a shot-boundaries dataset (eval `shots-vhs`) from one real video whose cuts a person has marked by looking at it.

Spliced video (spliced_shots.py) has exact cuts but none of a tape's faults: the torn first frames of a camcorder restart, the roll before the
signal locks, the lost signal after the recording ends. A real tape has them, and its cuts have to be marked. The marks are a text file: one
frame number a line (the first frame of the new shot, counted from 0), `#` starts a comment. Ends of a stretch with no picture are cuts too.

The video is not cut up: each item is a window of it (`clip`), so the eval has more than one item to bootstrap over. Windows are `--window`
seconds long, each edge moved away from any marked cut by at least two seconds. The video and the marks are private and stay in the data
directory.

    .venv/bin/python scripts/evals/marked_shots.py /path/to/tape.mp4 marks.txt --frames 31822 [--name shots-vhs] [--window 80] [--fps 25]

Items: input {path, clip: [start, end]}; reference {cuts (seconds in the file), tolerance_s}; tags {kind: "signal" for a window that holds a
mark commented `signal` (the picture comes or goes there), else "shots"}."""

import argparse
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from stratamodels.evals import core as C  # noqa: E402

CLEAR_S = Fraction(2)  # an edge this close to a cut would score the cut on a clip's first frames


def read_marks(text):
    """(frame, comment) for every marked cut, in order."""
    out = {}
    for n, line in enumerate(text.splitlines(), 1):
        frame, _, note = line.partition("#")
        if not frame.strip():
            continue
        if not frame.strip().isdigit():
            raise SystemExit(f"line {n}: not a frame number: {frame.strip()!r}")
        out[int(frame)] = note.strip()
    return sorted(out.items())


def windows(cuts, end, length):
    """Half-open windows over 0 to `end` (seconds), about `length` long, no inner edge within CLEAR_S of a cut."""
    edges = [Fraction(0)]
    at = Fraction(length)
    while at < end - length / 2:
        edge = at
        while any(abs(edge - c) < CLEAR_S for c in cuts):
            edge += Fraction(1, 2)
        if edge >= end - CLEAR_S:
            break
        edges.append(edge)
        at = edge + length
    return list(zip(edges, [*edges[1:], end], strict=True))


def build(path, marks, name, frames, fps=25, window=80, tolerance_s=0.5):
    rate = Fraction(fps)
    cuts = [(Fraction(f) / rate, note) for f, note in marks]
    if not cuts:
        raise SystemExit("no marked cuts")
    end = Fraction(frames) / rate
    if cuts[-1][0] >= end:
        raise SystemExit(f"a cut is marked at {float(cuts[-1][0]):g} s, past the video's end ({float(end):g} s)")
    items = []
    for k, (a, b) in enumerate(windows([t for t, _ in cuts], end, Fraction(window))):
        inside = [(t, note) for t, note in cuts if a < t < b]
        items.append(
            dict(
                id=f"window-{k + 1:02d}",
                input=dict(path=os.path.abspath(path), clip=[float(a), float(b)]),
                reference=dict(cuts=[round(float(t), 6) for t, _ in inside], tolerance_s=tolerance_s),
                tags=dict(kind="signal" if any("signal" in note for _, note in inside) else "shots"),
            )
        )
    return C.write_dataset(
        name,
        items,
        description=f"{len(items)} windows of one real tape ({float(end):.0f} s), {len(cuts)} cuts marked by eye",
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("marks")
    ap.add_argument("--name", default="shots-vhs")
    ap.add_argument("--frames", type=int, required=True, help="how many frames the video has (ffprobe -count_frames)")
    ap.add_argument("--fps", type=Fraction, default=Fraction(25))
    ap.add_argument("--window", type=float, default=80)
    a = ap.parse_args(argv)
    if not os.path.isfile(a.video):
        raise SystemExit(f"no video {a.video}")
    with open(a.marks, encoding="utf8") as f:
        marks = read_marks(f.read())
    print(build(a.video, marks, a.name, a.frames, a.fps, a.window))


if __name__ == "__main__":
    main()
