"""Estimate a VHS line's horizontal impulse response (for the video worker's cleanup.vs.desmear) from a high-contrast region of a clip:
dark marks on a light ground (writing on a card, text). Each row of the region is turned into its ideal two-level version (ink or ground)
and the kernel that best turns the ideal into what the tape shows is fitted by least squares, the tail smoothed. Prints TOML for a spec.

    .venv/bin/python scripts/evals/smear_kernel.py CLIP --box X Y W H [--times 0.5 1.5 ...] [--crop X Y W H] [--reach 12]

Needs ffmpeg (bwdif deinterlace, as the worker does) and numpy."""

import argparse
import json
import subprocess

import numpy as np


def frame(path, t, crop):
    """A grey frame at `t` seconds, deinterlaced (bwdif, top field first) and cropped like the worker's preparation."""
    x, y, w, h = crop
    r = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            path,
            "-frames:v",
            "1",
            "-vf",
            f"bwdif=mode=send_field:parity=tff,crop={w}:{h}:{x}:{y},format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(r.stdout, np.uint8).reshape(h, w).astype(float)


def fit(rows, left=4, right=60, smooth=1e4):
    """The kernel (taps -left..right) that maps each row's two-level ideal to the row, least squares with its tail (taps > 3) kept smooth."""
    a_rows, b = [], []
    for r in rows:
        paper, ink = np.percentile(r, 70), np.percentile(r, 3)
        if paper - ink < 80:
            continue
        ideal = np.where(r < (paper + ink) / 2, ink, paper) - paper
        for n in range(right, len(r) - left):
            a_rows.append(ideal[n - right : n + left + 1][::-1])
            b.append(r[n] - paper)
    if not b:
        raise SystemExit("no row in the box has enough contrast (dark marks on a light ground)")
    a, b = np.array(a_rows), np.array(b)
    taps = np.arange(-left, right + 1)
    d2 = np.diff(np.eye(len(taps)), 2, axis=0)
    w = np.where(taps > 3, 1.0, 0.0)[1:-1]
    h = np.linalg.solve(a.T @ a + smooth * (d2.T * w) @ d2, a.T @ b)
    return taps, h, float(np.sqrt(np.mean((a @ h - b) ** 2)))


def kernel(taps, h, reach=12):
    """The kernel cut to +-reach pixels (beyond that this fit is noise) and normalised to sum 1."""
    keep = np.abs(taps) <= reach
    hk = h[keep] / h[keep].sum()
    return [int(t) for t in taps[keep]], [round(float(v), 4) for v in hk]


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("clip")
    a.add_argument(
        "--box",
        nargs=4,
        type=int,
        required=True,
        metavar=("X", "Y", "W", "H"),
        help="the high-contrast region, in the cropped frame",
    )
    a.add_argument("--crop", nargs=4, type=int, default=[14, 0, 692, 568], metavar=("X", "Y", "W", "H"))
    a.add_argument("--times", nargs="+", type=float, default=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    a.add_argument("--reach", type=int, default=12)
    o = a.parse_args(argv)
    x, y, w, h = o.box
    rows = [r for t in o.times for r in frame(o.clip, t, o.crop)[y : y + h, x : x + w]]
    taps, hh, err = fit(rows)
    kt, kh = kernel(taps, hh, o.reach)
    print(f"# fitted on {len(rows)} rows, error {err:.1f} levels")
    print(f"desmear = {{ taps = {json.dumps(kt)}, h = {json.dumps(kh)} }}")


if __name__ == "__main__":
    main()
