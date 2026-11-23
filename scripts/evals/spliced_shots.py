"""Build the shot-boundaries dataset (eval `shots-spliced`) by splicing: stretches of uncut camera files are joined end to end, so every cut
is where the builder put it and the reference is exact, with nobody marking cuts by hand.

The sources must each be one continuous shot (a camera's own recording: a race's raw 360 or action-camera files, not an edited video or a
tape with camcorder stops in it), because a cut already inside a source would be scored as a false cut. Each item is a spliced video of
`--pieces` stretches of `--min-s` to `--max-s` seconds, and every item uses one kind of cut, so the report breaks down by `tags.kind`:

  files   each stretch from another file: the easy cut, two different scenes
  far     from the same file, at least `--far-s` apart: same camera, same light, a different place on the course
  near    from the same file, `--near-s` apart, forward or back: the hard cut, where the two sides show almost the same picture and only
          a jump in what moves gives it away (Jonathan's idea, 2026-11-23: cuts taken from inside existing clips are harder to detect)

Each stretch is decoded, deinterlaced where the frames say interlaced (yadif deint=interlaced), fitted inside `--size` (padded, not
stretched, so files of any shape splice) and resampled to `--fps`, with an exact count of frames; the stretches are then joined without
re-encoding. So cut k is at (frames before it) / fps seconds, exactly, and the reference keeps every stretch's source and times
(`pieces`) for looking at a miss. The spliced video is progressive: the deinterlacing candidates of the eval do not apply to it.

    .venv/bin/python scripts/evals/spliced_shots.py "<project folder>/clips" /path/to/more.mp4 --name spliced-shots [--items 4] [--seed 1]

Items: input {path: <data>/evals/media/<name>/<id>.mp4}; reference {cuts (seconds), tolerance_s, pieces: [{file, start, end, at}]};
tags {kind}. The same sources, seed and options give the same items; a spliced video already there is kept."""

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from stratamodels import settings  # noqa: E402
from stratamodels.evals import core as C  # noqa: E402

VIDEO = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".webm", ".insv", ".lrv")
KINDS = ("files", "far", "near")


@dataclass(frozen=True)
class Piece:
    file: str
    start: Fraction  # seconds into the file
    frames: int  # how many output frames it lasts


def sources(paths, ext=VIDEO):
    """The video files named, and those under the folders named (sorted, so the same folder always gives the same list)."""
    found = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs.sort()
                found += [os.path.join(root, n) for n in sorted(names) if n.lower().endswith(ext)]
        elif os.path.isfile(p):
            found.append(p)
        else:
            raise SystemExit(f"no file or folder {p}")
    return [os.path.abspath(f) for f in found]


def duration(path, run=subprocess.run):
    """The file's length in seconds by ffprobe, or None when it has no video ffprobe can read."""
    r = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        d = json.loads(r.stdout or "{}")
        return float(d["format"]["duration"]) if d.get("streams") else None
    except (KeyError, TypeError, ValueError):
        return None


def plan(kind, lengths, rng, pieces, fps, min_s, max_s, far_s, near_s):
    """The stretches of one item: `pieces` of them, every cut of `kind`. `lengths` maps each usable file to its seconds. Returns None when the
    sources cannot give this kind (one file for `files`, none long enough for `far`)."""
    files = sorted(lengths)
    lo, hi = round(min_s * fps), round(max_s * fps)
    if kind == "files" and len(files) < 2:
        return None
    if kind == "far":
        files = [f for f in files if lengths[f] >= far_s + 2 * max_s + 1]
    if kind == "near":
        files = [f for f in files if lengths[f] >= near_s[1] + 2 * max_s + 1]
    if not files:
        return None
    out = []
    home = rng.choice(files)
    for i in range(pieces):
        n = rng.randint(lo, hi)
        d = Fraction(n) / fps
        if kind == "files":
            f = rng.choice([x for x in files if not out or x != out[-1].file])
            room = Fraction(lengths[f]) - d - 1  # a second short of the end: a file's last frames are often cut short
            start = Fraction(rng.randrange(0, max(1, int(room * 10)))) / 10
        else:
            f = home
            room = Fraction(lengths[f]) - d - 1
            if i == 0:
                start = Fraction(rng.randrange(0, max(1, int(room * 10)))) / 10
            else:
                prev = out[-1]
                prev_end = prev.start + Fraction(prev.frames) / fps
                gap_lo, gap_hi = (far_s, float(room)) if kind == "far" else near_s
                choices = [
                    s
                    for s in (
                        prev_end + Fraction(rng.uniform(gap_lo, gap_hi)).limit_denominator(10),
                        prev_end - Fraction(rng.uniform(gap_lo, gap_hi)).limit_denominator(10),
                    )
                    if 0 <= s <= room
                ]
                if (
                    not choices
                ):  # the far side of the file is too close on both sides: start again near the beginning or end
                    choices = [s for s in (Fraction(0), room) if abs(s - prev_end) >= gap_lo]
                if not choices:
                    return None
                start = rng.choice(choices)
        out.append(Piece(f, start, n))
    return out


def cuts(pieces, fps):
    """Seconds into the spliced video where each stretch after the first begins."""
    at, out = 0, []
    for p in pieces[:-1]:
        at += p.frames
        out.append(Fraction(at) / fps)
    return out


def splice(pieces, dest, fps, size, run=subprocess.run):
    """Write the spliced video: each stretch encoded alike with an exact frame count, then joined without re-encoding."""
    w, h = size
    vf = (
        f"yadif=mode=0:deint=interlaced,scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}"
    )
    tmp = dest + ".parts"
    os.makedirs(tmp, exist_ok=True)
    parts = []
    for i, p in enumerate(pieces):
        part = os.path.join(tmp, f"{i:03d}.mp4")
        _ff(
            run,
            [
                "-ss",
                f"{float(p.start):.3f}",
                "-i",
                p.file,
                "-map",
                "0:v:0",
                "-vf",
                vf,
                "-frames:v",
                str(p.frames),
                "-an",
            ]
            + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(fps), part],
        )
        parts.append(part)
    listing = os.path.join(tmp, "parts.txt")
    with open(listing, "w", encoding="utf8") as f:
        f.writelines(f"file '{os.path.basename(x)}'\n" for x in parts)
    _ff(run, ["-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", "-movflags", "+faststart", dest + ".tmp.mp4"])
    os.replace(dest + ".tmp.mp4", dest)
    for x in [*parts, listing]:
        os.remove(x)
    os.rmdir(tmp)


def _ff(run, args):
    r = run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], capture_output=True, text=True, check=False)
    if r.returncode:
        raise SystemExit(f"ffmpeg failed: {(r.stderr or '').strip()[-400:]}")


def build(
    paths,
    name,
    items=4,
    pieces=8,
    seed=1,
    fps=25,
    size=(640, 360),
    min_s=2.0,
    max_s=6.0,
    far_s=60.0,
    near_s=(2.0, 12.0),
    tolerance_s=0.5,
    run=subprocess.run,
    make=True,
):
    files = sources(paths)
    lengths = {f: d for f in files if (d := duration(f, run)) is not None and d >= max_s + 2}
    if not lengths:
        raise SystemExit(f"no video of at least {max_s + 2:g} s in {', '.join(paths)}")
    rng = random.Random(seed)  # a dataset's choices, not a secret
    media_dir = os.path.join(settings.data_dir(), "evals", "media", name)
    out, skipped = [], []
    for kind in KINDS:
        for k in range(items):
            ps = plan(kind, lengths, rng, pieces, fps, min_s, max_s, far_s, near_s)
            if ps is None:
                skipped.append(kind)
                break
            iid = f"{kind}-{k + 1:02d}"
            dest = os.path.join(media_dir, f"{iid}.mp4")
            if make and not os.path.isfile(dest):
                os.makedirs(media_dir, exist_ok=True)
                splice(ps, dest, fps, size, run)
            at = [Fraction(0), *cuts(ps, fps)]
            out.append(
                dict(
                    id=iid,
                    input=dict(path=dest),
                    reference=dict(
                        cuts=[round(float(t), 6) for t in at[1:]],
                        tolerance_s=tolerance_s,
                        pieces=[
                            dict(
                                file=os.path.basename(p.file),
                                start=round(float(p.start), 3),
                                end=round(float(p.start + Fraction(p.frames) / fps), 3),
                                at=round(float(a), 6),
                            )
                            for p, a in zip(ps, at, strict=True)
                        ],
                    ),
                    tags=dict(kind=kind),
                )
            )
    if not out:
        raise SystemExit("the sources give no item: one short file cannot be spliced")
    n = sum(len(i["reference"]["cuts"]) for i in out)
    note = f"; no {', '.join(skipped)} items (the sources are too few or too short)" if skipped else ""
    return C.write_dataset(
        name,
        out,
        description=f"{len(out)} spliced videos, {n} known cuts, from {len(lengths)} uncut files (seed {seed}){note}",
    )


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("paths", nargs="+", help="uncut camera files, or folders of them")
    a.add_argument("--name", required=True, help="the dataset's name, e.g. spliced-shots")
    a.add_argument("--items", type=int, default=4, help="items of each kind (files, far, near)")
    a.add_argument("--pieces", type=int, default=8, help="stretches in each item (cuts = pieces - 1)")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--fps", type=int, default=25)
    a.add_argument("--size", default="640x360", help="the spliced video's size: each stretch is fitted inside it")
    a.add_argument("--min-s", type=float, default=2.0)
    a.add_argument("--max-s", type=float, default=6.0)
    a.add_argument("--far-s", type=float, default=60.0, help="far cuts: at least this far apart in the file")
    a.add_argument("--near-s", default="2,12", help="near cuts: this far apart in the file, from,to seconds")
    a.add_argument("--tolerance-s", type=float, default=0.5)
    o = a.parse_args(argv)
    w, h = (int(x) for x in o.size.lower().split("x"))
    near = tuple(float(x) for x in o.near_s.split(","))
    ds = build(
        o.paths, o.name, o.items, o.pieces, o.seed, o.fps, (w, h), o.min_s, o.max_s, o.far_s, near, o.tolerance_s
    )
    print(f"{len(ds.items)} items, version {ds.version}: {ds.description}")


if __name__ == "__main__":
    main()
