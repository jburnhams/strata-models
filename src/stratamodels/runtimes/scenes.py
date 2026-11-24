"""Where a video cuts from one shot to the next, from ffmpeg's scene score (the `scene` value of its `select` filter), role `shot-boundaries`.
Asked for by strata-reels, the video editor: a digitised VHS tape is hours of home video with a cut wherever the camcorder was stopped and
started, and the editor splits a tape into shots before it can describe, choose or trim anything. This is the plain baseline the role's eval
(`evals/specs/shots-spliced.toml`) measures learned detectors against, and it runs on the CPU (vram_gb = 0: the CPU lane), so a tape never waits
for the GPU.

How: ffmpeg decodes the video (deinterlaced with yadif, one frame per frame, so the two fields of an interlaced tape do not read as motion),
scales it to `width` pixels across, and scores every frame against the one before (0 = the same picture, 1 = nothing alike). A cut is a frame
whose score reaches `threshold`; of cuts closer together than `min_shot_s`, the highest wins (a camcorder restart on tape often has a few
frames of noise, each scoring high). Stretches of black (blackdetect) come back separately: a fade or an unrecorded gap is not a shot.

A tape's unrecorded end is not black all through: where the signal is lost, grey bars roll and flash between black frames, each flash a
"cut" (about a dozen in the last 7 s of a real family tape, 2026-11-24). So a stretch with no picture is found too: frames that, shrunk to
16x12 grey, do not vary along their rows (rolling bars, flat black or blue, and snow, which the shrinking averages out; on that tape they
measured up to 1.1 and the flattest real picture 5.8). A stretch of at least `blank_min_s` joins the black ones in the answer. When it
flashes (a would-be cut inside it), those cuts are dropped and its ends are cuts instead; a frame or two of torn picture at its ends belongs to
it, not to a shot of its own. One that does not flash adds no cut: pitch-dark footage of a night run has no picture either, and is not a gap.

options (models.toml): threshold (0.2), min_shot_s (0.6), width (160), deinterlace (true), binary (the ffmpeg command, else `ffmpeg` on PATH).

params:
  path        the video
  clip        [start, end] seconds to look at (times in the answer are still the file's)
  threshold, min_shot_s, width, deinterlace   override the model's options for this job
  black       find black stretches too (default true); black_min_s (0.4) and black_pixel (0.10, a pixel's luma up to which it is black)
  blank       count stretches with no picture as black, and their inside as one (default: as black); blank_min_s (1.0)
  scores      also answer every frame's score (a long list: 180,000 frames for a two-hour PAL tape)
Answer: {cuts: [{time, frame, score}], shots: [{start, end}], black: [{start, end}], fps, frames, duration, threshold, min_shot_s, width,
deinterlace, engine, seconds, model}, and `scores` when asked. Times are seconds in the file, `time` being the first frame of the new shot;
`frame` counts from the first frame looked at. `shots` cover the part looked at, end to end."""

import itertools
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from stratamodels.runtimes import RunError  # pyright: ignore[reportMissingTypeStubs]  (the package __init__ is not in the strict list yet)

TIMEOUT_S = (
    4 * 3600
)  # a three-hour tape decoded at 160 px with yadif takes minutes, not hours; this only stops a hung ffmpeg


BLANK_W, BLANK_H = (
    16,
    12,
)  # what a frame is shrunk to, to see whether there is a picture: a three-hour tape is 52 MB of it
# How much a row varies (standard deviation of its 8-bit greys, the frame's edges left out) below which there is no picture. One real tape,
# 2026-11-24: its lost signal and black up to 1.1, generated snow 1.7, its flattest picture (a static card) 5.8. Head-torch footage of night
# races goes under 4 for seconds at a time and under 2 only where it is as good as black, hence nearer the low side.
BLANK_ROW = 2.0


def binary(option: str | None = None) -> str | None:
    """The ffmpeg command to run, or None when there is none."""
    if option:
        return option if os.path.isfile(option) else shutil.which(option)
    return shutil.which("ffmpeg")


def load(model: Any, models_dir: str) -> dict[str, Any]:
    opts: dict[str, Any] = dict(model.options)
    exe = binary(opts.get("binary"))
    if not exe:
        raise RunError(
            "ffmpeg is not installed (pacman -S ffmpeg; apt install ffmpeg; brew install ffmpeg; choco install ffmpeg)"
        )
    try:
        out = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, text=True, timeout=30, check=False)
    except OSError as e:
        raise RunError(f"cannot run {exe}: {e}") from e
    return dict(
        exe=exe,
        version=(out.stdout or "").splitlines()[0].strip() if out.stdout else "ffmpeg",
        threshold=float(opts.get("threshold", 0.3)),
        min_shot_s=float(opts.get("min_shot_s", 0.6)),
        width=int(opts.get("width", 160)),
        deinterlace=bool(opts.get("deinterlace", True)),
        key=model.key,
    )


def parse_metadata(text: str) -> list[dict[str, Any]]:
    """ffmpeg's `metadata=mode=print` file: a `frame:N pts:P pts_time:T` line per frame, then its `key=value` lines. One dict per frame with
    `time` and whichever of `score`, `black_start`, `black_end` the frame carries."""
    frames: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("frame:"):
            fields = dict(part.split(":", 1) for part in line.split() if ":" in part)
            frames.append({"time": float(fields["pts_time"])})
        elif "=" in line and frames:
            key, value = line.split("=", 1)
            name = {
                "lavfi.scene_score": "score",
                "lavfi.black_start": "black_start",
                "lavfi.black_end": "black_end",
            }.get(key)
            if name:
                frames[-1][name] = float(value)
    return frames


def pick_cuts(scores: list[float], times: list[float], threshold: float, min_shot_s: float) -> list[int]:
    """Indices of the frames that start a new shot: every frame after the first scoring at least `threshold`, strongest first, keeping one
    only when no kept cut is nearer than `min_shot_s`. Returned in time order. The first frame is never a cut (it has nothing before it), but a
    cut close after it is kept: a clip may start just before a real cut."""
    order = sorted((i for i, s in enumerate(scores) if i > 0 and s >= threshold), key=lambda i: (-scores[i], i))
    kept: list[int] = []
    for i in order:
        if all(abs(times[i] - times[k]) >= min_shot_s for k in kept):
            kept.append(i)
    return sorted(kept)


def black_stretches(frames: list[dict[str, Any]], end: float, min_s: float = 0.0) -> list[dict[str, float]]:
    """Pairs blackdetect's start and end marks into stretches; one still open when the video ends closes at `end`. Stretches shorter than
    `min_s` are dropped here: blackdetect's own `d` does not filter the frame metadata ("added regardless of the minimum duration"). One
    still black when the part looked at ends is kept whatever its length: it may go on past the clip, so how long it is is not known."""
    out: list[dict[str, float]] = []
    open_at: float | None = None
    for f in frames:
        if "black_start" in f and open_at is None:
            open_at = f["black_start"]
        if "black_end" in f and open_at is not None:
            out.append({"start": round(open_at, 3), "end": round(f["black_end"], 3)})
            open_at = None
    out = [s for s in out if s["end"] - s["start"] >= min_s - 1e-9]
    if open_at is not None:
        out.append({"start": round(open_at, 3), "end": round(end, 3)})
    return out


def row_variation(grey: bytes) -> list[float]:
    """How much each frame varies along its rows, from its `BLANK_W` x `BLANK_H` greys one frame after another: the frame's own, or that of the
    mean of the five frames around it when that is less (snow is new every frame, so it averages out; a picture does not). The outer cells
    are left out: the head-switching noise along a tape's bottom lines and the black edges of the capture are there either way."""
    size = BLANK_W * BLANK_H
    cells = np.asarray(memoryview(grey)[: len(grey) // size * size], dtype=np.uint8).reshape(-1, BLANK_H, BLANK_W)[
        :, 1:-1, 1:-1
    ]
    own = np.asarray(cells, dtype=np.float32)
    if not len(own):
        return []
    n = len(own)
    mean = np.zeros_like(own)
    for k in range(
        -2, 3
    ):  # at the video's ends the window reflects, so it is still five pictures and not one five times
        picked = [i + k if 0 <= i + k < n else max(0, min(n - 1, i - k)) for i in range(n)]
        mean += own[np.asarray(picked, dtype=np.int64)]
    each = [np.asarray(np.mean(np.std(x, axis=2), axis=1), dtype=np.float32) for x in (own, mean / 5)]
    return [float(min(a, b)) for a, b in zip(*each, strict=True)]


def blank_runs(variation: list[float], min_frames: int) -> list[tuple[int, int]]:
    """Half-open frame ranges of at least `min_frames` with no picture (`BLANK_ROW`)."""
    out: list[tuple[int, int]] = []
    at = 0
    for blank, group in itertools.groupby(variation, lambda v: v < BLANK_ROW):
        n = sum(1 for _ in group)
        if blank and n >= min_frames:
            out.append((at, at + n))
        at += n
    return out


def widen_blank(
    runs: list[tuple[int, int]], scores: list[float], times: list[float], threshold: float, near_s: float
) -> list[tuple[int, int]]:
    """Each run grown over the torn frames at its ends: to the first frame scoring `threshold` less than `near_s` before it and the last one
    less than `near_s` after it (there the new picture starts). A run at the video's start or end stays there."""
    out: list[tuple[int, int]] = []
    for a, b in runs:
        first, last = a, b
        while first > 1 and times[a] - times[first - 1] < near_s:
            first -= 1
        while last < len(scores) - 1 and times[last + 1] - times[min(b, len(times) - 1)] < near_s:
            last += 1
        before = [i for i in range(first, a) if scores[i] >= threshold]
        after = [i for i in range(b + 1, last + 1) if i < len(scores) and scores[i] >= threshold]
        a, b = (min(before) if before else a), (max(after) if after else b)
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def merge(stretches: list[dict[str, float]]) -> list[dict[str, float]]:
    """Stretches in time order, those that touch or overlap as one."""
    out: list[dict[str, float]] = []
    for s in sorted(stretches, key=lambda s: s["start"]):
        if out and s["start"] <= out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
        else:
            out.append(dict(s))
    return out


def _number(params: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    value = params.get(key, default)
    try:
        n = float(value)
    except (TypeError, ValueError) as e:
        raise RunError(f"{key} must be a number: {value!r}") from e
    if not low <= n <= high:
        raise RunError(f"{key} must be from {low} to {high}: {n}")
    return n


def run(handle: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    path = params.get("path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise RunError(f"no such video: {path!r}")
    threshold = _number(params, "threshold", handle["threshold"], 0.0, 1.0)
    min_shot_s = _number(params, "min_shot_s", handle["min_shot_s"], 0.0, 3600.0)
    width = int(_number(params, "width", handle["width"], 32, 1920)) // 2 * 2
    deinterlace = bool(params.get("deinterlace", handle["deinterlace"]))
    black = bool(params.get("black", True))
    black_min_s = _number(params, "black_min_s", 0.4, 0.04, 3600.0)
    black_pixel = _number(params, "black_pixel", 0.10, 0.0, 1.0)
    blank = bool(params.get("blank", black))
    blank_min_s = _number(params, "blank_min_s", 1.0, 0.04, 3600.0)
    clip = params.get("clip")
    offset = 0.0
    cmd = [handle["exe"], "-nostdin", "-hide_banner", "-v", "error"]
    if clip is not None:
        if not isinstance(clip, (list, tuple)) or len(clip) != 2:  # pyright: ignore[reportUnknownArgumentType]
            raise RunError(f"clip must be [start, end] in seconds: {clip!r}")
        start, end = float(clip[0]), float(clip[1])  # pyright: ignore[reportUnknownArgumentType]
        if not 0 <= start < end:
            raise RunError(f"clip must be [start, end] with 0 <= start < end: {clip!r}")
        offset = start
        cmd += [
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{end - start:.6f}",
        ]  # before -i: ffmpeg seeks to the frame when it decodes, and times start at 0
    chain = (["yadif=mode=0"] if deinterlace else []) + [f"scale={width}:-2"]
    if black:
        chain.append(f"blackdetect=d={black_min_s}:pix_th={black_pixel}")
    chain += [
        "select='gte(scene\\,0)'",
        "metadata=mode=print:file=frames.txt",
    ]  # select on `scene` computes it for every frame; all are kept
    # every frame in both outputs, none doubled or dropped, so they pair up by their place
    every = ["-fps_mode", "passthrough"]
    if blank:
        small = f"scale={BLANK_W}:{BLANK_H}:flags=area,format=gray"
        # split after the analysis scale: both see the same deinterlaced frames, decoded once
        at = 2 if deinterlace else 1
        graph = f"[0:v:0]{','.join(chain[:at])},split[s][b];[s]{','.join(chain[at:])}[scored];[b]{small}[grey]"
        outputs = ["-filter_complex", graph, "-map", "[scored]", *every, "-f", "null", "-"]
        outputs += ["-map", "[grey]", *every, "-f", "rawvideo", "grey.raw"]
    else:
        outputs = ["-vf", ",".join(chain), *every, "-f", "null", "-"]
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="stratamodels-scenes-") as tmp:
        cmd += ["-i", os.path.abspath(path), "-an", "-sn", "-dn", *outputs]
        # cwd=tmp so the filter's file= is a plain name: a full path would need escaping for the filtergraph (a Windows drive's colon)
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, check=False, cwd=tmp)
        except subprocess.TimeoutExpired as e:
            raise RunError(f"ffmpeg took over {TIMEOUT_S} s on {path}") from e
        listing = os.path.join(tmp, "frames.txt")
        if out.returncode != 0 or not os.path.isfile(listing):
            raise RunError(f"ffmpeg failed on {path}: {(out.stderr or '').strip()[-500:]}")
        with open(listing, encoding="utf8") as f:
            frames = parse_metadata(f.read())
        grey = b""
        if blank and frames:
            with open(os.path.join(tmp, "grey.raw"), "rb") as f:
                grey = f.read()
    if not frames:
        raise RunError(f"{path}: no video frames (no video stream, or the clip is past its end)")
    times = [fr["time"] + offset for fr in frames]
    scores = [float(fr.get("score", 0.0)) for fr in frames]
    steps = sorted(b - a for a, b in itertools.pairwise(times) if b > a)
    frame_s = (
        steps[len(steps) // 2] if steps else 0.04
    )  # the median gap: one late or dropped frame does not change the rate
    end = times[-1] + frame_s
    runs: list[tuple[int, int]] = []
    if blank:
        runs = blank_runs(row_variation(grey)[: len(frames)], max(1, round(blank_min_s / frame_s)))
    # no cut inside a stretch that flashes; its ends are cuts (but not the video's own start and end), which outrank any cut near them
    flashing = [(a, b) for a, b in runs if any(s >= threshold for s in scores[a + 1 : b])]
    flashing = widen_blank(flashing, scores, times, threshold, min_shot_s)
    ends_at = sorted({i for run in flashing for i in run if 0 < i < len(frames)})
    outside = scores[:]
    for a, b in flashing:
        outside[a:b] = [0.0] * (b - a)
    cut_at = [
        i
        for i in pick_cuts(outside, times, threshold, min_shot_s)
        if all(abs(times[i] - times[k]) >= min_shot_s for k in ends_at)
    ]
    cut_at = sorted({*cut_at, *ends_at})
    edges = [times[0]] + [times[i] for i in cut_at] + [end]
    for fr in frames:  # blackdetect's marks are in the clip's time, like pts_time
        for k in ("black_start", "black_end"):
            if k in fr:
                fr[k] += offset
    dark = black_stretches(frames, end, black_min_s) if black else []
    dark += [
        {"start": round(times[a], 3), "end": round(times[b] if b < len(times) else end, 3)} for a, b in runs + flashing
    ]
    result: dict[str, Any] = dict(
        cuts=[{"time": round(times[i], 3), "frame": i, "score": round(scores[i], 4)} for i in cut_at],
        shots=[{"start": round(a, 3), "end": round(b, 3)} for a, b in itertools.pairwise(edges)],
        black=merge(dark),
        fps=round(1 / frame_s, 3),
        frames=len(frames),
        duration=round(end - times[0], 3),
        threshold=threshold,
        min_shot_s=min_shot_s,
        width=width,
        deinterlace=deinterlace,
        engine=handle["version"],
        seconds=round(time.monotonic() - t0, 3),
        model=handle["key"],
    )
    if params.get("scores"):
        result["scores"] = [round(s, 4) for s in scores]
    return result


def unload(handle: dict[str, Any]) -> None:
    handle.clear()  # nothing stays in memory between videos: each is its own ffmpeg process
