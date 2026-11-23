"""Where a video cuts from one shot to the next, from ffmpeg's scene score (the `scene` value of its `select` filter), role `shot-boundaries`.
Asked for by strata-reels, the video editor: a digitised VHS tape is hours of home video with a cut wherever the camcorder was stopped and
started, and the editor splits a tape into shots before it can describe, choose or trim anything. This is the plain baseline the role's eval
(`evals/specs/shots-vhs.toml`) measures learned detectors against, and it runs on the CPU (vram_gb = 0: the CPU lane), so a tape never waits
for the GPU.

How: ffmpeg decodes the video (deinterlaced with yadif, one frame per frame, so the two fields of an interlaced tape do not read as motion),
scales it to `width` pixels across, and scores every frame against the one before (0 = the same picture, 1 = nothing alike). A cut is a frame
whose score reaches `threshold`; of cuts closer together than `min_shot_s`, the highest wins (a camcorder restart on tape often has a few
frames of noise, each scoring high). Stretches of black (blackdetect) come back separately: a fade or an unrecorded gap is not a shot.

options (models.toml): threshold (0.3), min_shot_s (0.6), width (160), deinterlace (true), binary (the ffmpeg command, else `ffmpeg` on PATH).

params:
  path        the video
  clip        [start, end] seconds to look at (times in the answer are still the file's)
  threshold, min_shot_s, width, deinterlace   override the model's options for this job
  black       find black stretches too (default true); black_min_s (0.4) and black_pixel (0.10, a pixel's luma up to which it is black)
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

from stratamodels.runtimes import RunError  # pyright: ignore[reportMissingTypeStubs]  (the package __init__ is not in the strict list yet)

TIMEOUT_S = (
    4 * 3600
)  # a three-hour tape decoded at 160 px with yadif takes minutes, not hours; this only stops a hung ffmpeg


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


def black_stretches(frames: list[dict[str, Any]], end: float) -> list[dict[str, float]]:
    """Pairs blackdetect's start and end marks into stretches; one still open when the video ends closes at `end`."""
    out: list[dict[str, float]] = []
    open_at: float | None = None
    for f in frames:
        if "black_start" in f and open_at is None:
            open_at = f["black_start"]
        if "black_end" in f and open_at is not None:
            out.append({"start": round(open_at, 3), "end": round(f["black_end"], 3)})
            open_at = None
    if open_at is not None:
        out.append({"start": round(open_at, 3), "end": round(end, 3)})
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
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="stratamodels-scenes-") as tmp:
        cmd += ["-i", os.path.abspath(path), "-an", "-sn", "-dn", "-vf", ",".join(chain), "-f", "null", "-"]
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
    if not frames:
        raise RunError(f"{path}: no video frames (no video stream, or the clip is past its end)")
    times = [fr["time"] + offset for fr in frames]
    scores = [float(fr.get("score", 0.0)) for fr in frames]
    steps = sorted(b - a for a, b in itertools.pairwise(times) if b > a)
    frame_s = (
        steps[len(steps) // 2] if steps else 0.04
    )  # the median gap: one late or dropped frame does not change the rate
    end = times[-1] + frame_s
    cut_at = pick_cuts(scores, times, threshold, min_shot_s)
    edges = [times[0]] + [times[i] for i in cut_at] + [end]
    for fr in frames:  # blackdetect's marks are in the clip's time, like pts_time
        for k in ("black_start", "black_end"):
            if k in fr:
                fr[k] += offset
    result: dict[str, Any] = dict(
        cuts=[{"time": round(times[i], 3), "frame": i, "score": round(scores[i], 4)} for i in cut_at],
        shots=[{"start": round(a, 3), "end": round(b, 3)} for a, b in itertools.pairwise(edges)],
        black=black_stretches(frames, end) if black else [],
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
