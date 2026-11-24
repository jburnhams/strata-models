"""Learned and classic shot-boundary detectors as rivals to ffmpeg's scene score (runtimes/scenes.py), role `shot-boundaries`, run as a
venv-worker (see stratamodels/runtimes/worker.py) in the venv `shots` (`scripts/setup_worker_venv.sh shots`), so torch, OpenCV and
PySceneDetect never join the service's own dependencies (Jonathan, 2026-11-23). Eval `shots-spliced` scores them against the baseline.

Engines (models.toml options.engine):
  transnetv2  TransNet V2 (Soucek and Lokoc 2020, MIT), the PyTorch port in `transnetv2-pytorch` with the authors' weights in the wheel. It
              looks at 48 x 27 pixel frames, 100 at a time, and gives each frame the chance that a transition is on it; a run of frames over
              `threshold` (0.5) is one transition, and the new shot starts on the frame after the run's strongest frame (a hard cut's peak is
              the last frame of the old shot; a dissolve's is its middle). Runs on CUDA when the model has VRAM (device "cpu" keeps it off).
  adaptive    PySceneDetect's AdaptiveDetector (BSD-3-Clause): ContentDetector's HSV + edge difference, divided by the mean of the
              `window_width` frames either side, so a pan or a handheld shake that changes every frame a lot does not read as a cut.
              threshold is that ratio (3.0); min_content_val (15) is the plain difference a cut also needs.
  content     PySceneDetect's ContentDetector: the plain HSV + edge difference against `threshold` (27).

Every engine sees the same frames: ffmpeg decodes once, deinterlaced with yadif (one frame per frame, as the baseline does, so an interlaced
tape's two fields do not read as motion), and the same pass finds black stretches (blackdetect at 160 px, as the baseline) and every frame's
time, so the answer has the baseline's shape and an app can switch detectors without changing its code. Of cuts closer than `min_shot_s`,
the strongest wins (the baseline's rule: a camcorder restart is often a few noisy frames), applied the same way to every engine; the
libraries' own minimum scene lengths are off so they do not decide it differently.

params: path; clip [start, end] (times in the answer stay the file's); threshold, min_shot_s, width (adaptive and content: frame width to
analyse, default 256, as PySceneDetect's own downscaling gives), deinterlace, and the engine's own (adaptive: window_width, min_content_val;
adaptive and content: luma_only) override the model's options for this job; black (default true), black_min_s (0.4), black_pixel (0.10);
scores (also answer every frame's score).
Answer: {cuts: [{time, frame, score}], shots: [{start, end}], black: [{start, end}], transitions: [{start, end}] (transnetv2: runs longer
than one frame, a dissolve or fade), fps, frames, duration, threshold, min_shot_s, deinterlace, engine, device, settings (every setting the
job ran with), seconds, model}."""

import itertools
import os
import subprocess
import tempfile
import time

from protocol import ParamError, serve

ENGINES = ("transnetv2", "adaptive", "content")
DEFAULTS = dict(
    transnetv2=dict(threshold=0.5, min_shot_s=0.6),
    adaptive=dict(threshold=3.0, min_shot_s=0.6, width=256, window_width=2, min_content_val=15.0, luma_only=False),
    content=dict(threshold=27.0, min_shot_s=0.6, width=256, luma_only=False),
)
TN_W, TN_H = 48, 27  # TransNet V2's input size, fixed by its weights
WINDOW, STEP, PAD = 100, 50, 25  # TransNet V2's inference windows: 100 frames, the middle 50 kept (its predict_frames)
BATCH = 8  # windows per forward pass
TIMEOUT_S = 4 * 3600
M = {}


def load(model, models_dir):
    opts = model.get("options") or {}
    engine = opts.get("engine")
    if engine not in ENGINES:
        raise ValueError(f"options.engine must be one of {', '.join(ENGINES)}: {engine!r}")
    M.clear()
    M.update(key=model["key"], engine=engine, opts=opts)
    if engine == "transnetv2":
        from importlib.metadata import version

        import torch
        from transnetv2_pytorch import TransNetV2

        cuda = torch.cuda.is_available() and opts.get("device") != "cpu" and float(model.get("vram_gb") or 0) > 0
        M["device"] = "cuda" if cuda else "cpu"
        M["net"] = TransNetV2(device=M["device"])
        M["version"] = f"transnetv2-pytorch {version('transnetv2-pytorch')}, torch {torch.__version__}"
    else:
        import scenedetect

        M["device"] = "cpu"
        M["version"] = f"scenedetect {scenedetect.__version__} {engine}"
    return dict(engine=M["version"], device=M["device"])


def settings_of(engine, opts, params):
    """The settings a job runs with: the engine's defaults, then the model's options, then the job's params, each checked."""
    out = dict(DEFAULTS[engine], deinterlace=True)
    for source in (opts, params):
        for k in out:
            if k in source:
                out[k] = source[k]
    for k, v in out.items():
        if isinstance(DEFAULTS[engine].get(k, True), bool) or k == "deinterlace":
            if not isinstance(v, bool):
                raise ParamError(f"{k} must be true or false: {v!r}")
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError) as e:
            raise ParamError(f"{k} must be a number: {v!r}") from e
        if out[k] < 0:
            raise ParamError(f"{k} must not be negative: {v!r}")
    if engine == "transnetv2" and not 0 < out["threshold"] < 1:
        raise ParamError(f"threshold is a chance from 0 to 1 for transnetv2: {out['threshold']}")
    if "width" in out:
        out["width"] = max(32, min(1920, int(out["width"]))) // 2 * 2
    if "window_width" in out:
        out["window_width"] = max(1, int(out["window_width"]))
    return out


def parse_metadata(text):
    """ffmpeg's `metadata=mode=print` file, as runtimes/scenes.py reads it (copied: a worker imports nothing from stratamodels)."""
    frames = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("frame:"):
            fields = dict(part.split(":", 1) for part in line.split() if ":" in part)
            frames.append({"time": float(fields["pts_time"])})
        elif "=" in line and frames:
            key, value = line.split("=", 1)
            name = {"lavfi.black_start": "black_start", "lavfi.black_end": "black_end"}.get(key)
            if name:
                frames[-1][name] = float(value)
    return frames


def black_stretches(frames, end, min_s=0.0):
    """blackdetect's marks paired into stretches, as runtimes/scenes.py does (one still open at the end closes there, whatever its length)."""
    out, open_at = [], None
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


def pick_cuts(scores, times, candidates, min_shot_s):
    """The baseline's rule over an engine's candidate cut frames: strongest first, kept only when no kept cut is nearer than `min_shot_s`;
    frame 0 is never a cut. Returned in time order."""
    kept = []
    for i in sorted((i for i in set(candidates) if i > 0), key=lambda i: (-scores[i], i)):
        if all(abs(times[i] - times[k]) >= min_shot_s for k in kept):
            kept.append(i)
    return sorted(kept)


def runs_over(values, threshold):
    """[start, end] (inclusive) of each run of values over `threshold`."""
    out, start = [], None
    for i, v in enumerate(values):
        if v > threshold and start is None:
            start = i
        elif v <= threshold and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(values) - 1))
    return out


def transnet_windows(frames):
    """TransNet V2's inference windows over a stream of frames (its predict_frames, without holding the whole video: a three-hour tape is a
    million frames). The stream is padded with 25 copies of the first frame and enough of the last to fill the final window; each window
    is 100 frames, 50 apart, and the middle 50 of each are the answers. Yields lists of 100 frames."""
    buf, n, last = [], 0, None
    for f in frames:
        if not buf:
            buf = [f] * PAD
        buf.append(f)
        n += 1
        last = f
        while len(buf) >= WINDOW:
            yield buf[:WINDOW]
            buf = buf[STEP:]
    if n == 0:
        return
    buf += [last] * (PAD + STEP - (n % STEP or STEP))
    while len(buf) >= WINDOW:
        yield buf[:WINDOW]
        buf = buf[STEP:]


def transnet_scores(frames):
    """Each frame's transition chance (TransNet V2's single-frame head, after the sigmoid)."""
    import numpy as np
    import torch

    net, out, batch = M["net"], [], []

    def flush():
        x = torch.from_numpy(np.stack([np.stack(w) for w in batch])).to(net.device)
        with torch.no_grad():
            single, _ = net.predict_raw(x)
        out.extend(single[:, PAD : PAD + STEP, 0].cpu().numpy().reshape(-1).tolist())
        batch.clear()

    count = [0]

    def counted():
        for f in frames:
            count[0] += 1
            yield f

    for w in transnet_windows(counted()):
        batch.append(w)
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()
    return out[: count[0]]  # the last window's padding answers too


def scenedetect_scores(engine, s, frames):
    """Each frame's score and the frames PySceneDetect's detector calls cuts (its own minimum scene length off; the caller applies one rule
    to every engine). Scores: content_val for content, the adaptive ratio for adaptive."""
    from scenedetect.common import FrameTimecode
    from scenedetect.detectors import AdaptiveDetector, ContentDetector
    from scenedetect.stats_manager import StatsManager

    if engine == "adaptive":
        det = AdaptiveDetector(
            adaptive_threshold=s["threshold"],
            min_scene_len=0,
            window_width=s["window_width"],
            min_content_val=s["min_content_val"],
            luma_only=s["luma_only"],
        )
    else:
        det = ContentDetector(threshold=s["threshold"], min_scene_len=0, luma_only=s["luma_only"])
    stats = StatsManager()
    det.stats_manager = stats
    stats.register_metrics(det.get_metrics())
    key = det.get_metrics()[-1] if engine == "adaptive" else "content_val"
    cuts, n = [], 0
    # the frame rate only carries the frame number here (every length the detectors measure is off), so any rate does
    for i, f in enumerate(frames):
        cuts += [tc.frame_num for tc in det.process_frame(FrameTimecode(i, fps=100.0), f)]
        n = i + 1
    if n:
        cuts += [tc.frame_num for tc in det.post_process(FrameTimecode(n - 1, fps=100.0))]
    scores = [float(stats.get_metrics(i, [key])[0] or 0.0) for i in range(n)]
    return scores, cuts


def decode(path, s, engine, clip, tmp, black, black_min_s, black_pixel):
    """Starts ffmpeg; returns (process, frame iterator, listing path). One decode feeds the engine (raw frames on stdout, at its size) and
    the black and time listing (blackdetect, and select on the scene score so every frame has a line, at 160 px)."""
    import numpy as np

    # the frame size is given to ffmpeg exactly, never left to scale's -2: frames are read from stdout by their byte count, so the size
    # read must be the size written (a height worked out here could round differently from ffmpeg's and mix rows of two frames)
    w, h = (TN_W, TN_H) if engine == "transnetv2" else (s["width"], analysis_height(path, s["width"]))
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error"]
    if clip is not None:
        cmd += ["-ss", f"{clip[0]:.6f}", "-t", f"{clip[1] - clip[0]:.6f}"]
    listing = ["scale=160:-2"] + ([f"blackdetect=d={black_min_s}:pix_th={black_pixel}"] if black else [])
    listing += ["select='gte(scene\\,0)'", "metadata=mode=print:file=frames.txt", "nullsink"]
    head = "yadif=mode=0," if s["deinterlace"] else ""
    graph = f"[0:v]{head}split[a][b];[a]{','.join(listing)};[b]scale={w}:{h}[out]"
    pix = (
        "rgb24" if engine == "transnetv2" else "bgr24"
    )  # TransNet V2 was trained on RGB; PySceneDetect reads OpenCV's BGR
    cmd += [
        "-i",
        os.path.abspath(path),
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix,
        "-",
    ]
    err = open(os.path.join(tmp, "ffmpeg.err"), "w+")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, cwd=tmp)

    def frames():
        size = w * h * 3
        while True:
            b = p.stdout.read(size)
            if len(b) < size:
                return
            yield np.frombuffer(b, np.uint8).reshape(h, w, 3)

    return p, err, frames(), os.path.join(tmp, "frames.txt")


def analysis_height(path, width, run=subprocess.run):
    """An even height for `width` that keeps the picture's stored shape as ffmpeg shows it: turned a quarter when the stream carries a
    rotation of 90 or 270 degrees (ffmpeg turns such a video upright before the filters, by default)."""
    import json

    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:stream_side_data=rotation",
            "-show_entries",
            "stream_tags=rotate",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        st = json.loads(out.stdout or "{}")["streams"][0]
        vw, vh = int(st["width"]), int(st["height"])
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise ParamError(f"ffprobe could not read {path}: {(out.stderr or '').strip()[-300:]}") from e
    turn = [d.get("rotation") for d in st.get("side_data_list") or [] if "rotation" in d] or [
        (st.get("tags") or {}).get("rotate", 0)
    ]
    if round(abs(float(turn[0] or 0))) % 180 == 90:
        vw, vh = vh, vw
    return max(2, int(width * vh / vw / 2 + 0.5) * 2)  # halves rounded up, as ffmpeg's own -2 does


def _number(params, key, default, low, high):
    v = params.get(key, default)
    try:
        n = float(v)
    except (TypeError, ValueError) as e:
        raise ParamError(f"{key} must be a number: {v!r}") from e
    if not low <= n <= high:
        raise ParamError(f"{key} must be from {low} to {high}: {n}")
    return n


def run(params):
    params = {k: v for k, v in params.items() if k != "out_dir"}
    engine = M["engine"]
    path = params.get("path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise ParamError(f"no such video: {path!r}")
    known = set(DEFAULTS[engine]) | {"path", "clip", "deinterlace", "black", "black_min_s", "black_pixel", "scores"}
    unknown = set(params) - known
    if unknown:
        raise ParamError(f"unknown params for {engine}: {sorted(unknown)}")
    s = settings_of(engine, M["opts"], params)
    black = bool(params.get("black", True))
    black_min_s = _number(params, "black_min_s", 0.4, 0.04, 3600.0)
    black_pixel = _number(params, "black_pixel", 0.10, 0.0, 1.0)
    clip, offset = params.get("clip"), 0.0
    if clip is not None:
        if not isinstance(clip, (list, tuple)) or len(clip) != 2:
            raise ParamError(f"clip must be [start, end] in seconds: {clip!r}")
        clip = (float(clip[0]), float(clip[1]))
        if not 0 <= clip[0] < clip[1]:
            raise ParamError(f"clip must be [start, end] with 0 <= start < end: {clip!r}")
        offset = clip[0]
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="stratamodels-shots-") as tmp:
        p, err, frames, listing = decode(path, s, engine, clip, tmp, black, black_min_s, black_pixel)
        try:
            if engine == "transnetv2":
                scores, candidates = transnet_scores(frames), []
            else:
                scores, candidates = scenedetect_scores(engine, s, frames)
            p.stdout.read()
            code = p.wait(TIMEOUT_S)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
            err.seek(0)
            stderr = err.read()
            err.close()
        if code != 0 or not os.path.isfile(listing):
            raise ParamError(f"ffmpeg failed on {path}: {stderr.strip()[-500:]}")
        with open(listing, encoding="utf8") as f:
            meta = parse_metadata(f.read())
    if not meta or not scores:
        raise ParamError(f"{path}: no video frames (no video stream, or the clip is past its end)")
    result = answer(engine, s, meta, scores, candidates, offset, black_min_s if black else None)
    result.update(engine=M["version"], device=M["device"], seconds=round(time.monotonic() - t0, 3), model=M["key"])
    if params.get("scores"):
        result["scores"] = [round(x, 4) for x in scores[: result["frames"]]]
    return result


def answer(engine, s, meta, scores, candidates, offset, black_min_s):
    """The answer from ffmpeg's listing (`meta`, one per frame), the engine's per-frame scores and its candidate cut frames (TransNet V2's
    come from its runs here); times in the file's seconds (`offset`: where the clip starts); black stretches unless black_min_s is None."""
    n = min(len(meta), len(scores))  # the two branches see the same frames; this only guards a stream cut short
    meta, scores = meta[:n], scores[:n]
    times = [m["time"] + offset for m in meta]
    steps = sorted(b - a for a, b in itertools.pairwise(times) if b > a)
    frame_s = steps[len(steps) // 2] if steps else 0.04
    end = times[-1] + frame_s
    transitions = []
    candidates = list(candidates)
    if engine == "transnetv2":
        runs = runs_over(scores, s["threshold"])
        # the new shot starts after the run's strongest frame: a hard cut's peak is the old shot's last frame, a dissolve's its middle
        # (a run that peaks on the last frame looked at has its new shot after the end: no cut)
        peaks = [max(range(a, b + 1), key=lambda i: scores[i]) for a, b in runs]
        transitions = [{"start": round(times[a], 3), "end": round(times[b] + frame_s, 3)} for a, b in runs if b > a]
        cut_score = scores[:]  # a cut's score is its run's peak, which is the frame before it
        for peak in peaks:
            if peak + 1 < n:
                candidates.append(peak + 1)
                cut_score[peak + 1] = scores[peak]
    else:
        candidates = [c for c in candidates if c < n]
        cut_score = scores
    cut_at = pick_cuts(cut_score, times, candidates, s["min_shot_s"])
    edges = [times[0]] + [times[i] for i in cut_at] + [end]
    for m in meta:
        for k in ("black_start", "black_end"):
            if k in m:
                m[k] += offset
    return dict(
        cuts=[{"time": round(times[i], 3), "frame": i, "score": round(cut_score[i], 4)} for i in cut_at],
        shots=[{"start": round(a, 3), "end": round(b, 3)} for a, b in itertools.pairwise(edges)],
        black=black_stretches(meta, end, black_min_s) if black_min_s is not None else [],
        transitions=transitions,
        fps=round(1 / frame_s, 3),
        frames=n,
        duration=round(end - times[0], 3),
        threshold=s["threshold"],
        min_shot_s=s["min_shot_s"],
        deinterlace=s["deinterlace"],
        settings=s,
    )


if __name__ == "__main__":
    serve(load, run)
