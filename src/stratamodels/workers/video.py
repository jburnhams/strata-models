"""Video enhancement (deinterlace, upscale, restore) as a venv-worker (see stratamodels/runtimes/worker.py): one process per model, in that
model's own venv. Needs ffmpeg and ffprobe on the PATH; the spandrel engine also needs torch, numpy and spandrel; the CLI engines need their own
repository and venv (scripts/setup_worker_venv.sh video | seedvr2 | flashvsr-plus).

Every engine gets the same prepared clip and its result is finished the same way, so the candidates differ only in the enhancement:
  prepare   deinterlace (bwdif, send_field: 25i -> 50p, field order forced), crop the active picture (no black side bars, no head-switching
            lines), resize to square pixels (`prep_size`, default 768 x 576 for 4:3 PAL)
  enhance   the engine (below)
  finish    resize to `size` (default 1440 x 1080) with lanczos, x264, the source's audio; then the judging extras: `regions` cropped from the
            result and enlarged nearest-neighbour (the same box in every candidate, so a face or some text can be compared up close) and a
            few `stills` (PNG frames, for the evals' similarity)

Engines (models.toml options.engine):
  ffmpeg     classical, no model and no GPU: the prepared clip through optional ffmpeg `filters` (e.g. hqdn3d) and lanczos. The baseline.
  spandrel   a per-frame model that spandrel loads (Real-ESRGAN, RealPLKSR, SwinIR, DAT, SPAN...): options.file (under <models>/spandrel/ or
             absolute), options.tile (0: whole frames), options.half (default true when the model supports it).
  seedvr2    SeedVR2 (ByteDance, Apache-2.0), a one-step video diffusion restorer, through the standalone CLI of
             numz/ComfyUI-SeedVR2_VideoUpscaler: options.repo (default <models>/src/seedvr2), options.args (CLI flags, e.g.
             {dit_model = "seedvr2_ema_3b_fp8_e4m3fn.safetensors", blocks_to_swap = 16, batch_size = 5}).
  flashvsr   FlashVSR+ (OpenImagingLab, Apache-2.0), streaming one-step video diffusion SR, through `python -m inference.run_stream`:
             options.repo (default <models>/src/flashvsr-plus), options.args ({mode = "turbo"} ...). Its output size must be a multiple of
             128 on each side (`engine_size`, default 1536 x 1152 for 4:3).

params:
  path           the clip (anything ffmpeg reads; an interlaced source stays interlaced until prepare)
  deinterlace    "bwdif" (default), "yadif", or "none" (a progressive source)
  field_order    "tff" (default) or "bff": forced, never trusted from the file's flags
  crop           [x, y, w, h] of the source's active picture (default: the whole frame)
  prep_size      [w, h] (default [768, 576])
  size           [w, h] of the result (default [1440, 1080])
  regions        [{id, label, box: [x, y, w, h]}] in source pixels
  crop_width     width of each region's enlargement (default 960; the height keeps the box's shape)
  stills         times in seconds of the frames kept as PNG (default [1, 3, 5, 7], those past the end left out)
  crf            x264 quality of the result (default 14; the regions get crf - 2)
  seed           for the diffusion engines (default 42)
  out_dir        added by the runtime
Result: {path, width, height, fps, frames, seconds, engine, model, crops: [{id, label, box, path}], stills: [paths], enhance_s, fps_enhance,
peak_vram_mb (the engine's processes on the card, at their highest)}."""

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

E = {}


from protocol import ParamError, serve  # noqa: E402  (the shared JSON-lines loop, workers/protocol.py)

DEFAULTS = dict(
    deinterlace="bwdif",
    field_order="tff",
    prep_size=[768, 576],
    size=[1440, 1080],
    crop_width=960,
    stills=[1, 3, 5, 7],
    crf=14,
    seed=42,
)


# ---- ffmpeg -----------------------------------------------------------------------------------------------------------------------------
def ffmpeg(args, run=subprocess.run):
    r = run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {(r.stderr or b'').decode(errors='replace').strip()[-800:]}")
    return r


def probe(path, run=subprocess.run):
    """{width, height, fps, frames, seconds, audio} of a video file."""
    r = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,r_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise ParamError(f"cannot read {path!r}: {(r.stderr or '').strip()[-300:]}")
    d = json.loads(r.stdout or "{}")
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        raise ParamError(f"{path!r} has no video stream")
    num, _, den = str(v.get("r_frame_rate", "25/1")).partition("/")
    fps = float(num) / float(den or 1)
    secs = float((d.get("format") or {}).get("duration") or 0)
    frames = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else round(secs * fps)
    return dict(
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        frames=frames,
        seconds=secs,
        audio=any(s.get("codec_type") == "audio" for s in d.get("streams", [])),
    )


def _pair(v, name):
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, int) and x > 0 for x in v)):
        raise ParamError(f"{name} is [width, height] in whole pixels, not {v!r}")
    return int(v[0]), int(v[1])


def prep_filter(p, src):
    """The ffmpeg filter chain that turns the source into the prepared clip every engine is given."""
    f = []
    di, fo = p["deinterlace"], p["field_order"]
    if fo not in ("tff", "bff"):
        raise ParamError(f"field_order is tff or bff, not {fo!r}")
    if di == "bwdif":
        f.append(f"bwdif=mode=send_field:parity={fo}:deint=all")
    elif di == "yadif":
        f.append(f"yadif=mode=send_field:parity={fo}:deint=all")
    elif di != "none":
        raise ParamError(f"deinterlace is bwdif, yadif or none, not {di!r}")
    x, y, w, h = crop_box(p, src)
    if (x, y, w, h) != (0, 0, src["width"], src["height"]):
        f.append(f"crop={w}:{h}:{x}:{y}")
    pw, ph = _pair(p["prep_size"], "prep_size")
    f.append(f"scale={pw}:{ph}:flags=lanczos,setsar=1")
    return ",".join(f)


def crop_box(p, src):
    c = p.get("crop")
    if c is None:
        return 0, 0, src["width"], src["height"]
    if not (isinstance(c, (list, tuple)) and len(c) == 4 and all(isinstance(v, int) and v >= 0 for v in c)):
        raise ParamError(f"crop is [x, y, w, h] in whole pixels, not {c!r}")
    x, y, w, h = c
    if w <= 0 or h <= 0 or x + w > src["width"] or y + h > src["height"]:
        raise ParamError(f"crop {c} is not inside the {src['width']} x {src['height']} frame")
    return x, y, w, h


def region_box(box, crop, size):
    """A region's box in source pixels -> the same box in the result's pixels (even numbers, inside the frame), or None when it is outside the crop."""
    if not (isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
        raise ParamError(f"a region's box is [x, y, w, h], not {box!r}")
    cx, cy, cw, ch = crop
    W, H = size
    x0, y0 = max(box[0], cx), max(box[1], cy)
    x1, y1 = min(box[0] + box[2], cx + cw), min(box[1] + box[3], cy + ch)
    if x1 <= x0 or y1 <= y0:
        return None
    sx, sy = W / cw, H / ch
    ox, oy = int((x0 - cx) * sx) // 2 * 2, int((y0 - cy) * sy) // 2 * 2
    ow, oh = max(2, int((x1 - x0) * sx) // 2 * 2), max(2, int((y1 - y0) * sy) // 2 * 2)
    return [ox, oy, min(ow, W - ox), min(oh, H - oy)]


def finish_args(src_args, p, out, fps, audio_from=None, prefilter=None):
    """ffmpeg arguments that take the engine's frames (src_args: an input) to the result: lanczos to `size`, x264, the source's audio.
    `prefilter` runs first (the ffmpeg engine: the whole preparation and its own filters); audio_from may be the video's own input."""
    W, H = _pair(p["size"], "size")
    a = [*src_args]
    own = audio_from is not None and src_args[-1] == audio_from
    if audio_from and not own:
        a += ["-i", audio_from]
    a += ["-map", "0:v:0"]
    if audio_from:
        a += ["-map", f"{0 if own else 1}:a:0?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    a += [
        "-vf",
        (f"{prefilter}," if prefilter else "") + f"scale={W}:{H}:flags=lanczos,setsar=1,format=yuv420p",
        "-r",
        f"{fps:g}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(int(p["crf"])),
        "-movflags",
        "+faststart",
        out,
    ]
    return a


def extras(p, result, crop, stem, run=subprocess.run):
    """The judging extras of a finished result: each region cropped and enlarged nearest-neighbour, and PNG stills."""
    W, H = _pair(p["size"], "size")
    info = probe(result, run=run)
    crops = []
    for i, r in enumerate(p.get("regions") or []):
        if not isinstance(r, dict) or "box" not in r:
            raise ParamError(f"region {i} needs a box: {r!r}")
        b = region_box(r["box"], crop, (W, H))
        if b is None:
            continue
        cw = int(p["crop_width"]) // 2 * 2
        ch = max(2, round(cw * b[3] / b[2] / 2) * 2)
        rid = str(r.get("id") or f"region{i + 1}")
        path = f"{stem}.{rid}.mp4"
        ffmpeg(
            [
                "-i",
                result,
                "-an",
                "-vf",
                f"crop={b[2]}:{b[3]}:{b[0]}:{b[1]},scale={cw}:{ch}:flags=neighbor,setsar=1,format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(max(int(p["crf"]) - 2, 0)),
                "-movflags",
                "+faststart",
                path,
            ],
            run=run,
        )
        crops.append(dict(id=rid, label=str(r.get("label") or rid), box=list(r["box"]), path=path))
    stills = []
    for t in p.get("stills") or []:
        if not isinstance(t, (int, float)) or t < 0 or t >= info["seconds"]:
            continue
        path = f"{stem}.still{float(t):g}s.png"
        ffmpeg(["-ss", f"{float(t):.3f}", "-i", result, "-frames:v", "1", path], run=run)
        stills.append(path)
    return info, crops, stills


# ---- VRAM of child processes -------------------------------------------------------------------------------------------------------------
def descendants(pid, proc="/proc"):
    """pid and every process under it (Linux /proc; elsewhere just pid)."""
    kids = {}
    try:
        for d in os.listdir(proc):
            if d.isdigit():
                try:
                    with open(os.path.join(proc, d, "stat")) as f:
                        ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                    kids.setdefault(ppid, []).append(int(d))
                except (OSError, ValueError, IndexError):
                    continue
    except OSError:
        return {pid}
    out, todo = set(), [pid]
    while todo:
        q = todo.pop()
        if q not in out:
            out.add(q)
            todo += kids.get(q, [])
    return out


def gpu_mb(pids, run=subprocess.run):
    """MB of GPU memory the given processes hold now (nvidia-smi), 0 when it cannot be read."""
    try:
        r = run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    total = 0.0
    for line in (r.stdout or "").splitlines() if r.returncode == 0 else []:
        a = [s.strip() for s in line.split(",")]
        if len(a) >= 2 and a[0].isdigit() and int(a[0]) in pids:
            try:
                total += float(a[1])
            except ValueError:
                pass
    return total


class Peak:
    """Polls the GPU memory of a process tree while it runs; .mb is the highest seen."""

    def __init__(self, pid, every=0.5, sample=None):
        self.mb, self.pid, self.every = 0.0, pid, every
        self.sample = sample or (lambda: gpu_mb(descendants(self.pid)))
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._go, daemon=True)
        self.t.start()

    def _go(self):
        while True:
            self.mb = max(self.mb, self.sample())
            if self.stop.wait(self.every):
                break

    def done(self):
        self.stop.set()
        self.t.join(5)
        self.mb = max(self.mb, self.sample())
        return self.mb


def run_cli(cmd, cwd, log, popen=subprocess.Popen):
    """Run an engine's own CLI; its output goes to the worker's log. Returns the peak GPU memory of its processes (MB)."""
    p = popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    peak = Peak(p.pid)
    rc = p.wait()
    mb = peak.done()
    if rc != 0:
        raise RuntimeError(f"{os.path.basename(cmd[1]) if len(cmd) > 1 else cmd[0]} exited {rc}: see the worker's log")
    return mb


def cli_flags(args):
    """{"blocks_to_swap": 16, "vae_decode_tiled": true} -> ["--blocks_to_swap", "16", "--vae_decode_tiled"] (false leaves a flag out)."""
    out = []
    for k, v in (args or {}).items():
        flag = k if k.startswith("-") else f"--{k}"
        if v is True:
            out.append(flag)
        elif v is False or v is None:
            continue
        else:
            out += [flag, str(v)]
    return out


# ---- engines ----------------------------------------------------------------------------------------------------------------------------
def load(model, models_dir):
    o = model.get("options") or {}
    eng = o.get("engine")
    E.clear()
    E.update(engine=eng, name=model.get("name"), key=model.get("key"), options=o, models_dir=models_dir)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be on the PATH")
    if eng == "ffmpeg":
        return dict(engine=eng)
    if eng == "spandrel":
        import torch
        from spandrel import ModelLoader

        f = o.get("file") or model.get("name")
        path = f if os.path.isabs(str(f)) else os.path.join(models_dir, "spandrel", str(f))
        if not os.path.isfile(path):
            raise RuntimeError(f"missing {path}: run scripts/setup_worker_venv.sh video (it downloads the weights)")
        d = ModelLoader().load_from_file(path)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        half = bool(o.get("half", True)) and dev == "cuda" and d.supports_half
        m = d.model.eval().to(dev)
        if half:
            m = m.half()
        E.update(model=m, scale=d.scale, device=dev, half=half, torch=torch)
        return dict(engine=eng, scale=d.scale, architecture=d.architecture.name, half=half, device=dev)
    if eng in ("seedvr2", "flashvsr"):
        repo = o.get("repo") or os.path.join(models_dir, "src", "seedvr2" if eng == "seedvr2" else "flashvsr-plus")
        if not os.path.isdir(repo):
            raise RuntimeError(
                f"no {repo}: run scripts/setup_worker_venv.sh {'seedvr2' if eng == 'seedvr2' else 'flashvsr-plus'}"
            )
        E["repo"] = repo
        return dict(engine=eng, repo=repo)
    raise RuntimeError(f"unknown engine {eng!r} (ffmpeg, spandrel, seedvr2, flashvsr)")


def _tiles(n, tile, overlap):
    """Start offsets that cover 0..n in tiles of `tile` with at least `overlap` between them."""
    if tile <= 0 or n <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, n - tile, step)) + [n - tile]
    return sorted(set(starts))


def upscale_frame(frame, tile=0, overlap=16):
    """One RGB frame (H x W x 3 uint8 numpy) through the loaded spandrel model, whole or in overlapping tiles."""
    import numpy as np

    torch, m, s = E["torch"], E["model"], E["scale"]
    dt = torch.float16 if E["half"] else torch.float32
    x = torch.from_numpy(np.ascontiguousarray(frame)).to(E["device"]).permute(2, 0, 1)[None].to(dt) / 255.0
    h, w = frame.shape[:2]
    with torch.inference_mode():
        if tile <= 0 or (h <= tile and w <= tile):
            y = m(x)
        else:
            y = torch.zeros((1, 3, h * s, w * s), dtype=dt, device=x.device)
            wt = torch.zeros((1, 1, h * s, w * s), dtype=dt, device=x.device)
            for t in _tiles(h, tile, overlap):
                for l in _tiles(w, tile, overlap):
                    th, tw = min(tile, h - t), min(tile, w - l)
                    y[:, :, t * s : (t + th) * s, l * s : (l + tw) * s] += m(x[:, :, t : t + th, l : l + tw])
                    wt[:, :, t * s : (t + th) * s, l * s : (l + tw) * s] += 1
            y = y / wt
        return (y.clamp(0, 1) * 255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()


def enhance_spandrel(p, src, out, popen=subprocess.Popen):
    """Decode the prepared frames, upscale each, pipe them into the finishing encoder. Returns peak VRAM (MB)."""
    pw, ph = _pair(p["prep_size"], "prep_size")
    s = E["scale"]
    fps = src["fps"] * (2 if p["deinterlace"] != "none" else 1)
    dec = popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            p["path"],
            "-an",
            "-vf",
            prep_filter(p, src),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    enc = popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            *finish_args(
                ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{pw * s}x{ph * s}", "-r", f"{fps:g}", "-i", "-"],
                p,
                out,
                fps,
                p["path"] if src["audio"] else None,
            ),
        ],
        stdin=subprocess.PIPE,
    )
    import numpy as np

    torch = E["torch"]
    if E["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    n, size, done = 0, pw * ph * 3, False
    try:
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            y = upscale_frame(np.frombuffer(buf, np.uint8).reshape(ph, pw, 3), int(E["options"].get("tile", 0)))
            enc.stdin.write(y.tobytes())
            n += 1
        done = True
    finally:
        if not done:  # the model failed (out of VRAM...): a decoder blocked on its full pipe would never exit, so stop both before waiting
            for proc in (dec, enc):
                with contextlib.suppress(OSError):
                    proc.kill()
        with contextlib.suppress(OSError):
            enc.stdin.close()
        rc_e, rc_d = enc.wait(), dec.wait()
    if rc_d != 0 or rc_e != 0 or n == 0:
        raise RuntimeError(f"ffmpeg failed (decode exit {rc_d}, encode exit {rc_e}, {n} frames)")
    return torch.cuda.max_memory_reserved() / 2**20 if E["device"] == "cuda" else 0.0


def prepare_file(p, src, work, run=subprocess.run):
    """The prepared clip as a file, for engines with a CLI of their own: lossless x264 (every reader opens it), the source's audio left out."""
    path = os.path.join(work, "prep.mp4")
    ffmpeg(
        [
            "-i",
            p["path"],
            "-an",
            "-vf",
            prep_filter(p, src) + ",format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-qp",
            "0",
            path,
        ],
        run=run,
    )
    return path


def engine_cmd(p, prep, up):
    """The command line of a CLI engine, and the folder to run it in."""
    o, eng = E["options"], E["engine"]
    args = dict(o.get("args") or {})
    if eng == "seedvr2":
        W, H = _pair(p["size"], "size")
        base = dict(
            resolution=min(W, H),
            seed=p["seed"],
            output_format="mp4",
            video_backend="ffmpeg",
            model_dir=os.path.join(E["models_dir"], "seedvr2"),
        )
        return [
            sys.executable,
            os.path.join(E["repo"], "inference_cli.py"),
            prep,
            "--output",
            up,
            *cli_flags({**base, **args}),
        ], E["repo"]
    W, H = _pair(p.get("engine_size") or [1536, 1152], "engine_size")
    if W % 128 or H % 128:
        raise ParamError(f"FlashVSR+ needs an engine_size whose sides are multiples of 128, not {W} x {H}")
    base = dict(mode="turbo", width=W, height=H, seed=p["seed"])
    return [
        sys.executable,
        "-m",
        "inference.run_stream",
        "--input",
        prep,
        "--output",
        up,
        *cli_flags({**base, **args}),
    ], E["repo"]


def run(params, run_=subprocess.run, popen=subprocess.Popen):
    p = {**DEFAULTS, **(E["options"].get("defaults") or {}), **params}
    if not isinstance(p.get("path"), str) or not os.path.isfile(p["path"]):
        raise ParamError(f"no such clip: {p.get('path')!r}")
    out_dir = p.get("out_dir") or tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    src = probe(p["path"], run=run_)
    crop = crop_box(p, src)
    prep_filter(p, src)  # checks the params before any work
    stem = os.path.join(out_dir, f"video-{uuid.uuid4().hex[:12]}")
    out = stem + ".mp4"
    fps = src["fps"] * (2 if p["deinterlace"] != "none" else 1)
    audio = p["path"] if src["audio"] else None
    t0 = time.time()
    eng = E["engine"]
    if eng == "ffmpeg":
        extra = E["options"].get("filters")
        vf = prep_filter(p, src) + (f",{extra}" if extra else "")
        ffmpeg(finish_args(["-i", p["path"]], p, out, fps, audio, prefilter=vf), run=run_)
        peak = 0.0
    elif eng == "spandrel":
        peak = enhance_spandrel(p, src, out, popen=popen)
    else:
        work = tempfile.mkdtemp(prefix="video-", dir=out_dir)
        try:
            prep = prepare_file(p, src, work, run=run_)
            up = os.path.join(work, "up.mp4")
            cmd, cwd = engine_cmd(p, prep, up)
            log = open(os.path.join(work, "engine.log"), "w")
            try:
                peak = run_cli(cmd, cwd, log, popen=popen)
            finally:
                log.close()
                with open(os.path.join(work, "engine.log"), errors="replace") as f:
                    sys.stderr.write(f.read()[-20000:])
            found = (
                up
                if os.path.isfile(up)
                else next(
                    (
                        os.path.join(work, f)
                        for f in sorted(os.listdir(work))
                        if f.endswith(".mp4") and f not in ("prep.mp4",)
                    ),
                    None,
                )
            )
            if not found:
                raise RuntimeError(f"{eng} wrote no video")
            ffmpeg(finish_args(["-i", found], p, out, fps, audio), run=run_)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    enhance_s = time.time() - t0
    info, crops, stills = extras(p, out, crop, stem, run=run_)
    return dict(
        path=out,
        width=info["width"],
        height=info["height"],
        fps=info["fps"],
        frames=info["frames"],
        seconds=round(enhance_s, 3),
        duration_s=info["seconds"],
        engine=eng,
        model=E.get("key") or E.get("name"),
        crops=crops,
        stills=stills,
        enhance_s=round(enhance_s, 3),
        fps_enhance=round(info["frames"] / enhance_s, 3) if enhance_s > 0 else None,
        peak_vram_mb=round(peak, 1),
    )


if __name__ == "__main__":
    serve(load, run)
