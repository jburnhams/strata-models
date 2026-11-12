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
  region_stills  times of the frames kept as PNG of each region's crop (default [3]): what a vision judge is shown (one moment: 3 pictures a region keep a small judge within its context)
  crf            x264 quality of the result (default 14; the regions get crf - 2)
  seed           for the diffusion engines (default 42)
  cleanup        a VHS cleanup stage before any engine (see cleanup_filters): {chroma_shift = [cb_h, cb_v, cr_h, cr_v], filters = "...",
                 model = "1x model file under <models>/spandrel/", tile = 0}; the same stage can go before every upscaler, so a cleanup eval
                 picks it once with lanczos and the winner then chains
  quality        measure the result (default true): see quality(); readings that cannot be taken are left out and `quality.skipped`
                 says why (no numpy: only the ffmpeg readings), and never fail the job
  out_dir        added by the runtime
Result: {path, width, height, fps, frames, seconds, engine, model, crops: [{id, label, box, path, stills}], stills: [paths], enhance_s, fps_enhance,
peak_vram_mb (the engine's processes on the card, at their highest), quality {ssim, psnr, flicker, flicker_source, flicker_ratio,
sharpness, luma_noise, chroma_noise}}."""

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
    region_stills=[3],
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
    elif di not in ("none", *VS_DEINTERLACERS):  # qtgmc is done by the VapourSynth stage before ffmpeg sees the clip
        raise ParamError(f"deinterlace is bwdif, yadif, qtgmc or none, not {di!r}")
    x, y, w, h = crop_box(p, src)
    if (x, y, w, h) != (0, 0, src["width"], src["height"]):
        f.append(f"crop={w}:{h}:{x}:{y}")
    f += cleanup_filters(p.get("cleanup"))
    pw, ph = _pair(p["prep_size"], "prep_size")
    f.append(f"scale={pw}:{ph}:flags=lanczos,setsar=1")
    return ",".join(f)


CLEANUP_KEYS = ("chroma_shift", "filters", "planes", "vs", "model", "tile")


# ---- the VapourSynth stage: QTGMC and motion-compensated denoise (vsjetpack), before everything else -------------------------------------
VS_DEINTERLACERS = ("qtgmc",)
VS_PLANES = dict(all=None, luma=0, chroma=[1, 2])


def needs_vs(p):
    return p.get("deinterlace") in VS_DEINTERLACERS or bool((p.get("cleanup") or {}).get("vs"))


def vs_script(p, path):
    """A VapourSynth script (text) for the stage: the clip through bestsource with its field order set, deinterlaced (qtgmc: vsjetpack's
    QTempGaussMC; bwdif: the bwdif plugin, double rate) and denoised by cleanup.vs = {degrain = {tr, thsad, blksize}} (vsjetpack's
    mc_degrain: SMDegrain's motion-compensated temporal average, on the cleanup's `planes`). The output is progressive at the field rate."""
    tff = p["field_order"] == "tff"
    di = p["deinterlace"]
    lines = ["import vapoursynth as vs", "from vapoursynth import core", f"c = core.bs.VideoSource(source={path!r})"]
    if di != "none":
        lines.append(f"c = core.std.SetFieldBased(c, {2 if tff else 1})")
    if di == "qtgmc":
        lines += ["from vsdeinterlace import QTempGaussMC", f"c = QTempGaussMC().deinterlace(c, tff={tff})"]
    elif di == "bwdif":
        lines.append(f"c = core.bwdif.Bwdif(c, field={3 if tff else 2})")
    elif di != "none":
        raise ParamError(f"deinterlace {di!r} with a VapourSynth cleanup: use bwdif, qtgmc or none")
    c = p.get("cleanup") or {}
    v = c.get("vs")
    if v:
        if not isinstance(v, dict) or set(v) - {"degrain"}:
            raise ParamError(f"cleanup vs is {{degrain = {{tr, thsad, blksize}}}}, not {v!r}")
        d = v["degrain"] or {}
        if (
            not isinstance(d, dict)
            or set(d) - {"tr", "thsad", "blksize"}
            or not all(isinstance(x, int) and x > 0 for x in d.values())
        ):
            raise ParamError(f"degrain takes whole numbers tr, thsad, blksize, not {d!r}")
        planes = VS_PLANES.get(c.get("planes", "all"), "bad")
        if planes == "bad":
            raise ParamError(f"cleanup planes is all, luma or chroma, not {c.get('planes')!r}")
        lines += [
            "from vsdenoise import mc_degrain",
            f"c = mc_degrain(c, tr={d.get('tr', 2)}, thsad={d.get('thsad', 300)}, blksize={d.get('blksize', 16)}, planes={planes!r})",
        ]
    lines.append("c.set_output()")
    return "\n".join(lines) + "\n"


def vspipe_of(models_dir, options):
    """vspipe of the VapourSynth venv (scripts/setup_worker_venv.sh vapoursynth), or options.vspipe."""
    exe = options.get("vspipe") or os.path.join(models_dir, "venvs", "vapoursynth", "bin", "vspipe")
    if not os.path.isfile(exe):
        raise RuntimeError(
            f"no {exe}: run scripts/setup_worker_venv.sh vapoursynth (QTGMC and motion-compensated denoise need it)"
        )
    return exe


def run_vs(p, work, popen=subprocess.Popen):
    """The VapourSynth stage into a lossless intermediate (FFV1, the source's audio copied), and the params to run the rest on it: the
    deinterlacing is done (deinterlace "none") and the stage's denoise with it (cleanup.vs dropped)."""
    script = os.path.join(work, "stage.vpy")
    with open(script, "w") as f:
        f.write(vs_script(p, p["path"]))
    inter = os.path.join(work, "vs.mkv")
    vsp = popen([vspipe_of(E["models_dir"], E["options"]), "-c", "y4m", script, "-"], stdout=subprocess.PIPE)
    enc = popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "yuv4mpegpipe",
            "-i",
            "-",
            "-i",
            p["path"],
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-c:a",
            "copy",
            inter,
        ],
        stdin=vsp.stdout,
    )
    vsp.stdout.close()  # ffmpeg holds the pipe now: vspipe gets SIGPIPE if ffmpeg dies
    rc_e, rc_v = enc.wait(), vsp.wait()
    if rc_v != 0 or rc_e != 0:
        raise RuntimeError(
            f"the VapourSynth stage failed (vspipe exit {rc_v}, ffmpeg exit {rc_e}): see the worker's log"
        )
    c = {k: v for k, v in (p.get("cleanup") or {}).items() if k != "vs"}
    return {**p, "path": inter, "deinterlace": "none", "cleanup": c or None}


# mergeplanes inputs: 0 the original, 1 the filtered copy; output planes Y, U, V
PLANE_MAPS = dict(
    luma="map0s=1:map0p=0:map1s=0:map1p=1:map2s=0:map2p=2",
    chroma="map0s=0:map0p=0:map1s=1:map1p=1:map2s=1:map2p=2",
)


def cleanup_filters(c):
    """The ffmpeg part of a `cleanup` stage, run on the deinterlaced, cropped picture at its own pixels (before the resize, so a shift is in
    the tape's pixels): chroma_shift [cb_h, cb_v, cr_h, cr_v] (chroma moved against luma; negative is left or up; edges smeared), then
    `filters` (any ffmpeg filters) on the planes `planes` names: all (default), luma or chroma, the others left bit-identical. Its `model` (a 1x spandrel model) runs after the resize."""
    if not c:
        return []
    if not isinstance(c, dict) or set(c) - set(CLEANUP_KEYS):
        raise ParamError(f"cleanup is a table of {', '.join(CLEANUP_KEYS)}, not {c!r}")
    f = []
    cs = c.get("chroma_shift")
    if cs is not None:
        if not (
            isinstance(cs, (list, tuple)) and len(cs) == 4 and all(isinstance(v, int) and abs(v) <= 32 for v in cs)
        ):
            raise ParamError(f"chroma_shift is [cb_h, cb_v, cr_h, cr_v] in whole pixels, not {cs!r}")
        f += ["format=yuv444p", "chromashift=cbh={}:cbv={}:crh={}:crv={}:edge=smear".format(*cs)]
    if c.get("filters"):
        if not isinstance(c["filters"], str):
            raise ParamError("cleanup filters is a string of ffmpeg filters")
        planes = c.get("planes", "all")
        if planes == "all":
            f.append(c["filters"])
        elif planes in PLANE_MAPS:
            # Exactly one kind of plane: the filters run on a copy and mergeplanes takes the other planes from the original, bit for bit.
            # (A strength of 0 does not switch a plane off in every filter: hqdn3d reads 0 as "use the default".)
            fmt = "yuv444p" if cs is not None else "yuv420p"
            f.append(
                f"format={fmt},split[cl_orig][cl_in];[cl_in]{c['filters']}[cl_out];[cl_orig][cl_out]mergeplanes={PLANE_MAPS[planes]}:format={fmt}"
            )
        else:
            raise ParamError(f"cleanup planes is all, luma or chroma, not {planes!r}")
    return f


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
        shots = []
        for t in p.get("region_stills") or []:  # a frame of the crop, for a vision judge (it reads pictures, not video)
            if isinstance(t, (int, float)) and 0 <= t < info["seconds"]:
                shot = f"{stem}.{rid}.{float(t):g}s.png"
                ffmpeg(["-ss", f"{float(t):.3f}", "-i", path, "-frames:v", "1", shot], run=run)
                shots.append(shot)
        crops.append(dict(id=rid, label=str(r.get("label") or rid), box=list(r["box"]), path=path, stills=shots))
    stills = []
    for t in p.get("stills") or []:
        if not isinstance(t, (int, float)) or t < 0 or t >= info["seconds"]:
            continue
        path = f"{stem}.still{float(t):g}s.png"
        ffmpeg(["-ss", f"{float(t):.3f}", "-i", result, "-frames:v", "1", path], run=run)
        stills.append(path)
    return info, crops, stills


# ---- quality, measured (beside people's judgments, never instead of them) ----------------------------------------------------------------
def _ffmpeg_log(args, run=subprocess.run):
    """ffmpeg's log (stderr, at info level) of a run whose output is thrown away: the ssim/psnr summaries are written there."""
    r = run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "info", *args, "-f", "null", "-"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {(r.stderr or b'').decode(errors='replace').strip()[-400:]}")
    return (r.stderr or b"").decode(errors="replace")


def fidelity(p, src, result, run=subprocess.run):
    """How far the result strays from the tape: the result resized back to the prepared size (lanczos) against the prepared source (the same
    deinterlace and crop, no cleanup), SSIM (all planes) and PSNR in dB. Invented detail, colour drift and smearing all lower them; a cleanup
    that removes noise lowers them a little too, so they are context, not a verdict."""
    import re

    pw, ph = _pair(p["prep_size"], "prep_size")
    ref = prep_filter({**p, "cleanup": None}, src)
    fc = f"[0:v]scale={pw}:{ph}:flags=lanczos,setsar=1,format=yuv420p,split[a1][a2];[1:v]{ref},format=yuv420p,split[b1][b2];[a1][b1]ssim;[a2][b2]psnr"
    log = _ffmpeg_log(["-i", result, "-i", p["path"], "-filter_complex", fc], run=run)
    ssim = re.search(r"SSIM .*?All:([0-9.]+)", log)
    psnr = re.search(r"PSNR .*?average:([0-9.]+|inf)", log)
    return dict(
        ssim=float(ssim.group(1)) if ssim else None,
        psnr=(99.0 if psnr.group(1) == "inf" else float(psnr.group(1))) if psnr else None,
    )


def flicker(path, vf, run=subprocess.run):
    """Mean absolute difference of luma between consecutive frames (0-255), after `vf`: how much the picture moves or shimmers."""
    import re

    log = _ffmpeg_log(
        [
            "-i",
            path,
            "-an",
            "-vf",
            f"{vf},tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG",
        ],
        run=run,
    )
    vals = [
        float(v) for v in re.findall(r"lavfi\.signalstats\.YAVG=([0-9.]+)", log)[1:]
    ]  # the first blend is with nothing
    return sum(vals) / len(vals) if vals else None


def noise_sigma(plane):
    """Immerkaer's fast estimate of white noise in one plane (a 2-D numpy array): the standard deviation of what a Laplacian difference leaves."""
    import numpy as np

    a = plane.astype(np.float64)
    h, w = a.shape
    if h < 3 or w < 3:
        return 0.0
    c = (
        a[:-2, :-2]
        - 2 * a[:-2, 1:-1]
        + a[:-2, 2:]
        - 2 * a[1:-1, :-2]
        + 4 * a[1:-1, 1:-1]
        - 2 * a[1:-1, 2:]
        + a[2:, :-2]
        - 2 * a[2:, 1:-1]
        + a[2:, 2:]
    )
    return float(np.sqrt(np.pi / 2) * np.abs(c).sum() / (6 * (w - 2) * (h - 2)))


def sharpness(plane):
    """Variance of the Laplacian of luma: falls when a denoiser or upscaler blurs, rises with detail (and with noise, so read it with noise)."""
    import numpy as np

    a = plane.astype(np.float64)
    lap = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
    return float(lap.var())


def frames_at(path, times, size, vf="", run=subprocess.run):
    """yuv444p frames (numpy, 3 x h x w) of a video at `times`, resized to `size` with lanczos after `vf`."""
    import numpy as np

    w, h = size
    out = []
    for t in times:
        chain = (f"{vf}," if vf else "") + f"scale={w}:{h}:flags=lanczos,format=yuv444p"
        r = run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{float(t):.3f}",
                "-i",
                path,
                "-frames:v",
                "1",
                "-vf",
                chain,
                "-f",
                "rawvideo",
                "-",
            ],
            capture_output=True,
        )
        if r.returncode == 0 and len(r.stdout or b"") >= w * h * 3:
            out.append(np.frombuffer(r.stdout[: w * h * 3], np.uint8).reshape(3, h, w))
    return out


def quality(p, src, result, info, run=subprocess.run):
    """Programmatic readings of a result, at the prepared size so every candidate is measured alike: fidelity to the tape (ssim, psnr),
    temporal stability (flicker against the tape's own, flicker_ratio; over 1 shimmers more than the tape), and on stills luma sharpness,
    luma_noise and chroma_noise (Immerkaer sigma; chroma noise on VHS is the blotchy colour on flat areas)."""
    pw, ph = _pair(p["prep_size"], "prep_size")
    q = dict(fidelity(p, src, result, run=run))
    f_out = flicker(result, f"scale={pw}:{ph}:flags=lanczos", run=run)
    f_src = flicker(p["path"], prep_filter({**p, "cleanup": None}, src), run=run)
    q.update(
        flicker=f_out, flicker_source=f_src, flicker_ratio=(f_out / f_src) if f_out is not None and f_src else None
    )
    times = [t for t in (p.get("stills") or []) if isinstance(t, (int, float)) and 0 <= t < info["seconds"]]
    try:
        import numpy  # noqa: F401  (the stills' readings need it; a venv without it still gets the ffmpeg readings above)
    except ImportError:
        q["skipped"] = "sharpness, luma_noise and chroma_noise: numpy is not installed in this worker's venv"
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()}
    frames = frames_at(result, times, (pw, ph), run=run)
    if frames:
        q.update(
            sharpness=sum(sharpness(f[0]) for f in frames) / len(frames),
            luma_noise=sum(noise_sigma(f[0]) for f in frames) / len(frames),
            chroma_noise=sum((noise_sigma(f[1]) + noise_sigma(f[2])) / 2 for f in frames) / len(frames),
        )
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()}


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
    # a model registered with no VRAM runs in the scheduler's CPU lane (even beside an exclusive GPU lease): nothing of its may go on the card
    E["cpu_only"] = float(model.get("vram_gb") or 0) == 0
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be on the PATH")
    if eng == "ffmpeg":
        return dict(engine=eng)
    if eng == "spandrel":
        m = load_spandrel(o.get("file") or model.get("name"), models_dir, bool(o.get("half", True)))
        E.update(m)
        return dict(engine=eng, scale=m["scale"], architecture=m["architecture"], half=m["half"], device=m["device"])
    if eng in ("seedvr2", "flashvsr"):
        repo = o.get("repo") or os.path.join(models_dir, "src", "seedvr2" if eng == "seedvr2" else "flashvsr-plus")
        if not os.path.isdir(repo):
            raise RuntimeError(
                f"no {repo}: run scripts/setup_worker_venv.sh {'seedvr2' if eng == 'seedvr2' else 'flashvsr-plus'}"
            )
        E["repo"] = repo
        return dict(engine=eng, repo=repo)
    raise RuntimeError(f"unknown engine {eng!r} (ffmpeg, spandrel, seedvr2, flashvsr)")


def load_spandrel(f, models_dir, half=True, cpu=False):
    """A model spandrel loads (an upscaler, or a 1x cleanup model), on the card in half precision when it can: {model, scale, device, half, torch, architecture}."""
    import torch
    from spandrel import ModelLoader

    path = f if os.path.isabs(str(f)) else os.path.join(models_dir, "spandrel", str(f))
    if not os.path.isfile(path):
        raise RuntimeError(f"missing {path}: run scripts/setup_worker_venv.sh video (it downloads the weights)")
    d = ModelLoader().load_from_file(path)
    dev = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    half = half and dev == "cuda" and d.supports_half
    m = d.model.eval().to(dev)
    if half:
        m = m.half()
    return dict(model=m, scale=d.scale, device=dev, half=half, torch=torch, architecture=d.architecture.name)


def cleanup_model(p):
    """The cleanup stage's 1x model, loaded once and kept beside the engine's own (None when the stage has none)."""
    c = p.get("cleanup") or {}
    f = c.get("model")
    if not f:
        return None
    cache = E.setdefault("cleanups", {})
    if f not in cache:
        cache[f] = load_spandrel(
            f, E["models_dir"], cpu=bool(E.get("cpu_only"))
        )  # on the card only when the model was given VRAM
    if cache[f]["scale"] != 1:
        raise ParamError(f"a cleanup model keeps the size (1x); {f} is {cache[f]['scale']}x")
    return cache[f]


def _tiles(n, tile, overlap):
    """Start offsets that cover 0..n in tiles of `tile` with at least `overlap` between them."""
    if tile <= 0 or n <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, n - tile, step)) + [n - tile]
    return sorted(set(starts))


def upscale_frame(frame, tile=0, overlap=16, mm=None):
    """One RGB frame (H x W x 3 uint8 numpy) through a loaded spandrel model (`mm`, default the engine's), whole or in overlapping tiles."""
    import numpy as np

    mm = mm or E
    torch, m, s = mm["torch"], mm["model"], mm["scale"]
    dt = torch.float16 if mm["half"] else torch.float32
    x = torch.from_numpy(np.ascontiguousarray(frame)).to(mm["device"]).permute(2, 0, 1)[None].to(dt) / 255.0
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


def raw_input(w, h, fps):
    """ffmpeg input arguments for RGB frames written to its stdin."""
    return ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-"]


def pipe_frames(p, src, models, encode, popen=subprocess.Popen):
    """Decode the prepared frames, put each through `models` in turn ([(model dict, tile)]: a cleanup model, an upscaler), and write them to
    the encoder `encode(width, height)` gives the arguments of. Returns peak VRAM (MB)."""
    import numpy as np

    pw, ph = _pair(p["prep_size"], "prep_size")
    scale = 1
    for mm, _ in models:
        scale *= mm["scale"]
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
    enc = popen(["ffmpeg", "-nostdin", "-v", "error", "-y", *encode(pw * scale, ph * scale)], stdin=subprocess.PIPE)
    torch = models[0][0]["torch"]
    cuda = any(mm["device"] == "cuda" for mm, _ in models)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    n, size, done = 0, pw * ph * 3, False
    try:
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            y = np.frombuffer(buf, np.uint8).reshape(ph, pw, 3)
            for mm, tile in models:
                y = upscale_frame(y, tile, mm=mm)
            enc.stdin.write(y.tobytes())
            n += 1
        done = True
    finally:
        if not done:  # a model failed (out of VRAM...): a decoder blocked on its full pipe would never exit, so stop both before waiting
            for proc in (dec, enc):
                with contextlib.suppress(OSError):
                    proc.kill()
        with contextlib.suppress(OSError):
            enc.stdin.close()
        rc_e, rc_d = enc.wait(), dec.wait()
    if rc_d != 0 or rc_e != 0 or n == 0:
        raise RuntimeError(f"ffmpeg failed (decode exit {rc_d}, encode exit {rc_e}, {n} frames)")
    return torch.cuda.max_memory_reserved() / 2**20 if cuda else 0.0


def enhance_spandrel(p, src, out, popen=subprocess.Popen, cleanup=None):
    """The engine's upscaler on every prepared frame (after the cleanup model, when there is one), into the finishing encoder."""
    fps = src["fps"] * (2 if p["deinterlace"] != "none" else 1)
    models = ([(cleanup, int((p.get("cleanup") or {}).get("tile", 0)))] if cleanup else []) + [
        (E, int(E["options"].get("tile", 0)))
    ]
    audio = p["path"] if src["audio"] else None
    return pipe_frames(p, src, models, lambda w, h: finish_args(raw_input(w, h, fps), p, out, fps, audio), popen=popen)


def prepare_file(p, src, work, run=subprocess.run, cleanup=None, popen=subprocess.Popen):
    """The prepared clip as a file, for engines with a CLI of their own: lossless x264 (every reader opens it), the source's audio left out.
    With a cleanup model the frames go through it on the way. Returns (path, peak VRAM in MB)."""
    path = os.path.join(work, "prep.mp4")
    if cleanup:
        fps = src["fps"] * (2 if p["deinterlace"] != "none" else 1)
        tile = int((p.get("cleanup") or {}).get("tile", 0))
        enc = lambda w, h: [
            *raw_input(w, h, fps),
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-qp",
            "0",
            path,
        ]
        return path, pipe_frames(p, src, [(cleanup, tile)], enc, popen=popen)
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
    return path, 0.0


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
    tape, tape_src = p, src  # what quality() measures against: the tape, deinterlaced the plain way
    if needs_vs(p):
        vs_script(p, p["path"])  # checks the stage's params too
    stem = os.path.join(out_dir, f"video-{uuid.uuid4().hex[:12]}")
    out = stem + ".mp4"
    fps = src["fps"] * (2 if p["deinterlace"] != "none" else 1)
    audio = p["path"] if src["audio"] else None
    t0 = time.time()
    vs_work = tempfile.mkdtemp(prefix="video-vs-", dir=out_dir) if needs_vs(p) else None
    try:
        return _run(p, src, crop, tape, tape_src, stem, out, fps, audio, t0, vs_work, run_, popen)
    finally:
        if vs_work:
            shutil.rmtree(vs_work, ignore_errors=True)


def _run(p, src, crop, tape, tape_src, stem, out, fps, audio, t0, vs_work, run_, popen):
    if vs_work:  # the VapourSynth stage first; everything after reads its intermediate (progressive, at the field rate, with the audio)
        p = run_vs(p, vs_work, popen=popen)
        src = probe(p["path"], run=run_)
        fps, audio = src["fps"], (p["path"] if src["audio"] else None)
    out_dir = os.path.dirname(out)
    eng = E["engine"]
    cm = cleanup_model(p)
    if eng == "ffmpeg":
        extra = E["options"].get("filters")
        if cm:
            tile = int((p.get("cleanup") or {}).get("tile", 0))
            peak = pipe_frames(
                p,
                src,
                [(cm, tile)],
                lambda w, h: finish_args(raw_input(w, h, fps), p, out, fps, audio, prefilter=extra),
                popen=popen,
            )
        else:
            vf = prep_filter(p, src) + (f",{extra}" if extra else "")
            ffmpeg(finish_args(["-i", p["path"]], p, out, fps, audio, prefilter=vf), run=run_)
            peak = 0.0
    elif eng == "spandrel":
        peak = enhance_spandrel(p, src, out, popen=popen, cleanup=cm)
    else:
        work = tempfile.mkdtemp(prefix="video-", dir=out_dir)
        try:
            prep, prep_peak = prepare_file(p, src, work, run=run_, cleanup=cm, popen=popen)
            up = os.path.join(work, "up.mp4")
            cmd, cwd = engine_cmd(p, prep, up)
            log = open(os.path.join(work, "engine.log"), "w")
            try:
                peak = max(run_cli(cmd, cwd, log, popen=popen), prep_peak)
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
    # measured against the tape as bwdif deinterlaces it, whatever this candidate did, so every candidate is held to the same reference
    plain = {
        **tape,
        "deinterlace": "bwdif" if tape["deinterlace"] in VS_DEINTERLACERS else tape["deinterlace"],
        "cleanup": None,
    }
    q = None
    if tape.get("quality", True):
        try:
            q = quality(plain, tape_src, out, info, run=run_)
        except Exception as e:  # the readings are context: a reading that cannot be taken never fails the job (its metrics read as not measured)
            q = dict(skipped=f"{type(e).__name__}: {e}"[:300])
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
        quality=q,
    )


if __name__ == "__main__":
    serve(load, run)
