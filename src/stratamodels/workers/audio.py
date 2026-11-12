"""Noise taken out of recorded sound, run as a venv-worker (see stratamodels/runtimes/worker.py): one process per model, in that model family's
own venv (`scripts/setup_worker_venv.sh dfn | clearvoice | resemble | audio`). Needs numpy, scipy and soundfile, and ffmpeg on the PATH (the
filters of the chain are ffmpeg's, as in strata-360).

The chain, the same for every engine so they differ only in the denoiser (strata-360's `dsp.clean_for_playback`, moved here; it is the
baseline the evals compare against):
  1. decode to mono 48 kHz (ffmpeg's downmix), `clip` [start, end] seconds if given
  2. high-pass at `highpass` Hz (default 80: wind and handling rumble; 0 = off)
  3. the engine's denoiser, its result mixed back with a little of what it took out when `atten_lim_db` caps how much it may remove (0: no
     cap, to override a model's defaults)
     (DeepFilterNet does this natively; for the others: denoised + 10^(-cap/20) x (input - denoised), strata-360's MossFormer2 stand-in)
  4. `finish = "strata360"`: a little speech EQ (-2 dB at 250 Hz, +2 dB at 3 kHz) and speech-aware auto gain (ffmpeg speechnorm: quiet speech
     lifted, pauses not); "none" leaves it
  5. loudness to `lufs` (default -14, BS.1770 integrated, peak-limited at -1 dBFS): every output at the same level, so a judge never prefers
     the louder one. null keeps the level the denoiser left.

Engines (models.toml options.engine; options.name is the engine's own model name where it has several):
  none          no denoiser: the original through the same chain (the "did denoising help at all" anchor)
  afftdn        ffmpeg's adaptive FFT denoiser (classical; params nr, nf)
  noisereduce   spectral gating (noisereduce, Audacity's method; params stationary, prop_decrease)
  dfn3          DeepFilterNet3 (2023, 48 kHz): strata-360's denoiser today
  clearvoice    ClearerVoice-Studio: MossFormer2_SE_48K, MossFormerGAN_SE_16K, FRCRN_SE_16K (options.name)
  resemble      Resemble Enhance (generative): `mode` denoise (default) or enhance (also restores bandwidth; nfe, lambd, tau)

params: path (required: any file ffmpeg reads), clip, highpass, atten_lim_db, finish, lufs, and the engine's own (above); out_dir (added by
the runtime). A model entry's options.defaults are the settings an eval chose (the job's own win).
Result: {path (48 kHz mono 16-bit WAV), duration_s, sample_rate, seconds, engine, model, lufs_in, lufs_out, removed_db (how much quieter the
denoiser left the high-passed input: what it took out, before finishing)}."""

import os
import subprocess
import time
import uuid
from typing import Any

from protocol import ParamError, serve

SR = 48000
E = {}
RUN = [subprocess.run]  # ffmpeg is called through this (a fake in the unit tests)
KEYS = {
    "path",
    "clip",
    "highpass",
    "atten_lim_db",
    "finish",
    "lufs",
    "nr",
    "nf",
    "stationary",
    "prop_decrease",
    "mode",
    "nfe",
    "lambd",
    "tau",
    "out_dir",
}
ENGINES = ("none", "afftdn", "noisereduce", "dfn3", "clearvoice", "resemble")
FINISH = "equalizer=f=250:t=q:w=1:g=-2,equalizer=f=3000:t=q:w=0.9:g=2,speechnorm=e=6:r=0.0005:l=1:t=0.02"  # strata-360's clean_for_playback
LIMIT = "alimiter=limit=0.89:attack=5:release=60:level=0"


# ---- ffmpeg ---------------------------------------------------------------------------------------------------------------------------------
def _ffmpeg(args, data=None, what="ffmpeg"):
    p = RUN[0](["ffmpeg", "-nostdin", "-v", "error", *args], input=data, capture_output=True)
    if p.returncode != 0:
        raise ParamError(f"{what} failed: {p.stderr.decode(errors='replace').strip()[-300:]}")
    return p.stdout


def decode(path, clip=None, sr=SR):
    """The file (or its clip) as mono float32 at `sr`."""
    import numpy as np

    cut = ["-ss", f"{float(clip[0]):.3f}", "-to", f"{float(clip[1]):.3f}"] if clip else []
    raw = _ffmpeg([*cut, "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"], what=f"decoding {path}")
    return np.frombuffer(raw, np.float32).copy()


def ffilter(x, graph, sr=SR):
    """Mono float32 through an ffmpeg -af graph (the same length back: filters here keep it)."""
    import numpy as np

    x = np.asarray(x, np.float32)
    raw = _ffmpeg(
        [
            "-f",
            "f32le",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-i",
            "-",
            "-af",
            graph,
            "-f",
            "f32le",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-",
        ],
        x.tobytes(),
        what=f"ffmpeg -af {graph.split('=')[0]}",
    )
    return np.frombuffer(raw, np.float32).copy()


def fit(y, n):
    """y cut or zero-padded to n samples (a model may hand back a few more or fewer)."""
    import numpy as np

    y = np.asarray(y, np.float32).reshape(-1)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


# ---- loudness (ITU-R BS.1770, as strata-360's dsp.integrated_lufs) -----------------------------------------------------------------------
_KSOS = [
    [1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585],
    [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621],
]  # K-weighting at 48 kHz


def lufs(x):
    """Gated integrated loudness of a mono signal (absolute gate -70 LUFS, relative -10 LU); -70 for silence. A mono signal counts as both
    channels of a stereo pair (strata-360 measured it so), which reads 3 dB above the single-channel figure."""
    import numpy as np
    from scipy import signal

    sr = SR  # the K-weighting coefficients are for 48 kHz
    if not len(x):
        return -70.0
    y = signal.sosfilt(np.array(_KSOS), np.asarray(x, np.float64))
    b, h = int(0.4 * sr), int(0.1 * sr)
    starts = np.arange(0, max(len(y) - b, 0) + 1, h)
    ms = np.array([2 * np.mean(y[s : s + b] ** 2) for s in starts])
    lv = -0.691 + 10 * np.log10(np.maximum(ms, 1e-12))
    keep = lv > -70
    if not keep.any():
        return -70.0
    rel = -0.691 + 10 * np.log10(np.mean(ms[keep])) - 10
    keep &= lv > rel
    return float(-0.691 + 10 * np.log10(np.mean(ms[keep])))


def to_lufs(y, target):
    """Loudness to `target` with the peak limiter (it takes some back, so measure and correct again, as strata-360 does), then the limiter once
    more for safety."""
    import numpy as np

    for _ in range(3):
        cur = lufs(y)
        if not np.isfinite(cur) or cur <= -69.9 or abs(target - cur) < 0.4:
            break
        y = ffilter(y * 10 ** (float(np.clip(target - cur, -12, 30)) / 20), LIMIT)
    return ffilter(y, LIMIT)


# ---- engines --------------------------------------------------------------------------------------------------------------------------------
def load(model, models_dir):
    o = model.get("options") or {}
    eng = o.get("engine")
    if eng not in ENGINES:
        raise RuntimeError(f"unknown engine {eng!r} (known: {', '.join(ENGINES)})")
    E.clear()
    E.update(engine=eng, name=model.get("name"), options=o, key=model.get("key"))
    if eng in ("none", "afftdn"):
        return dict(device="cpu")
    if eng == "noisereduce":
        import noisereduce

        E["m"] = noisereduce
        return dict(device="cpu")
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E["device"] = dev
    if eng == "dfn3":
        from df.enhance import init_df

        E["m"] = init_df(log_level="ERROR")[:2]
        E["m"][0].to(dev)
        return dict(device=dev)
    if eng == "clearvoice":
        from clearvoice import ClearVoice

        E["m"] = ClearVoice(task="speech_enhancement", model_names=[o.get("model") or model.get("name")])
        return dict(device=dev)
    from resemble_enhance.enhancer import inference  # resemble

    E["m"] = inference
    return dict(device=dev)


def denoise(x, p):
    """The engine's denoised version of x (mono float32 at 48 kHz), the same length. Returns (y, capped natively)."""
    import numpy as np

    eng = E["engine"]
    m: Any = E.get("m")  # the engine's model or module, set by load
    if eng == "none":
        return x.copy(), False
    if eng == "afftdn":
        return ffilter(x, f"afftdn=nr={float(p.get('nr', 18))}:nf={float(p.get('nf', -40))}:tn=1"), False
    if eng == "noisereduce":
        y = m.reduce_noise(
            y=x,
            sr=SR,
            stationary=bool(p.get("stationary", True)),
            prop_decrease=float(p.get("prop_decrease", 1.0)),
        )
        return fit(y, len(x)), False
    import torch

    if eng == "dfn3":
        from df.enhance import enhance

        model, st = m
        y = enhance(
            model,
            st,
            torch.from_numpy(x[None]),
            atten_lim_db=float(p["atten_lim_db"]) if p.get("atten_lim_db") else None,
        )
        return fit(y[0].cpu().numpy(), len(x)), True
    import tempfile

    import soundfile as sf

    if eng == "clearvoice":  # file in, file out: ClearVoice resamples to the model's own rate and back is ours to do
        with tempfile.TemporaryDirectory() as d:
            sf.write(f"{d}/in.wav", x, SR, subtype="FLOAT")
            out = m(input_path=f"{d}/in.wav", online_write=False)
            m.write(out, output_path=f"{d}/out.wav")
            return fit(decode(f"{d}/out.wav"), len(x)), False
    t = torch.from_numpy(x)  # resemble: 44.1 kHz inside; it resamples itself
    mode = p.get("mode", "denoise")
    if mode == "denoise":
        y, sr = m.denoise(t, SR, E["device"])
    elif mode == "enhance":
        y, sr = m.enhance(
            t,
            SR,
            E["device"],
            nfe=int(p.get("nfe", 64)),
            solver="midpoint",
            lambd=float(p.get("lambd", 0.9)),
            tau=float(p.get("tau", 0.5)),
        )
    else:
        raise ParamError("mode is denoise or enhance")
    y = y.cpu().numpy().astype(np.float32)
    if sr != SR:
        with tempfile.TemporaryDirectory() as d:
            sf.write(f"{d}/o.wav", y, sr, subtype="FLOAT")
            y = decode(f"{d}/o.wav")
    return fit(y, len(x)), False


def _db(e):
    import numpy as np

    return 10 * np.log10(max(float(e), 1e-20))


def run(p):
    import numpy as np
    import soundfile as sf

    p = {**(E["options"].get("defaults") or {}), **p}
    unknown = set(p) - KEYS
    if unknown:
        raise ParamError(f"unknown params {sorted(unknown)}")
    path = p.get("path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise ParamError(f"no such audio file: {path!r}")
    clip = p.get("clip")
    if clip is not None and (not isinstance(clip, list) or len(clip) != 2 or float(clip[1]) <= float(clip[0])):
        raise ParamError("clip is [start, end] in seconds")
    finish = p.get("finish", "none")
    if finish not in ("none", "strata360"):
        raise ParamError("finish is none or strata360")
    cap = p.get("atten_lim_db")
    if cap is not None and float(cap) < 0:
        raise ParamError("atten_lim_db is a number of dB (0 or null: no cap)")
    cap = float(cap) if cap else None  # 0 = no cap: TOML has no null, and a model's defaults may set one
    target = p.get("lufs", -14.0)
    if target is not None and not -40 <= float(target) <= -6:
        raise ParamError("lufs is between -40 and -6 (or null to keep the level)")
    t0 = time.perf_counter()
    x = decode(path, clip)
    if not len(x):
        raise ParamError(f"no audio in {path}" + (f" between {clip[0]} and {clip[1]} s" if clip else ""))
    hp = float(p.get("highpass", 80) or 0)
    h = ffilter(x, f"highpass=f={hp:g}:poles=2") if hp > 0 else x
    y, native = denoise(h, p)
    if cap is not None and not native:  # keep a little of what was taken out: the place stays, artefacts shrink
        y = (y + 10 ** (-float(cap) / 20) * (h - y)).astype(np.float32)
    removed = _db(np.mean(h.astype(np.float64) ** 2)) - _db(np.mean(y.astype(np.float64) ** 2))
    if finish == "strata360":
        y = ffilter(y, FINISH)
    if target is not None:
        y = to_lufs(y, float(target))
    y = np.clip(fit(y, len(x)), -1, 1)
    out_dir = p.get("out_dir") or "."
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"denoise-{uuid.uuid4().hex[:12]}.wav")
    sf.write(out, y, SR, subtype="PCM_16")
    return dict(
        path=out,
        duration_s=round(len(y) / SR, 3),
        sample_rate=SR,
        seconds=round(time.perf_counter() - t0, 3),
        engine=E["engine"],
        model=E.get("name"),
        lufs_in=round(lufs(x), 2),
        lufs_out=round(lufs(y), 2),
        removed_db=round(float(removed), 2),
    )


if __name__ == "__main__":
    serve(load, run)
