"""Text to speech, run as a venv-worker (see stratamodels/runtimes/worker.py): one process per model, in that model's own venv. Needs only the
engine's own libraries, numpy and soundfile (and ffmpeg on the PATH for `speed` on engines with no speed of their own).

Engines (models.toml options.engine):
  kokoro             Kokoro-82M through kokoro-onnx (the files strata-360 used: kokoro-v1.0.onnx, voices-v1.0.bin), preset voices (bm_lewis...)
  chatterbox         Resemble AI Chatterbox (MIT): clones the voice of a reference recording; exaggeration, cfg_weight
  chatterbox-turbo   Chatterbox Turbo (MIT, 350M): faster, needs a reference recording of more than 5 s
  qwen3-custom       Qwen3-TTS CustomVoice (Apache 2.0): preset speakers, styled by an instruction
  qwen3-design       Qwen3-TTS VoiceDesign: the voice is described in words (instruct)
  qwen3-clone        Qwen3-TTS Base: clones a reference recording (with its transcript, voice_text, it imitates it more closely)

params (every engine): text (required); voice (a preset name, or the path of a reference recording for the cloning engines); voice_text (the
reference's words, or voice_text_file); instruct (how to speak, or for qwen3-design what the voice is); speed (1.0 = the model's own pace; native on kokoro, else a
pitch-keeping time stretch after); language (kokoro: en-gb / en-us, qwen3: English...); seed (default 0: the same params give the same
audio); temperature, top_p, exaggeration, cfg_weight (engine knobs; ignored where they do not apply); out_dir (added by the runtime).
Pace (workers/pacing.py): wpm (speak at this many words a minute, counted in syllables so short and long words even out) or fit_s (the
clip lasts this long); Kokoro says it again at the speed that lands there, the sampled models make `tries` takes (seeds seed..seed+tries-1)
and keep the nearest, then a pitch-keeping stretch (0.75 to 1.35x) closes the rest. max_pause (default 0.45 s; 0 keeps every pause) cuts
longer pauses between words to that, and edge (default 0.05 s) is the silence kept before and after the speech; both apply to every clip.
Result: {path, duration_s, sample_rate, seconds, engine, voice, seed, speech_s, rate_wpm, max_pause_s, long_pauses}, and with a target
{target_s, stretch, fit: ok | long | short (more than 5 % off), speed (Kokoro)}.

A model entry can carry the settings an eval chose as options.defaults ({voice = ..., speed = ...}): the role then speaks with them unless a job
says otherwise, so apps ask the role for speech and get the chosen voice."""

import os
import subprocess
import time
import uuid

E = {}


import pacing  # noqa: E402  (pace and pauses, workers/pacing.py: shared with the eval metrics)
from protocol import ParamError, serve  # noqa: E402  (the shared JSON-lines loop, workers/protocol.py)


def _seed(n):
    import random

    import numpy as np

    random.seed(n)
    np.random.seed(n)
    try:
        import torch

        torch.manual_seed(n)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(n)
    except ImportError:
        pass


def _ref(p, need=True):
    v = p.get("voice")
    if not v:
        if need:
            raise ParamError("this engine clones a voice: give voice = the path of a reference recording")
        return None
    if not os.path.isfile(v):
        raise ParamError(f"no such reference recording: {v!r}")
    return v


# ---- loading --------------------------------------------------------------------------------------------------------------------------------
def load(model, models_dir):
    o = model.get("options") or {}
    eng = o.get("engine")
    E.clear()
    E.update(engine=eng, name=model.get("name"), options=o)
    if eng == "kokoro":
        import logging

        from kokoro_onnx import Kokoro

        logging.getLogger("phonemizer").setLevel(logging.ERROR)
        d = o.get("dir") or os.path.join(models_dir, "tts", "kokoro")
        files = [os.path.join(d, f) for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin")]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            raise RuntimeError(
                f"missing {missing}: download them from https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
            )
        E["m"] = Kokoro(*files)
        return dict(voices=sorted(E["m"].get_voices()))
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if eng == "chatterbox":
        from chatterbox.tts import ChatterboxTTS

        E["m"] = ChatterboxTTS.from_pretrained(device=dev)
        return dict(device=dev)
    if eng == "chatterbox-turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        E["m"] = ChatterboxTurboTTS.from_pretrained(device=dev)
        return dict(device=dev)
    if eng in ("qwen3-custom", "qwen3-design", "qwen3-clone"):
        from qwen_tts import Qwen3TTSModel

        E["m"] = Qwen3TTSModel.from_pretrained(
            model["name"],
            device_map=f"{dev}:0" if dev == "cuda" else dev,
            dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
            attn_implementation=o.get("attn", "sdpa"),
        )
        spk = getattr(E["m"].model, "get_supported_speakers", None)
        return dict(device=dev, speakers=sorted(spk() or []) if spk else None)
    raise RuntimeError(f"unknown engine {eng!r}")


# ---- speaking -------------------------------------------------------------------------------------------------------------------------------
def speak(p):
    """(float32 mono samples, sample rate, voice used, whether `speed` was applied natively)."""
    import numpy as np

    eng, m, text = E["engine"], E["m"], p["text"]
    if eng == "kokoro":
        v = p.get("voice") or "bm_lewis"
        if v not in m.get_voices():
            raise ParamError(f"no Kokoro voice {v!r}")
        lang = p.get("language") or ("en-gb" if v[0] == "b" else "en-us")
        x, sr = m.create(text, voice=v, speed=float(p.get("speed") or 1.0), lang=lang)
        return np.asarray(x, np.float32), sr, v, True
    if eng == "chatterbox":
        ref = _ref(p, need=False)
        wav = m.generate(
            text,
            audio_prompt_path=ref,
            exaggeration=float(p.get("exaggeration", 0.5)),
            cfg_weight=float(p.get("cfg_weight", 0.5)),
            temperature=float(p.get("temperature", 0.8)),
        )
        return wav.squeeze(0).cpu().numpy().astype(np.float32), m.sr, ref or "built-in", False
    if eng == "chatterbox-turbo":
        ref = _ref(p)
        wav = m.generate(
            text, audio_prompt_path=ref, temperature=float(p.get("temperature", 0.8)), top_p=float(p.get("top_p", 0.95))
        )
        return wav.squeeze(0).cpu().numpy().astype(np.float32), m.sr, ref, False
    lang = p.get("language") or "English"
    kw = {k: p[k] for k in ("temperature", "top_p") if k in p}
    if eng == "qwen3-custom":
        spk = p.get("voice") or E["options"].get("voice") or "Ryan"
        wavs, sr = m.generate_custom_voice(
            text=text, speaker=spk, language=lang, instruct=p.get("instruct") or None, **kw
        )
        return np.asarray(wavs[0], np.float32), sr, spk, False
    if eng == "qwen3-design":
        ins = p.get("instruct")
        if not ins:
            raise ParamError("qwen3-design needs instruct: a description of the voice")
        wavs, sr = m.generate_voice_design(text=text, instruct=ins, language=lang, **kw)
        return np.asarray(wavs[0], np.float32), sr, "designed", False
    if eng == "qwen3-clone":
        ref = _ref(p)
        rt = p.get("voice_text")
        if not rt and p.get("voice_text_file"):
            rt = open(p["voice_text_file"]).read().strip()
        wavs, sr = m.generate_voice_clone(
            text=text, language=lang, ref_audio=ref, ref_text=rt, x_vector_only_mode=not rt, **kw
        )
        return np.asarray(wavs[0], np.float32), sr, ref, False
    raise ParamError(f"unknown engine {eng!r}")


def stretch(path, speed):
    """Speed a finished file up or down, keeping its pitch (ffmpeg atempo)."""
    tmp = path + ".tmp.wav"
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path, "-filter:a", f"atempo={speed:.4f}", tmp],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        raise RuntimeError(f"ffmpeg atempo failed: {r.stderr[-300:]}")
    os.replace(tmp, path)


def _render(p, max_pause, edge):
    """One synthesis, its long pauses cut (pacing.tighten): (samples, sample rate, voice, speed applied natively)."""
    import numpy as np

    x, sr, voice, native = speak(p)
    x = np.clip(x, -1, 1)
    return pacing.tighten(x, sr, max_pause, edge), sr, voice, native


def run(p):
    import numpy as np
    import soundfile as sf

    p = {
        **(E["options"].get("defaults") or {}),
        **p,
    }  # a model entry's own voice and settings (models.toml options.defaults); the job's win
    text = p.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ParamError("give text")
    text = text.strip()
    speed = float(p.get("speed") or 1.0)
    if not 0.5 <= speed <= 2.0:
        raise ParamError("speed is between 0.5 and 2.0")
    wpm, fit_s = p.get("wpm"), p.get("fit_s")
    if wpm is not None and not 60 <= float(wpm) <= 300:
        raise ParamError("wpm is between 60 and 300")
    if fit_s is not None and float(fit_s) <= 0:
        raise ParamError("fit_s is a length in seconds")
    max_pause = None if p.get("max_pause", 0.45) in (None, 0, False) else float(p.get("max_pause", 0.45))
    edge = float(p.get("edge", 0.05))
    tries = max(1, min(int(p.get("tries") or 1), 8))
    seed = int(p.get("seed") or 0)
    target = (
        (float(fit_s) - 2 * edge)
        if fit_s is not None
        else pacing.target_seconds(text, wpm)
        if wpm is not None
        else None
    )
    if (
        target is not None and target <= 0.05
    ):  # a line with nothing to say at a pace (no syllables), or a length all edges
        raise ParamError(
            f"nothing to fit: the target speech length is {target:.2f} s (fit_s must be more than 2 x edge, and wpm needs words)"
        )
    t0 = time.perf_counter()

    def once(params, s):
        _seed(s)
        x, sr, voice, native = _render(dict(params, text=text), max_pause, edge)
        a, b, _ = pacing.pauses(x, sr)
        return x, sr, voice, native, max(b - a, 1e-3)

    x, sr, voice, native, got = once(p, seed)
    if (
        target is not None and native
    ):  # Kokoro: say it again at the speed that lands on the target (its pace is near linear in speed)
        speed = min(max(speed * got / target, 0.5), 2.0)
        x, sr, voice, native, got = once(dict(p, speed=speed), seed)
    elif (
        target is not None
    ):  # sampled models: their timing differs by seed; keep the take nearest the target, stretch the rest
        for k in range(1, tries):
            cand = once(p, seed + k)
            if abs(np.log(cand[4] / target)) < abs(np.log(got / target)):
                x, sr, voice, native, got = cand
                seed = seed + k
    out_dir = p.get("out_dir") or "."
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"tts-{uuid.uuid4().hex[:12]}.wav")
    sf.write(path, x, sr, subtype="PCM_16")
    factor = 1.0
    if target is not None:
        factor = min(
            max(got / target, 0.75), 1.35
        )  # a pitch-keeping stretch beyond this sounds processed: the fit is then reported off
        if abs(factor - 1.0) > 0.02:
            stretch(path, factor)
    elif not native and abs(speed - 1.0) > 1e-3:
        stretch(path, speed)
    y, sr2 = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if (
        max_pause is not None and len(y) and (factor < 1.0 or (target is None and not native and speed < 1.0))
    ):  # a slowdown lengthened the pauses again
        y = pacing.tighten(y, sr2, max_pause, edge)
        sf.write(path, y, sr2, subtype="PCM_16")
    m = pacing.measure(text, y, sr2)
    dur = len(y) / sr2
    res = dict(
        path=path,
        duration_s=round(dur, 3),
        sample_rate=sr,
        seconds=round(time.perf_counter() - t0, 3),
        engine=E["engine"],
        voice=voice,
        seed=seed,
        **m,
    )
    if target is not None:
        err = m["speech_s"] / target - 1
        res.update(
            target_s=round(target, 3),
            stretch=round(factor, 3),
            fit="ok" if abs(err) <= 0.05 else "long" if err > 0 else "short",
        )
        if native:
            res["speed"] = round(speed, 3)
    return res


if __name__ == "__main__":
    serve(load, run)
