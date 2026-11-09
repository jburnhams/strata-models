# pyright: reportMissingImports=false
# (the GPU stack is the optional `gpu` dependency group; this module imports it inside load(), by design)
"""Whisper through faster-whisper (CTranslate2): float16 on the GPU, int8 on the CPU when there is none.

params: path (an audio or video file ffmpeg can read); language (ISO code, else detected); task (transcribe | translate); word_timestamps (default
true); vad_filter (default false: callers that cut by pauses do it themselves); beam_size (default 5); condition_on_previous_text (default false:
fewer runaway repeats); initial_prompt (names, jargon); clip (start, end seconds).
detect_language_only: true answers {language, language_probability, all_language_probs: {code: p}} without transcribing (strata-360 picks the
language of each speech chunk among the few it expects, then transcribes the chunk in it)."""

import os

from stratamodels.runtimes import RunError

KEYS = (
    "language",
    "task",
    "word_timestamps",
    "vad_filter",
    "beam_size",
    "condition_on_previous_text",
    "initial_prompt",
    "temperature",
)


def load(model, models_dir):
    import ctranslate2
    from faster_whisper import WhisperModel

    cuda = ctranslate2.get_cuda_device_count() > 0
    m = WhisperModel(
        model.name,
        device="cuda" if cuda else "cpu",
        compute_type=model.options.get("compute_type", "float16") if cuda else "int8",
        download_root=os.path.join(models_dir, "whisper"),
    )
    return dict(model=m)


def decode(path, rate=16000):
    """The file as mono float32 at 16 kHz, decoded by ffmpeg (faster-whisper's own decoder uses PyAV, whose API changed under it in PyAV 15+)."""
    import subprocess

    import numpy as np

    p = subprocess.run(  # noqa: S603  (fixed ffmpeg command line; the path is one argument; ffmpeg is found on PATH)
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-f", "s16le", "-ac", "1", "-ar", str(rate), "-"],  # noqa: S607
        capture_output=True,
    )
    if p.returncode != 0:
        raise RunError(f"ffmpeg could not decode {path}: {p.stderr.decode(errors='replace').strip()[-300:]}")
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0


def _r(v):
    return round(float(v), 3)


def run(handle, params):
    path = params.get("path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise RunError(f"no such audio file: {path!r}")
    unknown = set(params) - set(KEYS) - {"path", "clip", "detect_language_only"}
    if unknown:
        raise RunError(f"unknown params {sorted(unknown)}")
    if params.get("detect_language_only"):
        lang, prob, probs = handle["model"].detect_language(decode(path))[:3]
        return dict(
            language=lang, language_probability=_r(prob), all_language_probs={l: round(float(p), 4) for l, p in probs}
        )
    kw = dict(word_timestamps=True, vad_filter=False, beam_size=5, condition_on_previous_text=False)
    kw.update({k: params[k] for k in KEYS if k in params})
    if params.get("clip"):
        kw["clip_timestamps"] = [float(v) for v in params["clip"]]
    segs, info = handle["model"].transcribe(decode(path), **kw)
    out = []
    for s in segs:
        d = dict(
            start=_r(s.start),
            end=_r(s.end),
            text=s.text,
            avg_logprob=_r(s.avg_logprob),
            no_speech_prob=_r(s.no_speech_prob),
        )
        if s.words:
            d["words"] = [
                dict(start=_r(w.start), end=_r(w.end), word=w.word, probability=_r(w.probability)) for w in s.words
            ]
        out.append(d)
    return dict(
        language=info.language,
        language_probability=_r(info.language_probability),
        duration=_r(info.duration),
        segments=out,
    )


def unload(handle):
    handle.pop("model", None)
    import gc

    gc.collect()
