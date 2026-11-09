"""Whisper through faster-whisper (CTranslate2), run as a venv-worker (see stratamodels/runtimes/worker.py) in its own venv
(`scripts/setup_worker_venv.sh whisper`): CTranslate2 is built for CUDA 12, and in its own process its CUDA 12 libraries never meet the CUDA 13
ones the service's onnxruntime and torch load (the same cuDNN file name, libcudnn.so.9, is in both, and a process can load only one).
float16 on the GPU, int8 on the CPU when there is none. Needs ffmpeg on the PATH.

params: path (an audio or video file ffmpeg can read); language (ISO code, else detected); task (transcribe | translate); word_timestamps
(default true); vad_filter (default false: callers that cut by pauses do it themselves); beam_size (default 5); condition_on_previous_text
(default false: fewer runaway repeats); initial_prompt (names, jargon); temperature; clip [start, end] (seconds: only that part is decoded and
heard; times in the answer stay those of the whole file).
detect_language_only: true answers {language, language_probability, all_language_probs: {code: p}} without transcribing (strata-360 picks the
language of each speech chunk among the few it expects, then transcribes the chunk in it).
Answer: {language, language_probability, duration, segments: [{start, end, text, avg_logprob, no_speech_prob, words: [{start, end, word,
probability}]}]}."""

import os
import subprocess

from protocol import ParamError, serve

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
M = {}


def load(model, models_dir):
    import ctranslate2
    from faster_whisper import WhisperModel

    cuda = ctranslate2.get_cuda_device_count() > 0
    opts = model.get("options") or {}
    M["model"] = WhisperModel(
        model["name"],
        device="cuda" if cuda else "cpu",
        compute_type=opts.get("compute_type", "float16") if cuda else "int8",
        download_root=os.path.join(models_dir, "whisper"),
    )
    return dict(device="cuda" if cuda else "cpu")


def decode(path, clip=None, rate=16000, run=subprocess.run):
    """The file (or the clip of it) as mono float32 at 16 kHz, decoded by ffmpeg (faster-whisper's own decoder uses PyAV, whose API changed under
    it in PyAV 15+). Returns (samples, offset in seconds)."""
    import numpy as np

    cut = ["-ss", f"{float(clip[0]):.3f}", "-to", f"{float(clip[1]):.3f}"] if clip else []
    p = run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            *cut,
            "-i",
            path,
            "-vn",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(rate),
            "-",
        ],
        capture_output=True,
    )
    if p.returncode != 0:
        raise ParamError(f"ffmpeg could not decode {path}: {p.stderr.decode(errors='replace').strip()[-300:]}")
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0, float(clip[0]) if clip else 0.0


def _r(v):
    return round(float(v), 3)


def run(params):
    params = {k: v for k, v in params.items() if k != "out_dir"}  # the runtime adds it for workers that write files
    path = params.get("path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise ParamError(f"no such audio file: {path!r}")
    unknown = set(params) - set(KEYS) - {"path", "clip", "detect_language_only"}
    if unknown:
        raise ParamError(f"unknown params {sorted(unknown)}")
    clip = params.get("clip")
    if clip is not None and (not isinstance(clip, list) or len(clip) != 2):
        raise ParamError("clip is [start, end] in seconds")
    audio, off = decode(path, clip)
    if params.get("detect_language_only"):
        lang, prob, probs = M["model"].detect_language(audio)[:3]
        return dict(
            language=lang, language_probability=_r(prob), all_language_probs={k: round(float(v), 4) for k, v in probs}
        )
    kw = dict(word_timestamps=True, vad_filter=False, beam_size=5, condition_on_previous_text=False)
    kw.update({k: params[k] for k in KEYS if k in params})
    segs, info = M["model"].transcribe(audio, **kw)
    out = []
    for s in segs:
        d = dict(
            start=_r(s.start + off),
            end=_r(s.end + off),
            text=s.text,
            avg_logprob=_r(s.avg_logprob),
            no_speech_prob=_r(s.no_speech_prob),
        )
        if s.words:
            d["words"] = [
                dict(start=_r(w.start + off), end=_r(w.end + off), word=w.word, probability=_r(w.probability))
                for w in s.words
            ]
        out.append(d)
    return dict(
        language=info.language,
        language_probability=_r(info.language_probability),
        duration=_r(info.duration),
        segments=out,
    )


if __name__ == "__main__":
    serve(load, run)
