"""Speaker embeddings (role `speaker-embedding`): ECAPA-TDNN trained on VoxCeleb (SpeechBrain, Apache-2.0), run as a venv-worker (see
stratamodels/runtimes/worker.py) in its own venv (`scripts/setup_worker_venv.sh speaker`), because SpeechBrain wants a torchaudio the
service's torch does not match. One 192-number vector per stretch of speech, L2-normalised, so a dot product is the cosine similarity: two
stretches by the same voice sit near 1, by different voices near 0. Needs ffmpeg on the PATH.

What it is for (strata-voice, 2026-11-26): a found recording holds other people as well as the speaker; the app embeds each sentence and keeps
those that sound like the speaker's own studio takes. Telling who is who, how many voices there are and where one hands over to the next is
the app's (it knows whose voice it wants); this only measures.

params: items, a list of {path, clip?}: each a file ffmpeg can read and, optionally, [start, end] seconds of it. A file named by several items
is decoded once. A stretch shorter than MIN_SECONDS has too little voice to place and answers null in its place; one longer than MAX_SECONDS is
measured on its middle MAX_SECONDS (a sentence never needs more, and memory grows with length).
Answer: {dim, embeddings: [[float] | null, ...] in the order asked, seconds: [measured seconds, ...]}."""

import os
import subprocess

from protocol import ParamError, serve

RATE = 16000  # what the VoxCeleb model was trained on
MIN_SECONDS = 0.5
MAX_SECONDS = 30.0
BATCH = 16
MAX_ITEMS = 2000
M = {}


def load(model, models_dir):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    M["model"] = EncoderClassifier.from_hparams(
        source=model["name"],
        savedir=os.path.join(models_dir, "speechbrain", model["name"].replace("/", "--")),
        run_opts={"device": device},
    )
    M["device"] = device
    return dict(device=device)


def decode(path, run=subprocess.run):
    """The whole file as mono float32 at 16 kHz (ffmpeg's downmix)."""
    import numpy as np

    p = run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-f", "s16le", "-ac", "1", "-ar", str(RATE), "-"],
        capture_output=True,
    )
    if p.returncode != 0:
        raise ParamError(f"ffmpeg could not decode {path}: {p.stderr.decode(errors='replace').strip()[-300:]}")
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0


def stretch(audio, clip):
    """The samples of `clip` (or all of them), cut to the middle MAX_SECONDS."""
    a = audio
    if clip is not None:
        a = audio[max(0, int(float(clip[0]) * RATE)) : max(0, int(float(clip[1]) * RATE))]
    cap = int(MAX_SECONDS * RATE)
    if len(a) > cap:
        start = (len(a) - cap) // 2
        a = a[start : start + cap]
    return a


def pad(batch):
    """The arrays zero-padded to the longest, and each one's share of that length (so the model does not hear the padding)."""
    import numpy as np

    n = max(len(a) for a in batch)
    wavs = np.zeros((len(batch), n), np.float32)
    for i, a in enumerate(batch):
        wavs[i, : len(a)] = a
    return wavs, np.array([len(a) / n for a in batch], np.float32)


def normalise(e):
    import numpy as np

    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)


def embed(batch):
    """One normalised vector per array in `batch`. The torch glue alone; tests/gpu runs it."""
    import torch

    wavs, lens = pad(batch)
    dev = M["device"]
    with torch.no_grad():
        e = M["model"].encode_batch(torch.from_numpy(wavs).to(dev), torch.from_numpy(lens).to(dev))
    return normalise(e.squeeze(1).float().cpu().numpy())


def _items(params):
    unknown = set(params) - {"items", "out_dir"}  # the runtime adds out_dir for workers that write files
    if unknown:
        raise ParamError(f"unknown params {sorted(unknown)}")
    items = params.get("items")
    if not isinstance(items, list) or not items:
        raise ParamError("items is a list of {path, clip?}")
    if len(items) > MAX_ITEMS:
        raise ParamError(f"{len(items)} items; at most {MAX_ITEMS} in one job")
    out = []
    for i, it in enumerate(items):
        if not isinstance(it, dict) or not isinstance(it.get("path"), str) or not os.path.isfile(it["path"]):
            raise ParamError(f"item {i}: no such audio file: {it.get('path') if isinstance(it, dict) else it!r}")
        clip = it.get("clip")
        if clip is not None and (not isinstance(clip, list) or len(clip) != 2 or float(clip[1]) <= float(clip[0])):
            raise ParamError(f"item {i}: clip is [start, end] in seconds")
        out.append((it["path"], clip))
    return out


def run(params):
    items = _items(params)
    decoded = {}
    stretches = []
    for path, clip in items:
        if path not in decoded:
            decoded[path] = decode(path)
        stretches.append(stretch(decoded[path], clip))
    vectors = [None] * len(stretches)
    long_enough = [i for i, a in enumerate(stretches) if len(a) >= MIN_SECONDS * RATE]
    dim = 0
    for k in range(0, len(long_enough), BATCH):
        idx = long_enough[k : k + BATCH]
        for i, v in zip(idx, embed([stretches[i] for i in idx])):
            vectors[i] = [round(float(x), 5) for x in v]
            dim = len(v)
    return dict(dim=dim, embeddings=vectors, seconds=[round(len(a) / RATE, 3) for a in stretches])


if __name__ == "__main__":
    serve(load, run)
