"""The speaker-embedding venv-worker (stratamodels/workers/speaker.py) without SpeechBrain, torch or ffmpeg: a fake encoder over numpy, and
ffmpeg stood in for by a reader of the tiny generated WAVs (unit tests launch no external processes; the unit runners have no ffmpeg). The real model: tests/gpu (needs the speaker venv)."""

import os
import subprocess
import sys
import types
import wave

import numpy as np
import pytest

WORKERS = os.path.join(os.path.dirname(__file__), "..", "..", "src", "stratamodels", "workers")
sys.path.insert(0, os.path.abspath(WORKERS))
import protocol  # noqa: E402
import speaker as S  # noqa: E402


def _wav(path, seconds, rate=8000, freq=220.0):
    t = np.arange(int(seconds * rate)) / rate
    pcm = (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return str(path)


def _ffmpeg(cmd, capture_output):
    """Answers decode's ffmpeg call as ffmpeg would for these WAVs: the samples as s16le, mono, resampled to 16 kHz; anything else refused."""
    path, rate = cmd[cmd.index("-i") + 1], int(cmd[cmd.index("-ar") + 1])
    try:
        with wave.open(path, "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            source = w.getframerate()
    except (wave.Error, EOFError):
        return subprocess.CompletedProcess(cmd, 1, b"", b"Invalid data found when processing input")
    n = len(pcm) * rate // source
    out = np.interp(np.arange(n) * source / rate, np.arange(len(pcm)), pcm).astype(np.int16)
    return subprocess.CompletedProcess(cmd, 0, out.tobytes(), b"")


@pytest.fixture
def encoder(monkeypatch):
    """A fake `embed`: each stretch's vector is (its length in samples, its mean level), normalised, so tests can tell stretches apart."""
    seen = []

    def embed(batch):
        seen.append([len(a) for a in batch])
        v = np.array([[len(a), float(np.abs(a).mean()) * 1e5] for a in batch], np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(S, "embed", embed)
    real = S.decode
    monkeypatch.setattr(S, "decode", lambda path: real(path, run=_ffmpeg))
    return seen


def test_each_item_gets_a_vector_in_order_and_a_file_is_decoded_once(encoder, tmp_path, monkeypatch):
    a = _wav(tmp_path / "a.wav", 3.0)
    b = _wav(tmp_path / "b.wav", 1.0, freq=440.0)
    decodes = []
    real = S.decode
    monkeypatch.setattr(S, "decode", lambda p: decodes.append(p) or real(p))
    r = S.run(
        dict(
            items=[
                dict(path=a, clip=[0.0, 1.0]),
                dict(path=b),
                dict(path=a, clip=[1.0, 3.0]),
                dict(path=a, clip=[2.0, 2.2]),
            ]
        )
    )
    assert decodes == [a, b]  # a once, though three items name it
    assert r["seconds"] == [1.0, 1.0, 2.0, 0.2] and r["dim"] == 2
    assert all(abs(np.linalg.norm(v) - 1) < 1e-4 for v in r["embeddings"][:3])
    assert r["embeddings"][3] is None  # under MIN_SECONDS: too little voice to place
    assert encoder == [[16000, 16000, 32000]]  # decoded at 16 kHz, only the long-enough stretches embedded


def test_a_long_stretch_is_measured_on_its_middle(encoder, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "MAX_SECONDS", 1.0)
    a = _wav(tmp_path / "a.wav", 3.0)
    r = S.run(dict(items=[dict(path=a)]))
    assert r["seconds"] == [1.0] and encoder == [[16000]]
    x = np.arange(48000, dtype=np.float32)
    assert S.stretch(x, None)[0] == 16000  # the middle second of three


def test_items_go_in_batches(encoder, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "BATCH", 2)
    a = _wav(tmp_path / "a.wav", 3.0)
    S.run(dict(items=[dict(path=a, clip=[i * 0.5, i * 0.5 + 0.5]) for i in range(5)]))
    assert [len(b) for b in encoder] == [2, 2, 1]


def test_bad_requests_are_param_errors(encoder, tmp_path):
    a = _wav(tmp_path / "a.wav", 1.0)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    for p, msg in (
        (dict(), "items is a list"),
        (dict(items=[]), "items is a list"),
        (dict(items=[dict(path=a)], colour=1), "unknown params"),
        (dict(items=[dict(path="/nope.wav")]), "item 0: no such audio"),
        (dict(items=["x"]), "item 0: no such audio"),
        (dict(items=[dict(path=a), dict(path=a, clip=[2, 1])]), "item 1: clip is"),
        (dict(items=[dict(path=a, clip=[1])]), "item 0: clip is"),
        (dict(items=[dict(path=str(bad))]), "could not decode"),
    ):
        with pytest.raises(protocol.ParamError, match=msg):
            S.run(p)
    with pytest.raises(protocol.ParamError, match="at most"):
        S.run(dict(items=[dict(path=a)] * (S.MAX_ITEMS + 1)))


def test_out_dir_from_the_runtime_is_allowed(encoder, tmp_path):
    a = _wav(tmp_path / "a.wav", 1.0)
    assert S.run(dict(items=[dict(path=a)], out_dir=str(tmp_path)))["dim"] == 2


def test_decode_uses_ffmpeg_and_names_its_failure(tmp_path):
    calls = []

    def ffmpeg(cmd, capture_output):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, b"", b"Invalid data found")

    with pytest.raises(protocol.ParamError, match="Invalid data found"):
        S.decode(str(tmp_path / "x.mp3"), run=ffmpeg)
    assert calls[0][calls[0].index("-ar") + 1] == "16000" and calls[0][calls[0].index("-ac") + 1] == "1"


def test_pad_and_normalise():
    wavs, lens = S.pad([np.ones(4, np.float32), np.ones(2, np.float32)])
    assert wavs.shape == (2, 4) and wavs[1].tolist() == [1, 1, 0, 0] and lens.tolist() == [1.0, 0.5]
    assert np.allclose(
        S.normalise(np.array([[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]])), [[0.6, 0.8], [0.0, 1.0], [0.0, 0.0]]
    )


def test_load_picks_the_device_and_keeps_weights_under_models(monkeypatch):
    made = []
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    )
    inference = types.ModuleType("speechbrain.inference.speaker")
    inference.EncoderClassifier = types.SimpleNamespace(from_hparams=lambda **k: made.append(k) or "m")
    monkeypatch.setitem(sys.modules, "speechbrain", types.ModuleType("speechbrain"))
    monkeypatch.setitem(sys.modules, "speechbrain.inference", types.ModuleType("speechbrain.inference"))
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", inference)
    assert S.load(dict(name="speechbrain/spkrec-ecapa-voxceleb"), "/models") == dict(device="cpu")
    assert made[0]["savedir"] == os.path.join("/models", "speechbrain", "speechbrain--spkrec-ecapa-voxceleb")
    assert made[0]["run_opts"] == {"device": "cpu"} and S.M["model"] == "m"
