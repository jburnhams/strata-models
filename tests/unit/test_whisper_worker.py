"""The Whisper venv-worker (stratamodels/workers/whisper.py) and the shared worker loop (workers/protocol.py), without a model, ffmpeg or a
child process: a fake WhisperModel, a fake ffmpeg. The real thing: tests/gpu (needs the whisper venv)."""

import io
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

WORKERS = os.path.join(os.path.dirname(__file__), "..", "..", "src", "stratamodels", "workers")
sys.path.insert(0, os.path.abspath(WORKERS))
import protocol  # noqa: E402
import whisper as W  # noqa: E402


class FakeModel:
    def __init__(self):
        self.calls = []

    def detect_language(self, audio):
        return "fr", 0.91, [("fr", 0.91), ("en", 0.05)]

    def transcribe(self, audio, **kw):
        self.calls.append((len(audio), kw))
        word = types.SimpleNamespace(start=0.5, end=0.9, word=" Bonjour", probability=0.98)
        seg = types.SimpleNamespace(
            start=0.4, end=1.0, text=" Bonjour", avg_logprob=-0.2, no_speech_prob=0.01, words=[word]
        )
        bare = types.SimpleNamespace(start=1.2, end=1.4, text=" ...", avg_logprob=-1.0, no_speech_prob=0.5, words=None)
        return iter([seg, bare]), types.SimpleNamespace(language="fr", language_probability=0.9, duration=2.0)


@pytest.fixture
def model(monkeypatch, tmp_path):
    m = FakeModel()
    monkeypatch.setitem(W.M, "model", m)
    calls = []

    def ffmpeg(cmd, capture_output):
        calls.append(cmd)
        if "bad" in cmd[cmd.index("-i") + 1]:
            return subprocess.CompletedProcess(cmd, 1, b"", b"Invalid data found")
        return subprocess.CompletedProcess(cmd, 0, np.zeros(32000, np.int16).tobytes(), b"")

    real = W.decode
    monkeypatch.setattr(W, "decode", lambda path, clip=None: real(path, clip, run=ffmpeg))
    audio = tmp_path / "a.flac"
    audio.write_bytes(b"x")
    return m, calls, str(audio)


def test_transcribe_with_a_clip_decodes_only_the_clip_and_keeps_the_files_times(model):
    m, calls, audio = model
    r = W.run(dict(path=audio, language="fr", beam_size=1, clip=[10.0, 12.0], out_dir="/x"))
    cmd = calls[-1]
    assert cmd[cmd.index("-ss") + 1] == "10.000" and cmd[cmd.index("-to") + 1] == "12.000"
    assert (
        r["segments"][0]["start"] == 10.4
        and r["segments"][0]["words"][0]["start"] == 10.5
        and "words" not in r["segments"][1]
    )
    assert m.calls[-1][1]["beam_size"] == 1 and m.calls[-1][1]["language"] == "fr" and m.calls[-1][0] == 32000
    whole = W.run(dict(path=audio))
    assert "-ss" not in calls[-1] and whole["segments"][0]["start"] == 0.4 and whole["language"] == "fr"


def test_language_detection_alone(model):
    _, _, audio = model
    r = W.run(dict(path=audio, detect_language_only=True))
    assert r == dict(language="fr", language_probability=0.91, all_language_probs=dict(fr=0.91, en=0.05))


def test_bad_requests_are_param_errors(model, tmp_path):
    _, _, audio = model
    bad = tmp_path / "bad.flac"
    bad.write_bytes(b"x")
    for p, msg in (
        (dict(path="/nope.flac"), "no such audio"),
        (dict(path=audio, colour=1), "unknown params"),
        (dict(path=audio, clip=[1]), "clip is"),
        (dict(path=str(bad)), "could not decode"),
    ):
        with pytest.raises(protocol.ParamError, match=msg):
            W.run(p)


def test_load_picks_the_device(monkeypatch):
    made = []
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 0))
    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=lambda *a, **k: made.append((a, k)) or "m")
    )
    assert W.load(dict(name="small", options=dict(compute_type="float16")), "/models") == dict(device="cpu")
    assert (
        made[0][0] == ("small",)
        and made[0][1]["compute_type"] == "int8"
        and made[0][1]["download_root"] == os.path.join("/models", "whisper")
    )
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 1))
    assert W.load(dict(name="small"), "/models") == dict(device="cuda") and made[1][1]["compute_type"] == "float16"


def test_the_shared_loop_answers_every_kind_of_request(monkeypatch):
    def load(model, models_dir):
        return dict(loaded=model["name"])

    def run(params):
        if params.get("bad"):
            raise protocol.ParamError("bad param")
        if params.get("boom"):
            raise RuntimeError("CUDA error")
        return dict(echo=params)

    lines = [
        dict(op="load", model=dict(name="m"), models_dir="/m"),
        dict(op="run", params=dict(x=1)),
        dict(op="run", params=dict(bad=1)),
        dict(op="run", params=dict(boom=1)),
    ]
    stdin = io.StringIO(
        "\n".join(json.dumps(x) for x in lines)
        + "\nnot json\n"
        + json.dumps(dict(op="quit"))
        + "\n"
        + json.dumps(dict(op="run"))
        + "\n"
    )
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    protocol.serve(load, run, stdin=stdin, stdout=out)
    answers = [json.loads(x) for x in out.getvalue().splitlines()]
    assert answers[0] == dict(ok=True, result=dict(loaded="m")) and answers[1]["result"] == dict(echo=dict(x=1))
    assert (
        answers[2] == dict(ok=False, params=True, error="bad param")
        and answers[3]["ok"] is False
        and "CUDA error" in answers[3]["error"]
        and "params" not in answers[3]
    )
    assert len(answers) == 4  # nothing after quit
