"""The text-to-speech venv-worker (stratamodels/workers/tts.py) without a model: a fake Kokoro that records what it was fed. The real
engines: tests/gpu (needs the tts venvs). Covers `phonemes`, IPA said as written (an isolated sound such as /ɔɪ/ that no word spells)."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

WORKERS = os.path.join(os.path.dirname(__file__), "..", "..", "src", "stratamodels", "workers")
sys.path.insert(0, os.path.abspath(WORKERS))
import tts as T  # noqa: E402
from protocol import ParamError  # noqa: E402

SR = 24000


class FakeKokoro:
    def __init__(self):
        self.calls = []

    def get_voices(self):
        return ["bf_emma", "bm_lewis", "af_heart"]

    def create(self, text, voice, speed, lang, is_phonemes=False):
        self.calls.append(dict(text=text, voice=voice, speed=speed, lang=lang, is_phonemes=is_phonemes))
        t = np.arange(int(SR * 0.4)) / SR
        x = np.concatenate([np.zeros(SR // 10), 0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(SR // 10)])
        return x.astype(np.float32), SR


@pytest.fixture
def kokoro(monkeypatch):
    m = FakeKokoro()
    monkeypatch.setattr(T, "E", dict(engine="kokoro", name="Kokoro", options={}, m=m))
    return m


def test_text_goes_through_kokoros_own_phonemizer(kokoro, tmp_path):
    r = T.run(dict(text="ship", voice="bf_emma", out_dir=str(tmp_path)))
    assert kokoro.calls == [dict(text="ship", voice="bf_emma", speed=1.0, lang="en-gb", is_phonemes=False)]
    assert os.path.isfile(r["path"]) and r["voice"] == "bf_emma" and r["duration_s"] > 0.3


def test_phonemes_are_fed_as_written_and_text_is_only_the_label(kokoro, tmp_path):
    r = T.run(dict(text="/iː/", phonemes="iː", voice="bf_emma", out_dir=str(tmp_path)))
    assert kokoro.calls == [dict(text="iː", voice="bf_emma", speed=1.0, lang="en-gb", is_phonemes=True)]
    assert os.path.isfile(r["path"])


def test_phonemes_alone_need_no_text(kokoro, tmp_path):
    T.run(dict(phonemes="ʃ", out_dir=str(tmp_path)))
    assert kokoro.calls[0]["is_phonemes"] and kokoro.calls[0]["voice"] == "bm_lewis"


@pytest.mark.parametrize(
    ("params", "says"),
    [
        (dict(), "give text"),
        (dict(text="  "), "give text"),
        (dict(phonemes=""), "phonemes is a string of IPA"),
        (dict(phonemes=7), "phonemes is a string of IPA"),
        (dict(phonemes="iː", wpm=150), "wpm and fit_s"),
        (dict(phonemes="iː", fit_s=1.0), "wpm and fit_s"),
    ],
)
def test_refused(kokoro, tmp_path, params, says):
    with pytest.raises(ParamError, match=says):
        T.run(dict(params, out_dir=str(tmp_path)))
    assert kokoro.calls == []


def test_only_kokoro_takes_phonemes(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "E", dict(engine="chatterbox", name="Chatterbox", options={}, m=object()))
    with pytest.raises(ParamError, match="only the kokoro engine"):
        T.run(dict(phonemes="iː", out_dir=str(tmp_path)))


def _tone(seconds, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return np.concatenate([np.zeros(sr // 10), 0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(sr // 10)]).astype(
        np.float32
    )


class _Tensor:  # what Chatterbox returns: a torch tensor, (1, n)
    def __init__(self, x):
        self.x = x

    def squeeze(self, _):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.x


class FakeSampled:
    """Chatterbox (generate) and Qwen3-TTS (generate_*): sampled models, whose length differs by seed."""

    sr = SR

    def __init__(self):
        self.calls = []
        self.n = 0

    def _say(self, kw):
        self.calls.append(kw)
        self.n += 1
        return _tone(0.3 + 0.1 * self.n)

    def generate(self, text, **kw):
        return _Tensor(self._say(dict(kw, text=text)))

    def generate_custom_voice(self, **kw):
        return [self._say(kw)], SR

    def generate_voice_design(self, **kw):
        return [self._say(kw)], SR

    def generate_voice_clone(self, **kw):
        return [self._say(kw)], SR


@pytest.fixture
def stretched(monkeypatch):
    """ffmpeg's atempo, which the unit suite does not run (no external processes): the factors asked for, the file kept."""
    factors = []
    monkeypatch.setattr(T, "stretch", lambda path, speed: factors.append(speed))
    return factors


@pytest.fixture
def sampled(monkeypatch, stretched):
    def use(engine, options=None):
        m = FakeSampled()
        m.stretched = stretched
        monkeypatch.setattr(T, "E", dict(engine=engine, name=engine, options=options or {}, m=m))
        return m

    return use


@pytest.fixture
def ref(tmp_path):
    p = tmp_path / "ref.wav"
    p.write_bytes(b"RIFF")
    return str(p)


def test_kokoro_respeaks_at_the_speed_that_lands_on_a_target(kokoro, tmp_path):
    r = T.run(dict(text="Two hundred kilometres behind me.", wpm=160, out_dir=str(tmp_path)))
    assert len(kokoro.calls) == 2 and kokoro.calls[1]["speed"] != 1.0
    assert r["fit"] in ("ok", "long", "short") and "speed" in r and "target_s" in r


def test_an_unknown_kokoro_voice_is_refused(kokoro, tmp_path):
    with pytest.raises(ParamError, match="no Kokoro voice"):
        T.run(dict(text="hi", voice="xx_nobody", out_dir=str(tmp_path)))


@pytest.mark.parametrize(
    ("params", "says"),
    [
        (dict(text="hi", speed=3), "speed is between"),
        (dict(text="hi", wpm=20), "wpm is between"),
        (dict(text="hi", fit_s=0), "fit_s is a length"),
        (dict(text="...", wpm=150), "nothing to fit"),
    ],
)
def test_pace_refused(kokoro, tmp_path, params, says):
    with pytest.raises(ParamError, match=says):
        T.run(dict(params, out_dir=str(tmp_path)))


def test_chatterbox_with_and_without_a_reference(sampled, ref, tmp_path):
    m = sampled("chatterbox")
    assert T.run(dict(text="hello", out_dir=str(tmp_path)))["voice"] == "built-in"
    assert T.run(dict(text="hello", voice=ref, out_dir=str(tmp_path)))["voice"] == ref
    assert m.calls[1]["audio_prompt_path"] == ref


def test_cloning_needs_a_reference_that_exists(sampled, tmp_path):
    sampled("chatterbox-turbo")
    with pytest.raises(ParamError, match="clones a voice"):
        T.run(dict(text="hello", out_dir=str(tmp_path)))
    with pytest.raises(ParamError, match="no such reference"):
        T.run(dict(text="hello", voice=str(tmp_path / "missing.wav"), out_dir=str(tmp_path)))


def test_sampled_models_keep_the_take_nearest_a_target(sampled, ref, tmp_path):
    m = sampled("chatterbox-turbo")
    r = T.run(dict(text="One two three four five six.", voice=ref, fit_s=1.0, tries=3, out_dir=str(tmp_path)))
    assert len(m.calls) == 3 and "stretch" in r and r["seed"] in (0, 1, 2)


def test_qwen3_engines(sampled, ref, tmp_path):
    m = sampled("qwen3-custom", dict(voice="Ryan"))
    assert T.run(dict(text="hello", instruct="calmly", out_dir=str(tmp_path)))["voice"] == "Ryan"
    assert m.calls[0]["speaker"] == "Ryan" and m.calls[0]["instruct"] == "calmly"
    sampled("qwen3-design")
    with pytest.raises(ParamError, match="needs instruct"):
        T.run(dict(text="hello", out_dir=str(tmp_path)))
    assert T.run(dict(text="hello", instruct="a calm voice", out_dir=str(tmp_path)))["voice"] == "designed"
    m = sampled("qwen3-clone")
    words = tmp_path / "ref.txt"
    words.write_text("the reference's words\n")
    T.run(dict(text="hello", voice=ref, voice_text_file=str(words), out_dir=str(tmp_path)))
    assert m.calls[0]["ref_text"] == "the reference's words" and m.calls[0]["x_vector_only_mode"] is False


def test_a_slower_speed_on_a_sampled_model_is_a_stretch(sampled, tmp_path):
    m = sampled("chatterbox")
    r = T.run(dict(text="hello there", speed=0.8, out_dir=str(tmp_path)))
    assert m.stretched == [pytest.approx(0.8)]
    assert r["duration_s"] > 0.3


def test_stretch_runs_ffmpeg_atempo_and_replaces_the_file(monkeypatch, tmp_path):
    """The command line and both outcomes, with subprocess.run standing in for ffmpeg (tests/gpu runs the real one)."""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"old")
    seen = []

    def ok(cmd, **kw):
        seen.append(cmd)
        Path(cmd[-1]).write_bytes(b"new")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(T.subprocess, "run", ok)
    T.stretch(str(wav), 0.8)
    assert seen[0][0] == "ffmpeg" and "atempo=0.8000" in seen[0]
    assert wav.read_bytes() == b"new"
    monkeypatch.setattr(T.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=1, stderr="bad input"))
    with pytest.raises(RuntimeError, match="atempo failed: bad input"):
        T.stretch(str(wav), 1.2)


def test_an_unknown_engine(sampled, tmp_path):
    sampled("nope")
    with pytest.raises(ParamError, match="unknown engine"):
        T.run(dict(text="hello", out_dir=str(tmp_path)))
