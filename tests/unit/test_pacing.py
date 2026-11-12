"""Pace and pauses of synthetic speech (stratamodels.workers.pacing, shared by the TTS worker and the eval metrics), on synthetic tones."""

import wave

import numpy as np
import pytest

from stratamodels.evals import metrics as M
from stratamodels.workers import pacing as P

SR = 16000


def tone(s):
    return (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(SR * s)) / SR)).astype(np.float32)


def quiet(s):
    return np.zeros(int(SR * s), np.float32)


def test_words_as_said_and_their_syllables():
    assert (
        P.number_words(367) == "three hundred and sixty seven"
        and P.number_words(20) == "twenty"
        and P.number_words(2000) == "two thousand"
        and P.number_words(1200000) == "one two zero zero zero zero zero"
    )
    assert P.spoken("I took a DNF, at 3.") == ["i", "took", "a", "d", "n", "f", "at", "three"] and P.spoken(
        "the WWW"
    ) == ["the", "double", "you", "double", "you", "double", "you"]
    assert (
        P.syllables("Organizers were so supportive.") == 9
        and P.syllables("race") == 1
        and P.syllables("little") == 2
        and P.syllables("") == 0
    )
    assert P.target_seconds("one two three", 60) == pytest.approx(3 / (60 * 1.4 / 60))


def test_pauses_are_found_and_cut():
    x = np.concatenate([quiet(0.5), tone(1.0), quiet(2.0), tone(1.0), quiet(0.3), tone(0.5), quiet(0.7)])
    a, b, ps = P.pauses(x, SR)
    assert (
        a == pytest.approx(0.5, abs=0.02)
        and b == pytest.approx(5.3, abs=0.02)
        and [round(p, 1) for p in ps] == [2.0, 0.3]
    )
    y = P.tighten(x, SR, max_pause=0.4, edge=0.05)
    a, b, ps = P.pauses(y, SR)
    assert len(y) / SR == pytest.approx(0.05 + 1.0 + 0.4 + 1.0 + 0.3 + 0.5 + 0.05, abs=0.05) and max(
        ps
    ) == pytest.approx(0.4, abs=0.03)
    assert len(P.tighten(x, SR, max_pause=None, edge=0.0)) / SR == pytest.approx(
        4.8, abs=0.03
    )  # pauses kept, only the ends trimmed
    assert np.array_equal(P.tighten(quiet(1.0), SR), quiet(1.0)) and P.pauses(quiet(1.0), SR) == (0.0, 1.0, [])
    assert P.frames(np.zeros(3, np.float32), SR)[1].size == 0


def test_measure_and_wav_round_trip(tmp_path):
    x = np.concatenate([tone(1.0), quiet(0.8), tone(1.0)])
    m = P.measure("one two three four five six seven", x, SR)
    assert (
        m["long_pauses"] == 1
        and m["max_pause_s"] == pytest.approx(0.8, abs=0.03)
        and m["rate_wpm"] == pytest.approx(8 / 2.8 * 60 / 1.4, rel=0.03)
    )
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.repeat(x[:, None], 2, 1) * 32767).astype("<i2").tobytes())
    y, sr = P.read_wav(str(p))
    assert sr == SR and len(y) == len(x)
    it = dict(input=dict(text="one two three four five six seven"))
    assert (
        M.pace_of(it, dict(path=str(p)), None)["long_pauses"] == 1
        and M.pace_of(it, dict(rate_wpm=150.0), None)["rate_wpm"] == 150.0
    )
    assert (
        M.pace_of(it, None, "boom") is None
        and M.pace_of(it, dict(path=str(tmp_path / "missing.wav")), None) is None
        and M.pace_of(dict(input={}), dict(path=str(p)), None) is None
    )


def test_pace_metrics():
    its = [dict(input=dict(text="x"))] * 3
    even = [(dict(rate_wpm=160.0, long_pauses=0, max_pause_s=0.2), None)] * 3
    uneven = [(dict(rate_wpm=r, long_pauses=1, max_pause_s=0.9), None) for r in (100.0, 160.0, 220.0)]
    assert M.METRICS["pace_spread"].dataset(its, even) == 0 and M.METRICS["pace_spread"].dataset(
        its, uneven
    ) == pytest.approx(np.std([100, 160, 220]) / 160)
    assert np.isnan(M.METRICS["pace_spread"].dataset(its[:1], even[:1]))
    assert (
        M.METRICS["long_pauses"].score(its[0], *uneven[0]) == 1.0
        and M.METRICS["max_pause_s"].score(its[0], *uneven[0]) == 0.9
        and M.METRICS["long_pauses"].score(its[0], None, "x") is None
    )
    assert M.METRICS["max_pause_s"].score(its[0], None, "x") is None
    assert (
        M.METRICS["fit_err"].score(its[0], dict(speech_s=2.1, target_s=2.0), None) == pytest.approx(0.05)
        and M.METRICS["fit_err"].score(its[0], dict(speech_s=2.0), None) is None
    )
