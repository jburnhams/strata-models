"""Pure consonant sounds cut from a carrier syllable (stratamodels.workers.phonics, shared by the TTS worker and the eval metrics), on
synthetic carriers: a hiss, a hum or a click, then a vowel made of harmonics with formant peaks. The real Kokoro carriers: tests/gpu."""

import numpy as np
import pytest
from sounds import SR, carrier, hiss, hum, silence, vowel

from stratamodels.workers import phonics as P


@pytest.mark.parametrize(
    ("sound", "kind"),
    [
        ("s", "hiss"),
        ("ʃ", "hiss"),
        ("m", "nasal"),
        ("ɹ", "liquid"),
        ("z", "buzz"),
        ("p", "stop"),
        ("tʃ", "affricate"),
        ("w", "brief"),
        ("ks", "brief"),
        ("iː", "vowel"),
        ("ɔɪ", "vowel"),
        ("x", "other"),
    ],
)
def test_kind(sound, kind):
    assert P.kind(sound) == kind


def test_the_table_is_the_25_consonants_of_letters_and_sounds():
    assert len(P.CONSONANTS) == 25 and len(P.KINDS) == 25
    assert all(P.kind(ipa) == k for ipa, k in P.CONSONANTS.values())


def test_a_hiss_is_cut_from_before_the_vowel_and_held():
    x = carrier(hiss(0.12), vowel(0.25))
    a, b = P.span(x, SR, "s")
    assert 0.09 * SR <= a <= 0.12 * SR and 0.19 * SR <= b <= 0.225 * SR  # the hiss is 0.10-0.22 s
    y, info = P.cut(x, SR, "s", hold_s=0.5)
    m = P.measure(y, SR)
    assert info["kind"] == "hiss" and info["held_s"] == 0.5 and abs(len(y) / SR - 0.5) < 0.01
    assert m["high"] > 0.9 and m["vowel_leak"] == 0 and m["voiced"] < 0.2
    assert P.pure("s", m) == (True, "no hiss")


def test_the_whole_carrier_is_no_pure_sound():
    m = P.measure(carrier(hiss(0.05), vowel(0.25)), SR)
    assert m["vowel_leak"] > 0.5 and P.pure("s", m) == (False, "a vowel in it")


def test_a_nasal_keeps_its_hum_and_its_pitch_when_held():
    x = carrier(hum(0.08), vowel(0.25))
    a, b = P.span(x, SR, "m")
    assert b <= 0.185 * SR and b - a >= 0.04 * SR
    y, _ = P.cut(x, SR, "m", hold_s=0.4)
    m = P.measure(y, SR)
    assert m["voiced"] > 0.8 and m["low"] > 0.9 and m["vowel_leak"] == 0 and P.pure("m", m)[0]
    assert P.period(y[SR // 10 : SR // 10 + 2400], SR) == 120  # 200 Hz: grains laid at whole periods keep the pitch


def test_a_stop_keeps_its_burst_alone():
    burst = hiss(0.015, level=0.3)
    x = carrier(burst, silence(0.03), vowel(0.25))
    y, info = P.cut(x, SR, "t")
    assert info["kind"] == "stop" and "held_s" not in info and len(y) <= 0.15 * SR
    m = P.measure(y, SR)
    assert m["high"] > 0.8 and P.pure("t", m)[0]


def test_hold_0_keeps_a_sound_as_said():
    y, info = P.cut(carrier(hiss(0.12), vowel(0.25)), SR, "f", hold_s=0)
    assert "held_s" not in info and len(y) < 0.15 * SR


def test_no_vowel_no_cut():
    with pytest.raises(ValueError, match="no vowel"):
        P.cut(silence(0.3), SR, "s")
    assert P.onset(np.ones(100, np.float32), SR) is None


def test_hold_lengthens_noise_and_tones():
    n = P.hold(hiss(0.05), SR, 0.5)
    assert len(n) == SR // 2 and P.measure(n, SR)["high"] > 0.9
    assert P.hold(np.zeros(2, np.float32), SR, 0.1).shape == (2400,)


def test_period_of_a_tone_and_of_noise():
    assert P.period(hum(0.05, f0=150), SR) == 160
    assert P.period(hiss(0.05), SR) is None
    assert P.period(np.zeros(1000), SR) is None


def test_measure_of_silence():
    assert P.measure(np.zeros(100), SR)["sound_s"] == 0
    assert P.pure("s", P.measure(np.zeros(100), SR)) == (False, "no sound")


@pytest.mark.parametrize(
    ("sound", "m", "ok"),
    [
        ("iː", dict(sound_s=0.3, voiced=0.9, vowel_leak=0.2, high=0.0), True),
        ("iː", dict(sound_s=0.3, voiced=0.1, vowel_leak=0.0, high=0.0), False),
        ("z", dict(sound_s=0.3, voiced=0.0, vowel_leak=0.0, high=0.8), True),
        ("v", dict(sound_s=0.3, voiced=0.1, vowel_leak=0.0, high=0.1), False),
        ("p", dict(sound_s=0.5, voiced=0.0, vowel_leak=0.0, high=0.8), False),
        ("w", dict(sound_s=0.05, voiced=0.5, vowel_leak=0.0, high=0.0), True),
        ("l", dict(sound_s=0.4, voiced=0.2, vowel_leak=0.0, high=0.0), False),
    ],
)
def test_pure_by_kind(sound, m, ok):
    assert P.pure(sound, m)[0] is ok


def test_fade_on_a_tiny_clip():
    assert len(P.fade(np.ones(3, np.float32), SR)) == 3


# ---- the eval's metrics (evals/metrics.py sound_pure, vowel_leak) and dataset builder (scripts/evals/phonics_sounds.py) ------------------
def wav(path, x):
    import soundfile as sf

    sf.write(str(path), x, SR, subtype="PCM_16")
    return str(path)


def test_sound_metrics_measure_the_file(tmp_path):
    from stratamodels.evals import metrics as M

    item = dict(input=dict(sound="s", text="/s/"))
    held, _ = P.cut(carrier(hiss(0.12), vowel(0.25)), SR, "s")
    good = dict(path=wav(tmp_path / "a.wav", held))
    bad = dict(path=wav(tmp_path / "b.wav", carrier(hiss(0.05), vowel(0.25))))
    pure, leak = M.METRICS["sound_pure"].score, M.METRICS["vowel_leak"].score
    assert pure(item, good, None) is True and pure(item, bad, None) is False and leak(item, bad, None) > 0.5
    assert pure(item, None, "boom") is False and pure(dict(input=dict(text="hi")), None, "boom") is None
    assert pure(item, dict(path=str(tmp_path / "gone.wav")), None) is None and leak(item, {}, None) is None
    summary = M.summarise("sound_pure", [item, item], [(good, None), (bad, None)])
    assert summary["value"] == 0.5 and summary["n"] == 2


def builder():
    import importlib.util
    import os

    root = os.path.join(os.path.dirname(__file__), "..", "..")
    spec = importlib.util.spec_from_file_location(
        "phonics_sounds", os.path.join(root, "scripts", "evals", "phonics_sounds.py")
    )
    assert spec and spec.loader
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    return b


def test_the_dataset_is_the_25_consonants():
    its = builder().items()
    assert len(its) == 25 and len({i["id"] for i in its}) == 25
    assert its[0] == dict(
        id="c01-s",
        input=dict(text="/s/ (s)", sound="s"),
        reference=dict(sound="s"),
        tags=dict(kind="hiss", letters="s"),
    )
    assert {i["input"]["sound"] for i in its} == {ipa for ipa, _ in P.CONSONANTS.values()}


def test_a_persons_recordings_are_found_by_letters_or_ipa_trimmed_and_listed(tmp_path):
    import json

    import soundfile as sf

    b = builder()
    rec = tmp_path / "rec"
    rec.mkdir()
    sf.write(str(rec / "sh.wav"), np.stack([carrier(hiss(0.2))] * 2, 1), 48000)  # stereo, any rate
    wav(rec / "θ.wav", carrier(hiss(0.2)))
    wav(rec / "th-then.wav", carrier(hum(0.2)))
    wav(rec / "ck.wav", carrier(hiss(0.02)))
    wav(
        rec / "j.wav", carrier(hiss(0.03))
    )  # the j of jam (/dʒ/), not /j/: with no y.wav, the y of yes stays unrecorded
    (rec / "notes.txt").write_text("not a recording")
    rows, missing = b.recordings(str(rec), str(tmp_path / "out"), str(tmp_path / "person.jsonl"))
    assert (
        sorted(r["key"] for r in rows) == sorted(["ʃ", "θ", "ð", "k", "dʒ"]) and len(missing) == 20 and "y" in missing
    )
    listed = [json.loads(line) for line in (tmp_path / "person.jsonl").read_text(encoding="utf8").splitlines()]
    assert [r["key"] for r in listed] == [r["key"] for r in rows]
    x, _sr = sf.read(listed[0]["output"]["path"])
    assert x.ndim == 1 and len(x) < len(carrier(hiss(0.2)))  # mono, the silence trimmed to 50 ms a side


def test_recording_names_letters_first_and_ipa_only_where_unambiguous():
    b = builder()
    assert b.names("y", "j") == ["y"] and b.names("j", "dʒ") == ["j", "dʒ"] and b.names("sh", "ʃ") == ["sh", "ʃ"]
    assert b.names("th (then)", "ð") == ["th (then)", "th-then", "ð"] and b.names("c/k/ck", "k") == ["c", "k", "ck"]
