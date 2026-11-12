"""stratamodels.evals.similarity: text (words and sound), images (perceptual hash, structure, colour), JSON, and picking the kind."""

import numpy as np
import pytest
from PIL import Image

from stratamodels.evals import similarity as S


def tx(t):
    return dict(segments=[dict(text=t)])


def test_phonetic_keeps_a_word_its_rules_would_empty():
    assert S.phonetic("gh") == "gh" and S.phonetic("wh") == "wh" and S.text_similarity("gh wh", "gh wh") == 1.0


def test_phonetic_keys_match_words_that_sound_alike():
    assert (
        S.phonetic("there") == S.phonetic("their")
        and S.phonetic("four") == S.phonetic("for")
        and S.phonetic("knight") == S.phonetic("night")
    )
    assert (
        S.phonetic("phone") == S.phonetic("fone")
        and S.phonetic("cat") != S.phonetic("dog")
        and S.phonetic("42") == "42"
        and S.phonetic("'") == "'"
    )


def test_text_similarity_ranks_what_a_listener_would():
    same = S.similarity(tx(" Anders' face grew red."), tx("anders' face grew red"))
    spelling = S.similarity(tx("their bikes are over there"), tx("there bikes are over their"))
    words = S.similarity(tx("their bikes are over there"), tx("the hikes are over here"))
    assert same == 1.0 and 1.0 > spelling > words > 0 and S.similarity("", "", "text") == 1.0
    assert S.edit_similarity([], []) == 1.0 and S.edit_similarity("abc", "abd") == pytest.approx(2 / 3)


def test_image_similarity_orders_pictures(tmp_path):
    def pic(name, f):
        im = Image.new("RGB", (128, 128))
        px = im.load()
        for x in range(128):
            for y in range(128):
                px[x, y] = f(x, y)
        p = tmp_path / name
        im.save(p)
        return dict(path=str(p))

    a = pic("a.png", lambda x, y: (x * 2, y * 2, 100))
    a2 = pic("a2.png", lambda x, y: (min(x * 2 + 3, 255), y * 2, 100))
    b = pic("b.png", lambda x, y: (255 - y * 2, 30, x * 2))
    near, far = S.similarity(a, a2), S.similarity(a, b)
    assert S.similarity(a, a) == 1.0 and 1.0 > near > far and S.kind_of(a, b) == "image"
    assert (
        S.image_similarity(a["path"], str(tmp_path / "missing.png")) == 0.0
        and S.similarity(a, dict(path="x.txt"), "image") == 0.0
    )


def test_image_similarity_without_pillow_compares_bytes(tmp_path, monkeypatch):
    p, q = tmp_path / "p.png", tmp_path / "q.png"
    p.write_bytes(b"x")
    q.write_bytes(b"y")

    def no_pil(*a):
        raise ImportError("PIL")

    monkeypatch.setattr(S, "_image_features", no_pil)
    assert S.image_similarity(str(p), str(q)) == 0.0 and S.image_similarity(str(p), str(tmp_path / "p.png")) == 1.0
    assert S.image_similarity(str(p), str(tmp_path / "none.png")) == 0.0


def test_json_and_kinds():
    assert (
        S.similarity(dict(a=1, seconds=2.0), dict(a=1, seconds=9.0)) == 1.0
        and 0 < S.similarity(dict(a=1), dict(a=2)) < 1
    )
    assert (
        S.kind_of(tx("a"), tx("b")) == "text" and S.kind_of("a", "b") == "text" and S.kind_of(dict(a=1), [1]) == "json"
    )
    with pytest.raises(ValueError, match="no similarity"):
        S.similarity("a", "b", "smell")


def _wav(path, freqs, seconds=1.0, sr=24000):
    import wave

    t = np.arange(int(sr * seconds)) / sr
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs) * 0.5
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return str(path)


def test_audio_similarity_hears_voice_colour_and_length(tmp_path):
    low, low2, high, short = (
        _wav(tmp_path / "a.wav", [150, 300]),
        _wav(tmp_path / "b.wav", [150, 300]),
        _wav(tmp_path / "c.wav", [2000, 4000]),
        _wav(tmp_path / "d.wav", [150, 300], 0.5),
    )
    assert (
        S.similarity(dict(path=low), dict(path=low2)) == pytest.approx(1.0, abs=1e-6)
        and S.kind_of(dict(path=low), dict(path=high)) == "audio"
    )
    assert S.similarity(dict(path=low), dict(path=high)) < S.similarity(dict(path=low), dict(path=short)) < 1.0
    tiny = _wav(tmp_path / "e.wav", [200], 0.01)
    assert S.audio_similarity(tiny, low) == pytest.approx(0.01, abs=1e-3)
    (tmp_path / "bad.wav").write_bytes(b"nope")
    assert S.audio_similarity(str(tmp_path / "bad.wav"), low) == 0.0
    assert S.similarity(dict(path=low), dict(path="x.png"), "audio") == 0.0 and "audio" in S.KINDS


def test_text_similarity_reads_answers_and_captions_not_only_transcripts():
    assert (
        S.similarity(dict(text="a dog", model="m"), dict(text="a cat", model="m"), "text") < 1
    )  # chat answers: their text
    assert (
        S.similarity(
            dict(text="{}", json=dict(caption="a dog on a beach")),
            dict(caption="a dog on a beach", tags=["dog"]),
            "text",
        )
        == 1.0
    )
    assert (
        S.as_text(None) == ""
        and S.as_text("hi") == "hi"
        and S.as_text(dict(a=1)) == ""
        and S.as_text(tx("hello")) == "hello"
    )


def test_video_similarity_compares_the_stills_frame_for_frame(tmp_path):
    from PIL import Image

    def png(name, colour):
        p = tmp_path / name
        Image.new("RGB", (64, 48), colour).save(p)
        return str(p)

    a = dict(
        path="/g/video-a.mp4",
        stills=[png("a1.png", (200, 30, 30)), png("a2.png", (30, 30, 200))],
        crops=[dict(id="face", path="/g/a.face.mp4")],
        peak_vram_mb=1.0,
    )
    same = dict(path="/g/video-b.mp4", stills=list(a["stills"]), crops=[], peak_vram_mb=9.0)
    other = dict(path="/g/video-c.mp4", stills=[png("c1.png", (30, 200, 30))])
    assert S.kind_of(a, same) == "video" and S.video_similarity(a, same) == 1.0
    assert S.similarity(a, same) == 1.0 and S.similarity(a, other) < S.similarity(
        a, same
    )  # only the first stills are compared: the shorter list decides
    assert S.similarity(a, dict(path="/g/v.mp4")) == 0.0 and S.video_similarity("x", a) == 0.0
    assert (
        S.video_path(dict(path="x.MOV")) == "x.MOV" and S.video_path(dict(path="x.png")) is None and "video" in S.KINDS
    )
