"""The image-kind runtime (role `image-kind`): its kinds, the rules engine on drawn stand-ins for each kind of picture, and the vision-
language engine with the model's answers given in the handle. CLIP itself runs in tests/gpu (open_clip is in the gpu group)."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from stratamodels.runtimes import RunError, imagekind


class M:
    def __init__(self, **options):
        self.key, self.options = "image-kind-test", options


def save(tmp_path, name, a):
    p = tmp_path / name
    Image.fromarray(a.astype(np.uint8)).save(p)
    return str(p)


def paper(h=300, w=400):
    return np.full((h, w, 3), (236, 226, 200), np.float64)


def test_every_kind_has_a_group_and_an_unknown_one_is_other():
    assert {g for g, _, _ in imagekind.KINDS.values()} == set(imagekind.GROUPS)
    assert imagekind.kind_of("woodcut") == dict(kind="woodcut", group="illustration", line_art=True)
    assert imagekind.kind_of("halftone")["group"] == "photo"
    assert imagekind.kind_of("a sketch of a dog") == dict(kind="unknown", group="other", line_art=False)


def test_rules_tell_blank_lines_colour_tone_and_screens_apart(tmp_path):
    h = imagekind.load(M(engine="rules"), "")
    run = lambda a: imagekind.run(h, {"image": save(tmp_path, "x.png", a)})
    assert run(paper())["kind"] == "blank"
    lines = paper()
    lines[20:280:6, 40:360] = 20  # hatching: thin black lines on the paper
    lines[150:160, :] = 20
    r = run(lines)
    assert r["group"] == "illustration" and r["line_art"] and not r["colour"] and r["seconds"] >= 0
    colour = paper()
    colour[50:250, 50:350] = (200, 40, 40)
    colour[100:200, 100:300] = (40, 60, 190)
    assert run(colour)["colour"] is True
    yy, xx = np.mgrid[0:300, 0:400]
    tone = paper() * (0.3 + 0.7 * (xx / 400.0))[..., None]  # a smooth gradient: a photograph's continuous tone
    assert run(tone)["group"] == "photo"
    screen = np.where(((yy % 6) < 3) & ((xx % 6) < 3), 30.0, 220.0)  # a regular dot grid, the mark of a halftone
    screen = np.where((xx < 200)[..., None], np.repeat(screen[..., None], 3, -1), 128.0 + 0 * paper())
    assert run(screen)["kind"] in ("halftone", "photograph", "tonal-drawing")


def test_features_are_measured_apart_from_the_paper_tint():
    a = paper().astype(np.uint8)
    f = imagekind.features(a)
    assert f["colour"] < 0.05 and f["ink"] < 0.05  # yellowed paper is not colour, and not ink
    assert set(f) == {"mid", "ink", "colour", "screen", "edges", "spread"}


def test_params_and_options_are_checked(tmp_path):
    with pytest.raises(RunError, match="engine is rules, clip or vlm"):
        imagekind.load(M(engine="magic"), "")
    with pytest.raises(RunError, match="names no vision-language model"):
        imagekind.load(M(engine="vlm"), "")
    h = imagekind.load(M(engine="rules"), "")
    with pytest.raises(RunError, match="unknown params"):
        imagekind.run(h, {"image": "x", "colour": True})
    with pytest.raises(RunError, match="no such image"):
        imagekind.run(h, {"image": str(tmp_path / "none.png")})


def vlm(answers, **options):
    h = imagekind.load(M(engine="vlm", vlm="claude-haiku-5-5-api", **options), "")
    asked = []

    def call(p):
        asked.append(p)
        return answers.pop(0)

    h["vlm"] = call
    return h, asked


def test_vlm_answer_is_mapped_to_its_group(tmp_path):
    p = save(tmp_path, "x.png", paper())
    h, asked = vlm(
        [{"json": {"kind": "wood-engraving", "colour": False, "why": "white lines cut from black"}, "cost_usd": 0.002}]
    )
    out = imagekind.run(h, {"image": p, "context": "Punch and Judy, 1828"})
    assert (
        asked[0]["preset"] == "image-kind-v1"
        and asked[0]["context"] == "Punch and Judy, 1828"
        and asked[0]["image"] == p
    )
    assert (
        out["kind"] == "wood-engraving"
        and out["group"] == "illustration"
        and out["line_art"]
        and out["cost_usd"] == 0.002
    )
    assert out["why"] == "white lines cut from black"
    h, _ = vlm([{"text": '```json\n{"kind": "halftone", "colour": true}\n```'}])
    out = imagekind.run(h, {"image": p})
    assert (out["kind"], out["group"], out["colour"]) == ("halftone", "photo", True) and "cost_usd" not in out
    h, _ = vlm(["a woodcut, I think"])
    with pytest.raises(RunError, match="not JSON"):
        imagekind.run(h, {"image": p})


def test_the_preset_names_every_kind():
    from stratamodels.runtimes import llama

    text, schema = llama.preset("image-kind-v1")
    assert all(f"- {k}:" in text for k in imagekind.KINDS) and "{context}" in text
    assert schema["properties"]["kind"]["enum"] == list(imagekind.KINDS)


def test_unload_frees_the_reader():
    h, _ = vlm([])
    freed = []
    h["vlm_unload"] = lambda: freed.append(True)
    imagekind.unload(h)
    assert freed == [True] and h == {}


def test_drawn_kinds_stay_illustrations(tmp_path):
    im = Image.new("RGB", (300, 200), (240, 235, 220))
    d = ImageDraw.Draw(im)
    for x in range(10, 290, 4):
        d.line((x, 10, x + 40, 190), fill=(10, 10, 10), width=1)
    p = tmp_path / "e.png"
    im.save(p)
    out = imagekind.run(imagekind.load(M(engine="rules"), ""), {"image": str(p)})
    assert out["group"] == "illustration" and out["features"]["edges"] > 0.05


def test_rules_name_halftones_photographs_and_woodcuts_by_their_features(monkeypatch):
    base = dict(mid=0.1, ink=0.1, colour=0.0, screen=0.0, edges=0.1, spread=0.5)
    cases = [
        (dict(screen=0.8, mid=0.4), "halftone"),
        (dict(mid=0.6, edges=0.02), "photograph"),
        (dict(mid=0.6, edges=0.1), "tonal-drawing"),
        (dict(ink=0.5), "woodcut"),
        (dict(edges=0.2), "engraving"),
        (dict(colour=0.5, mid=0.2, edges=0.1), "hand-coloured"),
        (dict(colour=0.5, mid=0.5, edges=0.07), "colour-print"),
        (dict(colour=0.5, mid=0.5, edges=0.01), "photograph"),
    ]
    for change, kind in cases:
        monkeypatch.setattr(imagekind, "features", lambda rgb, c=change: dict(base, **c))
        assert imagekind.rules(np.zeros((2, 2, 3), np.uint8))["kind"] == kind, change


def test_the_reader_is_asked_through_the_delegate(tmp_path, monkeypatch):
    from stratamodels.runtimes import delegate

    got = []
    monkeypatch.setattr(
        delegate,
        "caller",
        lambda key, d, via, opts=None: got.append(key) or ((lambda p: {"json": {"kind": "woodcut"}}), lambda: None),
    )
    h = imagekind.load(M(engine="vlm", vlm="claude-haiku-5-5-api"), "")
    assert imagekind.run(h, {"image": save(tmp_path, "x.png", paper())})["kind"] == "woodcut"
    assert got == ["claude-haiku-5-5-api"]
