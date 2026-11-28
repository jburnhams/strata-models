"""The vectorise runtime with the real potrace command (installed in CI for this: apt, brew, and the project's own Windows build in
integration.yml): a drawn plate traced and drawn back, the stacked tones, ink under washes, and llm-clean's limit."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from stratamodels.evals import vector
from stratamodels.runtimes import RunError, vectorise


def plate(path, colour=False):
    im = Image.new("RGB", (360, 260), (236, 226, 200))
    d = ImageDraw.Draw(im)
    d.ellipse((40, 30, 200, 220), outline=(20, 20, 20), width=6)
    for x in range(220, 340, 8):  # hatching: parallel lines a tracer must keep apart
        d.line((x, 30, x + 20, 230), fill=(25, 25, 25), width=2)
    d.rectangle(
        (90, 100, 150, 160) if not colour else (60, 60, 330, 220),
        fill=(120, 120, 120) if not colour else (235, 140, 140),
    )  # a light wash, as a hand-colourist laid it
    im.save(path)
    return str(path)


class M:
    def __init__(self, **options):
        self.key, self.options = "vectorise-potrace", options


@pytest.fixture
def need_potrace():
    assert vectorise.binary(), "potrace must be installed for the integration tests (see integration.yml)"


def test_potrace_traces_the_plate_and_it_scores_well(tmp_path, need_potrace):
    p = plate(tmp_path / "p.png")
    h = vectorise.load(M(engine="potrace", scale=2), "")
    out = vectorise.run(h, {"image": p})
    assert out["paths"] == 1 and out["nodes"] > 20 and out["engine"] == "potrace"
    r = vector.readings({"input": {"image": p}}, out, None)
    assert r is not None and r["ink_f1"] > 0.8 and r["ssim"] > 0.5 and r["tone"] < 0.1 and r["colour"] < 8
    drawn = vectorise.render(open(out["out"]).read(), 360, 260)
    assert drawn[130, 280].mean() > 150 or drawn[130, 281].mean() > 150  # the hatching stays open lines, not one block


def test_tones_stack_greys_and_lines_colour_keeps_the_red(tmp_path, need_potrace):
    p = plate(tmp_path / "p.png")
    out = vectorise.run(vectorise.load(M(engine="potrace-tones", levels=3), ""), {"image": p})
    assert out["paths"] == 3
    mid = vectorise.render(open(out["out"]).read(), 360, 260)[130, 120].mean()
    assert 40 < mid < 220  # the grey square drawn grey
    c = plate(tmp_path / "c.png", colour=True)
    out = vectorise.run(vectorise.load(M(engine="lines-colour", colours=6), ""), {"image": c})
    r = vectorise.render(out["svg"], 360, 260)[130, 120]
    assert r[0] > 180 and r[1] < 170
    assert out["svg"] if out["bytes"] <= vectorise.INLINE else "svg" not in out


def test_llm_engines_built_on_the_trace(tmp_path, need_potrace):
    p = plate(tmp_path / "p.png")
    h = vectorise.load(M(engine="llm-clean", llm="m", clean_kb=0.1), "")
    with pytest.raises(RunError, match="more than a model can read"):
        vectorise.run(h, {"image": p})
    # a grey plate has no colour to ask for: the trace alone, no call
    h = vectorise.load(M(engine="llm-colour", llm="m"), "")
    h["llm"] = lambda q: pytest.fail("no call for a picture without colour")
    out = vectorise.run(h, {"image": p})
    assert "none" in out["colour"] and "cost_usd" not in out
    # a coloured one: the model's washes under the ink, in its own viewBox
    asked = []
    h["llm"] = lambda q: (
        asked.append(q)
        or {
            "text": '<svg viewBox="0 0 36 26"><rect x="9" y="10" width="6" height="6" fill="#eb8c8c"/></svg>',
            "cost_usd": 0.05,
        }
    )
    out = vectorise.run(h, {"image": plate(tmp_path / "c.png", colour=True)})
    assert asked[0]["preset"] == "vectorise-colour-v1" and out["cost_usd"] == 0.05
    assert vectorise.render(open(out["out"]).read(), 360, 260)[130, 120][0] > 150
    h = vectorise.load(M(engine="llm-clean", llm="m"), "")
    h["llm"] = lambda q: {"text": q["context"].split("The traced SVG:\n", 1)[1]}  # gives the trace back as it was
    out = vectorise.run(h, {"image": p})
    assert out["paths"] == 1


def test_potrace_failure_and_absence_are_reported(tmp_path, need_potrace, monkeypatch):
    with pytest.raises(RunError, match="are numbers"):
        vectorise._trace(np.ones((8, 8), bool), {"alphamax": "x"})
    with pytest.raises(RunError, match="potrace failed: potrace: invalid color"):
        vectorise._trace(np.ones((8, 8), bool), {}, fill="nocolour")
    monkeypatch.setattr(vectorise, "binary", lambda: None)
    with pytest.raises(RunError, match="potrace command is not installed"):
        vectorise._trace(np.ones((8, 8), bool), {})
