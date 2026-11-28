"""The vectorise runtime (role `vectorise`) without the potrace command: VTracer (a library), the SVG plumbing, and the engines that ask a
model, with the model's answers given in the handle (as page-build's grouper is in test_pagebuild.py). potrace itself runs in
tests/integration/test_vectorise_runtime.py (unit tests run no external processes); the scores of a drawing are in test_vector_metrics.py."""

import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from stratamodels.runtimes import RunError, vectorise


def plate(path, colour=False, size=(240, 180)):
    """Cream paper with a black ring and a bar, and a red patch when `colour`: a small stand-in for a book's picture."""
    im = Image.new("RGB", size, (236, 226, 200))
    d = ImageDraw.Draw(im)
    d.ellipse((40, 30, 150, 140), outline=(20, 20, 20), width=8)
    d.rectangle((170, 40, 190, 150), fill=(15, 15, 15))
    if colour:
        d.rectangle((60, 60, 120, 110), fill=(200, 40, 40))
    im.save(path)
    return str(path)


class M:
    def __init__(self, **options):
        self.key, self.options = "vectorise-test", options


def handle(answers=None, **options):
    """A handle whose model answers come from `answers` (a list, taken in order), recording what it was asked."""
    h = vectorise.load(M(**options), "")
    asked = []
    if answers is not None:

        def call(p):
            asked.append(p)
            a = answers.pop(0)
            return a if isinstance(a, dict) else {"text": a, "cost_usd": 0.01}

        h["llm"] = call
    return h, asked


CIRCLE = '<svg viewBox="0 0 240 180" width="999"><circle cx="95" cy="85" r="51" fill="none" stroke="#000" stroke-width="8"/></svg>'


def test_vtracer_draws_the_plate_into_the_data_folder(tmp_path):
    h, _ = handle(engine="vtracer")
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png"), "colours": 2})
    assert out["engine"] == "vtracer" and (out["width"], out["height"]) == (240, 180)
    assert out["out"].startswith(os.path.join(os.environ["STRATAMODELS_DATA"], "vectorise")) and os.path.isfile(
        out["out"]
    )
    assert out["path"].endswith(".png") and Image.open(out["path"]).size == (240, 180)
    assert out["svg"].startswith("<svg") and 'viewBox="0 0 240 180"' in out["svg"]  # small: inline too
    assert out["paths"] >= 1 and out["nodes"] > 4 and out["bytes"] == len(out["svg"].encode())
    drawn = vectorise.render(out["svg"], 240, 180)
    assert drawn[85, 175].mean() < 80 and drawn[5, 5].mean() > 200  # the bar is ink, the paper is white


def test_colour_vtracer_keeps_the_red_and_out_writes_where_asked(tmp_path):
    h, _ = handle(engine="vtracer")
    dest = tmp_path / "x.svg"
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png", colour=True), "out": str(dest), "background": "white"})
    assert out["out"] == str(dest) and dest.read_text().count("<rect") >= 1
    r = vectorise.render(dest.read_text(), 240, 180)[85, 90]
    assert r[0] > 150 and r[1] < 100


def test_a_big_svg_is_left_out_of_the_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorise, "INLINE", 10)
    h, _ = handle(engine="vtracer")
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png")})
    assert "svg" not in out and os.path.getsize(out["out"]) == out["bytes"]


def test_params_and_options_are_checked(tmp_path):
    p = plate(tmp_path / "p.png")
    with pytest.raises(RunError, match="engine is one of"):
        vectorise.load(M(engine="magic"), "")
    with pytest.raises(RunError, match="names no vision-language model"):
        vectorise.load(M(engine="llm"), "")
    h, _ = handle(engine="vtracer")
    with pytest.raises(RunError, match="unknown params"):
        vectorise.run(h, {"image": p, "colour": 3})
    with pytest.raises(RunError, match="no such image"):
        vectorise.run(h, {"image": str(tmp_path / "none.png")})
    with pytest.raises(RunError, match="engine is one of"):
        vectorise.run(h, {"image": p, "engine": "magic"})
    with pytest.raises(RunError, match="out is an .svg"):
        vectorise.run(h, {"image": p, "out": str(tmp_path / "x.png")})
    with pytest.raises(RunError, match="threshold"):
        vectorise._cut(np.zeros((4, 4), np.uint8), 1.5)
    with pytest.raises(RunError, match="levels"):
        vectorise.tones_svg(np.zeros((4, 4, 3), np.uint8), {"levels": 9})


def test_the_cut_is_otsu_or_a_share_of_the_way_from_ink_to_paper():
    g = np.concatenate([np.full(50, 20), np.full(50, 220)]).astype(np.uint8).reshape(10, 10)
    assert 20 <= vectorise._cut(g, None) < 220 and vectorise._cut(g, "otsu") == vectorise.otsu(g)
    assert vectorise._cut(g, 0.5) == 120 and vectorise._cut(g, 0.25) == 70


def test_svg_plumbing():
    assert vectorise.stats('<svg><path d="M0 0L1 1Z"/><circle r="2"/></svg>') == dict(bytes=47, paths=2, nodes=3)
    assert vectorise.svg_of("Here:\n```svg\n<svg viewBox='0 0 1 1'><path d='M0 0'/></svg>\n```").startswith(
        '<svg xmlns="http://www.w3.org/2000/svg"'
    )
    with pytest.raises(RunError, match="no <svg>"):
        vectorise.svg_of("I cannot draw that.")
    fitted = vectorise._fit(CIRCLE, 240, 180)
    assert 'width="240" height="180"' in fitted and 'width="999"' not in fitted and 'viewBox="0 0 240 180"' in fitted
    assert 'viewBox="0 0 10 20"' in vectorise._fit(
        '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>', 10, 20
    )
    assert vectorise._fit("no svg here", 1, 1) == "no svg here"
    with pytest.raises(RunError, match="does not draw"):
        vectorise.render("<svg", 10, 10)
    assert vectorise._body('<svg a="1"><g/></svg>') == "<g/>"


def test_llm_draws_and_refines(tmp_path):
    h, asked = handle([CIRCLE, CIRCLE.replace('r="51"', 'r="55"')], engine="llm", llm="claude-opus-5-5-api", refine=1)
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png")})
    assert asked[0]["preset"] == "vectorise-llm-v1" and 'viewBox="0 0 240 180"' in asked[0]["context"]
    assert (
        asked[1]["preset"] == "vectorise-refine-v1"
        and "Your SVG" in asked[1]["context"]
        and asked[1]["image"].endswith("compare.png")
    )
    assert (
        'r="55"' in out["svg"] and out["cost_usd"] == 0.02 and [s["step"] for s in out["steps"]] == ["draw", "refine 1"]
    )


def test_a_broken_refinement_keeps_the_drawing_before_it(tmp_path):
    h, _ = handle([CIRCLE, "Sorry, no."], engine="llm", llm="m", refine=1)
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png")})
    assert 'r="51"' in out["svg"] and "no <svg>" in out["steps"][1]["error"]


def test_an_answer_with_no_svg_fails_the_job(tmp_path):
    h, _ = handle(["I would rather not."], engine="llm", llm="m")
    with pytest.raises(RunError, match="no <svg>"):
        vectorise.run(h, {"image": plate(tmp_path / "p.png")})


def test_llm_plan_steers_the_tracer(tmp_path):
    plan = {"json": {"engine": "vtracer", "colours": 2, "threshold": 7, "why": "black line work"}, "cost_usd": 0.003}
    h, asked = handle([plan], engine="llm-plan", llm="m")
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png")})
    assert asked[0]["preset"] == "vectorise-plan-v1" and asked[0]["max_tokens"] == 1024
    assert out["plan"] == {
        "engine": "vtracer",
        "colours": 2,
        "why": "black line work",
    }  # a threshold out of range is dropped
    assert out["cost_usd"] == 0.003 and out["paths"] >= 1
    h, _ = handle(["not json"], engine="llm-plan", llm="m")
    with pytest.raises(RunError, match="plan is not JSON"):
        vectorise.run(h, {"image": plate(tmp_path / "q.png")})


# ---- the potrace engines with potrace's part played by a box per mask (the real command: tests/integration) ----------------------------
def boxes(monkeypatch):
    """_trace replaced by the box around each mask's ink, so the engines that stack, scale and colour traces run without the command."""
    fills = []

    def trace(mask, o, fill="#000000"):
        fills.append(fill)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return ""
        return f'<g fill="{fill}"><path d="M{xs.min()} {ys.min()}H{xs.max() + 1}V{ys.max() + 1}H{xs.min()}Z"/></g>'

    monkeypatch.setattr(vectorise, "_trace", trace)
    return fills


def test_potrace_scales_its_trace_back_to_the_pictures_pixels(tmp_path, monkeypatch):
    boxes(monkeypatch)
    h, _ = handle(engine="potrace")
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png"), "threshold": 0.5})
    assert 'transform="scale(0.5)"' in out["svg"] and out["engine"] == "potrace"
    drawn = vectorise.render(out["svg"], 240, 180)
    assert drawn[85, 100].mean() < 80  # inside the box around the ring and the bar
    one = vectorise.run(h, {"image": plate(tmp_path / "p.png"), "scale": 1})
    assert "scale(1)" in one["svg"]


def test_tones_stack_lightest_first_and_end_in_black(tmp_path, monkeypatch):
    fills = boxes(monkeypatch)
    h, _ = handle(engine="potrace-tones")
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png", colour=True), "levels": 3})
    assert len(fills) == 3 and fills[-1] == "#000000" and fills[0] != "#000000"
    assert out["svg"].count("<path") == 3


def test_lines_colour_lays_the_ink_over_the_washes(tmp_path, monkeypatch):
    boxes(monkeypatch)
    h, _ = handle(engine="lines-colour")
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png", colour=True), "colours": 4})
    assert "mix-blend-mode:multiply" in out["svg"] and out["svg"].index("multiply") > out["svg"].index("<path")


def test_the_potrace_command_is_checked_before_it_runs(monkeypatch):
    mask = np.zeros((8, 8), bool)
    monkeypatch.setattr(vectorise, "binary", lambda: None)
    with pytest.raises(RunError, match="potrace command is not installed"):
        vectorise._trace(mask, {})
    monkeypatch.setattr(vectorise, "binary", lambda: "/usr/bin/potrace")
    with pytest.raises(RunError, match="are numbers"):
        vectorise._trace(mask, {"turdsize": "many"})


def test_trace_reads_the_group_potrace_writes(monkeypatch):
    """subprocess played by a stand-in that writes potrace's SVG where -o says (the real command: tests/integration)."""
    import subprocess

    seen = []

    def fake(cmd, **kw):
        seen.append(cmd)
        out = cmd[cmd.index("-o") + 1]
        with open(out, "w", encoding="utf8") as f:
            f.write('<svg><metadata/><g transform="x" fill="#ff0000"><path d="M0 0"/></g></svg>' if "-C" in cmd else "")
        return subprocess.CompletedProcess(cmd, 1 if "bad" in cmd else 0, "", "potrace: bad")

    monkeypatch.setattr(vectorise, "binary", lambda: "potrace")
    monkeypatch.setattr(vectorise.subprocess, "run", fake)
    g = vectorise._trace(np.ones((4, 4), bool), {"turdsize": 5}, fill="#ff0000")
    assert g.startswith("<g") and g.endswith("</g>") and seen[0][seen[0].index("-t") + 1] == "5"
    with pytest.raises(RunError, match="potrace failed: potrace: bad"):
        vectorise._trace(np.ones((4, 4), bool), {}, fill="bad")


def test_a_big_picture_is_read_at_the_long_side(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorise, "LONG_SIDE", 120)
    assert vectorise.read(plate(tmp_path / "p.png")).shape == (90, 120, 3)


def test_unload_frees_the_drawing_model():
    h, _ = handle([], engine="llm", llm="m")
    freed = []
    h["llm_unload"] = lambda: freed.append(1)
    vectorise.unload(h)
    assert freed == [1] and h == {}


def test_llm_colour_pays_only_for_a_picture_with_colour(tmp_path, monkeypatch):
    boxes(monkeypatch)
    wash = '<svg viewBox="0 0 120 90"><rect x="30" y="30" width="30" height="25" fill="#c83030"/></svg>'
    h, asked = handle([], engine="llm-colour", llm="m")
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png")})
    assert asked == [] and out["colour"].startswith("none") and "cost_usd" not in out
    h, asked = handle([wash], engine="llm-colour", llm="m")
    big = plate(tmp_path / "c.png", colour=True)
    out = vectorise.run(h, {"image": big})
    assert asked[0]["preset"] == "vectorise-colour-v1" and '<rect x="30"' in out["svg"] and out["cost_usd"] == 0.01
    h, _ = handle(["no colours today"], engine="llm-colour", llm="m")
    out = vectorise.run(h, {"image": big})
    assert "no <svg>" in out["colour_error"]


def test_llm_clean_redraws_a_small_trace_and_refuses_a_big_one(tmp_path, monkeypatch):
    boxes(monkeypatch)
    h, asked = handle([CIRCLE], engine="llm-clean", llm="m")
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png")})
    assert "The traced SVG:" in asked[0]["context"] and "<path" in asked[0]["context"] and 'r="51"' in out["svg"]
    h, _ = handle([CIRCLE], engine="llm-clean", llm="m", clean_kb=0.01)
    with pytest.raises(RunError, match="llm-clean is for simple line art"):
        vectorise.run(h, {"image": plate(tmp_path / "p.png")})


def test_a_plan_in_text_naming_an_unknown_engine_falls_back_to_potrace(tmp_path, monkeypatch):
    boxes(monkeypatch)
    h, _ = handle(['{"engine": "magic", "threshold": 0.4, "levels": 3}'], engine="llm-plan", llm="m")
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png")})
    assert out["plan"]["engine"] == "potrace" and out["plan"]["threshold"] == 0.4 and "scale(0.5)" in out["svg"]
    h, _ = handle([{"json": {"engine": "potrace-tones", "levels": 2}}], engine="llm-plan", llm="m")
    assert vectorise.run(h, {"image": plate(tmp_path / "p.png")})["svg"].count("<path") == 2


def test_the_drawing_model_is_asked_through_the_delegate(tmp_path, monkeypatch):
    from stratamodels.runtimes import delegate

    got = []
    monkeypatch.setattr(
        delegate, "caller", lambda key, d, via, opts=None: got.append((key, via)) or ((lambda p: CIRCLE), lambda: None)
    )
    h, _ = handle(engine="llm", llm="claude-opus-5-5-api")
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png")})
    assert got == [("claude-opus-5-5-api", None)] and 'r="51"' in out["svg"]


def test_a_refinement_that_does_not_render_keeps_the_drawing_before_it(tmp_path):
    h, _ = handle(
        [CIRCLE, "<svg><path d='M0 0'></svg>"], engine="llm", llm="m", refine=1
    )  # an <svg> block, but not XML
    out = vectorise.run(h, {"image": plate(tmp_path / "p.png")})
    assert 'r="51"' in out["svg"] and "error" in out["steps"][1]


def test_fit_replaces_sizing_in_either_quote_style_and_fits_again():
    svg = "<svg viewBox='0 0 24 18' width='5' height=\"6\" preserveAspectRatio='xMidYMid'><path d='M0 0'/></svg>"
    once = vectorise._fit(svg, 240, 180)
    head = once[: once.index(">")]
    assert head.count("width=") == head.count("height=") == head.count("preserveAspectRatio=") == 1
    assert vectorise._fit(once, 240, 180) == once
    vectorise.render(once, 24, 18)


def test_the_drawing_models_load_options_reach_the_delegate(tmp_path, monkeypatch):
    from stratamodels.runtimes import delegate

    got = []
    monkeypatch.setattr(
        delegate, "caller", lambda key, d, via, opts: got.append(opts) or ((lambda p: CIRCLE), lambda: None)
    )
    h, _ = handle(engine="llm", llm="qwen3.5-9b-vl", llm_options={"ctx": 32768})
    vectorise.run(h, {"image": plate(tmp_path / "p.png")})
    assert got == [{"ctx": 32768}]
