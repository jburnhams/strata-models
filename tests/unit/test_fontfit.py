"""The font-match runtime (runtimes/fontfit.py): the catalogue, fetching faces from Google Fonts (faked), reading a vision model's shortlist,
and the render-and-fit on a block set in a known font at a known size and width. Pillow's own bundled face (Aileron) stands in for the
catalogue's files, so nothing is downloaded."""

import json
import os
import urllib.error
from email.message import Message

import numpy as np
import pytest
from PIL import Image, ImageFont

from stratamodels.runtimes import RunError
from stratamodels.runtimes import fontfit as F


@pytest.fixture
def ttf(tmp_path):
    p = tmp_path / "aileron.ttf"
    p.write_bytes(ImageFont.load_default(20).path.getvalue())
    return str(p)


@pytest.fixture
def models(tmp_path, ttf):
    """A models folder where 'tinos' and 'old-standard' are already fetched (both Aileron); any other face fails to fetch."""
    d = tmp_path / "models"
    for fid in ("tinos", "old-standard"):
        f = d / "fonts" / fid
        f.mkdir(parents=True)
        (f / "manifest.json").write_text(json.dumps({"roman": ttf, "italic": ttf}))
    return str(d)


def no_network(url):
    raise OSError("offline")


def handle(models):
    return dict(key="fontfit", models_dir=models, top=2, get=no_network)


def block(tmp_path, ttf, size=40.0, hscale=1.1, text=("Punch and Judy", "with the Baby")):
    """A grey scan of two lines set in the font at `size` px, stretched by `hscale`; its words and boxes as Tesseract gives them."""
    face = F.Face("tinos", {"roman": ttf})
    W, H = 700, 160
    canvas = np.zeros((H, W), dtype=bool)
    lines = []
    for i, ln in enumerate(text):
        x, words = 20.0, []
        for w in ln.split():
            m, left, base = face.word("roman", w, size, hscale, 0.0, 0.5)
            cols, rows = np.flatnonzero(m.any(0)), np.flatnonzero(m.any(1))
            y = 55 + i * 60
            F._paste(canvas, m, left, base, x, y)
            bw, top = int(cols[-1] - cols[0] + 1), int(y - (base - rows[0]))
            words.append(dict(text=w, box=[int(x), top, bw, int(rows[-1] - rows[0] + 1)], conf=95.0))
            x += bw + size * 0.3
        lines.append(dict(text=ln, box=F_union([w["box"] for w in words]), conf=95.0, block=1, words=words))
    g = np.where(canvas, 50, 220).astype(np.uint8)
    p = str(tmp_path / "block.png")
    Image.fromarray(g).save(p, dpi=(300, 300))
    return p, lines


def F_union(boxes):
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    return [x0, y0, max(b[0] + b[2] for b in boxes) - x0, max(b[1] + b[3] for b in boxes) - y0]


def test_the_catalogue_has_every_class_and_known_classes_only():
    cat = F.catalogue()
    assert len(cat) >= 50
    assert {v["class"] for v in cat.values()} == set(F.CLASSES)
    assert all(v["family"] and v["like"] for v in cat.values())


def test_the_proposer_preset_lists_the_catalogue():
    from stratamodels.runtimes import llama

    text = llama.preset("font-propose-v1")[0]
    assert all(fid in text for fid in F.catalogue())


def test_closeness():
    assert F.closeness("modern", "modern") == 1.0
    assert F.closeness("modern", "transitional") == F.closeness("slab", "clarendon") == 0.5
    assert F.closeness("modern", "slab") == 0.0
    assert F.closeness(None, "modern") == 0.0


@pytest.mark.parametrize(
    ("value", "want"),
    [
        (["old-standard", "tinos"], ["old-standard", "tinos"]),
        (
            '{"class": "modern", "faces": ["Libre Bodoni", "old-standard"], "why": "hairlines"}',
            ["libre-bodoni", "old-standard"],
        ),
        ("Playfair Display Black, then playfair-display", ["playfair-display-black", "playfair-display"]),
        (
            '{"faces": ["old-standard", "libra-bodoni", "prata"]}',
            ["old-standard", "libre-bodoni", "prata"],
        ),  # a near miss of an id
        (
            '{"class": "old-style", "faces": ["no-such-face"]}',
            [k for k, v in F.catalogue().items() if v["class"] == "old-style"],
        ),
        ("", []),
        (None, []),
        ("no idea", []),
    ],
)
def test_parse_candidates(value, want):
    assert F.parse_candidates(value) == want


def test_a_class_without_faces_proposes_the_class():
    got = F.parse_candidates('{"class": "blackletter", "faces": []}')
    assert got and all(F.catalogue()[f]["class"] == "blackletter" for f in got)


def meta(weights, styles=("normal", "italic"), fam="x"):
    """Fontsource's metadata for a family with these weights and styles (each a latin TrueType file)."""
    return {
        "defSubset": "latin",
        "variants": {
            str(w): {st: {"latin": {"url": {"ttf": f"https://cdn/{fam}/latin-{w}-{st}.ttf"}}} for st in styles}
            for w in weights
        },
    }


def test_fontsource_urls_pick_the_weight_its_italic_and_a_bold():
    m = meta([300, 400, 700])
    assert F.fontsource_urls(m) == {
        "roman": "https://cdn/x/latin-400-normal.ttf",
        "italic": "https://cdn/x/latin-400-italic.ttf",
        "bold": "https://cdn/x/latin-700-normal.ttf",
        "bold-italic": "https://cdn/x/latin-700-italic.ttf",
    }
    assert set(F.fontsource_urls(m, weight=900)) == {
        "roman",
        "italic",
    }  # a heavy face is its own bold: the nearest weight, no bold
    assert F.fontsource_urls(meta([700], styles=("normal",)))["roman"].endswith(
        "700-normal.ttf"
    )  # one weight only (UnifrakturCook)
    assert F.fontsource_urls(meta([400], styles=("italic",))) == {"roman": "https://cdn/x/latin-400-italic.ttf"}
    assert F.fontsource_urls({}) == {}
    assert F.fontsource_id("IM Fell DW Pica SC") == "im-fell-dw-pica-sc"


def test_fetch_reads_fontsource_saves_each_style_and_reuses_it(tmp_path):
    calls = []

    def get(url):
        calls.append(url)
        if url == F.FONTSOURCE + "old-standard-tt":
            return json.dumps(meta([400, 700], fam="os")).encode()
        if url.startswith(F.FONTSOURCE):
            raise OSError("404")
        if "700-italic" in url:  # a cut the metadata lists but the family lacks
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        return url.encode()

    files = F.fetch("old-standard", str(tmp_path), get)
    assert sorted(files) == ["bold", "italic", "roman"]
    assert open(files["bold"], "rb").read() == b"https://cdn/os/latin-700-normal.ttf"
    n = len(calls)
    assert F.fetch("old-standard", str(tmp_path), get) == files and len(calls) == n  # the manifest is reused
    with pytest.raises(RunError, match="no face"):
        F.fetch("nonesuch", str(tmp_path), get)
    with pytest.raises(RunError, match="Fontsource has no"):
        F.fetch("tinos", str(tmp_path), get)


def test_a_transient_failure_fails_the_fetch_and_records_nothing(tmp_path):
    def get(url):
        if url.startswith(F.FONTSOURCE):
            return json.dumps(meta([400, 700])).encode()
        if "italic" in url:
            raise OSError("timed out")  # not a 404: the style may well exist
        return b"ttf"

    with pytest.raises(RunError, match="cannot download old-standard italic"):
        F.fetch("old-standard", str(tmp_path), get)
    assert not (tmp_path / "fonts" / "old-standard" / "manifest.json").exists()


def test_fetch_takes_small_capitals_and_direct_files(tmp_path):
    def get(url):
        if url.startswith(F.FONTSOURCE):
            sc = url.endswith("-sc")
            return json.dumps(meta([400], styles=("normal",), fam="SC" if sc else "R")).encode()
        return b"font"

    files = F.fetch("im-fell-english", str(tmp_path), get)
    assert set(files) == {"roman", "small-caps"} and files["small-caps"].endswith("small-caps.ttf")
    direct = next(
        k for k, v in F.catalogue().items() if v.get("files")
    )  # a CTAN face: its own URLs, OpenType kept as .otf
    files = F.fetch(direct, str(tmp_path), get)
    assert set(files) == set(F.catalogue()[direct]["files"]) and files["roman"].endswith(".otf")


def test_fit_recovers_the_size_and_width_of_a_known_setting(tmp_path, ttf, models):
    p, lines = block(tmp_path, ttf, size=40.0, hscale=1.1)
    r = F.run(handle(models), dict(image=p, lines=lines, candidates=["tinos"], out=str(tmp_path / "fit.png")))
    assert r["face"] == "tinos" and r["mode"] == "fit"
    ln = r["blocks"][0]["lines"][0]
    assert ln["size_px"] == pytest.approx(40.0, rel=0.06)
    assert ln["size_pt"] == pytest.approx(ln["size_px"] * 72 / 300, abs=0.01)
    assert ln["hscale"] == pytest.approx(1.1, abs=0.08)
    assert r["ink_f1"] > 0.9
    assert [w["style"] for w in ln["words"]] and r["style"] == "roman"
    w0 = ln["words"][0]
    assert {"x", "baseline_y", "size_px", "size_pt", "hscale", "tracking_em", "weight_delta"} <= set(w0)
    assert abs(w0["x"] - lines[0]["words"][0]["box"][0]) <= 2  # its own placement: at most a pixel or two from its ink
    assert w0["size_px"] == ln["size_px"] and w0["hscale"] == ln["hscale"]
    assert w0["stroke_em"] >= 0.0 and (w0["stroke_em"] > 0) == (w0["weight_delta"] > 0)
    assert os.path.isfile(r["path"]) and Image.open(r["path"]).size == (700, 160)


def test_stroke_em_is_measured_from_the_threshold(ttf):
    face = F.Face("tinos", {"roman": ttf})
    assert F.stroke_em(face, "roman", "Hamburg", 60.0, 1.0, 0.0, 0.5) == 0.0
    light = F.stroke_em(face, "roman", "Hamburg", 60.0, 1.0, 0.0, 0.4)
    heavy = F.stroke_em(face, "roman", "Hamburg", 60.0, 1.0, 0.0, 0.2)
    assert 0 < light < heavy < 0.02  # a fraction of a pixel per side at 60 px


def test_naive_and_measure_modes_and_the_ranking(tmp_path, ttf, models):
    p, lines = block(tmp_path, ttf)
    ocr = tmp_path / "ocr.json"
    ocr.write_text(json.dumps(dict(lines=lines)))
    naive = F.run(handle(models), dict(image=p, ocr=str(ocr), mode="naive", out=str(tmp_path / "n.png")))
    assert naive["face"] == "tinos" and naive["faces"] == []
    measure = F.run(
        handle(models),
        dict(
            image=p,
            ocr=str(ocr),
            mode="measure",
            candidates='{"faces": ["old-standard"], "style": "italic"}',
            out=str(tmp_path / "m.png"),
        ),
    )
    assert measure["face"] == "old-standard" and measure["style"] == "italic"
    # every face: only the two fetched ones can be fitted (the rest fail to fetch and are skipped); the screen keeps the best `top`
    fit = F.run(handle(models), dict(image=p, ocr=str(ocr), top=1, out=str(tmp_path / "f.png")))
    assert [f["face"] for f in fit["faces"]] and len(fit["faces"]) == 2
    assert [f["screened"] for f in fit["faces"]] == [False, True]
    assert fit["ink_f1"] >= naive["ink_f1"]


def test_box_limits_the_lines(tmp_path, ttf, models):
    p, lines = block(tmp_path, ttf)
    r = F.run(
        handle(models),
        dict(image=p, lines=lines, candidates=["tinos"], box=[0, 0, 700, 80], out=str(tmp_path / "b.png")),
    )
    assert len(r["blocks"][0]["lines"]) == 1


@pytest.mark.parametrize(
    ("params", "match"),
    [
        (dict(image="nope.png"), "no such picture"),
        (dict(mode="guess"), "mode is fit"),
        (dict(candidates="nothing I know"), "no catalogue face"),
        (dict(lines=[]), "no words"),
    ],
)
def test_bad_jobs_are_refused(tmp_path, ttf, models, params, match):
    p, lines = block(tmp_path, ttf)
    with pytest.raises(RunError, match=match):
        F.run(handle(models), dict(dict(image=p, lines=lines), **params))


def test_a_file_that_is_not_a_picture_and_an_unreadable_ocr(tmp_path, ttf, models):
    bad = tmp_path / "x.png"
    bad.write_text("no")
    with pytest.raises(RunError, match="not a picture"):
        F.run(handle(models), dict(image=str(bad), lines=[]))
    p, _ = block(tmp_path, ttf)
    with pytest.raises(RunError, match="cannot read ocr"):
        F.run(handle(models), dict(image=p, ocr=str(tmp_path / "missing.json")))


def test_load_and_unload(models):
    class M:
        key = "fontfit"
        options = {"top": 3}

    h = F.load(M(), models)
    assert h["top"] == 3 and h["models_dir"] == models
    F.unload(h)
    assert h == {}


def test_without_candidates_the_proposer_gives_the_shortlist(tmp_path, ttf, models):
    p, lines = block(tmp_path, ttf)
    asked = []

    def proposer(params):
        asked.append(dict(params, size=Image.open(params["image"]).size))
        return {"text": '{"class": "modern", "faces": ["old-standard"]}', "cost_usd": 0.01}

    h = dict(handle(models), propose="claude-sonnet-5-5-api", proposer=proposer)
    r = F.run(h, dict(image=p, lines=lines, box=[0, 0, 700, 120], out=str(tmp_path / "p.png")))
    assert r["proposed"][0] == "old-standard" and r["face"] in ("old-standard", "tinos")
    assert r["proposer"]["model"] == "claude-sonnet-5-5-api" and r["proposer"]["cost_usd"] == 0.01
    assert asked[0]["system_preset"] == "font-propose-v1" and asked[0]["size"] == (700, 120)  # the block, cut out
    assert not os.path.exists(asked[0]["image"])  # and removed after
    # candidates given, or propose = false: no call
    F.run(h, dict(image=p, lines=lines, candidates=["tinos"], out=str(tmp_path / "q.png")))
    r = F.run(h, dict(image=p, lines=lines, propose=False, out=str(tmp_path / "r.png")))
    assert len(asked) == 1 and r["proposer"] is None


def test_a_failed_proposer_falls_back_to_every_face(tmp_path, ttf, models):
    p, lines = block(tmp_path, ttf)

    def proposer(params):
        raise RunError("Claude API 529: overloaded")

    h = dict(handle(models), propose="claude-sonnet-5-5-api", proposer=proposer)
    r = F.run(h, dict(image=p, lines=lines, out=str(tmp_path / "p.png")))
    # the two fetchable faces, screened
    assert "529" in r["proposer"]["error"] and r["proposed"] == [] and len(r["faces"]) == 2
    h["proposer"] = lambda params: {"text": "The picture is too faint to tell."}  # an answer that names no face
    r = F.run(h, dict(image=p, lines=lines, out=str(tmp_path / "q.png")))
    assert "no catalogue face" in r["proposer"]["error"] and len(r["faces"]) == 2


def test_the_proposer_is_loaded_from_the_registry_once_and_unloaded(models, monkeypatch):
    from stratamodels import runtimes

    calls = []

    class RT:
        @staticmethod
        def load(m, d):
            calls.append(("load", m.key))
            return {"m": m.key}

        @staticmethod
        def run(h, params):
            calls.append(("run", h["m"]))
            return {"text": "tinos"}

        @staticmethod
        def unload(h):
            calls.append(("unload", h["m"]))

    monkeypatch.setattr(runtimes, "get", lambda name: RT)

    class M:
        key = "fontfit"
        options = {"propose": "claude-sonnet-5-5-api"}

    h = F.load(M(), models)
    f = F._proposer(h)
    assert f({}) == {"text": "tinos"} and F._proposer(h) is f
    F.unload(h)
    assert calls == [
        ("load", "claude-sonnet-5-5-api"),
        ("run", "claude-sonnet-5-5-api"),
        ("unload", "claude-sonnet-5-5-api"),
    ]
    with pytest.raises(RunError, match="not in the registry"):
        F._proposer(dict(propose="no-such-model", models_dir=models))


def test_the_proposer_comes_from_the_database_registry_when_the_service_has_one(models, monkeypatch):
    from stratamodels import registry, runtimes, settings

    seen = []

    class Reg:
        models = {"db-only-vl": type("M", (), {"key": "db-only-vl", "runtime": "fake"})()}

    class RT:
        @staticmethod
        def load(m, d):
            return {}

        @staticmethod
        def run(h, params):
            return {"text": "tinos"}

    monkeypatch.setattr(settings, "engine", lambda: "ENGINE")
    monkeypatch.setattr(registry.Registry, "from_db", classmethod(lambda cls, e: seen.append(e) or Reg))
    monkeypatch.setattr(runtimes, "get", lambda name: RT)
    assert F._proposer(dict(propose="db-only-vl", models_dir=models))({}) == {"text": "tinos"} and seen == ["ENGINE"]
