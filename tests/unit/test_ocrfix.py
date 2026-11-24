"""The ocr-fix runtime (runtimes/ocrfix.py): a reader's boxes with a vision model's corrections of its lines. The reader and the fixer are
stand-ins handed in where the runtime loads registry models."""

import json

import pytest

from stratamodels.runtimes import RunError
from stratamodels.runtimes import ocrfix as X

OCR = dict(
    width=1000,
    height=2000,
    engine="doctr db_resnet50+parseq",
    unread=[],
    lines=[
        dict(
            text="SIGNOR BUOITZ",
            box=[100, 200, 400, 40],
            conf=90.0,
            block=1,
            words=[
                dict(text="SIGNOR", box=[100, 200, 180, 40], conf=95.0),
                dict(text="BUOITZ", box=[320, 200, 180, 40], conf=85.0),
            ],
        ),
        dict(
            text="INDIARUBBER",
            box=[100, 300, 500, 40],
            conf=80.0,
            block=2,
            words=[dict(text="INDIARUBBER", box=[100, 300, 500, 40], conf=80.0)],
        ),
        dict(
            text="~~",
            box=[900, 1900, 50, 50],
            conf=20.0,
            block=3,
            words=[dict(text="~~", box=[900, 1900, 50, 50], conf=20.0)],
        ),
        dict(
            text="GOODS.",
            box=[100, 400, 200, 40],
            conf=99.0,
            block=2,
            words=[dict(text="GOODS.", box=[100, 400, 200, 40], conf=99.0)],
        ),
    ],
)
OCR["text"] = "\n".join(ln["text"] for ln in OCR["lines"])


def test_the_lines_are_numbered_as_page_text_fix_reads_them():
    assert X.numbered(OCR).splitlines()[0] == "1 b1 [100,100,500,120] SIGNOR BUOITZ"
    with pytest.raises(RunError, match="page size"):
        X.numbered(dict(lines=[]))


def test_corrections_keep_boxes_share_a_line_among_new_words_and_drop_emptied_lines():
    fixes = X.corrections(
        dict(
            lines=[
                dict(n=1, text="SIGNOR  BLITZ"),
                dict(n="2", text="INDIA RUBBER"),
                dict(n=3, text=""),
                dict(n="x"),
                "junk",
                dict(n=1, text="later"),
            ]
        )
    )
    assert fixes == {1: "SIGNOR BLITZ", 2: "INDIA RUBBER", 3: ""}
    out, changed = X.apply(OCR, fixes)
    assert changed == 3 and out["text"] == "SIGNOR BLITZ\nINDIA RUBBER\nGOODS."
    first, second = out["lines"][0], out["lines"][1]
    assert [w["box"] for w in first["words"]] == [
        w["box"] for w in OCR["lines"][0]["words"]
    ]  # same count: each word keeps its box
    assert first["words"][1]["text"] == "BLITZ" and first["box"] == OCR["lines"][0]["box"]
    assert [w["box"] for w in second["words"]] == [
        [100, 300, 208, 40],
        [350, 300, 250, 40],
    ]  # 5 + space + 6 letters over 500 px
    assert all(w["conf"] == 80.0 for w in second["words"])
    assert out["unread"] == [
        dict(box=[900, 1900, 50, 50], conf=0.0)
    ]  # the emptied line: no print, its box kept as unread
    assert X.apply(OCR, {4: "GOODS."})[1] == 0  # the same text is no change


@pytest.fixture
def models(monkeypatch):
    calls = []

    def reader_run(h, params):
        calls.append(("reader", params))
        return json.loads(json.dumps(OCR))

    def fixer_run(h, params):
        calls.append(("fixer", params))
        if params["context"].startswith("1 b1 [100,100,500,120] SIGNOR BUOITZ"):
            return dict(text='```json\n{"lines": [{"n": 1, "text": "SIGNOR BLITZ"}]}\n```', cost_usd=0.002)
        raise RuntimeError("budget spent")

    fake = {
        "reader": type("R", (), {"run": staticmethod(reader_run)}),
        "fixer": type("F", (), {"run": staticmethod(fixer_run)}),
    }
    monkeypatch.setattr(X, "_model", lambda handle, key: (fake[key], None))
    return calls


def handle():
    return X.load(type("M", (), dict(key="doctr-haiku-fix", options=dict(reader="reader", fixer="fixer")))(), "/models")


def test_run_reads_with_the_reader_then_corrects_with_the_fixer(models, tmp_path):
    pic = tmp_path / "p.png"
    pic.write_bytes(b"x")
    r = X.run(handle(), dict(image=str(pic), straight=False))
    assert r["text"].startswith("SIGNOR BLITZ\n") and r["cost_usd"] == 0.002 and r["fixer"]["changed"] == 1
    assert (
        r["reader"] == "doctr db_resnet50+parseq"
        and r["engine"] == "doctr db_resnet50+parseq + fixer"
        and r["model"] == "doctr-haiku-fix"
    )
    assert models[0] == ("reader", dict(straight=False, path=str(pic)))  # the reader's own params go through
    assert models[1][1]["preset"] == "page-text-fix-v1" and models[1][1]["image"] == str(pic)


def test_a_given_ocr_answer_is_corrected_instead_and_a_failed_fix_keeps_the_readers_text(models, tmp_path):
    pic = tmp_path / "p.png"
    pic.write_bytes(b"x")
    other = dict(
        OCR, lines=OCR["lines"][1:], text="INDIARUBBER\n~~\nGOODS."
    )  # the fixer stand-in fails on any page but the first
    path = tmp_path / "tess.json"
    path.write_text(json.dumps(other))
    r = X.run(handle(), dict(image=str(pic), ocr=str(path)))
    assert [c[0] for c in models] == ["fixer"]  # no reading: the given answer is corrected
    assert r["text"] == "INDIARUBBER\n~~\nGOODS." and "budget spent" in r["fixer"]["error"] and r["cost_usd"] == 0.0
    empty = X.run(handle(), dict(image=str(pic), ocr=dict(OCR, lines=[])))
    assert empty["fixer"] == dict(model="fixer", seconds=0.0, changed=0)


def test_bad_jobs_and_models_are_refused(models, tmp_path):
    with pytest.raises(RunError, match="no such picture"):
        X.run(handle(), dict(image=str(tmp_path / "none.png")))
    pic = tmp_path / "p.png"
    pic.write_bytes(b"x")
    with pytest.raises(RunError, match="not valid JSON"):
        X.run(handle(), dict(image=str(pic), ocr="{bad"))
    with pytest.raises(RunError, match="no lines"):
        X.run(handle(), dict(image=str(pic), ocr=dict(width=1)))
    with pytest.raises(RunError, match="names no fixer"):
        X.load(type("M", (), dict(key="k", options=dict(reader="r")))(), "/models")
    h = handle()
    X.unload(h)
    assert h == {}
