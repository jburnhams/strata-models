"""The detect-then-read OCR venv-worker (stratamodels/workers/ocr.py) without docTR or torch: fake modules stand in for them. The real thing
runs in the ocr venv (scripts/setup_worker_venv.sh ocr)."""

import os
import sys
import types

import pytest

WORKERS = os.path.join(os.path.dirname(__file__), "..", "..", "src", "stratamodels", "workers")
sys.path.insert(0, os.path.abspath(WORKERS))
import ocr as O  # noqa: E402
from protocol import ParamError  # noqa: E402

PAGE = dict(
    dimensions=(1000, 800),  # docTR's (height, width)
    blocks=[
        dict(
            geometry=((0.1, 0.1), (0.6, 0.25)),
            artefacts=[dict(type="logo", geometry=((0.7, 0.1), (0.9, 0.3)), confidence=0.6)],
            lines=[
                dict(
                    geometry=((0.1, 0.1), (0.6, 0.15)),
                    words=[
                        dict(value="SIGNOR", geometry=((0.1, 0.1), (0.3, 0.15)), confidence=0.99),
                        dict(value="BLITZ", geometry=((0.4, 0.1), (0.6, 0.15)), confidence=0.97),
                    ],
                ),
                dict(geometry=((0.1, 0.2), (0.2, 0.25)), words=[dict(value=" ", geometry=((0.1, 0.2), (0.2, 0.25)))]),
            ],
        ),
        dict(
            geometry=((0.2, 0.9), (0.8, 0.95)),
            lines=[
                dict(
                    geometry=((0.2, 0.9), (0.8, 0.95)),
                    words=[dict(value="GOODS.", geometry=((0.2, 0.9), (0.8, 0.95)), confidence=0.8)],
                )
            ],
        ),
    ],
)


def test_a_doctr_page_is_the_ocr_roles_answer():
    a = O.answer(PAGE, "doctr x")
    assert (a["width"], a["height"], a["engine"]) == (800, 1000, "doctr x")
    assert a["text"] == "SIGNOR BLITZ\nGOODS."  # an empty line is dropped
    first, last = a["lines"]
    assert first["box"] == [80, 100, 400, 50] and first["block"] == 1 and first["conf"] == 98.0
    assert [w["box"] for w in first["words"]] == [[80, 100, 160, 50], [320, 100, 160, 50]]
    assert last["block"] == 3 and last["words"][0]["conf"] == 80.0  # block 2 is the first block's artefact
    # what the model found beyond the text: its blocks, the artefact as a picture, the word box it read as nothing
    assert a["blocks"] == [
        dict(block=1, box=[80, 100, 400, 150], kind="text"),
        dict(block=2, box=[560, 100, 160, 200], kind="picture"),
        dict(block=3, box=[160, 900, 480, 50], kind="text"),
    ]
    assert a["unread"] == [dict(box=[80, 200, 80, 50], conf=0.0)]
    assert "regions" not in a
    laid = O.answer(
        dict(PAGE, layout=[dict(type="Section-header", geometry=((0.1, 0.1), (0.6, 0.15)), confidence=0.7)]), "x"
    )
    assert laid["regions"] == [dict(box=[80, 100, 400, 50], kind="section-header", conf=70.0)]


@pytest.fixture
def fake_doctr(monkeypatch, tmp_path):
    made = []

    class Predictor:
        def __init__(self, **kw):
            made.append(kw)

        def to(self, device):
            self.device = device
            return self

        def __call__(self, doc):
            return types.SimpleNamespace(export=lambda: dict(pages=[PAGE]))

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    models = types.ModuleType("doctr.models")
    models.ocr_predictor = lambda **kw: Predictor(**kw)
    io = types.ModuleType("doctr.io")
    io.DocumentFile = types.SimpleNamespace(from_images=lambda p: [p])
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "doctr", types.ModuleType("doctr"))
    monkeypatch.setitem(sys.modules, "doctr.models", models)
    monkeypatch.setitem(sys.modules, "doctr.io", io)
    monkeypatch.delenv("DOCTR_CACHE_DIR", raising=False)
    monkeypatch.setattr(O, "M", {})
    return made


def test_load_and_run_through_the_predictor(fake_doctr, tmp_path):
    info = O.load(dict(key="doctr", name="doctr", options=dict(reco="crnn_vgg16_bn")), str(tmp_path))
    assert info == dict(device="cpu", engine="doctr db_resnet50+crnn_vgg16_bn")
    assert os.environ["DOCTR_CACHE_DIR"] == str(tmp_path / "doctr")
    pic = tmp_path / "p.png"
    pic.write_bytes(b"x")
    out = O.run(dict(image=str(pic)))
    assert out["text"] == "SIGNOR BLITZ\nGOODS." and out["seconds"] >= 0
    O.run(dict(path=str(pic), straight=False))  # a rotated page: a second predictor, made once
    O.run(dict(path=str(pic), straight=False))
    assert [m["assume_straight_pages"] for m in fake_doctr] == [True, False]
    assert not any(m["detect_layout"] for m in fake_doctr)
    O.load(dict(name="doctr", options=dict(layout=True)), str(tmp_path))
    assert fake_doctr[-1]["detect_layout"] and O.M["engine"] == "doctr db_resnet50+parseq+layout"
    with pytest.raises(ParamError, match="no such picture"):
        O.run(dict(path=str(tmp_path / "none.png")))
    with pytest.raises(ParamError, match="Tesseract's params"):
        O.run(dict(path=str(pic), psm=4))
    assert O.run(dict(image=str(pic), ocr="tesseract.json"))[
        "lines"
    ]  # an eval item's other input is no param of docTR's
    with pytest.raises(ValueError, match="no ocr engine"):
        O.load(dict(name="paddle"), str(tmp_path))
