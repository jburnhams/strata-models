import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location(
    "ocr_reference", os.path.join(ROOT, "scripts", "evals", "ocr_reference.py")
)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)  # type: ignore[union-attr]


def test_reads_each_page_once_records_its_source_and_stops_at_the_cap(tmp_path, monkeypatch, capsys):
    calls = []

    def fetch(base, books, out, items=None):
        return {i: (os.path.join(out, f"{i}.png"), 300) for i in items if i != "ward1874-p999"}

    def run(role, params, app, priority, model):
        calls.append((role, params["system_preset"], params["image"].endswith(".png"), app, model))
        cut = params["image"].endswith("ward1874-p043.png")
        return dict(
            text=" Punch and Judy. \n",
            cost_usd=0.0 if cut else 0.6,
            tokens_in=1500,
            tokens_out=300,
            finish="max_tokens" if cut else "end_turn",
        )

    monkeypatch.setattr(R, "from_strata_scan", fetch)
    monkeypatch.setattr(R.sm, "run", run)
    (tmp_path / "collier1828-p001.txt").write_text("already read\n")
    args = ["--strata-scan", "http://x/", "--book", "ward1874=1", "--out", str(tmp_path), "--max-usd", "1.0"]
    items = [
        "--item",
        "ward1874-p019=check",
        "--item",
        "ward1874-p999=misread",
        "--item",
        "collier1828-p001",
        "--item",
        "ward1874-p040=unclear",
        "--item",
        "ward1874-p041=garbled",
    ]
    assert R.main(args + items) == 0
    out = capsys.readouterr()
    assert (
        calls == [("vision-language", "ocr-read-v1", True, "evals", "claude-opus-5-5-api")] * 2
    )  # p999 missing, p001 already read, p041 over the cap
    assert "no such page" in out.err and "stopping: $1.2000 spent" in out.err
    src = json.loads((tmp_path / "SOURCES.json").read_text())
    assert (
        set(src) == {"ward1874-p019", "ward1874-p040"}
        and src["ward1874-p040"]["kind"] == "unclear"
        and src["ward1874-p040"]["cost_usd"] == 0.6
        and src["ward1874-p040"]["finish"] == "end_turn"
    )
    assert 0 < src["ward1874-p019"]["cer_exact_vs_typed"] < 1  # the hand-typed page exists: Opus is checked against it
    assert (
        "cer_exact_vs_typed" not in src["ward1874-p040"]
        and (tmp_path / "ward1874-p040.txt").read_text() == "Punch and Judy.\n"
    )


def test_pages_reads_local_pictures_into_their_text_folder(tmp_path, monkeypatch, capsys):
    """--pages: the page-layout pictures read where they lie (no strata-scan), each reading beside them in text/, every picture unless
    --item names some; a page's other files (its .ocr.json) are not pages."""
    for name in ("ward1874-leaf013.png", "collier1828-leaf080.png", "collier1828-leaf080.ocr.json"):
        (tmp_path / name).write_bytes(b"x")
    sent = []
    monkeypatch.setattr(
        R.sm,
        "run",
        lambda role, params, app, priority, model: (
            sent.append(params["image"]) or dict(text="Punch", cost_usd=0.03, finish="end_turn")
        ),
    )
    assert R.main(["--pages", str(tmp_path)]) == 0
    assert sorted(os.path.basename(p) for p in sent) == ["collier1828-leaf080.png", "ward1874-leaf013.png"]
    assert (tmp_path / "text" / "ward1874-leaf013.txt").read_text() == "Punch\n"
    assert set(json.loads((tmp_path / "text" / "SOURCES.json").read_text())) == {
        "collier1828-leaf080",
        "ward1874-leaf013",
    }
    sent.clear()
    assert R.main(["--pages", str(tmp_path), "--item", "ward1874-leaf999"]) == 0
    assert sent == [] and "no such page in" in capsys.readouterr().err


def test_one_source_of_pages(tmp_path):
    import pytest

    for args in ([], ["--pages", str(tmp_path), "--strata-scan", "http://x/"]):
        with pytest.raises(SystemExit):
            R.main(args)
