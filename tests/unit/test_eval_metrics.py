"""stratamodels.evals.metrics: scoring outputs against references, and summaries and comparisons with the right direction."""

import math

import pytest

from stratamodels.evals import metrics as M


def tx(text):
    return dict(segments=[dict(text=t) for t in text.split("|")])


def test_text_normalisation_and_edit_distance():
    assert M.normalise("Ask NOT, what your country's “flag”...") == ["ask", "not", "what", "your", "country's", "flag"]
    assert (
        M.edits(["a", "b", "c", "d"], ["a", "x", "c"]) == 2
        and M.edits([], ["a"]) == 1
        and M.transcript(tx(" Ask not| what")) == " Ask not  what"
        and M.transcript(None) == ""
    )
    assert M.transcript("plain") == "plain"


def test_wer_cer_count_failures_as_all_errors():
    it = dict(id="1", reference=dict(text="ask not what your country"))
    assert M.METRICS["wer"].score(it, tx("Ask not what our country"), None) == (1, 5)
    assert (
        M.METRICS["wer"].score(it, None, "boom") == (5, 5)
        and M.METRICS["wer"].score(dict(id="2"), tx("x"), None) is None
    )
    n, d = M.METRICS["cer"].score(it, tx("ask not what your country"), None)
    assert n == 0 and d == len("ask not what your country")
    assert M.METRICS["cer"].score(it, None, "x")[0] == d and M.METRICS["cer"].score(dict(id="2"), None, None) is None


def test_faces_found_missed_and_iou():
    assert M.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3) and M.iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0
    it = dict(id="f", reference=dict(box=[10, 10, 50, 50], boxes=[[10, 10, 50, 50], [100, 100, 140, 140]]))
    out = dict(views=[dict(faces=[dict(bbox=[11, 11, 50, 51], embedding=[1, 0]), dict(bbox=[300, 300, 310, 310])])])
    assert (
        M.METRICS["face_found"].score(it, out, None) is True and M.METRICS["face_found"].score(it, None, "x") is False
    )
    assert M.METRICS["faces_missed"].score(it, out, None) == (1, 2) and M.METRICS["faces_missed"].score(
        it, None, "x"
    ) == (2, 2)
    assert (
        M.METRICS["face_found"].score(dict(id="n"), out, None) is None
        and M.METRICS["faces_missed"].score(dict(id="n"), out, None) is None
    )
    assert (
        M.best_face(it, dict(faces=[dict(bbox=[200, 200, 210, 210])])) is None
        and M.faces_of(None) == []
        and M.best_face(dict(), out) is None
    )


def test_verification_separates_people():
    items, outs = [], []
    for k, (person, vec) in enumerate(
        [("a", [1, 0, 0]), ("a", [0.9, 0.1, 0]), ("b", [0, 1, 0]), ("b", [0.1, 0.9, 0]), ("c", [0, 0, 1])]
    ):
        items.append(dict(id=str(k), reference=dict(person=person, box=[0, 0, 10, 10])))
        outs.append((dict(faces=[dict(bbox=[0, 0, 10, 10], embedding=vec)]), None))
    v = M.verification(items, outs)
    assert v["auc"] == 1.0 and v["tar"] == 1.0 and v["pairs"] == 10
    outs[1] = (None, "failed")
    assert M.verification(items, outs)["auc"] < 1.0
    assert math.isnan(M.verification(items[:1], outs[:1])["auc"]) and math.isnan(
        M.verification(items[:2], outs[:2])["auc"]
    )
    assert math.isnan(M.verification([dict(id="x")], [(None, None)])["auc"])


def test_generic_metrics():
    it = dict(id="1", reference=dict(answer="Yes."), _run_s=0.5)
    assert (
        M.METRICS["exact_match"].score(it, dict(answer="yes"), None) is True
        and M.METRICS["exact_match"].score(it, "no", None) is False
    )
    assert (
        M.METRICS["exact_match"].score(it, None, "x") is False
        and M.METRICS["exact_match"].score(dict(id="2"), "y", None) is None
    )
    assert (
        M.METRICS["parsed"].score(it, dict(text='{"a": 1}'), None)
        and not M.METRICS["parsed"].score(it, dict(text="{a"), None)
        and M.METRICS["parsed"].score(it, dict(a=1), None)
    )
    assert (
        not M.METRICS["parsed"].score(it, None, "x")
        and M.METRICS["failed"].score(it, None, "x")
        and M.METRICS["latency_s"].score(it, {}, None) == 0.5
        and M.METRICS["latency_s"].score(it, {}, "x") is None
    )
    assert M.METRICS["wer"].to_json()["better"] == "lower"


def test_summaries_and_comparisons_know_which_way_is_better():
    items = [
        dict(id=str(i), reference=dict(text="a b c d e f g h i j", answer="x"), _run_s=1.0 + i % 3) for i in range(40)
    ]
    good = [(tx("a b c d e f g h i j"), None)] * 40
    bad = [(tx("a b c d e f g h x x"), None)] * 40
    s = M.summarise("wer", items, bad)
    assert s["value"] == pytest.approx(0.2) and s["n"] == 40
    c = M.compare("wer", items, bad, good)
    assert c["diff"] == pytest.approx(-0.2) and c["improves"] is True
    assert (
        M.compare("wer", items, good, bad)["improves"] is False
        and M.compare("wer", items, good, good)["improves"] is None
    )
    right = [(dict(answer="x"), None)] * 40
    wrong = [(dict(answer="y"), None)] * 30 + right[:10]
    b = M.compare("exact_match", items, wrong, right)
    assert b["improves"] is True and b["only_candidate"] == 30 and b["p"] < 0.001
    assert (
        M.summarise("exact_match", items, right)["value"] == 1.0
        and M.summarise("latency_s", items, right)["p95"] == 3.0
    )
    fitems = [dict(id=str(k), reference=dict(person="ab"[k % 2], box=[0, 0, 1, 1])) for k in range(12)]
    sep = [(dict(faces=[dict(bbox=[0, 0, 1, 1], embedding=[1, 0] if k % 2 == 0 else [0, 1])]), None) for k in range(12)]
    mix = [(dict(faces=[dict(bbox=[0, 0, 1, 1], embedding=[1, 0] if k % 3 else [0, 1])]), None) for k in range(12)]
    assert (
        M.summarise("verification_auc", fitems, sep)["value"] == 1.0
        and M.compare("verification_auc", fitems, mix, sep)["diff"] > 0
    )
    none = [(None, "x")] * 12
    assert (
        M.summarise("verification_auc", fitems, none)["value"] == 0.5
        and M.compare("verification_auc", fitems, none, none)["improves"] is None
    )  # every face missing: chance
