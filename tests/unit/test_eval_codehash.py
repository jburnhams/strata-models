"""Which code decides an analysis (stratamodels.evals.codehash)."""

import ast
import types

import pytest

from stratamodels.evals import codehash as H
from stratamodels.evals import metrics as M


@pytest.fixture(autouse=True)
def fresh():
    yield
    for f in (H._module, H._definition, H._code):
        f.cache_clear()


def spec(*metrics, judges=()):
    return types.SimpleNamespace(metrics=list(metrics), primary=metrics[0], guards={}, judges=list(judges))


def edited(monkeypatch, module, name, change):
    """The hashes as they would be after `change` was made to the text of one definition."""
    m = H._module(module)
    nodes = [ast.parse(change(ast.unparse(n))).body[0] for n in m.defs[name]]
    monkeypatch.setitem(m.defs, name, nodes)
    H._definition.cache_clear()
    H._code.cache_clear()


def test_an_edit_to_a_metric_redoes_only_the_experiments_that_use_it(monkeypatch):
    wer, ink = H.code(spec("wer")), H.code(spec("page_ink_f1"))
    assert wer != ink and wer == H.code(spec("wer"))
    edited(monkeypatch, "evals.metrics", "_wer", lambda s: s.replace("return", "x = 1\n    return", 1))
    assert H.code(spec("wer")) != wer and H.code(spec("page_ink_f1")) == ink


def test_code_a_metric_reaches_in_another_module_counts(monkeypatch):
    wer, ink = H.code(spec("wer")), H.code(spec("page_ink_f1"))
    edited(monkeypatch, "evals.ink", "ink_f1", lambda s: s + "\n    pass")
    assert H.code(spec("page_ink_f1")) != ink and H.code(spec("wer")) == wer


def test_how_a_metric_samples_counts(monkeypatch):
    before = H.code(spec("pace_spread"))
    edited(monkeypatch, "evals.metrics", "Metric", lambda s: s.replace("self.prepare is not None", "False"))
    assert H.code(spec("pace_spread")) != before


def test_a_docstring_does_not_count(monkeypatch):
    before = H.code(spec("wer"))
    edited(monkeypatch, "evals.core", "analyse", lambda s: s.replace("Everything a report says", "All a report says"))
    assert H.code(spec("wer")) == before
    edited(monkeypatch, "evals.core", "analyse", lambda s: s + "\n    return 1")
    assert H.code(spec("wer")) != before


def test_judge_code_counts_only_for_an_experiment_with_judges(monkeypatch):
    plain, judged = H.code(spec("wer")), H.code(spec("wer", judges=["j"]))
    edited(monkeypatch, "evals.judges", "agreement", lambda s: s + "\n    pass")
    assert H.code(spec("wer")) == plain and H.code(spec("wer", judges=["j"])) != judged


def test_what_a_made_metric_closed_over_tells_it_from_its_siblings():
    roots, extra = H._metric_roots("block_f1")
    assert extra and H.code(spec("block_f1")) != H.code(spec("role_acc"))
    assert all(m.startswith("evals.") for m, _ in roots)


def test_every_metric_has_its_code_found():
    for n, m in M.METRICS.items():
        roots, extra = H._metric_roots(n)
        assert extra is None or all(r in H._reach(roots) for r in roots), n
        assert H.code(spec(n))
        assert roots or extra is None or (m.score is None and m.dataset is None)


def test_an_unknown_metric_is_left_out():
    assert H.code(spec("nope"))
