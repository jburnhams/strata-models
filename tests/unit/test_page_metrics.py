"""Layout metrics of a rebuilt page (evals/metrics.py: block_f1, figure_f1, role_acc, order_tau) against a cleaned reference page, both as
strata_python.pages descriptions (output["page"] is what page-build drew, or what a layout reader returned)."""

import pytest

from stratamodels.evals import metrics as M


def block(box, role="body", order=None):
    b = dict(box=box, role=role, lines=[])
    if order is not None:
        b["order"] = order
    return b


REF = dict(
    width=1000,
    height=1400,
    dpi=300,
    blocks=[
        block([100, 50, 800, 40], "running-head", 0),
        block([100, 150, 800, 300], "body", 1),
        block([100, 500, 800, 300], "body", 2),
        block([450, 1300, 100, 30], "folio", 3),
    ],
    figures=[dict(box=[200, 850, 600, 400])],
)
ITEM = dict(reference=dict(page=REF))


def score(name, output, error=None, item=ITEM):
    return M.METRICS[name].score(item, output, error)


def test_a_perfect_rebuild_scores_full_marks():
    out = dict(page=REF)
    assert score("block_f1", out) == 1.0
    assert score("figure_f1", out) == 1.0
    assert score("role_acc", out) == 1.0
    assert score("order_tau", out) == 1.0


def test_merged_blocks_match_nothing_and_wrong_roles_and_order_count():
    got = dict(
        blocks=[
            block([100, 150, 800, 650], "body", 0),  # the two body blocks run together: matches neither
            block([105, 52, 790, 38], "heading", 2),  # the running head, called a heading
            block([450, 1300, 100, 30], "folio", 1),  # read before the running head: order reversed for this pair
        ]
    )
    out = dict(page=got)
    assert score("block_f1", out) == pytest.approx(2 * 2 / (4 + 3))
    assert score("role_acc", out) == 0.5
    assert score("order_tau", out) == -1.0
    assert score("figure_f1", out) == 0.0  # the figure was missed


def test_reading_order_falls_back_to_list_order_like_the_renderer():
    # no order numbers: as listed, which is the reference's order
    got = dict(blocks=[block(b["box"]) for b in REF["blocks"]])
    assert score("order_tau", dict(page=got)) == 1.0
    assert M.match_boxes(REF["blocks"], list(reversed(REF["blocks"]))) == [(0, 3), (1, 2), (2, 1), (3, 0)]


def test_failures_and_missing_parts():
    assert score("block_f1", None, error="boom") == 0.0
    assert score("role_acc", None, error="boom") is None
    assert score("block_f1", dict(page=REF), item={}) is None  # no reference page
    assert score("block_f1", dict(text="no description")) == 0.0
    no_figs = dict(reference=dict(page=dict(REF, figures=[])))
    assert score("figure_f1", dict(page=dict(REF, figures=[])), item=no_figs) is None  # nothing to find, nothing found
    one = dict(page=dict(REF, blocks=REF["blocks"][:1]))
    assert score("order_tau", one) is None  # one block found: no order to compare
    assert score("role_acc", dict(page=dict(blocks=[]))) is None


def test_summarise_runs_over_pages():
    s = M.summarise("block_f1", [ITEM, ITEM], [(dict(page=REF), None), (None, "boom")])
    assert s["value"] == 0.5
    assert s["n"] == 2
