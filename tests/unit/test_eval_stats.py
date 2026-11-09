"""stratamodels.evals.stats: the intervals and tests every report relies on, checked against known answers."""

import math

import pytest

from stratamodels.evals import stats as S


def test_wilson_matches_the_textbook_values():
    p, lo, hi = S.wilson(8, 10)
    assert p == 0.8 and lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert S.wilson(0, 20)[1] == 0 and S.wilson(20, 20)[2] == 1 and all(math.isnan(x) for x in S.wilson(0, 0))


def test_bootstrap_mean_is_reproducible_and_skips_missing():
    a = S.bootstrap_mean([1, 2, 3, 4, 5, None, float("nan")])
    assert a == S.bootstrap_mean([1, 2, 3, 4, 5]) and a[0] == 3 and a[1] < 3 < a[2]
    assert S.bootstrap_mean([7.0]) == (7.0, 7.0, 7.0) and math.isnan(S.bootstrap_mean([])[0])


def test_paired_difference_finds_a_consistent_shift_and_not_noise():
    a = [0.1 + 0.05 * (i % 7) for i in range(40)]
    r = S.paired_diff(a, [x + 0.02 for x in a])
    assert r["diff"] == pytest.approx(0.02) and r["low"] == pytest.approx(0.02) and r["n"] == 40
    noise = S.paired_diff(a, [x + (0.01 if i % 2 else -0.01) for i, x in enumerate(a)])
    assert noise["low"] < 0 < noise["high"] and noise["p"] > 0.5
    shifted = S.paired_diff(a, [x + 0.02 + (0.01 if i % 2 else -0.01) for i, x in enumerate(a)])
    assert shifted["low"] > 0 and shifted["p"] < 0.001
    assert (
        S.paired_diff([1, None], [2, 3])["n"] == 1
        and S.paired_diff([], [])["n"] == 0
        and S.paired_diff([1, 1], [1, 1])["p"] == 1.0
    )


def test_ratio_and_its_paired_difference():
    v, lo, hi = S.bootstrap_ratio([1, 2, 0, 3], [10, 10, 10, 10])
    assert v == pytest.approx(0.15) and lo <= v <= hi
    r = S.paired_ratio_diff([2] * 30, [1] * 30, [10] * 30)
    assert r["diff"] == pytest.approx(-0.1) and r["high"] < 0 and r["p"] < 0.001
    assert math.isnan(S.bootstrap_ratio([], [])[0]) and S.paired_ratio_diff([], [], [])["n"] == 0


def test_mcnemar_exact_and_holm():
    assert S.mcnemar([1] * 10, [1] * 10)["p"] == 1.0
    r = S.mcnemar([1] * 8 + [0] * 2, [0] * 8 + [1] * 2)
    assert (r["only_a"], r["only_b"]) == (8, 2) and r["p"] == pytest.approx(0.1094, abs=1e-3)
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06]) and math.isnan(
        S.holm([float("nan"), 0.02])[0]
    )


def test_bradley_terry_orders_and_bounds():
    j = (
        [("a", "b", "a")] * 8
        + [("a", "b", "b")] * 2
        + [("b", "c", "b")] * 7
        + [("b", "c", "c")] * 3
        + [("a", "c", "tie")] * 2
        + [("a", "a", "a"), ("z", "a", "z")]
    )
    s = S.bradley_terry(j, ["a", "b", "c"])
    assert s["a"] > s["b"] > s["c"] and sum(s.values()) == pytest.approx(3)
    ci = S.bradley_terry_ci(j, ["a", "b", "c"], b=100)
    assert all(lo <= v <= hi for v, lo, hi in ci.values())
    assert math.isnan(S.bradley_terry_ci([], ["a", "b"])["a"][1])


def test_items_needed_and_quantile():
    assert (
        S.items_needed([0.01, -0.01, 0.03, -0.02, 0.0], 0.005) > 30
        and S.items_needed([0.1, 0.1], 0.01) is None
        and S.items_needed([0.1], 0.01) is None
    )
    assert S.quantile([1, 2, 3, 4], 0.5) == 2.5 and math.isnan(S.quantile([None], 0.5))
