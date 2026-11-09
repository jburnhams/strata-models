"""The statistics every eval uses, in one place, so every write-up means the same thing by "95 % interval" and "p".

Items are the unit of resampling: a run's score is a mean over items, and two candidates are always compared on the SAME items (paired), which
removes the item-to-item variation that would otherwise swamp the difference. Methods:
  proportion          Wilson score interval (correct near 0 and 1, unlike the normal approximation).
  mean                percentile bootstrap over items (B = 4000, fixed seed: a report is reproducible).
  paired difference   bootstrap of the per-item differences (B - A), and a two-sided p from the share of resampled means on the other side of 0
                      (sign-flip permutation, exact enough at B = 4000).
  paired binary       McNemar exact test on the discordant pairs (A right, B wrong / the reverse).
  several candidates  Holm's step-down correction of the p values of the comparisons with the baseline.
  pairwise judgments  Bradley-Terry strengths (MM algorithm), with bootstrap intervals over judgments, and the win share per pair (Wilson).
  sample size         items needed to see a given difference with 80 % power (normal approximation on the paired differences).
Only numpy."""
import math

import numpy as np

B = 4000
SEED = 20261109
Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """(share, low, high) for k successes in n trials."""
    if n == 0: return (float('nan'),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def _rng(seed): return np.random.default_rng(SEED if seed is None else seed)


def bootstrap_mean(values, b=B, seed=None, level=0.95):
    """(mean, low, high): percentile bootstrap of the mean over items. NaN values (an item the metric does not apply to) are left out."""
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0: return (float('nan'),) * 3
    if v.size == 1: return float(v[0]), float(v[0]), float(v[0])
    idx = _rng(seed).integers(0, v.size, size=(b, v.size)); means = v[idx].mean(axis=1); a = (1 - level) / 2
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired_diff(a, b_, b=B, seed=None, level=0.95):
    """Compare two candidates on the same items: dict(n, mean_a, mean_b, diff (B minus A), low, high, p). Items missing in either are dropped."""
    pairs = [(x, y) for x, y in zip(a, b_) if x is not None and y is not None and not math.isnan(x) and not math.isnan(y)]
    if not pairs: return dict(n=0, mean_a=float('nan'), mean_b=float('nan'), diff=float('nan'), low=float('nan'), high=float('nan'), p=float('nan'))
    x = np.array([p[0] for p in pairs], float); y = np.array([p[1] for p in pairs], float); d = y - x; n = d.size; rng = _rng(seed)
    if n == 1 or np.all(d == d[0]): return dict(n=n, mean_a=float(x.mean()), mean_b=float(y.mean()), diff=float(d.mean()), low=float(d.mean()), high=float(d.mean()), p=1.0 if d[0] == 0 else float('nan'))
    idx = rng.integers(0, n, size=(b, n)); means = d[idx].mean(axis=1); a_ = (1 - level) / 2
    signs = rng.choice((-1.0, 1.0), size=(b, n)); null = (signs * d).mean(axis=1)                  # sign-flip permutation: under "no difference" each item's difference is as likely negative
    p = float((np.sum(np.abs(null) >= abs(d.mean()) - 1e-12) + 1) / (b + 1))
    return dict(n=n, mean_a=float(x.mean()), mean_b=float(y.mean()), diff=float(d.mean()), low=float(np.quantile(means, a_)), high=float(np.quantile(means, 1 - a_)), p=p)


def mcnemar(a_right, b_right):
    """Exact McNemar test for paired right/wrong outcomes: dict(only_a, only_b, p) (two-sided binomial on the discordant pairs)."""
    only_a = sum(1 for x, y in zip(a_right, b_right) if x and not y); only_b = sum(1 for x, y in zip(a_right, b_right) if y and not x); n = only_a + only_b
    if n == 0: return dict(only_a=0, only_b=0, p=1.0)
    k = min(only_a, only_b); tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return dict(only_a=only_a, only_b=only_b, p=min(1.0, 2 * tail))


def holm(pvalues):
    """Holm-adjusted p values, in the order given (NaN stays NaN)."""
    idx = [i for i, p in enumerate(pvalues) if p is not None and not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvalues[i]); out = list(pvalues); m = len(order); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i])); out[i] = running
    return out


def bradley_terry(judgments, names, iters=200):
    """Strengths (summing to len(names)) from (a, b, winner) judgments, winner in (a, b, 'tie'); a tie counts half a win to each. MM algorithm."""
    k = len(names); ix = {n: i for i, n in enumerate(names)}; w = np.zeros((k, k))
    for a, b, win in judgments:
        if a not in ix or b not in ix or a == b: continue
        i, j = ix[a], ix[b]
        if win == a: w[i, j] += 1
        elif win == b: w[j, i] += 1
        else: w[i, j] += 0.5; w[j, i] += 0.5
    n = w + w.T; s = np.ones(k)
    for _ in range(iters):
        wins = w.sum(axis=1); denom = np.array([sum(n[i, j] / (s[i] + s[j]) for j in range(k) if j != i and n[i, j]) for i in range(k)])
        new = np.where(denom > 0, wins / np.where(denom > 0, denom, 1), s); new = np.maximum(new, 1e-9); new *= k / new.sum()
        if np.allclose(new, s, rtol=1e-8): s = new; break
        s = new
    return dict(zip(names, (float(x) for x in s)))


def bradley_terry_ci(judgments, names, b=1000, seed=None, level=0.95):
    """{name: (strength, low, high)}: Bradley-Terry with a bootstrap over judgments."""
    est = bradley_terry(judgments, names)
    if not judgments: return {n: (est[n], float('nan'), float('nan')) for n in names}
    rng = _rng(seed); j = list(judgments); boots = {n: [] for n in names}
    for _ in range(b):
        s = bradley_terry([j[i] for i in rng.integers(0, len(j), len(j))], names)
        for n in names: boots[n].append(s[n])
    a = (1 - level) / 2
    return {n: (est[n], float(np.quantile(boots[n], a)), float(np.quantile(boots[n], 1 - a))) for n in names}


def items_needed(diffs, effect, power_z=0.8416212335729143):
    """Items needed for a paired comparison to find a true difference of `effect` with 80 % power at 5 % two-sided, from the spread of the
    per-item differences seen so far (None when they do not vary)."""
    d = np.asarray([x for x in diffs if x is not None and not math.isnan(x)], float)
    if d.size < 2 or not effect: return None
    sd = float(d.std(ddof=1))
    return None if sd == 0 else int(math.ceil(((Z95 + power_z) * sd / abs(effect)) ** 2))


def quantile(values, q):
    v = [x for x in values if x is not None and not math.isnan(x)]
    return float(np.quantile(v, q)) if v else float('nan')


def bootstrap_ratio(nums, dens, b=B, seed=None, level=0.95):
    """(sum(nums) / sum(dens), low, high) with items resampled: for corpus-level rates such as WER (all errors over all reference words)."""
    n_ = np.asarray(nums, float); d_ = np.asarray(dens, float)
    if n_.size == 0 or d_.sum() == 0: return (float('nan'),) * 3
    idx = _rng(seed).integers(0, n_.size, size=(b, n_.size)); r = n_[idx].sum(1) / np.maximum(d_[idx].sum(1), 1e-12); a = (1 - level) / 2
    return float(n_.sum() / d_.sum()), float(np.quantile(r, a)), float(np.quantile(r, 1 - a))


def paired_ratio_diff(num_a, num_b, dens, b=B, seed=None, level=0.95):
    """Corpus rate of B minus that of A on the same items (same denominators): dict like paired_diff. p from swapping A and B per item at random."""
    x = np.asarray(num_a, float); y = np.asarray(num_b, float); d = np.asarray(dens, float); n = x.size
    if n == 0 or d.sum() == 0: return dict(n=0, mean_a=float('nan'), mean_b=float('nan'), diff=float('nan'), low=float('nan'), high=float('nan'), p=float('nan'))
    ra, rb = x.sum() / d.sum(), y.sum() / d.sum(); rng = _rng(seed)
    idx = rng.integers(0, n, size=(b, n)); diffs = (y[idx].sum(1) - x[idx].sum(1)) / np.maximum(d[idx].sum(1), 1e-12); a_ = (1 - level) / 2
    swap = rng.random((b, n)) < 0.5; null = (np.where(swap, x, y).sum(1) - np.where(swap, y, x).sum(1)) / d.sum()
    p = float((np.sum(np.abs(null) >= abs(rb - ra) - 1e-12) + 1) / (b + 1))
    return dict(n=n, mean_a=float(ra), mean_b=float(rb), diff=float(rb - ra), low=float(np.quantile(diffs, a_)), high=float(np.quantile(diffs, 1 - a_)), p=p)
