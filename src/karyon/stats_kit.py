"""stats_kit — the smallest honest statistics for the wet-data benchmark, stdlib only.

`bio/` is deliberately dependency-free (the same posture as `t0/dfm.py` standing in for DnaChisel),
so the benchmark hand-rolls the three statistics it needs rather than importing scipy:

  * `spearman` / `pearson` — does a predictor's RANKING track measured ON/OFF? (Q1, Q3)
  * `mann_whitney` -> `auroc` — does a contract's pass/fail PARTITION separate good from bad? (Q2)

The one real footgun is a **constant predictor**: the binding proxy is `30` on every designed
switch (`switch == rc(trigger)` by construction), and a correlation of a zero-variance vector is
`0/0`. Rather than emit `nan` or crash, every function returns a typed `Degenerate` carrying the
reason, so the benchmark can render "constant by construction — a gate, not a predictor" as the
finding it is. All ranking is tie-aware (average ranks); the Mann-Whitney variance is tie-corrected.

    cd bio/probe && python stats_kit.py        # self-tests against hand-computed values
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Corr:
    rho: float
    n: int
    method: str          # "spearman" | "pearson"


@dataclass(frozen=True)
class MannWhitney:
    auroc: float         # P(a random group-A value outranks a random group-B value); ties = 0.5
    u: float
    p: float             # two-sided, tie-corrected normal approximation
    n_a: int
    n_b: int


@dataclass(frozen=True)
class Degenerate:
    """A statistic that cannot be computed meaningfully (constant input / too few points)."""

    reason: str
    n: int


Result = Corr | Degenerate


def rank_avg(xs: list[float]) -> list[float]:
    """1-based ranks with ties assigned their average rank (the Spearman / Mann-Whitney convention)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0                       # mean of the 1-based ranks i+1 .. j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _clean(xs: list[float | None], ys: list[float | None]) -> tuple[list[float], list[float]]:
    """Drop index-aligned pairs where either value is missing."""
    a, b = [], []
    for x, y in zip(xs, ys):
        if x is not None and y is not None:
            a.append(float(x))
            b.append(float(y))
    return a, b


def _pearson_raw(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None                                    # a vector is constant -> undefined
    return sxy / math.sqrt(sxx * syy)


def pearson(xs: list[float | None], ys: list[float | None]) -> Result:
    a, b = _clean(xs, ys)
    if len(a) < 3:
        return Degenerate(f"n<3 after dropping missing (n={len(a)})", len(a))
    r = _pearson_raw(a, b)
    if r is None:
        return Degenerate("zero variance (a vector is constant)", len(a))
    return Corr(r, len(a), "pearson")


def spearman(xs: list[float | None], ys: list[float | None]) -> Result:
    a, b = _clean(xs, ys)
    if len(a) < 3:
        return Degenerate(f"n<3 after dropping missing (n={len(a)})", len(a))
    if len(set(a)) == 1 or len(set(b)) == 1:
        return Degenerate(
            f"zero variance — constant predictor (distinct x={len(set(a))}, y={len(set(b))})",
            len(a))
    r = _pearson_raw(rank_avg(a), rank_avg(b))
    if r is None:                                      # unreachable given the check above, but safe
        return Degenerate("zero rank variance", len(a))
    return Corr(r, len(a), "spearman")


def _two_sided_p(z: float) -> float:
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def mann_whitney(group_a: list[float | None], group_b: list[float | None]) -> MannWhitney | Degenerate:
    """U / AUROC for group_a vs group_b, tie-aware. AUROC = P(A outranks B), 0.5 = no separation."""
    a = [float(v) for v in group_a if v is not None]
    b = [float(v) for v in group_b if v is not None]
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return Degenerate(f"an empty group (n_a={n_a}, n_b={n_b})", n_a + n_b)
    pooled = a + b
    ranks = rank_avg(pooled)
    r_a = sum(ranks[:n_a])
    u_a = r_a - n_a * (n_a + 1) / 2.0
    auroc = u_a / (n_a * n_b)
    n = n_a + n_b
    tie = sum(t ** 3 - t for t in Counter(pooled).values())
    var = (n_a * n_b / 12.0) * ((n + 1) - tie / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return MannWhitney(auroc, u_a, 1.0, n_a, n_b)
    z = (u_a - n_a * n_b / 2.0) / math.sqrt(var)
    return MannWhitney(auroc, u_a, _two_sided_p(z), n_a, n_b)


def fmt(r: Result | MannWhitney | Degenerate) -> str:
    """Compact rendering for the benchmark report."""
    if isinstance(r, Corr):
        return f"{r.method} ρ={r.rho:+.3f} (n={r.n})"
    if isinstance(r, MannWhitney):
        return f"AUROC={r.auroc:.3f} p={r.p:.1e} (n={r.n_a}/{r.n_b})"
    return f"DEGENERATE — {r.reason}"


# --------------------------------------------------------------------------- #
# Screen-QC additions (still stdlib-only): multiple-testing, AUPRC, a CI, and a
# label-free mean→spread trend for moderating count log-fold-changes.
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg FDR q-values, returned in INPUT order, with step-up monotonicity enforced.
    q_(k) = min over j≥k of (n/j)·p_(j), clamped to 1."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    running = 1.0
    for rank in range(n - 1, -1, -1):                  # largest p first (step-up)
        i = order[rank]
        running = min(running, pvals[i] * n / (rank + 1))
        q[i] = min(running, 1.0)
    return q


def average_precision(scores: list[float], labels: list[bool]) -> float:
    """Area under the precision–recall curve (AUPRC), step estimator; higher `score` ⇒ more positive.
    Equals mean precision over the positives' ranks. No positives ⇒ 0.0. Ties broken stably."""
    pos = sum(1 for v in labels if v)
    if pos == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    ap = 0.0
    for seen, i in enumerate(order, start=1):
        if labels[i]:
            tp += 1
            ap += tp / seen                            # precision@rank, ΔRecall = 1/pos
    return ap / pos


def bootstrap_ci(items: list, statistic: Callable[[list], float],
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for `statistic` over `items` (resample-with-replacement). Deterministic
    given `seed`; returns (lo, hi) at the central 1−alpha mass. Empty input ⇒ (nan, nan)."""
    n = len(items)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = sorted(statistic([items[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def mean_variance_trend(means: list[float], values: list[float], n_bins: int = 20
                        ) -> Callable[[float], float]:
    """A label-free mean→spread trend: split points into `n_bins` equal-count bins by `means`, take a
    ROBUST spread (1.4826·MAD) of `values` per bin, and return a callable that piecewise-linearly
    interpolates spread at a given mean (clamped past the end bins). Used to MODERATE per-guide
    log-fold-changes — low-count guides have a wider null spread, so their extreme LFCs are discounted.
    Robust spread resists the essential-guide minority. Empty input ⇒ a constant 1.0."""
    pts = sorted(zip(means, values))
    n = len(pts)
    if n == 0:
        return lambda m: 1.0
    nb = max(1, min(n_bins, n))
    centers: list[float] = []
    spreads: list[float] = []
    for b in range(nb):
        chunk = pts[b * n // nb:(b + 1) * n // nb]
        if not chunk:
            continue
        vs = [v for _, v in chunk]
        med = statistics.median(vs)
        mad = statistics.median([abs(v - med) for v in vs])
        centers.append(statistics.fmean([m for m, _ in chunk]))
        spreads.append(1.4826 * mad if mad > 0 else 1e-9)
    def trend(m: float) -> float:
        if m <= centers[0]:
            return spreads[0]
        if m >= centers[-1]:
            return spreads[-1]
        for k in range(1, len(centers)):
            if m <= centers[k]:
                t = (m - centers[k - 1]) / (centers[k] - centers[k - 1] + 1e-12)
                return spreads[k - 1] + t * (spreads[k] - spreads[k - 1])
        return spreads[-1]
    return trend


if __name__ == "__main__":
    def approx(x, y, tol=1e-9):
        assert abs(x - y) < tol, f"{x} != {y}"

    # Spearman: perfect monotone (with a nonlinear map) -> +1; reversed -> -1.
    r = spearman([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]); assert isinstance(r, Corr); approx(r.rho, 1.0)
    r = spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]); approx(r.rho, -1.0)
    # Pearson: exact linear -> +1.
    r = pearson([1, 2, 3], [2, 4, 6]); assert isinstance(r, Corr); approx(r.rho, 1.0)
    # Tie handling: hand-computed Spearman for x=[1,2,2,3], y=[10,20,30,40].
    #   rank_avg(x)=[1,2.5,2.5,4], rank_avg(y)=[1,2,3,4]; pearson of those = 0.948683...
    r = spearman([1, 2, 2, 3], [10, 20, 30, 40]); approx(r.rho, 0.9486832980505138, 1e-12)
    # Degenerate: constant predictor.
    d = spearman([5, 5, 5, 5], [1, 2, 3, 4]); assert isinstance(d, Degenerate), d
    d = pearson([1, 2, 3], [7, 7, 7]); assert isinstance(d, Degenerate), d
    # Missing-value pairs are dropped before n<3 check.
    d = spearman([1, None, 2], [None, 5, 6]); assert isinstance(d, Degenerate) and d.n == 1, d
    # Mann-Whitney / AUROC: fully separated groups -> AUROC 1.0; identical -> 0.5.
    mw = mann_whitney([4, 5, 6], [1, 2, 3]); assert isinstance(mw, MannWhitney); approx(mw.auroc, 1.0)
    mw = mann_whitney([1, 2, 3], [4, 5, 6]); approx(mw.auroc, 0.0)
    mw = mann_whitney([1, 2, 3], [1, 2, 3]); approx(mw.auroc, 0.5)
    # AUROC with one tie across groups: A=[2,3], B=[1,2] -> pairs (2v1)=1,(2v2)=.5,(3v1)=1,(3v2)=1 => 3.5/4.
    mw = mann_whitney([2, 3], [1, 2]); approx(mw.auroc, 0.875)
    mw = mann_whitney([], [1, 2]); assert isinstance(mw, Degenerate), mw

    # Benjamini–Hochberg: hand-computed. p=[0.005,0.04,0.5] -> q=[0.015,0.06,0.5].
    q = benjamini_hochberg([0.005, 0.04, 0.5])
    approx(q[0], 0.015); approx(q[1], 0.06); approx(q[2], 0.5)
    # Step-up monotonicity: a tiny p after a big one can't make a later q smaller than an earlier one.
    q = benjamini_hochberg([0.04, 0.005, 0.5]); approx(q[1], 0.015); approx(q[0], 0.06)
    assert benjamini_hochberg([]) == []
    # Average precision: AUPRC. Perfect ranking -> 1.0; one interleave -> 0.8333…
    approx(average_precision([0.9, 0.8, 0.1, 0.05], [True, True, False, False]), 1.0)
    approx(average_precision([0.9, 0.8, 0.7, 0.6], [True, False, True, False]), (1.0 + 2 / 3) / 2)
    approx(average_precision([0.1, 0.2], [False, False]), 0.0)        # no positives
    # Bootstrap CI: a constant sample has a degenerate (point) CI; bounds ordered otherwise.
    lo, hi = bootstrap_ci([5.0] * 8, lambda s: sum(s) / len(s), n_boot=200, seed=1)
    approx(lo, 5.0); approx(hi, 5.0)
    lo, hi = bootstrap_ci(list(range(100)), lambda s: sum(s) / len(s), n_boot=500, seed=1)
    assert lo < hi and 40 < (lo + hi) / 2 < 60, (lo, hi)
    # Mean-variance trend: heteroscedastic by construction (wide at low mean, narrow at high) -> trend
    # must be larger at the low end. Means 0..199; spread shrinks with the mean.
    rng = random.Random(0)
    ms = list(range(200))
    vs = [rng.gauss(0, 1.0 + (200 - m) / 50.0) for m in ms]           # sd ~5 at m=0, ~1 at m=200
    tr = mean_variance_trend(ms, vs, n_bins=10)
    assert tr(10) > tr(190), (tr(10), tr(190))
    print("stats_kit self-tests pass.")
