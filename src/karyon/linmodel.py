"""linmodel — the smallest honest LEARNED model for the bio probes, stdlib only.

Why this exists: [BENCHMARK_RESULT.md] showed the sound / hand-feature layer ranks toehold ON/OFF at
only ρ≈0.13 — a feasibility GATE, not a performance predictor; the signal "lives in learned,
full-sequence models." This module is that learned core, kept to `bio/`'s dependency-free posture
(no numpy / no sklearn, same as `t0/dfm.py` standing in for DnaChisel):

  * a k-mer FREQUENCY spectrum featurizer (composition is scale-comparable across sequence lengths);
  * a Bayesian ridge regressor whose closed form gives BOTH a point prediction (`w = A⁻¹b`) AND a
    principled predictive uncertainty (`xᵀA⁻¹x`, the posterior variance up to the noise scale) from
    ONE fit — the two quantities the active-learning probe needs, with no ensemble.

The model accumulates `A = λI + Σ xxᵀ` and `b = Σ y x` INCREMENTALLY (`observe`), so the AL loop
adds a batch of labels in O(batch · p²) and refactors once per round in O(p³) — never refits from
scratch. The one nontrivial primitive is an SPD solve; `A` is SPD by construction (λ>0), so a
hand-rolled Cholesky is safe. Reused by the predictor-margin probes (RBS, promoter): same model,
new sequences.

    cd bio/probe && python linmodel.py        # self-tests (Cholesky solve/inverse + featurizer + fit)
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

_BASES = "ACGT"


# --------------------------------------------------------------------------- #
# Featurizer: intercept + concatenated k-mer frequency spectra.
# --------------------------------------------------------------------------- #
def kmer_keys(k: int) -> list[str]:
    """All 4^k k-mers in a fixed (lexicographic) order."""
    return ["".join(p) for p in itertools.product(_BASES, repeat=k)]


# Precompute keys + lookup for the small k we use; non-ACGT k-mers map to None and are dropped.
_KMER_KEYS: dict[int, list[str]] = {k: kmer_keys(k) for k in (1, 2, 3, 4)}
_KMER_IDX: dict[int, dict[str, int]] = {
    k: {km: i for i, km in enumerate(keys)} for k, keys in _KMER_KEYS.items()
}


def _kmer_freqs(seq: str, k: int) -> list[float]:
    """Frequency of each k-mer (count / number of valid windows); a length-invariant composition."""
    idx = _KMER_IDX[k]
    counts = [0.0] * len(_KMER_KEYS[k])
    total = 0
    for i in range(len(seq) - k + 1):
        j = idx.get(seq[i:i + k])
        if j is not None:                       # skip windows with a non-ACGT character (N, gap)
            counts[j] += 1.0
            total += 1
    if total:
        counts = [c / total for c in counts]
    return counts


def feature_dim(ks: tuple[int, ...]) -> int:
    """Length of `featurize(seq, ks)` — the leading intercept plus each k-mer block."""
    return 1 + sum(len(_KMER_KEYS[k]) for k in ks)


def featurize(seq: str, ks: tuple[int, ...] = (1, 2, 3)) -> list[float]:
    """Intercept (1.0) + concatenated k-mer frequency spectra. Callers may append extra scalars."""
    out = [1.0]
    for k in ks:
        out.extend(_kmer_freqs(seq, k))
    return out


def position_onehot(seq: str, length: int) -> list[float]:
    """Per-position base indicators: length×4 values, 1.0 at each position's base (all-zero for a
    position that is non-ACGT or past the end). For short FIXED-length elements (a Shine-Dalgarno
    hexamer, a fixed RBS window) this beats composition — there, *which base sits where* is the signal."""
    out = [0.0] * (length * 4)
    idx = _KMER_IDX[1]                          # base -> 0..3 (reuse the mono-nucleotide index)
    for i in range(min(len(seq), length)):
        j = idx.get(seq[i])
        if j is not None:
            out[i * 4 + j] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Linear algebra: SPD solve + inverse via Cholesky (plain nested lists, stdlib).
# --------------------------------------------------------------------------- #
def cholesky(A: list[list[float]]) -> list[list[float]]:
    """Lower-triangular L with A = L Lᵀ for symmetric positive-definite A."""
    p = len(A)
    L = [[0.0] * p for _ in range(p)]
    for i in range(p):
        Li = L[i]
        for j in range(i + 1):
            Lj = L[j]
            s = sum(Li[k] * Lj[k] for k in range(j))
            if i == j:
                d = A[i][i] - s
                if d <= 0.0:
                    raise ValueError("matrix is not positive-definite (Cholesky failed)")
                Li[j] = math.sqrt(d)
            else:
                Li[j] = (A[i][j] - s) / Lj[j]
    return L


def _solve_lower(L: list[list[float]], b: list[float]) -> list[float]:
    """Forward-substitute L y = b."""
    p = len(L)
    y = [0.0] * p
    for i in range(p):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    return y


def _solve_upper(L: list[list[float]], y: list[float]) -> list[float]:
    """Back-substitute Lᵀ x = y, reading Lᵀ off the lower factor L (Lᵀ[i][k] = L[k][i])."""
    p = len(L)
    x = [0.0] * p
    for i in reversed(range(p)):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, p))) / L[i][i]
    return x


def chol_solve(L: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b given A = L Lᵀ."""
    return _solve_upper(L, _solve_lower(L, b))


def chol_inv(L: list[list[float]]) -> list[list[float]]:
    """A⁻¹ given A = L Lᵀ, one solve per identity column."""
    p = len(L)
    cols = [chol_solve(L, [1.0 if i == c else 0.0 for i in range(p)]) for c in range(p)]
    return [[cols[c][r] for c in range(p)] for r in range(p)]


# --------------------------------------------------------------------------- #
# The model.
# --------------------------------------------------------------------------- #
@dataclass
class BayesRidge:
    """Ridge regression as a Gaussian posterior: A = λI + Σ xxᵀ, b = Σ y x.

    `predict` uses the MAP weights A⁻¹b; `variance` returns xᵀA⁻¹x — the posterior predictive
    variance up to the (constant) noise scale, which is all an acquisition RANKING needs. Both reuse
    one Cholesky factor of A, recomputed lazily after `observe` dirties it."""

    p: int
    lam: float = 1.0
    A: list[list[float]] = field(default=None, repr=False)   # type: ignore[assignment]
    b: list[float] = field(default=None, repr=False)         # type: ignore[assignment]
    n: int = 0

    def __post_init__(self) -> None:
        self.A = [[self.lam if i == j else 0.0 for j in range(self.p)] for i in range(self.p)]
        self.b = [0.0] * self.p
        self._L: list[list[float]] | None = None
        self._w: list[float] | None = None
        self._Ainv: list[list[float]] | None = None

    def observe(self, x: list[float], y: float) -> None:
        """Fold one labeled point into A and b (rank-1 update); invalidate the cached factor."""
        A, b, p = self.A, self.b, self.p
        for i in range(p):
            xi = x[i]
            if xi == 0.0:                       # k-mer spectra are sparse-ish; skip empty rows
                continue
            Ai = A[i]
            for j in range(p):
                xj = x[j]
                if xj != 0.0:
                    Ai[j] += xi * xj
            b[i] += y * xi
        self.n += 1
        self._L = self._w = self._Ainv = None

    def observe_all(self, xs: list[list[float]], ys: list[float]) -> None:
        for x, y in zip(xs, ys):
            self.observe(x, y)

    def _factor(self) -> list[list[float]]:
        if self._L is None:
            self._L = cholesky(self.A)
        return self._L

    def weights(self) -> list[float]:
        if self._w is None:
            self._w = chol_solve(self._factor(), self.b)
        return self._w

    def predict(self, x: list[float]) -> float:
        return sum(wi * xi for wi, xi in zip(self.weights(), x))

    def _ainv(self) -> list[list[float]]:
        if self._Ainv is None:
            self._Ainv = chol_inv(self._factor())
        return self._Ainv

    def variance(self, x: list[float]) -> float:
        """xᵀA⁻¹x — relative predictive uncertainty (the noise scale is a common factor, dropped)."""
        Ainv = self._ainv()
        return sum(xi * sum(Ainv[i][j] * x[j] for j in range(self.p))
                   for i, xi in enumerate(x) if xi != 0.0)


# --------------------------------------------------------------------------- #
# Self-tests.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    def approx(a: float, b: float, tol: float = 1e-9) -> None:
        assert abs(a - b) < tol, f"{a} != {b}"

    # Cholesky solve against a hand-checkable SPD system: A = [[4,2],[2,3]], A x = [1,1].
    #   det=8, A⁻¹ = [[3,-2],[-2,4]]/8 -> x = [1/8, 2/8] = [0.125, 0.25].
    A = [[4.0, 2.0], [2.0, 3.0]]
    L = cholesky(A)
    x = chol_solve(L, [1.0, 1.0])
    approx(x[0], 0.125); approx(x[1], 0.25)
    # A·A⁻¹ = I.
    Ainv = chol_inv(L)
    for i in range(2):
        for j in range(2):
            approx(sum(A[i][k] * Ainv[k][j] for k in range(2)), 1.0 if i == j else 0.0, 1e-9)
    print("1. cholesky solve + inverse correct on a hand-checked SPD system")

    # Featurizer: frequencies sum to 1 per k-block; dimension matches feature_dim.
    f = featurize("ACGTACGT", ks=(1, 2))
    assert len(f) == feature_dim((1, 2)) == 1 + 4 + 16
    approx(sum(f[1:5]), 1.0)                    # the four mono-nt frequencies
    approx(sum(f[5:21]), 1.0)                   # the sixteen di-nt frequencies
    approx(f[1 + _KMER_IDX[1]["A"]], 0.25)      # "ACGTACGT": 2 A of 8 -> 0.25
    # non-ACGT characters are dropped, not crashed on:
    assert featurize("ACGTN", ks=(1,))[1 + _KMER_IDX[1]["A"]] > 0.0
    print("2. featurizer: per-k frequencies normalize, dimension exact, non-ACGT dropped")

    # The model recovers a known linear signal: y = w_true · x on random-ish x -> high test ρ.
    import random
    rng = random.Random(0)
    p = feature_dim((1, 2, 3))
    w_true = [rng.uniform(-1, 1) for _ in range(p)]
    seqs = ["".join(rng.choice(_BASES) for _ in range(60)) for _ in range(400)]
    vecs = [featurize(s, (1, 2, 3)) for s in seqs]
    ys = [sum(wi * xi for wi, xi in zip(w_true, v)) for v in vecs]
    model = BayesRidge(p, lam=1e-6)
    model.observe_all(vecs[:300], ys[:300])
    preds = [model.predict(v) for v in vecs[300:]]
    truth = ys[300:]
    # Pearson r of preds vs truth, inline (no stats_kit dependency in the model layer).
    n = len(preds)
    mp, mt = sum(preds) / n, sum(truth) / n
    cov = sum((a - mp) * (b - mt) for a, b in zip(preds, truth))
    vp = sum((a - mp) ** 2 for a in preds)
    vt = sum((b - mt) ** 2 for b in truth)
    r = cov / math.sqrt(vp * vt)
    assert r > 0.99, f"ridge failed to recover a clean linear signal (r={r:.3f})"
    print(f"3. BayesRidge recovers a clean linear signal (held-out r={r:.4f})")

    # Variance is non-negative and SHRINKS where data is dense: a point near many observations has
    # lower xᵀA⁻¹x than a far-out direction. Sanity: variance of a seen-ish x < a wild x.
    seen_like = vecs[0]
    wild = [v * 50.0 for v in vecs[0]]          # same direction, blown-up magnitude -> higher xᵀA⁻¹x
    assert 0.0 <= model.variance(seen_like) < model.variance(wild)
    print("4. predictive variance is non-negative and grows away from the observed region")

    # Positional one-hot: right base set at the right offset; out-of-range positions stay zero.
    oh = position_onehot("AC", 3)
    assert len(oh) == 12 and sum(oh) == 2.0
    assert oh[0 * 4 + _KMER_IDX[1]["A"]] == 1.0 and oh[1 * 4 + _KMER_IDX[1]["C"]] == 1.0
    assert sum(oh[8:12]) == 0.0                 # position 2 is past the end -> all zero
    print("5. positional one-hot places the right base at the right offset")

    print("\nlinmodel self-tests pass.")
