"""promoter_predictor — the PROMOTER predictor margin: a cheap learned core vs the Promoter Calculator,
in-distribution at scale. The edge's SECOND data point.

Earlier evaluation settled the RBS predictor margin: in-distribution at scale a cheap
k-mer+positional stdlib ridge BEAT the RBS Calculator (OSTIR) de-novo (ρ +0.88 vs +0.65), after LOSING
out-of-distribution at small scale. That gap — win in-distribution, lose
out — is the edge mechanism: "the proprietary same-assay dataloop manufactures the in-distribution-at-
scale regime where the cheap predictor wins." It rests on ONE substrate (RBS). This probe asks whether
it GENERALIZES to a second — bacterial σ70 promoter strength (La Fleur/Salis 2022, Urtecho set, 10,898
promoters) — reusing [linmodel.py]'s featurizers + the numpy fast-path pattern [rbs_hollerer_predictor.py]
established, and the Promoter Calculator's OWN prediction column already cached by [promoter_data.py] as
the baseline (no ViennaRNA, no fetch, fully offline).

REGIME (same as Höllerer, stated plainly): ONE library, one assay, random train/eval split →
de-novo := held-out SEQUENCES from the same library (in-distribution at scale). NOT cross-library LOSO —
only one library exists here.

THE HONEST CAVEAT (load-bearing): the baseline is the published
Promoter Calculator's number ON THIS SET, so it may be IN-SAMPLE for the calculator (it may have trained
on these rows). That makes the calc *advantaged*, so:
  * a learned-core WIN in-distribution is CONSERVATIVE (it beats a possibly-cheating baseline) → strong;
  * a learned-core LOSS at 10.9k is INCONCLUSIVE on the edge but a real "promoter is more data-hungry
    than RBS" finding. The scale curve below shows which way it is trending.

THE LEARNED ARMS (all reuse [linmodel.py]; a new substrate cost only this file):
  1. core      — k-mer (1,2,3) freq + GC, stdlib BayesRidge. The discovery probe's exact ρ≈0.51 setup.
  2. rich      — k-mer (1,2,3,4) + GC (add tetramers).
  3. rich+pos  — rich + positional one-hot over the fixed 150-nt window (which-base-where; the σ70
                 −35/−10/spacer signal sits at positions).
  rich/rich+pos fit via a numpy normal-equations ridge (IDENTICAL math to BayesRidge — proved to 1e-13
  in the tests; numpy is importable, NOTHING is installed) since p≈342/942 is too slow for pure-Python.

BASELINE — the deposited Promoter Calculator column (`Record.calc_pred`, "Predicted log(TX/Txref)"),
SIGN-ALIGNED (it is sign-inverted vs strength: more negative = stronger → we rank by -calc_pred), scored
on the SAME eval rows. Metric: within-eval Spearman ρ vs measured log-strength (rank, scale-free).

    python -m karyon.promoter_predictor
    python -m karyon.promoter_predictor --seeds 3 --n-eval 3000
"""

from __future__ import annotations

import argparse
import random

from . import linmodel as lm
from . import promoter_data as pd
from . import stats_kit as sk

try:
    import numpy as np
except ImportError:                                  # rich arms are numpy-accelerated; degrade to core
    np = None

PROMOTER_LEN = pd.PROMOTER_LEN          # 150 (the Urtecho fixed promoter length)
KS_CORE = (1, 2, 3)
KS_RICH = (1, 2, 3, 4)
LAM = 1.0
SEED_BASE = 0                            # random-split seeds are SEED_BASE + s


# --------------------------------------------------------------------------- #
# Featurizers (all over linmodel) — a new substrate cost only these.
# --------------------------------------------------------------------------- #
def _gc(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


def _core_feat(seq: str) -> list[float]:
    return lm.featurize(seq, KS_CORE) + [_gc(seq)]


def _rich_feat(seq: str) -> list[float]:
    return lm.featurize(seq, KS_RICH) + [_gc(seq)]


def _rich_pos_feat(seq: str) -> list[float]:
    return lm.featurize(seq, KS_RICH) + lm.position_onehot(seq, PROMOTER_LEN) + [_gc(seq)]


# --------------------------------------------------------------------------- #
# Learned arms — stdlib BayesRidge, plus a math-identical numpy fast path.
# --------------------------------------------------------------------------- #
def _fit_stdlib(X: list[list[float]], y: list[float]) -> list[float]:
    """Fit linmodel.BayesRidge (pure stdlib) and return its MAP weights."""
    m = lm.BayesRidge(len(X[0]), lam=LAM)
    m.observe_all(X, y)
    return m.weights()


def _fit_numpy(X, y) -> list[float]:
    """The SAME ridge (A=λI+XᵀX, b=Xᵀy, w=A⁻¹b) vectorized — identical to BayesRidge to ~1e-13 (proved
    in the tests), only faster, so the p≈342/942 rich arms are tractable. numpy is importable; nothing
    is installed."""
    Xn = np.asarray(X, dtype=float)
    yn = np.asarray(y, dtype=float)
    A = LAM * np.eye(Xn.shape[1]) + Xn.T @ Xn
    b = Xn.T @ yn
    return list(np.linalg.solve(A, b))


def _predict_one(w: list[float], x: list[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


def _spearman(xs, ys) -> float | None:
    r = sk.spearman(xs, ys)
    return r.rho if isinstance(r, sk.Corr) else None


# --------------------------------------------------------------------------- #
# One arm: feature matrix + a fit fn (stdlib list-of-lists, or a numpy matrix).
# --------------------------------------------------------------------------- #
class _Arm:
    def __init__(self, label, X, fit, is_np):
        self.label, self.X, self.fit, self.is_np = label, X, fit, is_np

    def rho(self, tr: list[int], ev: list[int], y: list[float], truth: list[float]) -> float | None:
        ytr = [y[i] for i in tr]
        if self.is_np:
            w = self.fit(self.X[tr], ytr)
            preds = [float(v) for v in self.X[ev] @ np.asarray(w)]
        else:
            w = self.fit([self.X[i] for i in tr], ytr)
            preds = [_predict_one(w, self.X[i]) for i in ev]
        return _spearman(preds, truth)


def _build_arms(records) -> list[_Arm]:
    """Featurize all records once per arm (cached across seeds). numpy arms hold a float matrix; the
    stdlib core holds a list-of-lists."""
    seqs = [r.seq for r in records]
    arms = [_Arm("core  (k-mer 1,2,3 + GC, stdlib)", [_core_feat(s) for s in seqs], _fit_stdlib, False)]
    if np is not None:
        arms.append(_Arm("rich  (k-mer 1-4 + GC, numpy)",
                         np.asarray([_rich_feat(s) for s in seqs], dtype=float), _fit_numpy, True))
        arms.append(_Arm("rich+pos  (k-mer 1-4 + positional + GC, numpy)",
                         np.asarray([_rich_pos_feat(s) for s in seqs], dtype=float), _fit_numpy, True))
    return arms


# --------------------------------------------------------------------------- #
# The Promoter Calculator baseline (deposited column; sign-aligned).
# --------------------------------------------------------------------------- #
def _calc_rho(records, ev: list[int], y: list[float]) -> tuple[float | None, int]:
    """Within-eval Spearman ρ of the deposited Promoter-Calc prediction vs measured log-strength, on the
    SAME eval rows. `calc_pred` is sign-inverted (more negative = stronger), so rank by -calc_pred → a
    POSITIVE ρ directly comparable to the learned arms. Returns (ρ, n rows scored)."""
    rows = [i for i in ev if records[i].calc_pred is not None]
    rho = _spearman([-records[i].calc_pred for i in rows], [y[i] for i in rows])
    return rho, len(rows)


def _split(records, n_eval: int, seed: int) -> tuple[list[int], list[int]]:
    """A random train/eval split over the unique promoters → de-novo = held-out SEQUENCES. eval keeps
    only calc-present rows so every arm AND the calc are scored on identical rows; train is the rest
    (training never needs the calc column). Shared by `run` and `scale_curve` so they split identically."""
    rng = random.Random(SEED_BASE + seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    ev = [i for i in idx[:n_eval] if records[i].calc_pred is not None]
    tr = idx[n_eval:]
    return tr, ev


# --------------------------------------------------------------------------- #
# Headline: in-distribution de-novo ρ, learned arms vs the calculator, N seeds.
# --------------------------------------------------------------------------- #
def run(seeds: int = 3, n_eval: int = 2500, refresh: bool = False) -> dict:
    records = pd.load_records(refresh=refresh)
    n = len(records)
    y = [r.strength for r in records]                # log-strength; rank-equivalent to raw TX
    arms = _build_arms(records)

    print("\n=== promoter_predictor: learned cores vs the Promoter Calculator, in-distribution de-novo ===")
    print(f"  La Fleur/Salis Urtecho σ70 set; {n} unique 150-nt promoters; random train/eval split → "
          f"de-novo = held-out SEQUENCES, one library.")
    print(f"  within-eval Spearman ρ vs measured log(TX) (rank, scale-free), SAME eval rows; "
          f"{seeds} seeds, n_eval={n_eval}.")
    print(f"  ⚠ baseline = the PUBLISHED Promoter-Calc column on this very set (possibly in-sample → "
          f"advantaged); a learned WIN is conservative, a LOSS inconclusive. See the scale curve.\n")

    per_arm: dict[str, list[float]] = {a.label: [] for a in arms}
    calc_rhos: list[float] = []
    n_scored = 0
    for s in range(seeds):
        tr, ev = _split(records, n_eval, s)
        truth = [y[i] for i in ev]
        for a in arms:
            r = a.rho(tr, ev, y, truth)
            if r is not None:
                per_arm[a.label].append(r)
        cr, n_scored = _calc_rho(records, ev, y)
        if cr is not None:
            calc_rhos.append(cr)

    print(f"      {'arm':<44}{'dim':>6}{'mean ρ':>10}   per-seed ρ")
    arm_means: dict[str, float] = {}
    for a in arms:
        rs = per_arm[a.label]
        if not rs:
            continue
        m = sum(rs) / len(rs)
        arm_means[a.label] = m
        dim = (a.X.shape[1] if a.is_np else len(a.X[0]))
        print(f"      {a.label:<44}{dim:>6}{m:>+10.3f}   {['%+.3f' % v for v in rs]}")
    calc_mean = sum(calc_rhos) / len(calc_rhos) if calc_rhos else None
    print(f"      {'Promoter Calculator (deposited, sign-aligned)':<44}{'—':>6}"
          f"{_fmt(calc_mean):>10}   {['%+.3f' % v for v in calc_rhos]}  (n={n_scored}/seed)")

    best_label = max(arm_means, key=arm_means.get) if arm_means else None
    best = arm_means.get(best_label) if best_label else None
    print()
    if best is None or calc_mean is None:
        print("  Incomplete: a learned arm or the baseline produced no ρ.")
    else:
        gap = best - calc_mean
        tag = best_label.split("(")[0].strip()
        if best >= calc_mean:
            print(f"  WIN (in-distribution): the learned core BEATS the Promoter Calculator de-novo at "
                  f"scale (best {best:+.3f} [{tag}] vs Calc {calc_mean:+.3f}; Δ={gap:+.3f}).")
            print(f"  The edge mechanism GENERALIZES past RBS — and conservatively, since the calc may be "
                  f"in-sample here. The same-assay dataloop manufactures exactly this regime.")
        else:
            print(f"  LOSS (in-distribution @ {n - n_eval} train): the Promoter Calculator still ranks "
                  f"higher (best learned {best:+.3f} [{tag}] vs Calc {calc_mean:+.3f}; Δ={gap:+.3f}).")
            print(f"  Inconclusive on the edge (the calc may be in-sample/advantaged) but a real finding: "
                  f"promoter predictor-margin is MORE data-hungry than RBS at this scale. Scale curve ↓.")
    return {"arm_means": arm_means, "calc_mean": calc_mean, "best_label": best_label,
            "best": best, "n_eval": n_eval, "seeds": seeds, "n_scored": n_scored}


# --------------------------------------------------------------------------- #
# Scale curve: does more in-distribution data trend the core toward/past the calc?
# --------------------------------------------------------------------------- #
def scale_curve(n_eval: int = 2500, train_sizes=(300, 1000, 3000, 10000), seed: int = 0,
                refresh: bool = False) -> dict:
    """The Höllerer 'wins in-distribution by ~N same-library seqs' analysis on promoter: fix the eval
    set, grow n_train for the BEST featurizer, watch ρ vs the (constant) calc. Single seed — a trend
    line, not a CI."""
    records = pd.load_records(refresh=refresh)
    y = [r.strength for r in records]
    arms = _build_arms(records)
    best_arm = arms[-1]                              # rich+pos if numpy else core (highest-capacity arm)

    pool, ev = _split(records, n_eval, seed)         # pool = the train pool to grow
    truth = [y[i] for i in ev]
    calc_rho, n_scored = _calc_rho(records, ev, y)

    print(f"\n  ·· scale curve ({best_arm.label.split('(')[0].strip()}, seed {seed}, "
          f"eval={len(ev)}): ρ vs n_train; calc (constant) = {_fmt(calc_rho)}")
    print(f"      {'n_train':>9}{'ρ':>10}{'vs calc':>10}")
    curve = {}
    for nt in train_sizes:
        if nt > len(pool):
            nt = len(pool)
        tr = pool[:nt]
        r = best_arm.rho(tr, ev, y, truth)
        curve[nt] = r
        delta = (r - calc_rho) if (r is not None and calc_rho is not None) else None
        print(f"      {nt:>9}{_fmt(r):>10}{('%+.3f' % delta) if delta is not None else 'n/a':>10}")
        if nt == len(pool):
            break
    return {"calc_rho": calc_rho, "curve": curve, "n_eval": len(ev), "arm": best_arm.label}


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Promoter predictor margin: learned cores vs Promoter Calc")
    ap.add_argument("--seeds", type=int, default=3, help="random-split seeds for the headline")
    ap.add_argument("--n-eval", type=int, default=2500, help="held-out eval size (calc-present rows kept)")
    ap.add_argument("--no-curve", action="store_true", help="skip the n_train scale curve")
    ap.add_argument("--refresh", action="store_true", help="re-fetch the dataset")
    args = ap.parse_args()
    try:
        run(seeds=args.seeds, n_eval=args.n_eval, refresh=args.refresh)
        if not args.no_curve:
            scale_curve(n_eval=args.n_eval)
    except pd.DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
