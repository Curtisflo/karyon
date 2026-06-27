"""rbs_hollerer_predictor — the BIG-DATA predictor margin: a richer learned core vs the RBS Calculator.

Earlier evaluation settled the SMALL-data RBS predictor margin: on 394 SynBioMTS constructs
(leave-one-study-out, "de-novo" = a held-out *lab*), the RBS Calculator/OSTIR WON +0.665 vs the cheap
stdlib core's +0.500. It left ONE exception open — the only remaining way the predictor margin could be
a real edge: does a STRONGER learned model on a BIG dataset beat the calculator de-novo? The literature
(Höllerer et al. 2020, *Nat. Commun.* 11:3551 — their SAPIENs ResNet ensemble) says yes; we had only
tested cheap cores on small data. This probe closes it on Höllerer's own 300k RBS set
([rbs_hollerer_data.py]), reusing rbs_predictor.py's OSTIR baseline + Spearman mechanics verbatim.

THE REGIME DIFFERENCE (stated plainly, it changes what "de-novo" means):
  * SynBioMTS = 5 small studies from different labs → de-novo := a held-out STUDY (distribution shift
    across labs/designs). The cheap core overfit the training labs and lost.
  * Höllerer = ONE 300k MPRA library, one assay, one construct → de-novo := held-out SEQUENCES from the
    same library (a random train/test split — exactly how SAPIENs is benchmarked). No lab-shift; the
    question is purely "with enough data + capacity, does a learned model out-rank the biophysics?"

THE THREE LEARNED ARMS (all reuse [linmodel.py]'s featurizers — a new regime cost only this file):
  1. core      — k-mer (1,2,3) frequency, the EXACT featurizer rbs_predictor.py used. The cheap baseline.
  2. rich       — k-mer (1,2,3,4) + POSITIONAL one-hot over the 17-mer window. "FIRST try richer features
                 in the existing stdlib linmodel.BayesRidge" — which-base-where is the signal in a fixed
                 17-nt element (linmodel.position_onehot exists for exactly this). Stdlib only.
  3. rich+np    — the SAME rich features, fit via a numpy-vectorized normal-equations ridge (IDENTICAL
                 math to BayesRidge — verified to 1e-13; numpy is already importable, sklearn is NOT, so
                 nothing is installed). numpy only buys speed → it can train on ALL ~150k sequences, the
                 honest "stronger cheap model on big data" arm.

BASELINE — the RBS Calculator via OSTIR (open, ViennaRNA; already installed from probe #2; nothing new
installed). The 17-mer sits DIRECTLY 5' of the bxb1-sfGFP ATG (paper Methods: "the 17 bases directly
upstream of the bxb1 start codon" were randomized; no spacer). The flanks are CONSTANT across all 300k
constructs, so OSTIR's RANKING across the library is driven by the variable 17-mer — verified robust:
ρ varies only ≈0.61–0.65 across very different reconstructed leaders and even a different CDS (sfGFP vs
bxb1). The reconstruction below is documented and bounded, NOT a verbatim plasmid pull (the plasmid
GenBank is auth-gated on Addgene); `GAGCTCGCAT` is the REAL pre-RBS constant from the uASPIre config
(`SEQ_CONSTANT`), the bxb1 CDS start is the canonical Bxb1 integrase N-terminus in standard E. coli codons.

    python -m karyon.rbs_hollerer_predictor                 # default: r3, 8000-seq eval
    python -m karyon.rbs_hollerer_predictor --rep r2 --n-eval 5000
"""

from __future__ import annotations

import argparse
import csv
import random

from . import linmodel as lm
from . import rbs_hollerer_data as hd
from . import stats_kit as sk

try:
    import numpy as np
except ImportError:                                  # the strong arm is numpy-accelerated; degrade
    np = None

try:
    import ostir
except ImportError:                                  # baseline dep; learned arms still run from cache
    ostir = None

# --- learned-arm feature configs (all via linmodel) ------------------------- #
WIN = 17                       # the variable region length (== the whole sequence here)
KS_CORE = (1, 2, 3)            # the cheap core: rbs_predictor.py's exact featurizer
KS_RICH = (1, 2, 3, 4)         # richer: add tetramers …
LAM = 1.0
SEED = 0

# --- OSTIR construct reconstruction (documented; see module docstring) ------ #
# leader (a standard ribosome-distal 5' leader) + the REAL pre-RBS constant, then [17-mer], then the
# bxb1 ATG + canonical Bxb1 integrase N-terminus (std E. coli codons). ATG is the first bxb1 codon.
_LEADER = "AAGAAGGAGATATACATACT" + "GAGCTCGCAT"
_BXB1 = "ATGCGTGCACTGGTTGTTATTCGTCTGAGCCGTGTTACCGATGCAACCACCAGCCCGGAACGT"


def _core_feat(seq: str) -> list[float]:
    return lm.featurize(seq, KS_CORE)


def _rich_feat(seq: str) -> list[float]:
    return lm.featurize(seq, KS_RICH) + lm.position_onehot(seq, WIN)


# --------------------------------------------------------------------------- #
# Learned arms — stdlib BayesRidge, plus a math-identical numpy fast path.
# --------------------------------------------------------------------------- #
def _fit_stdlib(X: list[list[float]], y: list[float]) -> list[float]:
    """The mandated first attempt: fit linmodel.BayesRidge (pure stdlib) and return its MAP weights."""
    m = lm.BayesRidge(len(X[0]), lam=LAM)
    m.observe_all(X, y)
    return m.weights()


def _fit_numpy(X: list[list[float]], y: list[float]) -> list[float]:
    """The SAME ridge (A=λI+XᵀX, b=Xᵀy, w=A⁻¹b) vectorized in numpy — identical to BayesRidge to ~1e-13
    (proved in tests), only faster, so it can train on all ~150k sequences. numpy is already importable;
    nothing is installed. No analytic-uncertainty path is needed here (this probe is point-prediction)."""
    Xn = np.asarray(X, dtype=float)
    yn = np.asarray(y, dtype=float)
    A = LAM * np.eye(Xn.shape[1]) + Xn.T @ Xn
    b = Xn.T @ yn
    return list(np.linalg.solve(A, b))


def _predict(w: list[float], x: list[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


# --------------------------------------------------------------------------- #
# OSTIR baseline (cached, mirrors rbs_predictor.py).
# --------------------------------------------------------------------------- #
def _ostir_cache_path(rep: str):
    return hd._cache_path(rep).parent / f"rbs_hollerer_ostir_{rep}.csv"


def _ostir_one(seq17: str) -> float | None:
    full = _LEADER + seq17 + _BXB1
    start = len(_LEADER) + len(seq17) + 1          # 1-based position of the bxb1 ATG's A
    try:
        res = ostir.run_ostir(full, start=start, threads=1, verbosity=0)
    except Exception:
        return None
    match = [d for d in res if d.get("start_position") == start]
    chosen = match[0] if match else (res[0] if res else None)
    return None if chosen is None else chosen.get("expression")


def ostir_predict(seqs: list[str], rep: str, refresh: bool = False) -> dict[str, float | None]:
    """17-mer -> OSTIR predicted expression (None if it failed), cached to `~/.cache/karyon/`. Only the eval
    subset is scored (OSTIR on the full 150k is unnecessary; the ranking estimate is tight at ~thousands)."""
    path = _ostir_cache_path(rep)
    cache: dict[str, float | None] = {}
    if path.exists() and not refresh:
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                cache[row["sequence"]] = float(row["expr"]) if row["expr"] else None
    todo = [s for s in seqs if s not in cache]
    if todo and ostir is None:
        print(f"  [ostir] not installed — RBS-Calc baseline n/a ({len(cache)} cached)")
        return cache
    if todo:
        print(f"  [ostir] scoring {len(todo)} eval constructs ({len(cache)} cached)…")
        for s in todo:
            cache[s] = _ostir_one(s)
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sequence", "expr"])
            for s, e in cache.items():
                w.writerow([s, "" if e is None else e])
    return cache


# --------------------------------------------------------------------------- #
# The big-data de-novo comparison.
# --------------------------------------------------------------------------- #
def _spearman(xs, ys) -> float | None:
    r = sk.spearman(xs, ys)
    return r.rho if isinstance(r, sk.Corr) else None


def run(rep: str = "r3", n_eval: int = 8000, n_train: int | None = None,
        refresh: bool = False, refresh_ostir: bool = False) -> dict:
    records = hd.load_records(rep=rep, refresh=refresh)
    rng = random.Random(SEED)
    rng.shuffle(records)
    eval_recs = records[:n_eval]
    train_recs = records[n_eval:] if n_train is None else records[n_eval:n_eval + n_train]
    eval_seqs = [r.sequence for r in eval_recs]
    eval_truth = [r.ifp for r in eval_recs]

    ostir_pred = ostir_predict(eval_seqs, rep, refresh=refresh_ostir)
    have_ostir = any(v is not None for v in ostir_pred.values())
    ost_rho = _spearman([ostir_pred.get(s) for s in eval_seqs], eval_truth) if have_ostir else None

    print("\n=== rbs_hollerer_predictor: learned cores vs the RBS Calculator (OSTIR), big-data de-novo ===")
    print(f"  Höllerer 2020 300k RBS ({rep}); {len(records)} variants; train={len(train_recs)} / "
          f"held-out eval={len(eval_recs)} (random split — de-novo = unseen SEQUENCES, one library).")
    print(f"  within-set Spearman ρ vs measured IFP480 (rank, scale-free), on the SAME eval sequences.\n")

    arms = [("core  (k-mer 1,2,3, stdlib)", _core_feat, _fit_stdlib)]
    arms.append(("rich  (k-mer 1-4 + positional, stdlib)", _rich_feat, _fit_stdlib))
    if np is not None:
        arms.append(("rich  (k-mer 1-4 + positional, numpy)", _rich_feat, _fit_numpy))

    Xeval_cache: dict[int, list[list[float]]] = {}
    print(f"      {'learned arm':<40}{'dim':>6}{'ρ vs IFP480':>14}")
    results = {}
    for label, feat, fit in arms:
        Xtr = [feat(r.sequence) for r in train_recs]
        w = fit(Xtr, [r.ifp for r in train_recs])
        key = len(Xtr[0])
        if key not in Xeval_cache:
            Xeval_cache[key] = [feat(s) for s in eval_seqs]
        Xev = Xeval_cache[key]
        rho = _spearman([_predict(w, x) for x in Xev], eval_truth)
        results[label] = rho
        print(f"      {label:<40}{len(Xtr[0]):>6}{_fmt(rho):>14}")
    print(f"      {'RBS Calculator (OSTIR)':<40}{'—':>6}{_fmt(ost_rho):>14}")

    best_label = max((k for k in results if results[k] is not None),
                     key=lambda k: results[k], default=None)
    best = results.get(best_label)
    print()
    if not have_ostir:
        print("  RBS Calculator baseline unavailable (OSTIR/ViennaRNA not installed) — only the learned")
        print(f"  arms' de-novo ρ are reported. Best learned arm: {best_label} (ρ={_fmt(best)}).")
    elif best is not None and ost_rho is not None:
        gap = best - ost_rho
        if best >= ost_rho:
            print(f"  WIN: the learned core BEATS the RBS Calculator de-novo at scale "
                  f"(best {best:+.3f} [{best_label.split('(')[0].strip()}] vs OSTIR {ost_rho:+.3f}; "
                  f"Δ={gap:+.3f}).")
            print(f"  The SynBioMTS gap (calc +0.665 vs core +0.500) CLOSES and REVERSES with big data + "
                  f"richer features — the predictor margin IS real, but it is data-hungry, not $0-desk.")
        else:
            print(f"  RBS Calculator still wins de-novo (best learned {best:+.3f} < OSTIR {ost_rho:+.3f}; "
                  f"Δ={gap:+.3f}). Biophysics wins even at scale here.")
    return {"rep": rep, "n_train": len(train_recs), "n_eval": len(eval_recs),
            "learned": results, "ostir": ost_rho, "best_label": best_label, "best": best}


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Big-data RBS predictor margin: learned cores vs RBS Calc")
    ap.add_argument("--rep", default="r3", choices=hd._REPS, help="replicate (default r3, smallest 300k)")
    ap.add_argument("--n-eval", type=int, default=8000, help="held-out eval/test size (also the OSTIR budget)")
    ap.add_argument("--n-train", type=int, default=None, help="cap training size (default: all the rest)")
    ap.add_argument("--refresh", action="store_true", help="re-fetch the dataset")
    ap.add_argument("--refresh-ostir", action="store_true", help="recompute OSTIR predictions")
    args = ap.parse_args()
    try:
        run(rep=args.rep, n_eval=args.n_eval, n_train=args.n_train,
            refresh=args.refresh, refresh_ostir=args.refresh_ostir)
    except hd.DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
