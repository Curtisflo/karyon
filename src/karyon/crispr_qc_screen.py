"""crispr_qc_screen — the REAL-SCREEN confirmation for the CRISPRi QC layer.

[crispr_qc.py](./crispr_qc.py) validated the legible QC layer against Horlbeck *activity scores* — a
*derived* relative-activity. The one honest caveat there: an activity score is itself a fitted quantity,
so "the flag predicts low activity" is one step removed from "the flag predicts a guide that actually
goes dark in a screen." This module removes that caveat by joining the real **K562 growth-phenotype
screen** (eLife 19760 Supplementary Data 7) onto the activity set: each matched guide now carries
(sequence, activity, **measured gamma**), and 3,756 non-targeting controls define the screen's null band.

The demonstration — fully non-circular, on real screen readout:

  * **the link is real** — within ESSENTIAL genes (a real growth hit: best gamma < −0.30) the measured
    activity tracks gamma at ρ ≈ −0.7: weak guides on essential genes show ~0 gamma = **silent failures**.
  * **the legible flag predicts it from sequence ALONE** — train the QC model on activity for one set of
    genes, then on HELD-OUT (gene-disjoint) essential genes the *sequence-only* predicted activity
    separates guides that actually produce a phenotype (gamma below the NT null) from the silent ones.
    The model never saw gamma, nor this gene's activity. Sequence → legible flag → real-screen silence.
  * **Contract 3 (NT-control calibration)** — the non-targeting controls set the null band; an
    essential-gene guide inside that band "produced no phenotype," and if the legible layer also flags it
    weak, its non-hit is a silent failure (untested), not a true negative.

Reuses [crispr_qc.py](./crispr_qc.py) verbatim (features, `fit_model`, `hard_contracts`, `check_guides`,
the gene-disjoint split, `_auroc`) and [stats_kit](./stats_kit.py).

    cd karyon/probe && python crispr_qc_screen.py --seeds 3
"""

from __future__ import annotations

import argparse
import random
import statistics

from . import crispr_qc as qc
from . import crispr_qc_data as cd
from .crispr_qc_data import Record, ScreenRecord
from .linmodel import BayesRidge
from .stats_kit import Corr, fmt, spearman

NULL_K = 3.0            # a guide has a real phenotype if gamma < (NT mean − NULL_K·NT sd)
ESS_GAMMA = -0.30       # a gene is a growth hit (essential here) if its best guide's gamma is below this
MIN_GUIDES = 4          # ...and it has at least this many screened guides


def null_floor(ntc: list[float]) -> float:
    """The phenotype threshold from the non-targeting controls: gamma below it is beyond the null."""
    return statistics.fmean(ntc) - NULL_K * statistics.pstdev(ntc)


def essential_subset(srecs: list[ScreenRecord]) -> dict[str, list[ScreenRecord]]:
    """Genes that are real growth hits — at least MIN_GUIDES screened and a best guide below ESS_GAMMA.
    Silent failures only make sense in a gene the screen CAN see; a non-essential gene's flat guides are
    true negatives, not silent failures."""
    by_gene: dict[str, list[ScreenRecord]] = {}
    for r in srecs:
        by_gene.setdefault(r.gene, []).append(r)
    return {g: vs for g, vs in by_gene.items()
            if len(vs) >= MIN_GUIDES and min(r.gamma for r in vs) < ESS_GAMMA}


def _fit(recs, ys: list[float]) -> BayesRidge:
    m = BayesRidge(len(qc.FEATURE_NAMES), lam=1.0)
    m.observe_all([qc.features(r.seq) for r in recs], ys)
    return m


def evaluate(full_recs: list[Record], srecs: list[ScreenRecord], floor: float, seeds: int = 3):
    """Fixed-test, gene-disjoint evaluation. The test set is ALL essential-gene guides (n≈294); the model
    trains on an 80% subsample of the NON-essential genes (so the test genes never appear in training).
    Fixing the test removes the small-held-out-set noise that makes a per-seed split swing wildly; seeds
    vary only the training subsample, so the AUROCs report the estimate's stability, not sampling luck."""
    ess = essential_subset(srecs)
    test = [r for vs in ess.values() for r in vs]
    has_pheno = [r.gamma < floor for r in test]                # real phenotype = gamma beyond the NT null
    pool = [r for r in full_recs if r.gene not in ess]         # gene-disjoint training pool
    pool_genes = sorted({r.gene for r in pool})
    aurocs, shuf = [], []
    for seed in range(seeds):
        rng = random.Random(seed)
        keep = set(rng.sample(pool_genes, int(len(pool_genes) * 0.8)))
        tr = [r for r in pool if r.gene in keep]
        m = _fit(tr, [r.activity for r in tr])
        aurocs.append(qc._auroc([m.predict(qc.features(r.seq)) for r in test], has_pheno))
        sy = [r.activity for r in tr]
        rng.shuffle(sy)
        ms = _fit(tr, sy)
        shuf.append(qc._auroc([ms.predict(qc.features(r.seq)) for r in test], has_pheno))
    return test, has_pheno, aurocs, shuf


def run(seeds: int = 3) -> dict:
    full = cd.load_records()
    srecs, ntc = cd.load_screen()
    floor = null_floor(ntc)
    ess_all = essential_subset(srecs)
    ea = [r.activity for vs in ess_all.values() for r in vs]
    eg = [r.gamma for vs in ess_all.values() for r in vs]

    print(f"\nCRISPRi screen-QC — REAL K562 growth screen ({len(srecs)} guides matched to activity, "
          f"{len(ntc)} NT controls)")
    print(f"NT null band (Contract 3): gamma < {floor:+.3f} = a real phenotype "
          f"(mean {statistics.fmean(ntc):+.3f} ± {NULL_K:.0f}·{statistics.pstdev(ntc):.3f})")
    print(f"essential genes (best gamma < {ESS_GAMMA}, ≥{MIN_GUIDES} guides): {len(ess_all)} ({len(ea)} guides)")
    rho = spearman(ea, eg)
    print(f"the link is real — within-essential ρ(activity, gamma) = {fmt(rho)}; "
          f"global = {fmt(spearman([r.activity for r in srecs], [r.gamma for r in srecs]))}\n")

    test, has_pheno, aurocs, shuf = evaluate(full, srecs, floor, seeds)
    mean = lambda xs: sum(xs) / len(xs)
    print(f"  sequence-only QC predicts REAL phenotype (fixed test n={len(test)}, gene-disjoint):")
    print(f"    AUROC               " + "  ".join(f"{a:+.3f}" for a in aurocs) + f"    mean {mean(aurocs):+.3f}")
    print(f"    noise baseline      " + "  ".join(f"{a:+.3f}" for a in shuf) + f"    mean {mean(shuf):+.3f}")

    # The clean punchline: median real gamma by predicted-activity tercile (one gene-disjoint model over
    # the non-essential pool). Monotonic = the legible score ranks real-screen knockdown strength.
    m = _fit([r for r in full if r.gene not in ess_all], [r.activity for r in full if r.gene not in ess_all])
    scored = sorted((m.predict(qc.features(r.seq)), r.gamma) for r in test)
    t = len(scored) // 3
    bands = [("weakest-predicted ⅓", scored[:t]), ("middle ⅓", scored[t:2 * t]),
             ("strongest-predicted ⅓", scored[2 * t:])]
    print("\n  median REAL gamma by predicted-activity tercile (a phenotype is more negative):")
    for label, seg in bands:
        gam = [g for _, g in seg]
        print(f"    {label:<24} median gamma {statistics.median(gam):+.3f}  "
              f"(phenotype rate {sum(g < floor for g in gam) / len(gam):.0%}, n={len(gam)})")

    # The binary flag, honestly: too coarse at guide level on the real screen — hard rules have real
    # false positives (a GC-rich or run-bearing guide can still knock down), so recall is low.
    flagged = [v.flagged for v in qc.check_guides(m, test)]
    silent = [not h for h in has_pheno]
    frec, fprec = qc._recall_precision(flagged, silent)
    print(f"\n  binary flag at guide level (conservative — the continuous score is the better signal here): "
          f"silent-failure recall {frec:.2f} / precision {fprec:.2f}")

    return {
        "within_ess_rho": rho.rho if isinstance(rho, Corr) else 0.0,
        "global_rho": (lambda r: r.rho if isinstance(r, Corr) else 0.0)(
            spearman([r.activity for r in srecs], [r.gamma for r in srecs])),
        "auroc_mean": mean(aurocs), "auroc_min": min(aurocs), "shuf_auroc_mean": mean(shuf),
        "tercile_gammas": [statistics.median([g for _, g in seg]) for _, seg in bands],
        "n_matched": len(srecs), "n_essential_guides": len(test),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Real-screen confirmation for the CRISPRi QC layer.")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    try:
        run(seeds=args.seeds)
    except cd.DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
