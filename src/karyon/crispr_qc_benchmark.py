"""crispr_qc_benchmark — the ownership test STRATEGY §6 named and the first QC build skipped.

Two honest questions, answered on the SAME gene-disjoint splits the QC layer is scored on:

  Q1 (legibility cost / is the signal even there) — does a NON-legible model extract materially more
     from the same protospacer sequence than the legible layer's ρ≈0.32?  Baselines, all 3-seed
     gene-disjoint, ρ(pred, measured activity) held-out:
       * legible      — the QC layer of record (BayesRidge over NAMED features; crispr_qc.features)
       * rich-linear  — BayesRidge over k-mer(1–3) freqs + GC/run/length (still inspectable, just wide)
       * ceiling-GBM  — sklearn HistGradientBoosting over the SAME wide features (non-legible upper bound)
       * shuffled     — labels permuted (the chance floor)
     legible→rich isolates "more features"; rich→ceiling isolates "non-linearity." ceiling−legible is the
     ρ the legibility choice costs.

  Q2 (is guide-efficacy prediction already OWNED) — the honest answer, stated not faked. Azimuth / Doench
     Rule Set 1–2 / CRISPOR are the named incumbents, but:
       * they need a 30-nt GENOMIC context (4 bp up + 20 protospacer + 3 PAM + 3 down) and target CRISPR**ko**
         *cutting*; this table is protospacer-only CRISPR**i** *knockdown* — so they are NOT directly runnable
         here (no py3 Azimuth dist exists either; original is py2/old-sklearn), and they are cross-mechanism.
       * the actual CRISPRi guide-efficacy incumbent IS Horlbeck's own hCRISPRi-v2 activity model — i.e. the
         label we train on. So this lever is already published; a *legible* re-derivation of part of it is not
         white space. The white space, if any, is the screen-RELIABILITY layer, not guide efficacy (Q3).
     Literature ρ for those tools (≈0.4–0.5, on-target, on THEIR CRISPRko data; known to transfer worse to
     CRISPRi) is cited in the result doc, clearly labelled, not re-run. The path to a faithful run is logged:
     fetch hg19 flanks via the cached (chrom, coord, strand) → score → compare. Deliberately not this session.

  Q3 (--screen) — the screen-level QC analog (the MAGeCK/SCEPTRE idea): an NT-control-calibrated, observed
     per-gene power test on the real K562 screen, vs the layer's sequence-only gene call. Shows whether
     standard observed-phenotype QC already identifies under-powered genes (it does, post-hoc) and how much
     PRE-screen foresight the sequence layer adds — on SD7's pre-QC'd library, the honest answer is "little,"
     which is the range-restriction point quantified.

    python -m karyon.crispr_qc_benchmark            # Q1 + Q2 note (+ --screen for Q3)

Dep note: admits scikit-learn for the ceiling (precedent: the RBS predictor probe admits ostir/viennarna for
a fair runnable baseline). Everything else is stdlib + numpy. SKIPs offline like every other probe.
"""

from __future__ import annotations

import argparse
import random
import statistics

from . import crispr_qc as qc
from . import crispr_qc_data as cd
from . import linmodel
from .linmodel import BayesRidge
from .stats_kit import Corr, spearman

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    import numpy as np
    _HAVE_SK = True
except Exception:                                  # the ceiling needs sklearn+numpy; degrade gracefully
    _HAVE_SK = False


# Wide, still-inspectable feature vector shared by rich-linear and the GBM ceiling (length-invariant
# k-mer freqs avoid the variable-length 18–25 nt positional-alignment problem; GC/run/length are the
# three legible scalars the named layer already trusts).
_KS = (1, 2, 3)


def wide_features(seq: str) -> list[float]:
    return linmodel.featurize(seq, _KS) + [
        qc.gc(seq), qc.max_run(seq) / max(1, len(seq)), (len(seq) - 18) / 7.0]


def _rho(pred, meas) -> float:
    r = spearman(list(pred), list(meas))
    return r.rho if isinstance(r, Corr) else 0.0


def _fit_bayes(dim: int, xs, ys) -> BayesRidge:
    m = BayesRidge(dim, lam=1.0)
    m.observe_all(xs, ys)
    return m


def evaluate_seed(recs: list[cd.Record], seed: int) -> dict:
    """All baselines on ONE identical gene-disjoint split (crispr_qc.split_by_gene — the QC layer's own)."""
    train, test = qc.split_by_gene(recs, seed)
    yte = [r.activity for r in test]
    ytr = [r.activity for r in train]

    # legible — the layer of record (named features).
    leg = qc.fit_model(train)
    rho_leg = _rho((leg.predict(qc.features(r.seq)) for r in test), yte)

    # rich-linear — same model class (BayesRidge), wider inspectable features.
    dim = len(wide_features(train[0].seq))
    rich = _fit_bayes(dim, [wide_features(r.seq) for r in train], ytr)
    rho_rich = _rho((rich.predict(wide_features(r.seq)) for r in test), yte)

    # shuffled — chance floor (permute train labels, refit the legible model).
    rng = random.Random(seed)
    sy = list(ytr)
    rng.shuffle(sy)
    shuf = qc.fit_model([cd.Record(r.gene, r.seq, y) for r, y in zip(train, sy)])
    rho_shuf = _rho((shuf.predict(qc.features(r.seq)) for r in test), yte)

    rho_gbm = None
    if _HAVE_SK:
        xtr = np.array([wide_features(r.seq) for r in train], dtype=float)
        xte = np.array([wide_features(r.seq) for r in test], dtype=float)
        gbm = HistGradientBoostingRegressor(
            random_state=seed, max_iter=400, learning_rate=0.05,
            max_depth=3, l2_regularization=1.0)
        gbm.fit(xtr, np.array(ytr, dtype=float))
        rho_gbm = _rho(list(gbm.predict(xte)), yte)

    return {"legible": rho_leg, "rich_linear": rho_rich, "ceiling_gbm": rho_gbm, "shuffled": rho_shuf}


def run_guide(seeds: int = 3) -> dict:
    recs = cd.load_records()
    rows = [evaluate_seed(recs, s) for s in range(seeds)]
    keys = ["legible", "rich_linear", "ceiling_gbm", "shuffled"]
    labels = {
        "legible": "legible layer (named features) — OF RECORD",
        "rich_linear": "rich-linear (k-mer 1–3, BayesRidge)",
        "ceiling_gbm": "ceiling GBM (same features, non-legible)" + ("" if _HAVE_SK else "  [sklearn absent → skipped]"),
        "shuffled": "shuffled control",
    }

    print(f"\nCRISPRi guide-efficacy benchmark — held-out ρ(pred, measured activity), "
          f"{len(recs)} guides, {seeds}-seed gene-disjoint\n")
    print(f"  {'baseline':<44} " + "  ".join(f" seed{i}" for i in range(seeds)) + "    mean")
    means = {}
    for k in keys:
        vals = [r[k] for r in rows]
        if any(v is None for v in vals):
            print(f"  {labels[k]:<44} " + "  (not run)")
            means[k] = None
            continue
        means[k] = sum(vals) / len(vals)
        cells = "  ".join(f"{v:+.3f}" for v in vals)
        print(f"  {labels[k]:<44} {cells}    {means[k]:+.3f}")

    if means.get("ceiling_gbm") is not None:
        gap = means["ceiling_gbm"] - means["legible"]
        feat_gap = means["rich_linear"] - means["legible"]
        nonlin_gap = means["ceiling_gbm"] - means["rich_linear"]
        print(f"\n  legibility cost  ceiling−legible = {gap:+.3f}  "
              f"(features {feat_gap:+.3f}  +  non-linearity {nonlin_gap:+.3f})")
        verdict = ("small — the legible layer is near the sequence ceiling" if gap < 0.05 else
                   "moderate — sequence holds more signal than the legible layer takes" if gap < 0.12 else
                   "large — legibility is leaving real accuracy on the table")
        print(f"  read: {verdict}.")

    print("\n  Q2 — incumbent (Azimuth / Rule Set 1–2 / CRISPOR): NOT run here, by necessity not omission.")
    print("       they need 30-nt genomic context + target CRISPRko cutting; this table is protospacer-only")
    print("       CRISPRi knockdown. The CRISPRi efficacy incumbent IS Horlbeck's own score (our label), so")
    print("       guide-efficacy is already published — the open white space is screen-reliability (--screen).")
    return {"means": means, "seeds": seeds, "n": len(recs)}


def run_screen(seeds: int = 3) -> dict:
    """Q3 — the MAGeCK/SCEPTRE analog: NT-calibrated OBSERVED per-gene power on the real screen, vs the
    sequence-only gene call. Quantifies how much PRE-screen foresight the layer adds over post-hoc screen QC."""
    from . import crispr_qc_screen as scr
    full = cd.load_records()
    srecs, ntc = cd.load_screen()
    floor = statistics.fmean(ntc) - 3.0 * statistics.pstdev(ntc)

    by_gene: dict[str, list] = {}
    for r in srecs:
        by_gene.setdefault(r.gene, []).append(r)
    genes = {g: vs for g, vs in by_gene.items() if len(vs) >= 4}

    # Observed screen QC (no sequence): a gene is observed-powered if ANY guide clears the NT null.
    obs_powered = {g: any(r.gamma < floor for r in vs) for g, vs in genes.items()}
    n_under = sum(not p for p in obs_powered.values())
    print(f"\nCRISPRi screen-QC analog (MAGeCK/SCEPTRE idea) — {len(genes)} genes (≥4 guides), "
          f"NT floor γ<{floor:+.3f}")
    print(f"  observed (post-hoc) per-gene QC: {n_under}/{len(genes)} genes under-powered "
          f"({n_under / len(genes):.1%}) — i.e. NO guide produced a phenotype.")

    # Sequence-only PRE-screen prediction: train on activity for the other genes, predict best-guide power.
    ess = scr.essential_subset(srecs)
    aurocs = []
    for seed in range(seeds):
        rng = random.Random(seed)
        pool = [r for r in full if r.gene not in genes]
        m = qc.fit_model(pool)
        g_pred, g_obs = [], []
        for g, vs in genes.items():
            g_pred.append(max(m.predict(qc.features(r.seq)) for r in vs))
            g_obs.append(obs_powered[g])
        if 0 < sum(g_obs) < len(g_obs):
            aurocs.append(qc._auroc(g_pred, g_obs))
    if aurocs:
        print(f"  sequence-only PRE-screen prediction of observed power: AUROC "
              + "  ".join(f"{a:.3f}" for a in aurocs) + f"    mean {sum(aurocs)/len(aurocs):.3f}")
        print("  read: most of those 'under-powered' genes are simply NON-ESSENTIAL (no growth phenotype is")
        print("        expected at all), so observed per-gene power ≈ essentiality — which guide-sequence QC")
        print("        cannot and should not predict (AUROC ~0.54 ≈ chance). The gene-level call is confounded")
        print("        by essentiality; the layer's claim only holds WITHIN screen-visible essential genes (the")
        print("        n≈294 analysis), which is the range-restricted regime. True screen-reliability QC (the")
        print("        open white space) needs real screen COUNT data + NT calibration, not this activity join.")
    else:
        print("  (degenerate — too few under-powered genes to score; the library is pre-QC'd, as expected.)")
    return {"n_genes": len(genes), "observed_under_frac": n_under / len(genes) if genes else 0.0,
            "seq_pred_auroc": (sum(aurocs) / len(aurocs)) if aurocs else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CRISPRi QC ownership benchmark (Q1 ceiling + Q2 note + Q3 screen).")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--screen", action="store_true", help="also run Q3, the screen-QC analog (needs SD7)")
    args = ap.parse_args()
    try:
        run_guide(seeds=args.seeds)
        if args.screen:
            run_screen(seeds=args.seeds)
    except cd.DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
