"""screen_qc — a legible reliability/QC layer over a bulk CRISPR screen's NON-hits (the value-add).

The incumbent (`screen_baseline.py`) emits a gene-level q-value: hit / non-hit. That scalar throws
away the within-gene guide structure. This layer reads that structure back — purely from the COUNTS,
NT/NEG-calibrated, never from the gold standard — and qualifies every NON-hit:

  * **trustworthy true-negative** — the gene's guides were abundant, consistent, and showed no
    depletion → the non-hit is real; OR
  * **under-powered silent failure** — the screen lacked the power to call this gene (guides at the
    count floor / too few usable guides / strong-but-diluted disagreement) → the non-hit is *not* to
    be trusted, with a human-readable reason.

The contracts measure GUIDE POWER, which is essentiality-INDEPENDENT (the benchmark's Q3 lesson:
a gene-level "is it essential?" call is confounded by essentiality; "did the screen have power here?"
is not). The DRC-spine doctrine ported to counts — every flag carries a legible reason
(`crispr_qc.hard_contracts` is the template).

This module also runs the leakage-free evaluation (`run`): does flagging silent failures ADD
information the FDR scalar discarded? The guards (all pre-registered; see docs/screen-power.md):
  Q1 recall   — of CEGv2 essentials the baseline missed, how many are flagged under-powered;
  Q2 precision— on a HELD-OUT NEGv1 half (disjoint from the calibration half), the false-flag rate;
  Q3 non-redundancy — |spearman(under-power score, baseline −log10 q)| < 0.6 (else it's just a softer
                      FDR and the thesis FAILS, said plainly);
  Q4 lift     — a PARAMETER-FREE rank-combine of (baseline, QC) beats baseline-alone AUPRC for
                essential recovery within the non-hit pile (bootstrap CIs), or "no lift" honestly.

    python -m karyon.screen_qc --seeds 50
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass

from . import screen_qc_data as sd
from .screen_baseline import call_genes, guide_stats, GuideStat
from .stats_kit import Corr, average_precision, bootstrap_ci, rank_avg, spearman

# Pre-registered thresholds (set BEFORE the e2e run; each a field-standard constant or derived from
# the NEG-calibration half WITHOUT reference to CEGv2 — never tuned to recover more essentials).
NULL_K = 3.0               # null band = mean − 3σ of calibration LFCs (same as crispr_qc_screen.py)
MIN_INITIAL = 30.0         # a guide below this many normalized initial reads can't show dropout (power floor)
FLOOR_FRAC = 0.5           # gene under-powered if ≥ this fraction of its guides are below the floor
MIN_USABLE_GUIDES = 3      # fewer usable (≥MIN_INITIAL) guides than this ⇒ under-powered
DISP_K = 3.0               # dispersion threshold = calib median + 3·MAD of within-gene usable-LFC spread


@dataclass(frozen=True)
class GeneReliability:
    """The QC verdict on a baseline NON-hit: trustworthy true-negative vs under-powered silent failure."""

    gene: str
    trustworthy: bool          # True = a confident true-negative; False = under-powered (don't trust)
    reasons: list[str]         # human-readable reasons the under-power flag fired (empty ⇒ trustworthy)
    under_power_score: float   # continuous severity (count-power, NOT a restatement of the baseline q)
    n_guides: int
    n_usable: int
    n_cleared_null: int


def null_band(calib_lfcs: list[float], direction: str = "deplete") -> float:
    """The threshold beyond which a guide's LFC clears the non-essential / control null. `deplete`: a
    LOW LFC (mean − NULL_K·σ) — essential drop-out. `enrich`: a HIGH LFC (mean + NULL_K·σ) — positive
    selection / resistance. Lets the dispersion contract port to an enrichment screen."""
    if len(calib_lfcs) < 2:
        return float("-inf") if direction == "deplete" else float("inf")
    m, s = statistics.fmean(calib_lfcs), NULL_K * statistics.pstdev(calib_lfcs)
    return m - s if direction == "deplete" else m + s


def _dispersion_threshold(calib_gene_guides: list[list[GuideStat]]) -> float:
    """median + DISP_K·MAD of within-gene usable-guide LFC spread among the CALIBRATION (non-essential)
    genes — a label-free 'how much do healthy genes' guides normally disagree?' baseline."""
    disps = []
    for guides in calib_gene_guides:
        usable = [g.lfc for g in guides if g.init_count >= MIN_INITIAL]
        if len(usable) >= 2:
            disps.append(statistics.pstdev(usable))
    if not disps:
        return float("inf")
    med = statistics.median(disps)
    mad = statistics.median([abs(d - med) for d in disps]) or 1e-9
    return med + DISP_K * 1.4826 * mad


def reliability_contracts(guides: list[GuideStat], null: float, disp_thresh: float,
                          direction: str = "deplete") -> tuple[list[str], float, int, int]:
    """Deterministic, legible reasons a NON-hit is UNDER-POWERED (empty reasons ⇒ trustworthy). Returns
    (reasons, under_power_score, n_usable, n_cleared_null). The count-power contracts (floor, usable
    guides) are direction-AGNOSTIC; only 'cleared the null' flips for an enrichment screen."""
    cleared = (lambda x: x < null) if direction == "deplete" else (lambda x: x > null)
    n = len(guides)
    n_floor = sum(1 for g in guides if g.init_count < MIN_INITIAL)
    usable = [g for g in guides if g.init_count >= MIN_INITIAL]
    n_usable = len(usable)
    n_cleared = sum(1 for g in usable if cleared(g.lfc))
    reasons: list[str] = []

    # Contract A — count power floor: too many guides too sparse to show dropout.
    floor_frac = n_floor / n if n else 0.0
    if floor_frac >= FLOOR_FRAC:
        reasons.append(f"{n_floor}/{n} guides below {MIN_INITIAL:.0f}-read floor (can't show dropout)")
    # Contract B — too few usable guides to power a call.
    scarcity = max(0.0, (MIN_USABLE_GUIDES - n_usable) / MIN_USABLE_GUIDES)
    if n_usable < MIN_USABLE_GUIDES:
        reasons.append(f"only {n_usable} usable guides ≥{MIN_INITIAL:.0f} reads (under-powered)")
    # Contract C — strong-but-diluted: usable guides disagree AND ≥1 cleared the null (real signal diluted).
    disp_term = 0.0
    if n_usable >= 2:
        disp = statistics.pstdev([g.lfc for g in usable])
        if disp > disp_thresh and n_cleared >= 1:
            reasons.append(f"guides disagree (sd={disp:.2f}; {n_cleared} cleared null) — diluted signal")
        if disp_thresh not in (0.0, float("inf")):
            disp_term = min(1.0, disp / disp_thresh) if n_cleared >= 1 else 0.0

    under_power_score = floor_frac + scarcity + disp_term
    return reasons, under_power_score, n_usable, n_cleared


def qualify(non_hits: list[str], gene_guides: dict[str, list[GuideStat]],
            null: float, disp_thresh: float) -> dict[str, GeneReliability]:
    """Run the reliability contracts over every baseline non-hit gene."""
    out: dict[str, GeneReliability] = {}
    for gene in non_hits:
        guides = gene_guides[gene]
        reasons, ups, n_usable, n_cleared = reliability_contracts(guides, null, disp_thresh)
        out[gene] = GeneReliability(gene, not reasons, reasons, ups, len(guides), n_usable, n_cleared)
    return out


# --------------------------------------------------------------------------- #
# The leakage-free evaluation.
# --------------------------------------------------------------------------- #
def _split_genes(genes: list[str], seed: int) -> tuple[set[str], set[str]]:
    """Gene-disjoint calibration/evaluation halves (the NEGv1 double-use guard)."""
    g = sorted(genes)
    random.Random(seed).shuffle(g)
    half = len(g) // 2
    return set(g[:half]), set(g[half:])


# Q1 recall is sensitive to the NEGv1 calibration/eval split; it needs ~25+ seeds to
# converge (~53%). Too few (e.g. 3) is noisy and can spuriously fail the >50% gate.
def run(seeds: int = 50, sc=None, refs=None, label: str = "leukemia dropout (Wang 2014)") -> dict:
    """The legible screen-reliability QC evaluation. `sc`/`refs` default to the MAGeCK demo (Wang 2014);
    pass a different `ScreenCounts`/`ReferenceSets` to run the SAME Q1–Q4 machinery on any bulk screen
    (the avenue-1 naturally-low-power probe reuses this)."""
    sc = sc if sc is not None else sd.load_counts()
    gs = guide_stats(sc)
    calls = call_genes(gs)
    refs = refs if refs is not None else sd.load_references(sc.genes())

    gene_guides: dict[str, list[GuideStat]] = {}
    for g in gs:
        gene_guides.setdefault(g.gene, []).append(g)
    call_by = {c.gene: c for c in calls}
    non_hit = {g for g, c in call_by.items() if not c.significant}
    ceg_nonhit = sorted(refs.essential & non_hit)
    neg_genes = sorted(refs.nonessential)
    controls = sd.detect_controls(sc.rows)

    def neg_log_q(gene: str) -> float:
        return -statistics.log(max(call_by[gene].q, 1e-12), 10)

    q1s, q2s, q3s, lifts = [], [], [], []
    last = None
    for seed in range(seeds):
        calib_g, eval_g = _split_genes(neg_genes, seed)
        # Null band: control sgRNAs if present, else the NEGv1 CALIBRATION half's guide LFCs.
        if controls:
            calib_lfcs = [g.lfc for g in gs if g.sgrna in {c.sgrna for c in controls}]
        else:
            calib_lfcs = [g.lfc for cg in calib_g for g in gene_guides.get(cg, [])]
        null = null_band(calib_lfcs)
        disp_thresh = _dispersion_threshold([gene_guides[cg] for cg in calib_g if cg in gene_guides])

        verdicts = qualify(sorted(non_hit), gene_guides, null, disp_thresh)

        # Q1 recall — of the CEGv2 essentials the baseline missed, how many flagged under-powered.
        flagged_ceg = [g for g in ceg_nonhit if not verdicts[g].trustworthy]
        q1 = len(flagged_ceg) / len(ceg_nonhit) if ceg_nonhit else float("nan")
        # Q2 precision — false-flag rate on the HELD-OUT NEG half's non-hits (never used for calibration).
        neg_eval_nonhit = [g for g in eval_g if g in non_hit]
        flagged_neg = [g for g in neg_eval_nonhit if not verdicts[g].trustworthy]
        q2 = len(flagged_neg) / len(neg_eval_nonhit) if neg_eval_nonhit else float("nan")
        # Q3 non-redundancy — under-power score vs baseline −log10 q over the CEGv2 non-hits.
        ups = [verdicts[g].under_power_score for g in ceg_nonhit]
        nlq = [neg_log_q(g) for g in ceg_nonhit]
        r = spearman(ups, nlq)
        q3 = abs(r.rho) if isinstance(r, Corr) else float("nan")
        # Q4 lift — within the non-hit pile (CEGv2 vs held-out NEG), does a PARAMETER-FREE rank-combine
        # of (baseline residual, QC under-power) beat baseline alone at ranking the missed essentials?
        pool = ceg_nonhit + neg_eval_nonhit
        labels = [True] * len(ceg_nonhit) + [False] * len(neg_eval_nonhit)
        if len(set(labels)) == 2:
            base = [call_by[g].auroc for g in pool]            # baseline's residual essentiality signal
            qc = [verdicts[g].under_power_score for g in pool]
            comb = [a + b for a, b in zip(rank_avg(base), rank_avg(qc))]   # parameter-free
            ap_base = average_precision(base, labels)
            ap_comb = average_precision(comb, labels)
            lifts.append((ap_base, ap_comb, pool, base, comb, labels))
        q1s.append(q1); q2s.append(q2); q3s.append(q3)
        last = (verdicts, null, disp_thresh, flagged_ceg)

    # Report -----------------------------------------------------------------
    mean = lambda xs: statistics.fmean([x for x in xs if x == x]) if any(x == x for x in xs) else float("nan")
    print(f"\n  screen [{label}]: {len(sc.rows)} sgRNAs / {len(sc.genes())} genes; "
          f"baseline non-hits {len(non_hit)}; controls {len(controls)} "
          f"({'NTC null' if controls else 'NEGv1-as-null'})")
    print(f"  CEGv2∩non-hit (Q1 denom) {len(ceg_nonhit)}   NEGv1 {len(neg_genes)} "
          f"(split ~{len(neg_genes)//2}/{len(neg_genes)-len(neg_genes)//2} calib/eval)")

    q1, q2, q3 = mean(q1s), mean(q2s), mean(q3s)
    print(f"\n  Q1 recall (CEGv2 silent failures flagged) : {q1:.1%}   {'PASS' if q1 > 0.50 else 'FAIL'} (>50%)")
    print(f"  Q2 false-flag (held-out NEGv1 non-hits)   : {q2:.1%}   {'PASS' if q2 < 0.20 else 'FAIL'} (<20%)")
    print(f"  Q3 |ρ(under-power, baseline −log10 q)|     : {q3:.3f}   {'PASS' if q3 < 0.60 else 'FAIL'} (<0.60)")

    lift_pass = False
    ap_base = ap_comb = float("nan")
    if lifts:
        ap_base = mean([x[0] for x in lifts])
        ap_comb = mean([x[1] for x in lifts])
        # bootstrap CI on the representative (first) seed's pool
        _, _, pool, base, comb, labels = lifts[0]
        items = list(zip(base, comb, labels))
        ci_base = bootstrap_ci(items, lambda s: average_precision([i[0] for i in s], [i[2] for i in s]), seed=0)
        ci_comb = bootstrap_ci(items, lambda s: average_precision([i[1] for i in s], [i[2] for i in s]), seed=0)
        lift_pass = ci_comb[0] > ci_base[1]
        print(f"  Q4 AUPRC (non-hit pile): baseline {ap_base:.3f}  combined {ap_comb:.3f}  "
              f"(Δ{ap_comb - ap_base:+.3f}; CIs base{_fmt_ci(ci_base)} comb{_fmt_ci(ci_comb)})")
        print(f"     {'LIFT (CIs separate)' if lift_pass else 'no clean lift (CIs overlap) — reported honestly'}")

    # A few concrete flagged silent failures, with their reasons (the legibility payoff).
    if last:
        verdicts, _, _, flagged_ceg = last
        print("\n  example flagged CEGv2 silent failures (with reasons):")
        for g in flagged_ceg[:5]:
            v = verdicts[g]
            print(f"    {g:<10} q={call_by[g].q:.2f} n={v.n_guides} usable={v.n_usable} → {v.reasons}")

    thesis = (q1 > 0.50 and q2 < 0.20 and q3 < 0.60)
    print(f"\n  THESIS (Q1∧Q2∧Q3): {'SUPPORTED' if thesis else 'NOT supported'}"
          f"{' + Q4 lift' if lift_pass else ''}")
    return {"q1": q1, "q2": q2, "q3": q3, "thesis": thesis, "lift": lift_pass,
            "base_auprc": ap_base, "comb_auprc": ap_comb,
            "n_ceg_nonhit": len(ceg_nonhit), "n_nonhit": len(non_hit)}


def _fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.3f},{ci[1]:.3f}]"


def depth_stress(n_stress: int = 40, target_initial: float = 12.0) -> dict:
    """Validate the (normally dormant) count-floor contracts: take CEGv2 essentials the full-depth
    baseline CORRECTLY CALLS, read-depth-thin them below the power floor (fold-change preserved → still
    truly essential, just under-sequenced), and show the baseline loses them while the QC floor/usable
    contracts flag the resulting silent failures with a count reason. The depth-induced-failure analog
    of the dispersion result — generalizing the layer beyond one well-sequenced library."""
    sc = sd.load_counts()
    refs = sd.load_references(sc.genes())
    call_by0 = {c.gene: c for c in call_genes(guide_stats(sc))}
    called_ceg = sorted(g for g in refs.essential if g in call_by0 and call_by0[g].significant)
    if not called_ceg:
        print("  no called CEGv2 essentials to stress"); return {}
    stressed = set(called_ceg[:n_stress])

    sc2 = sd.downsample_counts(sc, stressed, target_initial, seed=0)
    gs2 = guide_stats(sc2)
    call_by2 = {c.gene: c for c in call_genes(gs2)}
    gene_guides2: dict[str, list[GuideStat]] = {}
    for g in gs2:
        gene_guides2.setdefault(g.gene, []).append(g)
    neg = [g for g in refs.nonessential if g in gene_guides2]
    null = null_band([x.lfc for cg in neg for x in gene_guides2[cg]])
    disp = _dispersion_threshold([gene_guides2[cg] for cg in neg])

    dropped = [g for g in sorted(stressed) if not call_by2[g].significant]
    verdicts = qualify(dropped, gene_guides2, null, disp)
    floor_flagged = [g for g in dropped if not verdicts[g].trustworthy
                     and any(("floor" in r or "usable" in r) for r in verdicts[g].reasons)]
    # Precision under stress: held-out NEG genes (unstressed, well-sequenced) must stay trustworthy.
    neg_nonhit = [g for g in neg if not call_by2[g].significant]
    neg_v = qualify(neg_nonhit, gene_guides2, null, disp)
    neg_flagged = sum(1 for g in neg_nonhit if not neg_v[g].trustworthy)

    print(f"\n  depth-stress: {len(stressed)} called CEGv2 essentials thinned to ~{target_initial:.0f} "
          f"initial reads (fold-change preserved)")
    print(f"  baseline lost to non-hit under low depth : {len(dropped)}/{len(stressed)}")
    print(f"  of those, flagged by count-floor/usable   : {len(floor_flagged)}/{len(dropped)} "
          f"{'PASS' if dropped and len(floor_flagged) / len(dropped) > 0.5 else 'FAIL'} (>50%)")
    print(f"  precision: unstressed NEGv1 non-hits flagged: {neg_flagged}/{len(neg_nonhit)} "
          f"(the dormant-contract stress doesn't false-flag well-sequenced genes)")
    print("  example depth-induced silent failures (with reasons):")
    for g in floor_flagged[:5]:
        v = verdicts[g]
        print(f"    {g:<10} was q={call_by0[g].q:.2f} (CALLED) → now non-hit, "
              f"usable={v.n_usable}/{v.n_guides} → {v.reasons}")
    return {"stressed": len(stressed), "dropped": len(dropped), "floor_flagged": len(floor_flagged),
            "neg_flagged": neg_flagged, "n_neg_nonhit": len(neg_nonhit)}


def low_power_gate(k_grid: tuple[int, ...] = (6, 5, 4, 3, 2), frac: float = 0.5, seed: int = 0) -> dict:
    """Pre-registered fail-fast gate (2026-06-09) for the question: is the full-depth Q4 'no AUPRC
    lift' a real null, or just a CEILING artifact (the baseline already ranks essentials within the
    non-hit pile at ~0.98, so nothing can beat it)? To find out, open head-room with a genuinely
    LOWER-power regime and re-measure the non-hit-pile AUPRC the Q4 combine sits on.

    The manipulation is GUIDE-dropping (`subsample_guides`), NOT read-depth thinning: thinning depth
    preserves each guide's fold-change, so the moderated baseline absorbs it and stays at ceiling
    (verified). Dropping guides attacks the axis the gene-level rank-sum actually uses. NEGv1 is held
    at full depth, so the draw is label-blind (depth ⟂ essentiality) and the calibration set stays
    clean.

    The gate's verdict: head-room does NOT open. Guide-loss
    collapses the baseline's SIGNIFICANCE (q rises → genes fall to non-hit) but PRESERVES its EFFECT
    SIZE (the rank-sum auroc), and the non-hit-pile AUPRC ranks by effect size — so it stays pinned at
    ~0.98 at every k. The Q4 null is therefore STRUCTURAL on a clean dropout screen, and `run_low_power`
    was deliberately NOT built (it would only re-derive a structural null). Re-ranking essentials is the
    baseline's job and it is already maxed; the QC layer's value is the orthogonal power-QUALIFICATION
    axis (Q1/Q2/Q3) — legibility, not accuracy."""
    sc = sd.load_counts()
    refs = sd.load_references(sc.genes())
    ceg, neg = set(refs.essential), set(refs.nonessential)
    pool = [g for g in sorted(sc.genes()) if g not in neg]      # subsample everything EXCEPT NEGv1
    target = set(random.Random(seed).sample(pool, int(len(pool) * frac)))

    def pile(sc_):
        call_by = {c.gene: c for c in call_genes(guide_stats(sc_))}
        nonhit = {g for g, c in call_by.items() if not c.significant}
        ceg_nh, neg_nh = sorted(ceg & nonhit), sorted(neg & nonhit)
        base = [call_by[g].auroc for g in ceg_nh + neg_nh]
        labels = [True] * len(ceg_nh) + [False] * len(neg_nh)
        ap = average_precision(base, labels) if len(set(labels)) == 2 else float("nan")
        called = sum(1 for g in ceg if g in call_by and call_by[g].significant)
        med = lambda gs_, f: statistics.median([f(call_by[g]) for g in gs_]) if gs_ else float("nan")
        return ap, len(ceg_nh), called, med(ceg_nh, lambda c: c.auroc), med(ceg_nh, lambda c: c.q)

    print(f"\n  subsample pool: {len(target)} of {len(pool)} non-NEG genes (label-blind {frac:.0%}, seed {seed})")
    print(f"  {'regime':<15}{'CEG-called':>11}{'CEG-nonhit':>12}{'pile-AUPRC':>12}{'med-auroc':>11}{'med-q':>8}")
    ap0, nnh0, called0, auroc0, q0 = pile(sc)
    print(f"  {'full-depth':<15}{called0:>11}{nnh0:>12}{ap0:>12.3f}{auroc0:>11.3f}{q0:>8.3f}")
    by_k, off_ceiling = {}, False
    for k in k_grid:
        ap, nnh, called, auroc, q = pile(sd.subsample_guides(sc, target, k, seed=seed))
        print(f"  {'k=' + str(k) + ' guides':<15}{called:>11}{nnh:>12}{ap:>12.3f}{auroc:>11.3f}{q:>8.3f}")
        by_k[k] = ap
        off_ceiling = off_ceiling or ap <= 0.92
    print(f"\n  GATE — non-hit-pile AUPRC drops below 0.92 at some k?  "
          f"{'YES → head-room; build run_low_power' if off_ceiling else 'NO → Q4 null is STRUCTURAL; full build skipped'}")
    print("  Read: guide-loss collapses SIGNIFICANCE (q ↑, more silent failures) but PRESERVES EFFECT")
    print("        SIZE (auroc) → within-pile ranker unbreakable; QC value = qualification, not accuracy.")
    return {"full_depth_auprc": ap0, "by_k": by_k, "off_ceiling": off_ceiling}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Legible screen-reliability QC layer over a bulk CRISPR screen.")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--depth-stress", action="store_true",
                    help="stress-test the count-floor contracts: thin called essentials below the power floor")
    ap.add_argument("--low-power-gate", action="store_true",
                    help="2026-06-09 gate: does a lower-power (guide-dropped) regime open Q4 head-room? (no — structural)")
    cli = ap.parse_args()
    try:
        if cli.depth_stress:
            print("Screen-reliability QC — depth-stress: do the count-floor contracts catch under-sequenced genes?")
            depth_stress()
        elif cli.low_power_gate:
            print("Screen-reliability QC — low-power gate: is the Q4 'no accuracy lift' a ceiling artifact?")
            low_power_gate()
        else:
            print("Screen-reliability QC — qualifying the baseline's NON-hits (under-powered vs trustworthy)")
            run(seeds=cli.seeds)
    except sd.DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
