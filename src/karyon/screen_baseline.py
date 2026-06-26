"""screen_baseline — a legible, minimal reimplementation of the MAGeCK-style screen-analysis core.

This is the INCUMBENT the QC layer is measured against — not a strawman and not a byte-match to
MAGeCK, but a faithful-in-spirit deterministic pipeline whose credibility is established by the
field-standard check: it must separate Hart CEGv2 (core-essential) from NEGv1 (non-essential).

The pipeline (each step legible, stdlib-only):

  1. **median-ratio normalization** (DESeq size factors) — correct for library-size / depth so the
     two cell lines and the initial/final arms are comparable.
  2. **per-sgRNA moderated depletion z** — normalized log2 fold-change (initial→final), standardized
     by a label-free mean→spread trend (`stats_kit.mean_variance_trend`): low-count guides have a
     wider null spread, so their extreme LFCs are discounted. This is the "negative-binomial
     moderation" in spirit, without a full GLM/MLE.
  3. **per-gene rank-sum** — each gene's guides vs the global guide pool on the depletion score
     (a tie-aware Mann-Whitney computed from one global ranking, so it's O(N log N), not O(genes·N)).
     A gene whose guides are systematically more depleted than the pool gets a small one-sided p.
  4. **Benjamini–Hochberg FDR** — gene p → q; `significant = q < FDR_ALPHA` is the baseline's
     hit/non-hit verdict. The non-hits are what the QC layer (`screen_qc.py`) then qualifies.

Full α-RRA (a permutation null over rank prefixes) and the per-sgRNA NB GLM are deliberately NOT
built — they don't move CEGv2/NEGv1 recovery enough to justify the complexity, and they fight the
stdlib charter. Optionally cross-check against real MAGeCK output if it's ever cached (not required).

    cd karyon/probe && python screen_baseline.py     # run the pipeline; print the B1/B2 credibility gate
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass

from . import screen_qc_data as sd
from .stats_kit import (average_precision, benjamini_hochberg, mann_whitney,
                       mean_variance_trend, rank_avg)

FDR_ALPHA = 0.05            # pre-registered gene-level FDR for the hit/non-hit call
_PSEUDO = 0.5              # fold-change pseudocount
_TREND_BINS = 30          # mean→spread bins for the moderation


@dataclass(frozen=True)
class GuideStat:
    """One sgRNA's depletion summary (the QC layer reads `init_count` and `lfc` too)."""

    sgrna: str
    gene: str
    lfc: float                  # normalized log2 fold-change, initial→final (depletion < 0)
    init_count: float           # mean normalized initial count (the dropout-power proxy)
    depletion: float            # moderated score, higher = more depleted (= −z)


@dataclass(frozen=True)
class GeneCall:
    """The baseline's gene-level verdict — a lossy scalar (q) the QC layer argues throws away signal."""

    gene: str
    lfc: float                  # mean of its guides' LFC (directionality / reporting)
    auroc: float                # rank-sum: fraction its guides outrank the pool on depletion (essentiality)
    p: float                    # one-sided depletion p (rank-sum)
    q: float                    # Benjamini–Hochberg FDR
    n_guides: int
    significant: bool           # q < FDR_ALPHA → a screen HIT; else a non-hit (QC-layer territory)


# --------------------------------------------------------------------------- #
# 1. Median-ratio normalization (DESeq size factors).
# --------------------------------------------------------------------------- #
def size_factors(sc: sd.ScreenCounts) -> dict[str, float]:
    """DESeq median-of-ratios size factors: per sample, exp(median over sgRNAs of
    log(count) − log(geometric-mean-across-samples)), using only sgRNAs positive in every sample."""
    logref: dict[str, float] = {}
    for r in sc.rows:
        cs = [r.counts[s] for s in sc.samples]
        if all(c > 0 for c in cs):
            logref[r.sgrna] = sum(math.log(c) for c in cs) / len(cs)
    factors: dict[str, float] = {}
    for s in sc.samples:
        ratios = [math.log(r.counts[s]) - logref[r.sgrna]
                  for r in sc.rows if r.sgrna in logref and r.counts[s] > 0]
        factors[s] = math.exp(statistics.median(ratios)) if ratios else 1.0
    return factors


# --------------------------------------------------------------------------- #
# 2. Per-sgRNA moderated depletion score.
# --------------------------------------------------------------------------- #
def guide_stats(sc: sd.ScreenCounts, direction: str = "deplete") -> list[GuideStat]:
    """Normalized LFC per sgRNA + a moderated hit score (LFC standardized by the trended null spread at
    that guide's initial-count level). `direction='deplete'` (default): essential genes DROP OUT, hit =
    low LFC, score = −(lfc−med)/sd. `direction='enrich'`: positive-selection / resistance genes RISE,
    hit = high LFC, score = +(lfc−med)/sd. Only the sign flips; `call_genes` ranks by the score
    identically either way (the `.depletion` field is the directional hit score)."""
    sign = -1.0 if direction == "deplete" else 1.0
    sf = size_factors(sc)
    sgrna: list[str] = []
    gene: list[str] = []
    lfcs: list[float] = []
    inits: list[float] = []
    for r in sc.rows:
        init = statistics.fmean([r.counts[s] / sf[s] for s in sc.initial])
        fin = statistics.fmean([r.counts[s] / sf[s] for s in sc.final])
        sgrna.append(r.sgrna)
        gene.append(r.gene)
        lfcs.append(math.log2((fin + _PSEUDO) / (init + _PSEUDO)))
        inits.append(init)
    xs = [math.log10(c + 1.0) for c in inits]          # trend over (log) initial count = the power axis
    trend = mean_variance_trend(xs, lfcs, n_bins=_TREND_BINS)
    med = statistics.median(lfcs)
    out = []
    for sg, g, lfc, init, x in zip(sgrna, gene, lfcs, inits, xs):
        sd_x = trend(x) or 1e-9
        out.append(GuideStat(sg, g, lfc, init, depletion=sign * (lfc - med) / sd_x))
    return out


# --------------------------------------------------------------------------- #
# 3+4. Per-gene rank-sum + BH-FDR.
# --------------------------------------------------------------------------- #
def call_genes(gstats: list[GuideStat]) -> list[GeneCall]:
    """Per-gene one-sided rank-sum on the depletion score (its guides vs the global pool), BH-corrected.
    Uses ONE global tie-aware ranking, so it's O(N log N) for the whole screen."""
    scores = [g.depletion for g in gstats]
    ranks = rank_avg(scores)
    n = len(scores)
    tie = sum(t ** 3 - t for t in Counter(scores).values())     # tie correction over the full pool
    by_gene: dict[str, list[tuple[GuideStat, float]]] = {}
    for g, rk in zip(gstats, ranks):
        by_gene.setdefault(g.gene, []).append((g, rk))

    genes, lfcs, aurocs, pvals, ns = [], [], [], [], []
    for gene, items in by_gene.items():
        n_a = len(items)
        n_b = n - n_a
        u_a = sum(rk for _, rk in items) - n_a * (n_a + 1) / 2.0
        auroc = u_a / (n_a * n_b) if n_b else 0.5
        var_u = (n_a * n_b / 12.0) * ((n + 1) - tie / (n * (n - 1))) if n > 1 and n_b else 0.0
        if var_u > 0:
            z = (u_a - n_a * n_b / 2.0) / math.sqrt(var_u)
            p = 0.5 * math.erfc(z / math.sqrt(2.0))             # one-sided: P(rank-sum this depleted)
        else:
            p = 1.0
        genes.append(gene)
        lfcs.append(statistics.fmean([gg.lfc for gg, _ in items]))
        aurocs.append(auroc)
        pvals.append(p)
        ns.append(n_a)
    qs = benjamini_hochberg(pvals)
    return [GeneCall(g, lfc, a, p, q, n_a, q < FDR_ALPHA)
            for g, lfc, a, p, q, n_a in zip(genes, lfcs, aurocs, pvals, qs, ns)]


def call_screen(sc: sd.ScreenCounts) -> list[GeneCall]:
    """The full deterministic incumbent: normalize → moderated guide depletion → gene rank-sum → BH."""
    return call_genes(guide_stats(sc))


# --------------------------------------------------------------------------- #
# B1/B2 credibility gate (CEGv2 vs NEGv1 recovery).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BaselineGate:
    auroc: float                # B1: essential vs non-essential separation by the gene score
    auprc: float
    ceg_median_lfc: float       # B2: directionality
    neg_median_lfc: float
    n_ceg: int
    n_neg: int
    n_ceg_significant: int      # essentials the baseline CALLS (q<FDR_ALPHA)
    n_ceg_missed: int           # essentials left non-significant — the Q1 silent-failure denominator


def gate(calls: list[GeneCall], refs: sd.ReferenceSets) -> BaselineGate:
    by = {c.gene: c for c in calls}
    ceg = [by[g] for g in refs.essential if g in by]
    neg = [by[g] for g in refs.nonessential if g in by]
    pos = [c.auroc for c in ceg]
    neg_s = [c.auroc for c in neg]
    mw = mann_whitney(pos, neg_s)
    auroc = mw.auroc if hasattr(mw, "auroc") else 0.5
    auprc = average_precision(pos + neg_s, [True] * len(pos) + [False] * len(neg_s))
    return BaselineGate(
        auroc, auprc,
        statistics.median([c.lfc for c in ceg]) if ceg else float("nan"),
        statistics.median([c.lfc for c in neg]) if neg else float("nan"),
        len(ceg), len(neg),
        sum(c.significant for c in ceg),
        sum(not c.significant for c in ceg))


if __name__ == "__main__":
    print("Baseline incumbent — median-ratio norm → moderated guide depletion → gene rank-sum → BH\n")
    try:
        sc = sd.load_counts()
    except sd.DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
    calls = call_screen(sc)
    n_hit = sum(c.significant for c in calls)
    print(f"  genes called             : {len(calls)}  ({n_hit} significant at q<{FDR_ALPHA})")
    try:
        refs = sd.load_references(sc.genes())
    except sd.DatasetUnavailable as e:
        print(f"SKIP (references) — {e}")
        raise SystemExit(0)
    g = gate(calls, refs)
    print("\n  B1 — CEGv2 vs NEGv1 separation (gene rank-sum AUROC as the essentiality score)")
    print(f"     AUROC {g.auroc:.3f}   AUPRC {g.auprc:.3f}   (n_ceg={g.n_ceg}, n_neg={g.n_neg})")
    print(f"     {'PASS' if g.auroc > 0.85 and g.auprc > 0.80 else 'FAIL'} "
          f"(gate: AUROC>0.85 and AUPRC>0.80)")
    print("\n  B2 — directionality (median LFC; essentials should deplete, below non-essentials)")
    print(f"     CEGv2 median LFC {g.ceg_median_lfc:+.3f}   NEGv1 median LFC {g.neg_median_lfc:+.3f}")
    print(f"     {'PASS' if g.ceg_median_lfc < g.neg_median_lfc and g.ceg_median_lfc < 0 else 'FAIL'}")
    print(f"\n  silent-failure denominator: {g.n_ceg_missed}/{g.n_ceg} CEGv2 essentials left "
          f"NON-significant ({g.n_ceg_significant} called) — the Q1 recall is measured on these.")
