"""crispr_qc — a LEGIBLE reliability/QC layer over a CRISPRi screen (the karyon thesis, made real).

The thesis (karyon/the design notes §2): AI authors a *legible* reliability / QC layer over an
existing loop — a design-rule-check (DRC) + contracts doctrine ported to biology — NOT a black-box
predictor. This module is the first build of that thesis on the most bootstrappable substrate: CRISPRi
functional-genomics screens.

The problem it addresses: in a CRISPRi screen a gene scored "no phenotype / non-hit" is **ambiguous**
— either a true negative, or a **silent failure** (the guide never knocked the gene down, so the gene
was never actually tested). ~40–50% of guides are ineffective, so the silent-failure tail is large and
a screen that ignores it mislabels under-powered genes as true negatives.

The QC layer is a small library of **contracts**, exactly this DRC shape:

  1. HARD contracts (`hard_contracts`) — deterministic, legible RULES that need no fitting: a `TTTT`
     Pol-III terminator truncates the sgRNA, extreme GC loads poorly, a long homopolymer mis-synthesizes.
     A guide that violates one is structurally compromised regardless of any model — the DRC spine.
  2. A THIN transparent layer (`fit_model` over `features`) — a ridge over *named, auditable* sequence
     features (`linmodel.BayesRidge`, the same stdlib core the other probes use) for the graded efficacy
     where mechanism runs out. Coefficients are inspectable; this is "explicit spine + thin statistical
     layer," the honest doctrine (pure-rules is refuted — the field trends hybrid).

`check_guides` flags a guide (silent-failure risk) when a hard contract fires OR predicted activity is
low, and every flag carries a reason. `gene_report` rolls flags up to "this gene is under-powered — its
non-hit is untrustworthy." Validation is **non-circular**: the flag is built from sequence alone and
scored against the INDEPENDENT measured activity (held-out, gene-disjoint split).

Scope/honesty: features are sequence-intrinsic only (the ~40% of the Horlbeck set needing no genome
tracks — TSS distance / nucleosome / DNase are omitted by design), so this is a *lower bound* on the
full QC layer — a positive result is conservative. We do NOT benchmark ρ against Azimuth/CRISPOR (that
is the commoditized predictor-margin lever the thesis rejects); the claim is that a *legible* layer
recovers the silent-failure tail with auditable reasons.

    python -m karyon.crispr_qc --seeds 3        # the QC evaluation + legible coefficients
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from . import crispr_qc_data
from .crispr_qc_data import Record
from .linmodel import BayesRidge
from .stats_kit import Corr, fmt, mann_whitney, spearman

# An sgRNA below this CRISPRi activity is an ineffective guide — the "silent-failure" tail. 0.20 picks
# out ≈44% of the Horlbeck library, matching the field's "~40–50% of guides are ineffective."
SILENT_FAIL_ACTIVITY = 0.20
# A gene is "powered" if at least one of its guides is effective (clears the bar above); a gene whose
# every guide is below it is under-powered — a screen's non-hit for it is untrustworthy. A realistic
# un-optimized CRISPRi library carries this many guides per gene (Horlbeck v2's ~12 is the pre-QC'd
# exception that makes gene-level failure vanish — the contrast the C2 evaluation draws).
NAIVE_K = 3

_COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


# --------------------------------------------------------------------------- #
# Legible sequence features (all computable from the protospacer alone — no genome tracks).
# --------------------------------------------------------------------------- #
def gc(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


def max_run(seq: str) -> int:
    """Longest single-base homopolymer run."""
    best = run = 0
    prev = ""
    for b in seq:
        run = run + 1 if b == prev else 1
        prev = b
        best = max(best, run)
    return best


def hairpin_score(seq: str, k: int = 4) -> float:
    """A deterministic self-complementarity proxy for sgRNA secondary structure (no ViennaRNA): the
    fraction of k-mer windows whose reverse complement also occurs in the guide — i.e. the guide's
    propensity to base-pair with itself and fold instead of loading. Legible by construction."""
    if len(seq) < 2 * k:
        return 0.0
    wins = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    present = set(wins)
    return sum(1 for w in wins if revcomp(w) in present) / len(wins)


# The named feature vector, intercept first. Order is fixed and mirrored by FEATURE_NAMES so fitted
# coefficients can be printed against their meaning (the legibility the thesis is about).
_PAM_WINDOW = 4                                   # PAM-proximal (3') positions to one-hot
FEATURE_NAMES = (
    ["intercept", "gc", "seed_gc(3'6)", "length", "max_run", "has_TTTT",
     "fracA", "fracC", "fracG", "fracT", "hairpin", "starts_G"]
    + [f"pam-{p}:{b}" for p in range(_PAM_WINDOW, 0, -1) for b in "ACGT"]
)


def features(seq: str) -> list[float]:
    """Named legible features for the thin statistical layer (aligned to FEATURE_NAMES)."""
    n = len(seq)
    seed = seq[-6:]
    tail = seq[-_PAM_WINDOW:]
    onehot = [0.0] * (_PAM_WINDOW * 4)
    pad = _PAM_WINDOW - len(tail)
    for i, b in enumerate(tail):
        j = "ACGT".find(b)
        if j >= 0:
            onehot[(pad + i) * 4 + j] = 1.0
    return [
        1.0,
        gc(seq),
        gc(seed),
        (n - 18) / 7.0,                           # length 18..25 -> ~0..1
        max_run(seq) / n,
        1.0 if "TTTT" in seq else 0.0,
        seq.count("A") / n, seq.count("C") / n, seq.count("G") / n, seq.count("T") / n,
        hairpin_score(seq),
        1.0 if seq[:1] == "G" else 0.0,
    ] + onehot


# --------------------------------------------------------------------------- #
# Contract 1a — HARD deterministic rules (the DRC spine; no fitting).
# --------------------------------------------------------------------------- #
def hard_contracts(seq: str) -> list[str]:
    """Deterministic structural violations — each a human-readable reason. Empty = passes the rules."""
    reasons = []
    if "TTTT" in seq:
        reasons.append("TTTT: Pol-III terminator truncates the sgRNA")
    g = gc(seq)
    if g < 0.20:
        reasons.append(f"GC {g:.0%} <20%: poor RISC loading")
    elif g > 0.80:
        reasons.append(f"GC {g:.0%} >80%: over-stable / poor specificity")
    if max_run(seq) >= 5:
        reasons.append(f"homopolymer run {max_run(seq)}: synthesis/folding risk")
    return reasons


# --------------------------------------------------------------------------- #
# The combined QC verdict.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GuideVerdict:
    gene: str
    seq: str
    predicted: float           # predicted CRISPRi activity (the thin layer)
    flagged: bool              # silent-failure risk (hard rule OR predicted-weak)
    reasons: list[str]         # human-readable reasons the flag fired


def fit_model(train: list[Record], lam: float = 1.0) -> BayesRidge:
    """Fit the transparent ridge over named features on measured activity."""
    m = BayesRidge(len(FEATURE_NAMES), lam=lam)
    m.observe_all([features(r.seq) for r in train], [r.activity for r in train])
    return m


def check_guides(model: BayesRidge, recs: list[Record]) -> list[GuideVerdict]:
    """Run the contracts over each guide — predicted activity + hard rules → a flag with reasons."""
    out = []
    for r in recs:
        pred = model.predict(features(r.seq))
        reasons = hard_contracts(r.seq)
        if pred < SILENT_FAIL_ACTIVITY:
            reasons = reasons + [f"predicted activity {pred:+.2f} < {SILENT_FAIL_ACTIVITY}"]
        out.append(GuideVerdict(r.gene, r.seq, pred, bool(reasons), reasons))
    return out


def gene_report(verdicts: list[GuideVerdict]) -> dict[str, dict]:
    """Roll guide flags up to a per-gene QC verdict: a gene every guide of which is flagged is
    under-powered — its screen non-hit is untrustworthy."""
    by_gene: dict[str, list[GuideVerdict]] = {}
    for v in verdicts:
        by_gene.setdefault(v.gene, []).append(v)
    report = {}
    for gene, vs in by_gene.items():
        n_flag = sum(v.flagged for v in vs)
        best_pred = max(v.predicted for v in vs)
        report[gene] = {
            "n_guides": len(vs),
            "n_flagged": n_flag,
            "best_predicted": best_pred,
            "under_powered": best_pred < SILENT_FAIL_ACTIVITY,   # no guide predicted to be effective
        }
    return report


# --------------------------------------------------------------------------- #
# Evaluation — non-circular, gene-disjoint split, 3 seeds.
# --------------------------------------------------------------------------- #
def _rho(a, b) -> float:
    r = spearman(a, b)
    return r.rho if isinstance(r, Corr) else 0.0


def _auroc(values, is_positive) -> float:
    """AUROC that a higher `value` marks the positive class (here: higher measured activity ⇒ NOT a
    silent failure). Reuses the tie-aware Mann-Whitney in stats_kit."""
    pos = [v for v, p in zip(values, is_positive) if p]
    neg = [v for v, p in zip(values, is_positive) if not p]
    mw = mann_whitney(pos, neg)
    return mw.auroc if hasattr(mw, "auroc") else 0.5


@dataclass
class SeedResult:
    rho: float                 # held-out Spearman(predicted, measured activity)
    qc_auroc: float            # continuous predicted activity separates the measured silent-failure tail
    hard_recall: float         # DRC spine alone: fraction of the measured tail the hard rules catch
    hard_precision: float      # ...and how clean those catches are
    flag_recall: float         # full flag (hard OR predicted-weak) operating point
    flag_precision: float
    flag_frac: float           # fraction of guides the full flag fires on
    gene_auroc: float          # C2: predicted best-of-k separates powered vs under-powered genes (naive lib)
    gene_under_frac: float     # fraction of genes under-powered in the naive k-guide library
    shuf_rho: float            # noise baseline: shuffled-label held-out ρ
    shuf_auroc: float


def _recall_precision(flag: list[bool], truth: list[bool]) -> tuple[float, float]:
    tp = sum(1 for f, t in zip(flag, truth) if f and t)
    return tp / max(1, sum(truth)), tp / max(1, sum(flag))


def _naive_gene_eval(test: list[Record], model: BayesRidge, seed: int) -> tuple[float, float]:
    """Contract 2 on a realistic library: draw NAIVE_K random guides per gene (no efficacy pre-QC) and
    ask whether the legible best-of-k prediction separates genes that DO get an effective guide from the
    under-powered ones whose every sampled guide is weak — the genes a screen would silently mis-call."""
    rng = random.Random(seed * 7 + 1)
    by_gene: dict[str, list[Record]] = {}
    for r in test:
        by_gene.setdefault(r.gene, []).append(r)
    g_pred, g_powered = [], []
    for vs in by_gene.values():
        if len(vs) < NAIVE_K:
            continue
        sample = rng.sample(vs, NAIVE_K)
        g_pred.append(max(model.predict(features(r.seq)) for r in sample))
        g_powered.append(max(r.activity for r in sample) >= SILENT_FAIL_ACTIVITY)
    if not g_powered:
        return 0.5, 0.0
    return _auroc(g_pred, g_powered), 1.0 - sum(g_powered) / len(g_powered)


def split_by_gene(recs: list[Record], seed: int, test_frac: float = 0.30
                  ) -> tuple[list[Record], list[Record]]:
    """Gene-disjoint train/test split — no gene appears in both, so held-out scores can't be inflated by
    memorizing a gene's other guides (the honest validation the non-circularity claim rests on)."""
    genes = sorted({r.gene for r in recs})
    rng = random.Random(seed)
    rng.shuffle(genes)
    test_genes = set(genes[:int(len(genes) * test_frac)])
    return ([r for r in recs if r.gene not in test_genes],
            [r for r in recs if r.gene in test_genes])


def evaluate_seed(recs: list[Record], seed: int, test_frac: float = 0.30) -> SeedResult:
    train, test = split_by_gene(recs, seed, test_frac)
    rng = random.Random(seed)
    model = fit_model(train)
    verdicts = check_guides(model, test)
    pred = [v.predicted for v in verdicts]
    meas = [r.activity for r in test]
    weak = [m < SILENT_FAIL_ACTIVITY for m in meas]               # the measured silent-failure tail
    not_weak = [not w for w in weak]

    # Guide level: the continuous QC score (predicted activity, higher = NOT weak) is the threshold-free
    # separator of the tail; the binary flag is one operating point on it.
    qc_auroc = _auroc(pred, not_weak)
    hard = [bool(hard_contracts(r.seq)) for r in test]
    hard_recall, hard_precision = _recall_precision(hard, weak)
    flag_recall, flag_precision = _recall_precision([v.flagged for v in verdicts], weak)

    gene_auroc, gene_under = _naive_gene_eval(test, model, seed)

    # Noise baseline: shuffle training labels, refit — signal must collapse to ρ≈0, AUROC≈0.5.
    sy = [r.activity for r in train]
    rng.shuffle(sy)
    shuf_model = BayesRidge(len(FEATURE_NAMES), lam=1.0)
    shuf_model.observe_all([features(r.seq) for r in train], sy)
    shuf_pred = [shuf_model.predict(features(r.seq)) for r in test]

    return SeedResult(
        _rho(pred, meas), qc_auroc, hard_recall, hard_precision,
        flag_recall, flag_precision, sum(v.flagged for v in verdicts) / len(verdicts),
        gene_auroc, gene_under, _rho(shuf_pred, meas), _auroc(shuf_pred, not_weak))


def run(seeds: int = 3) -> None:
    recs = crispr_qc_data.load_records()
    n_gene = len({r.gene for r in recs})
    tail = sum(r.activity < SILENT_FAIL_ACTIVITY for r in recs) / len(recs)
    # Context: in the pre-QC'd v2 library almost every gene already has an effective guide — gene-level
    # failure is rare BY DESIGN, which is exactly why C2 is evaluated on a naive un-optimized library.
    best = {}
    for r in recs:
        best[r.gene] = max(best.get(r.gene, -9.9), r.activity)
    full_under = sum(b < SILENT_FAIL_ACTIVITY for b in best.values()) / n_gene
    print(f"\nCRISPRi screen-QC — legible reliability layer over {len(recs)} guides / {n_gene} genes "
          f"(Horlbeck 2016 activity scores)")
    print(f"silent-failure tail = measured activity < {SILENT_FAIL_ACTIVITY} ({tail:.0%} of guides); "
          f"gene under-powered in full v2 library = {full_under:.1%} (well-powered by design)\n")

    results = [evaluate_seed(recs, s) for s in range(seeds)]

    def line(label, attr, fmt_="{:+.3f}"):
        vals = [getattr(r, attr) for r in results]
        cells = "  ".join(fmt_.format(v) for v in vals)
        print(f"  {label:<36} {cells}    mean {fmt_.format(sum(vals) / len(vals))}")

    print(f"  {'metric':<36} " + "  ".join(f" seed{i}" for i in range(seeds)))
    line("guide ρ(pred, measured)", "rho")
    line("QC-score AUROC (silent-fail tail)", "qc_auroc")
    line("  hard-rule recall (DRC spine)", "hard_recall")
    line("  hard-rule precision", "hard_precision")
    line("full-flag recall", "flag_recall")
    line("full-flag precision", "flag_precision")
    line("fraction flagged", "flag_frac")
    line(f"C2 gene-power AUROC (naive k={NAIVE_K})", "gene_auroc")
    line(f"C2 under-powered genes (naive k={NAIVE_K})", "gene_under_frac")
    line("noise baseline ρ (shuffled)", "shuf_rho")
    line("noise baseline QC AUROC", "shuf_auroc")

    # Legible coefficients — the whole point: every effect is named and inspectable.
    model = fit_model(recs)
    ranked = sorted(zip(FEATURE_NAMES, model.weights()), key=lambda t: -abs(t[1]))
    print("\n  legible coefficients (top 12 by |weight|; activity rises with +, falls with −):")
    for name, wi in ranked[:12]:
        print(f"     {name:<16} {wi:+.3f}")

    # A few example flagged guides with their reasons — the auditable QC output.
    verdicts = check_guides(model, recs)
    examples = [v for v in verdicts if v.flagged and len(v.reasons) >= 2][:4]
    print("\n  example flags (auditable reasons):")
    for v in examples:
        print(f"     {v.gene:<8} {v.seq:<24} → {'; '.join(v.reasons)}")

    def mean(attr):
        return sum(getattr(r, attr) for r in results) / len(results)

    return {
        "rho_mean": mean("rho"),
        "qc_auroc_mean": mean("qc_auroc"),
        "hard_recall_mean": mean("hard_recall"),
        "flag_precision_mean": mean("flag_precision"),
        "gene_auroc_mean": mean("gene_auroc"),
        "gene_under_frac_mean": mean("gene_under_frac"),
        "shuf_rho_mean": mean("shuf_rho"),
        "shuf_auroc_mean": mean("shuf_auroc"),
        "n_guides": len(recs),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Legible CRISPRi screen-QC evaluation.")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    try:
        run(seeds=args.seeds)
    except crispr_qc_data.DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
