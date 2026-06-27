"""retro_honesty — the legible reliability/QC layer over the USPTO-50k retrosynthesis eval.

The karyon pivot thesis: AI authors a *legible* reliability / QC / qualification
layer over an existing loop — not a black-box predictor. Here the "existing loop" is a retrosynthesis
benchmark, and the documented failure is **leakage-inflated accuracy** (RetroXpert USPTO-50k Top-1
70.4%→62.1% after a leak fix, RSC d4dd00007b). This module is a DRC/contracts doctrine ported one
domain further: a deterministic, per-reaction **leakage audit** that (a) measures how much train↔test
overlap the benchmark carries, (b) shows that overlap inflates a transparent baseline, and (c) reports the
honest (leakage-free) number — every flag carrying an auditable reason.

Reuses the engine verbatim: `contracts.ContractSet` (the leakage DRC is HARD contracts — zero fitting) and
`stats_kit` (Mann-Whitney AUROC, bootstrap CI). The baseline is `retro_baseline` (retrosim-lite).

Two prediction arms, deliberately:
  * **reaction class 1..10** — the headline inflation arm. Class correctness is NOT defined by duplication
    (a product has many same-class non-duplicate neighbours), so "leakage explains correctness" is a real,
    non-circular empirical claim — P2 (inflation) and P3 (AUROC) are evaluated here.
  * **reactant recovery** — a bounded sub-result: a stdlib retriever (no template transfer) recovers exact
    reactants only on near-duplicate reactions (~1%), so its "accuracy" IS the duplication rate. That bounds
    the question and motivates the RDKit template arm (the faithful retrosim, deferred — see the RESULT doc).

Pre-registered verdict (set before running):
  P1  leakage prevalence ≥ 10% of standard-split test reactions (model-free).
  P2  class top-1 inflation: standard split − leakage-free partition ≥ 10 absolute points.
  P3  explanatory (make-or-break): nn-similarity predicts class correctness at AUROC ≥ 0.65.
  Measured, not pre-judged: does the layer buy *accuracy* (a better model) or only *legibility* (an honest
  number + auditable reasons)? The honest-eval harness is itself the candidate deliverable.

    python -m karyon.retro_honesty --seeds 3
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from dataclasses import dataclass

from . import contracts
from . import stats_kit
from .retro_baseline import Outcome, class_accuracy, run_baseline, topk_reactant_accuracy
from .uspto_data import (
    DatasetUnavailable,
    Reaction,
    Split,
    load_reactions,
    patent_disjoint_split,
    random_split,
)

_TAU = 0.7          # near-duplicate threshold: a train product at Jaccard ≥ 0.7 is "near-trivially close"


# --------------------------------------------------------------------------- #
# The leakage DRC — HARD contracts (no calibration). Each names a distinct, legible leakage mechanism.
#   design = an Outcome (carries nn_sim + exact_in_train); ctx = a TrainView of the train side.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainView:
    patents: set[str]
    reactions: set[tuple[str, str]]      # (product, reactant_sig) — full-reaction duplication
    tau: float


def leakage_contracts() -> contracts.ContractSet:
    cs = contracts.ContractSet("retro-leakage")
    cs.add(contracts.Contract(
        "EXACT_PRODUCT_IN_TRAIN",
        lambda o, ctx: "product appears verbatim in train" if o.exact_in_train else None))
    cs.add(contracts.Contract(
        "NEAR_DUP_PRODUCT",
        lambda o, ctx: (f"a train product is near-identical (Jaccard {o.nn_sim:.2f} ≥ {ctx.tau})"
                        if o.nn_sim >= ctx.tau else None)))
    cs.add(contracts.Contract(
        "SAME_PATENT_IN_TRAIN",
        lambda o, ctx: ("a reaction from the same patent is in train"
                        if o.rxn.rid and o.rxn.rid in ctx.patents else None)))
    cs.add(contracts.Contract(
        "REACTION_DUP",
        lambda o, ctx: ("the full reaction (product + reactants) is duplicated in train"
                        if (o.rxn.product, o.rxn.reactant_sig) in ctx.reactions else None)))
    return cs


def _train_view(split: Split, tau: float) -> TrainView:
    return TrainView(
        patents={r.rid for r in split.train if r.rid},
        reactions={(r.product, r.reactant_sig) for r in split.train},
        tau=tau)


@dataclass(frozen=True)
class Audit:
    split: Split
    outcomes: list[Outcome]
    verdicts: list[contracts.Verdict]    # one per test outcome, parallel

    @property
    def leaked(self) -> list[bool]:
        return [not v.ok for v in self.verdicts]

    @property
    def clean(self) -> list[Outcome]:
        return [o for o, v in zip(self.outcomes, self.verdicts) if v.ok]


def audit_split(split: Split, *, tau: float = _TAU) -> Audit:
    _, outcomes = run_baseline(split)
    cs, tv = leakage_contracts(), _train_view(split, tau)
    verdicts = [cs.evaluate(o, tv) for o in outcomes]
    return Audit(split, outcomes, verdicts)


# --------------------------------------------------------------------------- #
# Metrics on an audit.
# --------------------------------------------------------------------------- #
def contract_fire_rates(audit: Audit) -> dict[str, float]:
    cs = leakage_contracts()
    n = len(audit.outcomes) or 1
    fired = Counter()
    for v in audit.verdicts:
        for name in v.fired:
            fired[name] += 1
    return {c.name: fired.get(c.name, 0) / n for c in cs.contracts}


def majority_class_floor(outcomes: list[Outcome]) -> float:
    """Accuracy of the trivial 'predict the commonest class' baseline on this partition."""
    n = len(outcomes) or 1
    counts = Counter(o.rxn.klass for o in outcomes)
    return (max(counts.values()) / n) if counts else 0.0


def auroc_sim_explains(outcomes: list[Outcome], correct: str) -> stats_kit.MannWhitney | stats_kit.Degenerate:
    """AUROC that nn-similarity ranks the CORRECT predictions above the incorrect ones.
    `correct` selects the arm: 'class' (non-circular) or 'reactant' (the circular sanity check)."""
    hit = (lambda o: o.class_correct) if correct == "class" else (lambda o: o.reactant_rank == 1)
    return stats_kit.mann_whitney([o.nn_sim for o in outcomes if hit(o)],
                                  [o.nn_sim for o in outcomes if not hit(o)])


def mean_reactant_overlap(outcomes: list[Outcome]) -> float:
    return statistics.fmean([o.reactant_overlap for o in outcomes]) if outcomes else 0.0


# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
def _decile_curve(outcomes: list[Outcome], bins: int = 10) -> list[tuple[float, float]]:
    """(mean nn_sim, class accuracy) per equal-count similarity bin — the inflation curve."""
    ordered = sorted(outcomes, key=lambda o: o.nn_sim)
    n = len(ordered)
    out = []
    for b in range(bins):
        chunk = ordered[b * n // bins:(b + 1) * n // bins]
        if chunk:
            out.append((statistics.fmean([o.nn_sim for o in chunk]), class_accuracy(chunk)))
    return out


def _headline(split: Split, tau: float) -> dict:
    a = audit_split(split, tau=tau)
    full_cls = class_accuracy(a.outcomes)
    clean_cls = class_accuracy(a.clean)
    return {
        "prevalence": statistics.fmean([1.0 if x else 0.0 for x in a.leaked]),
        "full_class": full_cls,
        "clean_class": clean_cls,
        "inflation": full_cls - clean_cls,
        "floor": majority_class_floor(a.outcomes),
        "clean_frac": len(a.clean) / (len(a.outcomes) or 1),
        "audit": a,
    }


def run(seeds: int = 3, tau: float = _TAU) -> None:
    print("Loading USPTO-50k\n")
    try:
        rxns = load_reactions()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    # Multi-seed headline on the random (standard-style, leaky) split.
    hs = [_headline(random_split(rxns, seed=s), tau) for s in range(seeds)]
    mean = lambda key: statistics.fmean([h[key] for h in hs])
    pj = _headline(patent_disjoint_split(rxns, seed=0), tau)
    a0 = hs[0]["audit"]                      # seed-0 detail

    print(f"=== Leakage audit — USPTO-50k, random split, {seeds} seeds (τ_neardup={tau}) ===\n")
    print("Per-contract fire rate on the standard-split test set (seed 0):")
    for name, rate in contract_fire_rates(a0).items():
        print(f"  {name:<24} {rate:6.1%}")
    print(f"  {'ANY (leaked)':<24} {mean('prevalence'):6.1%}   <- P1 (model-free leakage prevalence)")

    print("\nClass-prediction inflation (the non-circular arm; majority-class floor shown):")
    print(f"  standard split      top1 = {mean('full_class'):.1%}")
    print(f"  leakage-free part.  top1 = {mean('clean_class'):.1%}   ({mean('clean_frac'):.0%} of test survives the audit)")
    print(f"  patent-disjoint     top1 = {pj['full_class']:.1%}   (independent leakage removal)")
    print(f"  majority-class floor     = {mean('floor'):.1%}")
    print(f"  INFLATION (standard − leakage-free) = {mean('inflation'):+.1%}   <- P2")

    # §5 decomposition: split the standard→clean drop into leakage-removed vs genuine generalization.
    infl = mean("inflation")
    gen = mean("clean_class") - mean("floor")
    print("\n§5 decomposition (don't bill genuine difficulty as leakage):")
    print(f"  inflation attributable to leakage      = {infl:+.1%}")
    print(f"  genuine generalization (clean − floor) = {gen:+.1%}   "
          f"(clean still {'≫' if gen > 0.10 else '≈'} floor ⇒ the model {'generalizes' if gen > 0.10 else 'is leakage-bound'})")

    # P3: does similarity explain class correctness? (non-circular). + the reactant sanity check (circular).
    au_cls = auroc_sim_explains(a0.outcomes, "class")
    au_rct = auroc_sim_explains(a0.outcomes, "reactant")
    tagged = list(zip(a0.outcomes, a0.leaked))      # carry each outcome's leaked flag through the resample
    ci = stats_kit.bootstrap_ci(
        tagged,
        lambda s: class_accuracy([o for o, _ in s]) - class_accuracy([o for o, lk in s if not lk]),
        n_boot=1000, seed=0)
    print("\nExplanatory power — does nn-similarity predict correctness?")
    print(f"  AUROC(sim → class_correct)    = {stats_kit.fmt(au_cls)}   <- P3 (non-circular)")
    print(f"  AUROC(sim → reactant_recover) = {stats_kit.fmt(au_rct)}   (circular by design — instrument check)")
    print(f"  inflation gap 95% CI (bootstrap) = [{ci[0]:+.1%}, {ci[1]:+.1%}]")

    print("\nSimilarity-stratified class accuracy (the inflation curve, seed 0):")
    for sim, acc in _decile_curve(a0.outcomes):
        bar = "█" * round(acc * 40)
        print(f"  nn_sim≈{sim:.2f}  acc={acc:5.1%}  {bar}")

    # The bounded reactant arm.
    full_react = topk_reactant_accuracy(a0.outcomes)
    clean_react = topk_reactant_accuracy(a0.clean)
    print("\nReactant-recovery (BOUNDED — stdlib retriever, no template transfer):")
    print(f"  exact top1: standard {full_react[1]:.1%}  → leakage-free {clean_react[1]:.1%}  "
          f"(≈ the duplicate rate; a near-dup detector, not retrosim's 37%)")
    print(f"  top-1 reactant-molecule overlap: standard {mean_reactant_overlap(a0.outcomes):.1%}  "
          f"→ leakage-free {mean_reactant_overlap(a0.clean):.1%}")

    # Pre-registered verdict.
    p1 = mean("prevalence") >= 0.10
    p2 = mean("inflation") >= 0.10
    auroc_cls = au_cls.auroc if isinstance(au_cls, stats_kit.MannWhitney) else 0.5
    p3 = auroc_cls >= 0.65
    print("\n=== PRE-REGISTERED VERDICT ===")
    print(f"  P1 leakage prevalence ≥10%        : {'PASS' if p1 else 'FAIL'}  ({mean('prevalence'):.1%})")
    print(f"  P2 class inflation ≥10 pts        : {'PASS' if p2 else 'FAIL'}  ({mean('inflation'):+.1%})")
    print(f"  P3 sim explains class AUROC ≥0.65 : {'PASS' if p3 else 'FAIL'}  ({auroc_cls:.3f})")
    print("  Measured (not pre-judged): the deliverable is the legible honest-eval harness (a QC product),")
    print("  not a better retro model. Faithful reactant-level inflation needs the RDKit template arm (deferred).")
    print("  Qualification only; no broader conclusion is drawn here.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Legible leakage/honesty audit over the USPTO-50k retro eval.")
    ap.add_argument("--seeds", type=int, default=3, help="random-split seeds to average the headline over")
    ap.add_argument("--tau", type=float, default=_TAU, help="near-duplicate Jaccard threshold")
    cli = ap.parse_args()
    run(seeds=cli.seeds, tau=cli.tau)
