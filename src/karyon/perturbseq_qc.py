"""perturbseq_qc — a legible silent-failure QC layer over a real single-cell Perturb-seq screen.

The in-domain heavyweight the bulk screen-QC probe named but did not take (SCREEN_QC_RESULT.md § "natural
next probes"): the **SCEPTRE/Replogle single-cell analog.** A genome-scale Perturb-seq screen calls each
perturbation a transcriptomic **hit** or **no-phenotype** via a calibrated test (here Replogle's deposited
energy-distance p-value — the role SCEPTRE plays for single-cell screens). That call is ambiguous on the
no-phenotype side: a perturbation shows no phenotype either because the gene knockdown genuinely has no
transcriptomic consequence (a **true negative**) or because the guide **never knocked the target down** (a
**silent failure** — the null is an artifact). Bulk screens can only *infer* this; Perturb-seq **measures it
directly** via on-target knockdown — the cleanest possible silent-failure ground truth, and the reason the
in-domain single-cell substrate is the sharpest test of the screen-QC thesis.

The legible QC layer (the choane DRC spine, ported to Perturb-seq; `contracts` engine reused verbatim):
  * `WEAK_KNOCKDOWN`     — residual on-target expression above a floor (the guide failed → null untrustworthy);
  * `KNOCKDOWN_UNMEASURED` — on-target knockdown not measurable (target undetected → null unqualifiable);
  * `LOW_CELL_COUNT`    — too few cells to call a phenotype (the single-cell power floor).
Each verdict names which rule fired and why. The incumbent caller is *consumed* (deposited), not reimplemented
— karyon's owned, legible part is this qualification layer over the tool's output.

Pre-registered (mechanism-based, thresholds fixed before the verdict; mirrors the bulk Q1–Q4):
  P1  silent-failures flagged: ≥ 15% of no-phenotype targeting perturbations carry a QC flag.
  P2  precision: among clear hits (the perturbation demonstrably worked), weak-knockdown flag ≤ 20%.
  P3  non-redundancy (make-or-break): |ρ(knockdown, phenotype-p)| within the no-phenotype pile < 0.30 —
      the flag is NOT a softer restatement of the deposited significance.
  P4  decision-relevant: weak-knockdown is ≥ 2× enriched in the no-phenotype pile vs the hit pile.

stdlib + `contracts` + `stats_kit`; the substrate needs the optional h5py reader (SKIP-if-absent).

    python -m karyon.perturbseq_qc
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass, replace

from . import contracts
from . import stats_kit
from .perturbseq_data import (
    DatasetUnavailable,
    Perturbation,
    controls,
    load_perturbations,
    targeting,
)

WEAK_KD_FLOOR = 0.5      # residual target expression above which on-target knockdown is deemed failed (<50% KD)
MIN_CELLS = 25           # cells below which a perturbation is under-powered to call a phenotype
HIT_Q = 0.05             # the deposited energy-test p-value threshold for "has a transcriptomic phenotype"
STRONG_HIT_Q = 1e-3      # "the perturbation demonstrably worked" — the precision guard's clear-hit set


@dataclass(frozen=True)
class QCView:
    weak_kd_floor: float
    min_cells: int


# --------------------------------------------------------------------------- #
# The legible QC layer (reused contracts engine).
# --------------------------------------------------------------------------- #
def qc_contracts() -> contracts.ContractSet:
    cs = contracts.ContractSet("perturbseq-reliability")
    cs.add(contracts.Contract(
        "WEAK_KNOCKDOWN",
        lambda p, ctx: (f"on-target knockdown failed — {p.knockdown_resid:.0%} of {p.target} still expressed "
                        f"(> {ctx.weak_kd_floor:.0%} floor); a 'no phenotype' here is untrustworthy")
        if (p.knockdown_measured and p.knockdown_resid > ctx.weak_kd_floor) else None,
        weight=2.0))
    cs.add(contracts.Contract(
        "KNOCKDOWN_UNMEASURED",
        lambda p, ctx: f"on-target knockdown of {p.target} not measurable (target undetected) — null unqualifiable"
        if not p.knockdown_measured else None))
    cs.add(contracts.Contract(
        "LOW_CELL_COUNT",
        lambda p, ctx: f"only {p.n_cells} cells — under-powered to call a phenotype"
        if p.n_cells < ctx.min_cells else None))
    return cs


def qualify(p: Perturbation, view: QCView, cs: contracts.ContractSet | None = None) -> contracts.Verdict:
    return (cs or qc_contracts()).evaluate(p, view)


# --------------------------------------------------------------------------- #
# Analysis helpers.
# --------------------------------------------------------------------------- #
def _weak_kd(p: Perturbation, floor: float = WEAK_KD_FLOOR) -> bool:
    return p.knockdown_measured and p.knockdown_resid > floor


def _frac(items: list, pred) -> float:
    return sum(1 for x in items if pred(x)) / (len(items) or 1)


def run_one(*, perts: list[Perturbation] | None = None, view: QCView | None = None) -> dict | None:
    if perts is None:
        try:
            perts = load_perturbations()
        except DatasetUnavailable as e:
            print(f"  SKIP — {e}")
            return None
    view = view or QCView(WEAK_KD_FLOOR, MIN_CELLS)
    cs = qc_contracts()

    tgt, ctl = targeting(perts), controls(perts)
    hits = [p for p in tgt if p.energy_p < HIT_Q]
    nohit = [p for p in tgt if p.energy_p >= HIT_Q]
    strong = [p for p in tgt if p.energy_p < STRONG_HIT_Q]

    # B1 — is the deposited caller credible (calibrated on controls)?
    cal_t = _frac(tgt, lambda p: p.energy_p < HIT_Q)
    cal_c = _frac(ctl, lambda p: p.energy_p < HIT_Q)

    # Q1 — silent-failures flagged among no-phenotype targeting perturbations
    q1 = _frac(nohit, lambda p: not cs.evaluate(p, view).ok)
    weak_nohit = _frac(nohit, _weak_kd)
    weak_hit = _frac(hits, _weak_kd)
    enrich = weak_nohit / (weak_hit or 1e-9)                       # P4

    # Q2 — precision: weak-knockdown flag among demonstrably-working (clear-hit) perturbations
    q2 = _frac(strong, _weak_kd)

    # Q3 — non-redundancy within the no-phenotype pile (knockdown vs the deposited p-value)
    sub = [p for p in nohit if p.knockdown_measured]
    r = stats_kit.spearman([p.knockdown_resid for p in sub], [p.energy_p for p in sub])
    rho = r.rho if isinstance(r, stats_kit.Corr) else float("nan")

    # Q4 — the partition the significance scalar cannot make
    untrust = [p for p in nohit if _weak_kd(p)]
    trust = [p for p in nohit if p.knockdown_measured and not _weak_kd(p)]
    unmeasured = [p for p in nohit if not p.knockdown_measured]

    print(f"\n=== REPLOGLE K562-ESSENTIAL PERTURB-SEQ (single-cell screen QC) ===")
    print(f"  perturbations: {len(tgt)} targeting / {len(ctl)} controls")
    print(f"  B1 incumbent calibration (deposited energy-test): targeting {cal_t:.0%} hit vs controls {cal_c:.0%}"
          f"  ({'credible' if cal_t > 0.5 > cal_c else 'weak'})")
    print(f"  no-phenotype targeting perturbations (the silent-failure denominator): {len(nohit)}")
    print(f"\n  Q1 flagged among no-phenotype:        {q1:.1%}   (bar ≥15%)   <- P1")
    for c in cs.contracts:
        f = _frac(nohit, lambda p, n=c.name: n in cs.evaluate(p, view).fired)
        print(f"       {c.name:<22} {f:6.1%}")
    print(f"  Q2 weak-KD flag among clear hits:     {q2:.1%}   (bar ≤20%, precision)   <- P2")
    print(f"  Q3 |ρ(knockdown, energy-p)| in no-hit:{abs(rho):.3f}  (bar <0.30, non-redundancy)   <- P3")
    print(f"  Q4 weak-KD enrichment no-hit vs hit:  {enrich:.1f}×   ({weak_nohit:.1%} vs {weak_hit:.1%})   <- P4")
    print(f"     → the no-phenotype pile partitions: {len(untrust)} untrustworthy (weak-KD silent failures) · "
          f"{len(trust)} trustworthy negatives (strong-KD) · {len(unmeasured)} unmeasurable")

    examples = sorted(untrust, key=lambda p: -p.knockdown_resid)[:6]
    if examples:
        print(f"  example silent failures (essential gene, 'no phenotype', but KD failed):")
        for p in examples:
            print(f"       {p.target:<10} residual expr {p.knockdown_resid:.0%}, energy-p {p.energy_p:.2f}, {p.n_cells} cells")

    return {"cal_t": cal_t, "cal_c": cal_c, "n_nohit": len(nohit), "q1": q1, "q2": q2,
            "rho": abs(rho), "enrich": enrich, "weak_nohit": weak_nohit, "weak_hit": weak_hit,
            "n_untrust": len(untrust), "n_trust": len(trust)}


def _shuffle_knockdown(perts: list[Perturbation], seed: int) -> list[Perturbation]:
    """Permute on-target knockdown across targeting perturbations — the negative control: it must destroy the
    weak-KD ↔ no-phenotype relationship (enrichment → ~1×)."""
    tgt_idx = [i for i, p in enumerate(perts) if not p.is_control]
    folds = [perts[i].knockdown_resid for i in tgt_idx]
    random.Random(seed).shuffle(folds)
    out = list(perts)
    for j, i in enumerate(tgt_idx):
        out[i] = replace(perts[i], knockdown_resid=folds[j])
    return out


def run() -> None:
    print("Silent-failure QC over a real single-cell Perturb-seq screen (Replogle K562-essential)\n")
    try:
        perts = load_perturbations()
    except DatasetUnavailable as e:
        print(f"  SKIP — {e}")
        return
    real = run_one(perts=perts)
    shuf = run_one(perts=_shuffle_knockdown(perts, seed=0))

    p1 = real["q1"] >= 0.15
    p2 = real["q2"] <= 0.20
    p3 = real["rho"] < 0.30
    p4 = real["enrich"] >= 2.0
    print("\n=== PRE-REGISTERED VERDICT ===")
    print(f"  P1 flagged≥15%        {'PASS' if p1 else 'FAIL'} ({real['q1']:.0%})")
    print(f"  P2 precision≤20%      {'PASS' if p2 else 'FAIL'} ({real['q2']:.0%})")
    print(f"  P3 non-redundant<0.30 {'PASS' if p3 else 'FAIL'} (|ρ|={real['rho']:.3f})")
    print(f"  P4 enrichment≥2×      {'PASS' if p4 else 'FAIL'} ({real['enrich']:.1f}×)")
    print(f"\n  NEGATIVE CONTROL (knockdown shuffled): enrichment {real['enrich']:.1f}× → {shuf['enrich']:.1f}× "
          f"(must collapse toward 1× — confirms the silent-failure signal is real, not mechanical)")
    print("  Read: does the legible screen-QC layer add value in the IN-DOMAIN single-cell regime? Perturb-seq's")
    print("  direct on-target knockdown gives the ground-truth silent-failure label bulk screens lacked; the QC")
    print("  layer flags untrustworthy nulls NON-REDUNDANTLY with the deposited significance — qualification,")
    print("  not accuracy.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Silent-failure QC over the Replogle K562-essential Perturb-seq screen.")
    ap.add_argument("--weak-kd-floor", type=float, default=WEAK_KD_FLOOR)
    ap.add_argument("--min-cells", type=int, default=MIN_CELLS)
    cli = ap.parse_args()
    if cli.weak_kd_floor != WEAK_KD_FLOOR or cli.min_cells != MIN_CELLS:
        run_one(view=QCView(cli.weak_kd_floor, cli.min_cells))
    else:
        run()
