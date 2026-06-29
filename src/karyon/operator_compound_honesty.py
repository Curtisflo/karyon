"""operator_compound_honesty — pre-registered test that qualifying COMPOUNDS, + the reliability-crossover sweep.

A single round of readout qualification protects the model (the operator's no-regression+qualify proofs);
this harness asks whether that protection **compounds over recursive cycles**. It puts a model-degrading
readout-corruption failure mode into the recursive loop on a headroom substrate (the σ70 promoter pool —
~8.7k designs / ~435 winners, so recall has room to diverge) and measures whether the gated−ungated
held-out-ρ gap *grows* cycle-over-cycle.

Pre-registered predictions (the corruption RATE is calibrated to the model-degrading regime; the
falsifiers — the shuffle control and the clean anchor — are NOT calibrated):

  PI-1 (compounding) — in the degrading regime (random/bulk corruption) the gated−ungated ρ-gap WIDENS over
       cycles: mean final-gap > early-gap, final > 0, mean per-cycle slope > 0.
  PI-2 (mechanism = ρ-protection) — the ungated arm's held-out ρ DECLINES across cycles (poisoned) and ends
       below the gated arm; the gated arm holds/rises.
  PI-3 (negative control + crossover) — the shuffle control (flags ⊥ corruption) collapses the compounding,
       the clean anchor is flat, AND the gate is net-negative at low corruption / net-positive at high (the
       conditional-value crossover: qualifying pays only once the tool is unreliable enough).

    python -m karyon.operator_compound_honesty --seeds 8
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field

from . import loop as lp
from .operator_compound import (RECALL, RHO, compound_config, first_last, rho_trajectory,
                                run_pair, slope)

SWEEP_RATES = [0.0, 0.30, 0.45, 0.60]      # the crossover map
DEGRADE_RATE = 0.60                         # the model-degrading regime (the ungated arm's ρ collapses)


@dataclass
class Agg:
    """Per-(mode, rate) aggregate over seeds. ρ-gap = gated − ungated held-out ρ; >0 ⇒ qualifying helps."""
    mode: str
    rate: float
    seeds: int
    slope_rho: float           # mean per-cycle ρ-gap slope (compounding signature)
    early_rho: float           # mean ρ-gap after cycle 1
    final_rho: float           # mean ρ-gap at the final cycle
    ungated_end: float         # mean ungated final held-out ρ (poisoned)
    gated_end: float           # mean gated final held-out ρ (protected)
    ungated_decline: float     # mean (ungated final ρ − cycle-1 ρ); <0 ⇒ the model degraded over cycles
    recall_gap_final: float    # mean final recall gap (diagnostic, confounded)
    flagged: float             # mean readouts the gate dropped
    slopes: list = field(default_factory=list)


def aggregate(sub: lp.Substrate, cfg, mode: str, rate: float, seeds: int) -> Agg:
    sl, er, fr, ue, ge, ud, rg, fl = ([] for _ in range(8))
    for s in range(seeds):
        pr = run_pair(sub, cfg, s, mode, rate)
        gh, gr = pr.gaps(RHO), pr.gaps(RECALL)
        sl.append(slope(gh))
        e, f = first_last(gh)
        er.append(e); fr.append(f)
        g1, gl = rho_trajectory(pr.gated)
        p1, pl = rho_trajectory(pr.plain)
        ue.append(pl); ge.append(gl); ud.append(pl - p1)
        rg.append(first_last(gr)[1]); fl.append(pr.n_flagged())
    m = statistics.mean
    return Agg(mode, rate, seeds, m(sl), m(er), m(fr), m(ue), m(ge), m(ud), m(rg), m(fl), sl)


def _fmt(a: Agg) -> str:
    return (f"  {a.mode:>10} r={a.rate:.0%} | ρ-gap slope {a.slope_rho:+.4f}/cyc  "
            f"early {a.early_rho:+.3f} → final {a.final_rho:+.3f}  | "
            f"ρ_end gated {a.gated_end:+.2f} vs ungated {a.ungated_end:+.2f} "
            f"(ungated Δ {a.ungated_decline:+.2f}) | recall-gap {a.recall_gap_final:+.1%} | drop {a.flagged:.0f}")


def run(seeds: int = 8) -> int:
    try:
        sub = lp.promoter_substrate()
    except lp.pmd.DatasetUnavailable as e:
        print(f"SKIP — {e}")
        return 0
    cfg = compound_config()
    print(f"=== operator-compound HONESTY | {sub.name} | budget {cfg.budget} "
          f"({cfg.cycles}×{cfg.batch}) | {seeds} seeds ===\n")

    print("RELIABILITY CROSSOVER — does the gate pay, and where? (ρ-gap > 0 ⇒ qualifying helps)")
    sweep = {}
    for mode in ("random", "saturation"):
        for rate in SWEEP_RATES:
            a = aggregate(sub, cfg, mode, rate, seeds)
            sweep[(mode, rate)] = a
            print(_fmt(a))
    print()

    rnd = sweep[("random", DEGRADE_RATE)]
    shf = aggregate(sub, cfg, "shuffle", DEGRADE_RATE, seeds)
    cln = sweep[("random", 0.0)]
    print("CONTROLS @ degrading rate")
    print(_fmt(rnd)); print(_fmt(shf)); print(_fmt(cln)); print()

    pi1 = (rnd.final_rho > rnd.early_rho) and (rnd.final_rho > 0) and (rnd.slope_rho > 0)
    pi2 = (rnd.ungated_end < rnd.gated_end - 0.10) and (rnd.ungated_decline < 0)
    # Directional: breaking the flag↔corruption coupling removes the FAVORABLE compounding (disjoint flags
    # in fact make it net-costly — slope/final go negative).
    shuffle_collapses = shf.slope_rho < rnd.slope_rho / 2.0 and shf.final_rho < rnd.final_rho / 2.0
    clean_flat = abs(cln.slope_rho) < 1e-6 and abs(cln.final_rho) < 1e-6
    lo = sweep[("random", 0.30)]
    crossover = (lo.gated_end - lo.ungated_end) < (rnd.gated_end - rnd.ungated_end)
    pi3 = shuffle_collapses and clean_flat and crossover

    for name, ok, detail in [
        ("PI-1 compounding (random ρ-gap widens + positive slope)", pi1,
         f"early {rnd.early_rho:+.3f} → final {rnd.final_rho:+.3f}, slope {rnd.slope_rho:+.4f}"),
        ("PI-2 mechanism (ungated ρ collapses below gated)", pi2,
         f"gated_end {rnd.gated_end:+.2f} vs ungated_end {rnd.ungated_end:+.2f}, ungated Δ {rnd.ungated_decline:+.2f}"),
        ("PI-3 controls (shuffle collapses · clean flat · crossover)", pi3,
         f"shuffle slope {shf.slope_rho:+.4f} (vs {rnd.slope_rho:+.4f}); clean final {cln.final_rho:+.3f}; "
         f"net@30% {lo.gated_end - lo.ungated_end:+.2f} < net@60% {rnd.gated_end - rnd.ungated_end:+.2f}"),
    ]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")

    n_pass = sum((pi1, pi2, pi3))
    # A single grep-able headline (the reproduce driver extracts the slope from this line).
    print(f"\nHEADLINE — qualifying COMPOUNDS: random@{DEGRADE_RATE:.0%} ρ-gap slope {rnd.slope_rho:+.4f}/cyc "
          f"(early {rnd.early_rho:+.3f} → final {rnd.final_rho:+.3f}; gated ρ {rnd.gated_end:+.2f} vs ungated "
          f"{rnd.ungated_end:+.2f}); shuffle control {shf.slope_rho:+.4f}; {n_pass}/3 PASS")
    return 0 if n_pass == 3 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-registered compounding test + crossover sweep.")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    return run(seeds=args.seeds)


if __name__ == "__main__":
    raise SystemExit(main())
