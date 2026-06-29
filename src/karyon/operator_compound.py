"""operator_compound — does qualifying the readouts COMPOUND its advantage over recursive cycles?

The legible operator (`dbtl_operator`) qualifies each readout before folding it into its surrogate. A
single round of that is "qualify protects the model" — keep corrupt labels out and the predictor stays
honest. This module asks the *loop-dynamics* question: in a recursive design-build-test-learn loop, does
the gated arm's advantage over an ingest-everything arm **grow cycle-over-cycle** (a self-reinforcing
flywheel) or merely establish itself early (front-load)?

The single variable is the readout gate: GATED qualifies readouts (`readout=pc.READOUT`, QA–QD); UNGATED
ingests everything (`readout=` an empty set). The design DRC is OFF on both arms, so qualification is the
only difference — the mirror of the no-regression anchor. `dbtl_operator.single_run` is reused verbatim;
corruption enters through the stock `RetrospectiveAssay(stress=)` hook (`noisy_assay`).

The metric is **held-out ρ on CLEAN truth** (label-unconfounded model quality). Recall is reported as a
diagnostic only — it counts a design as "found" once *measured*, so it credits the ungated arm for
corrupt-measured winners and penalizes the gate for dropping them, which is the wrong signal for model
protection.

    python -m karyon.operator_compound_honesty --seeds 8     # the pre-registered test + crossover sweep
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from . import dbtl_operator as dop
from . import loop as lp
from . import noisy_assay as na
from . import promoter_contracts as pc
from .contracts import ContractSet
from .loop import LoopConfig

# The ungated readout gate: an empty ContractSet evaluates ok=True, so every non-None readout — corrupt
# included — is ingested. (single_run still drops a None value, i.e. a true no-data dropout; our corruption
# sets a wrong VALUE, not None, so it flows into the ungated surrogate.)
_NO_READOUT = ContractSet("readout-noop")

# idx into run.curve tuples (n_measured, recall, best_n, rho).
RECALL, RHO = 1, 3


def compound_config(**over) -> LoopConfig:
    """Generous budget so recall has room to diverge; greedy construct is essential — it is the channel
    through which a poisoned model picks worse designs."""
    base = dict(seeds=1, seed_size=64, cycles=10, batch=48, propose_m=200,
                beta=1.0, top_q=0.05, test_frac=0.20, lam=1.0, best_n=20)
    base.update(over)
    return LoopConfig(**base)


def _run(sub: lp.Substrate, cfg: LoopConfig, seed: int, assay, *, gated: bool) -> dop.OperatorRun:
    return dop.single_run(
        sub, cfg, seed, gate=False,                       # design DRC OFF — readout gate is the only variable
        assay=assay,
        readout=(pc.READOUT if gated else _NO_READOUT),
        readout_ctx=(pc.readout_ctx() if gated else {}),
        construct="greedy", choose="ucb",
    )


@dataclass
class PairResult:
    mode: str
    rate: float
    seed: int
    gated: dop.OperatorRun
    plain: dop.OperatorRun

    def gaps(self, idx: int) -> list[float]:
        """Per-cycle gated−ungated gap on metric `idx`, over the shared cycle range (gap[0] ≈ 0: both arms
        start from the same clean seed, so a non-zero LATER gap is the gate's accumulated effect)."""
        g, p = self.gated.curve, self.plain.curve
        return [g[k][idx] - p[k][idx] for k in range(min(len(g), len(p)))]

    def n_flagged(self) -> int:
        return self.gated.n_flagged


def run_pair(sub: lp.Substrate, cfg: LoopConfig, seed: int, mode: str, rate: float,
             *, sens: float = 0.75, spec: float = 0.05) -> PairResult:
    """One gated-vs-ungated pair under the same (deterministic) corruption — fair by construction."""
    gated = _run(sub, cfg, seed, na.make_assay(sub.truth, mode, rate, seed, sens=sens, spec=spec), gated=True)
    plain = _run(sub, cfg, seed, na.make_assay(sub.truth, mode, rate, seed, sens=sens, spec=spec), gated=False)
    return PairResult(mode, rate, seed, gated, plain)


# --------------------------------------------------------------------------- #
# Compounding metrics — a positive slope of the gated−ungated gap over cycles is the compounding signature
# (vs a flat/decaying gap = front-loading). We report both the least-squares slope and the first/last gap.
# --------------------------------------------------------------------------- #
def slope(ys: list[float]) -> float:
    """Least-squares slope of ys against x = 0..n−1 (per-cycle gap growth). 0 if < 2 points or degenerate."""
    n = len(ys)
    if n < 2:
        return 0.0
    xbar = (n - 1) / 2.0
    ybar = sum(ys) / n
    num = sum((i - xbar) * (y - ybar) for i, y in enumerate(ys))
    den = sum((i - xbar) ** 2 for i in range(n))
    return num / den if den else 0.0


def first_last(gaps: list[float]) -> tuple[float, float]:
    """(early, final) gap — early = gap after cycle 1 (first post-seed round), final = last cycle."""
    return (gaps[1] if len(gaps) > 1 else 0.0), (gaps[-1] if gaps else 0.0)


def rho_trajectory(run: dop.OperatorRun) -> tuple[float, float]:
    """(ρ after cycle 1, ρ at the final cycle) on held-out CLEAN truth — the model-protection read."""
    c = run.curve
    return (c[1][RHO] if len(c) > 1 else c[0][RHO]), c[-1][RHO]


def _print_pair(pr: PairResult) -> None:
    gr, gh = pr.gaps(RECALL), pr.gaps(RHO)
    rf, rl = first_last(gr)
    g_r1, g_rl = rho_trajectory(pr.gated)
    p_r1, p_rl = rho_trajectory(pr.plain)
    print(f"  [{pr.mode:>10} r={pr.rate:.0%} seed {pr.seed}]  "
          f"recall gated {pr.gated.final_recall:.1%} vs ungated {pr.plain.final_recall:.1%}  "
          f"| gap recall first {rf:+.1%} → last {rl:+.1%} (slope {slope(gr):+.3%}/cyc)  "
          f"| gap ρ slope {slope(gh):+.3f}/cyc")
    print(f"               held-out ρ: gated {g_r1:+.2f}→{g_rl:+.2f}  ungated {p_r1:+.2f}→{p_rl:+.2f}  "
          f"(gate flagged {pr.n_flagged()} readouts)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Does qualifying readouts COMPOUND over recursive cycles?")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rate", type=float, default=0.60)
    ap.add_argument("--modes", default="random,saturation,shuffle,clean")
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()

    try:
        sub = lp.promoter_substrate()
    except lp.pmd.DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    cfg = compound_config(cycles=args.cycles, batch=args.batch)
    print(f"=== operator-compound: qualify over recursive cycles | {sub.name} | "
          f"budget {cfg.budget} ({cfg.cycles}×{cfg.batch}) ===")
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"\n-- mode={mode} (rate {args.rate:.0%}) --")
        for s in range(args.seeds):
            _print_pair(run_pair(sub, cfg, s, mode, args.rate))


if __name__ == "__main__":
    main()
