"""dbtl_operator — the connected legible autonomous DBTL operator (Layer 3, the centerpiece).

This is the build the design notes §6 specifies: the two previously-disconnected spines wired into ONE
loop behind the `Assay` boundary, with the legible contract engine gating both ends —

    CONSTRUCT (rank the pool) → DRC-GATE (reject un-buildable designs, with reasons) → CHOOSE
    → ASSAY.emit_order/ingest → QUALIFY (trust only well-powered readouts, with reasons)
    → UPDATE on trustworthy only → recurse

— accumulating a per-decision provenance trail (consumed by `audit.py`). `loop.py` proved the cores
*integrate*; this proves the *legible operator* exists: every rejected design and every distrusted
readout carries a human-readable reason, and the run emits an auditable report. The differentiator vs a
black-box operator is exactly that legibility.

Reuse, not reinvention: the cores (`linmodel`/`acquisition`/`constructive_core`), the `Substrate` seam
and metric helpers (`loop._topset/_recall/_best_n_mean/_rho_on`, replicated setup), the contract engine
(`promoter_contracts.DESIGN/READOUT`), and the `assay` boundary. The genuinely new logic is the two
contract gates + the assay indirection + the run-state/provenance.

NO-REGRESSION ANCHOR (test_operator): with `gate=False` on the σ70 promoter pool — where every proposal
is measurable and a clean `RetrospectiveAssay` returns trustworthy readouts — the operator reduces
EXACTLY to `loop.py`'s L strategy (greedy construct + UCB choose), so its curve matches `loop`'s,
proving the gate/assay wrapper changed nothing in the cores.

(Named `dbtl_operator`, not `operator`, because the stdlib `operator` module is pre-imported at startup
and shadows a local `operator.py` — the design notes §6 calls it `operator.py`; this is that file.)

    cd karyon/probe && python dbtl_operator.py --seeds 1
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

from . import acquisition as acq
from . import constructive_core as cc
from . import linmodel as lm
from . import loop as lp
from . import promoter_contracts as pc
from .assay import Assay, Readout, RetrospectiveAssay
from .contracts import ContractSet
from .loop import SEED_BASE, LoopConfig


# --------------------------------------------------------------------------- #
# Run-state / provenance — every decision a cycle made, for the audit trail.
# --------------------------------------------------------------------------- #
@dataclass
class CycleRecord:
    cycle: int
    n_proposed: int
    rejected: list[tuple[str, list[str]]]      # (design, reasons) the DRC-gate removed
    chosen: list[str]                          # the batch CHOOSE committed to measure
    flagged: list[tuple[str, list[str]]]       # readouts QUALIFY distrusted (not used to update)
    n_qualified: int                           # trustworthy readouts folded into the model
    n_measured: int                            # cumulative trustworthy measurements
    recall: float                              # top-q recall after this cycle
    best_n: float
    rho: float                                 # predictor held-out ρ (context)


@dataclass
class OperatorRun:
    substrate: str
    gate: bool
    cycles: list[CycleRecord] = field(default_factory=list)
    curve: list[tuple] = field(default_factory=list)   # (n, recall, best_n, rho) — comparable to loop's
    measured: set[str] = field(default_factory=set)
    design_ctx: dict = field(default_factory=dict)

    @property
    def final_recall(self) -> float:
        return self.curve[-1][1] if self.curve else 0.0

    @property
    def n_rejected(self) -> int:
        return sum(len(c.rejected) for c in self.cycles)

    @property
    def n_flagged(self) -> int:
        return sum(len(c.flagged) for c in self.cycles)

    def reject_reasons(self) -> dict[str, int]:
        """Tally of which design contract did the rejecting (the legible 'why' rolled up)."""
        tally: dict[str, int] = {}
        for c in self.cycles:
            for _, reasons in c.rejected:
                # the first reason names the firing contract family; count the whole reason string head
                key = reasons[0].split(":")[0] if reasons else "unknown"
                tally[key] = tally.get(key, 0) + 1
        return tally


def _setup(sub: lp.Substrate, seed: int, cfg: LoopConfig):
    """Replicates loop._run_seed's setup EXACTLY (same shuffles/seeding) so plain mode is comparable to
    loop's L: held-out test split, design space, measure-truth, winners, and the seed set."""
    rng = random.Random(SEED_BASE + seed)
    acquirable = [s for s in sub.feasible_pool if s in sub.truth]
    rng.shuffle(acquirable)
    n_test = int(len(acquirable) * cfg.test_frac)
    test_seqs = acquirable[:n_test]
    test_set = set(test_seqs)
    design_space = [s for s in sub.feasible_pool if s not in test_set]
    measure_truth = {s: sub.truth[s] for s in design_space if s in sub.truth}
    topset = lp._topset(measure_truth, cfg.top_q)
    seed_pool = list(measure_truth.keys())
    rng.shuffle(seed_pool)
    seed_set = seed_pool[: cfg.seed_size]
    return design_space, measure_truth, test_seqs, topset, seed_set


def single_run(sub: lp.Substrate, cfg: LoopConfig, seed: int = 0, *, gate: bool = True,
               assay: Assay | None = None, design: ContractSet = pc.DESIGN,
               readout: ContractSet = pc.READOUT, readout_ctx: dict | None = None,
               construct: str = "greedy", choose: str = "ucb") -> OperatorRun:
    """One closed operator run. `gate=True` enforces the design DRC + readout qualification (the legible
    operator); `gate=False` is the plain DBTL baseline (the no-regression anchor vs loop's L)."""
    feat = sub.featurize
    design_ctx = pc.calibrate_design(sub.feasible_pool)       # calibrate C5/C6 to the buildable reference
    rctx = readout_ctx if readout_ctx is not None else pc.readout_ctx()
    assay = assay if assay is not None else RetrospectiveAssay(sub.truth)
    design_space, measure_truth, test_seqs, topset, seed_set = _setup(sub, seed, cfg)

    model = lm.BayesRidge(len(feat(seed_set[0])), lam=cfg.lam)
    model.observe_all([feat(s) for s in seed_set], [measure_truth[s] for s in seed_set])
    measured = set(seed_set)
    curve = [(len(measured), lp._recall(measured, topset),
              lp._best_n_mean(measured, sub.truth, cfg.best_n), lp._rho_on(model, feat, test_seqs, sub.truth))]
    run = OperatorRun(sub.name, gate, curve=curve, measured=measured, design_ctx=design_ctx)
    rng = random.Random(SEED_BASE + seed + 1)                 # matches loop's L (off=1)

    for c in range(cfg.cycles):
        pool = [s for s in design_space if s not in measured]
        if not pool:
            break
        # CONSTRUCT — rank the unmeasured design space (greedy = construct-core argmax; random = baseline).
        if construct == "random":
            rng.shuffle(pool)
            ranked = pool
        else:
            ranked = cc.gen_constructive_exhaustive(pool, model, len(pool),
                                                    featurize=feat, is_feasible=sub.is_feasible)
        proposals = ranked[: cfg.propose_m]

        # DRC-GATE — reject un-buildable designs BEFORE spending a measurement, each with a reason.
        rejected: list[tuple[str, list[str]]] = []
        if gate:
            passing = []
            for s in proposals:
                v = design.evaluate(s, design_ctx)
                (passing.append(s) if v.ok else rejected.append((s, v.messages)))
        else:
            passing = list(proposals)
        if not passing:
            run.cycles.append(CycleRecord(c + 1, len(proposals), rejected, [], [], 0,
                                          len(measured), *curve[-1][1:]))
            continue

        # CHOOSE — acquire the batch from the DRC-approved set.
        Xp = [feat(s) for s in passing]
        order = acq.acquire(model, Xp, list(range(len(passing))), choose, len(passing),
                            rng=rng, beta=cfg.beta)
        chosen = [passing[i] for i in order[: cfg.batch]]

        # ASSAY — emit the order, ingest the readouts (retrospective lookup or wet files).
        readouts = assay.ingest(assay.emit_order(chosen, cycle=c + 1))

        # QUALIFY — trust only well-powered readouts; UPDATE the model on those alone.
        flagged: list[tuple[str, list[str]]] = []
        upd_x, upd_y, qualified = [], [], []
        for r in readouts:
            q = readout.evaluate(r, rctx)
            if q.ok and r.value is not None:
                upd_x.append(feat(r.design))
                upd_y.append(r.value)
                qualified.append(r.design)
            else:
                flagged.append((r.design, q.messages or ["no data (build dropout)"]))
        if upd_x:
            model.observe_all(upd_x, upd_y)
            measured.update(qualified)

        curve.append((len(measured), lp._recall(measured, topset),
                      lp._best_n_mean(measured, sub.truth, cfg.best_n),
                      lp._rho_on(model, feat, test_seqs, sub.truth)))
        run.cycles.append(CycleRecord(c + 1, len(proposals), rejected, chosen, flagged,
                                      len(qualified), len(measured), *curve[-1][1:]))
    run.curve = curve
    run.measured = measured
    return run


def run(cfg: LoopConfig | None = None, seeds: int = 1, refresh: bool = False) -> dict:
    """Run the legible operator on the σ70 promoter substrate and print a short headline (the full
    legible report is audit.run_report)."""
    cfg = cfg or LoopConfig()
    sub = lp.promoter_substrate(refresh=refresh)
    runs = [single_run(sub, cfg, s, gate=True) for s in range(seeds)]
    print(f"\n=== operator: legible autonomous DBTL on {sub.name} (gate ON, {seeds} seed[s]) ===")
    for i, r in enumerate(runs):
        print(f"  seed {i}: final top-{cfg.top_q:.0%} recall {r.final_recall:.1%}; "
              f"DRC-rejected {r.n_rejected} designs, readout-flagged {r.n_flagged}; "
              f"reject reasons {r.reject_reasons()}")
    return {"runs": runs, "cfg": cfg, "substrate": sub.name}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Legible autonomous DBTL operator (σ70 promoter).")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    try:
        run(seeds=args.seeds, refresh=args.refresh)
    except lp.pmd.DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
