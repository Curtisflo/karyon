"""audit — the legibility surface (Layer 4): an operator run's per-decision provenance + a run report.

This is the artifact that makes "legible" a *product property* rather than a slogan: the concrete wedge vs a black-box operator. `dbtl_operator` accumulates, every cycle, which
designs the DRC rejected and why, which were chosen, and which readouts qualification distrusted and
why. `audit` turns that run-state into (a) a structured, serializable `provenance` trail and (b) a
human-readable `run_report` in which a reviewer can SEE why every decision was made.

Pure rendering over `OperatorRun` — no model, no data, stdlib-only.
"""

from __future__ import annotations

from .dbtl_operator import CycleRecord, OperatorRun


def _reason_head(reason: str) -> str:
    """The contract-family head of a reason string (e.g. 'weak −35 box' from 'weak −35 box: best ...')."""
    return reason.split(":")[0].strip() if reason else "unknown"


def _tally(pairs: list[tuple[str, list[str]]]) -> dict[str, int]:
    """Roll up (item, reasons) pairs into a {reason-head: count} tally."""
    out: dict[str, int] = {}
    for _, reasons in pairs:
        key = _reason_head(reasons[0]) if reasons else "unknown"
        out[key] = out.get(key, 0) + 1
    return out


def provenance(run: OperatorRun) -> dict:
    """A structured, serializable provenance trail — the machine-readable audit record of the whole run.
    Every rejection and every distrusted readout is preserved with its reason."""
    return {
        "substrate": run.substrate,
        "gate": run.gate,
        "design_calibration": {"max_run": run.design_ctx.get("max_run"),
                               "rare_motifs": sorted(run.design_ctx.get("rare_motifs", []))},
        "final_recall": run.final_recall,
        "n_rejected": run.n_rejected,
        "n_flagged": run.n_flagged,
        "reject_reasons": run.reject_reasons(),
        "cycles": [
            {
                "cycle": c.cycle,
                "proposed": c.n_proposed,
                "rejected": [{"design": s, "reasons": r} for s, r in c.rejected],
                "chosen": list(c.chosen),
                "flagged": [{"design": s, "reasons": r} for s, r in c.flagged],
                "n_qualified": c.n_qualified,
                "n_measured": c.n_measured,
                "recall": c.recall,
                "rho": c.rho,
            }
            for c in run.cycles
        ],
    }


def _cycle_lines(c: CycleRecord, *, sample: int = 2) -> list[str]:
    rej_tally = _tally(c.rejected)
    flag_tally = _tally(c.flagged)
    rej_str = ", ".join(f"{k}×{v}" for k, v in rej_tally.items()) or "none"
    head = (f"  cycle {c.cycle}: proposed {c.n_proposed} → DRC-rejected {len(c.rejected)} ({rej_str}); "
            f"measured {len(c.chosen)} → qualified {c.n_qualified}"
            + (f", flagged {len(c.flagged)}" if c.flagged else "")
            + f"  ⟹ recall {c.recall:.1%} · ρ {c.rho:+.2f}")
    lines = [head]
    for s, reasons in c.rejected[:sample]:
        lines.append(f"        ✗ {s[:30]}… — {reasons[0] if reasons else ''}")
    for s, reasons in c.flagged[:sample]:
        lines.append(f"        ⚠ {s[:30]}… readout distrusted — {reasons[0] if reasons else ''}")
    return lines


def run_report(run: OperatorRun, *, title: str = "legible autonomous DBTL operator") -> str:
    """The human-readable run report — the legibility payoff. Shows, per cycle, what was proposed,
    rejected (and why), measured, and which readouts were distrusted (and why)."""
    cal = run.design_ctx
    L = [f"\n=== {title} — {run.substrate} (gate {'ON' if run.gate else 'OFF'}) ===",
         f"  final top-q recall {run.final_recall:.1%}  |  "
         f"DRC-rejected {run.n_rejected} designs, readout-flagged {run.n_flagged} measurements",
         f"  design DRC calibrated to the buildable reference: max homopolymer run {cal.get('max_run')}, "
         f"avoidable motifs {sorted(cal.get('rare_motifs', []))}",
         ""]
    for c in run.cycles:
        L.extend(_cycle_lines(c))
    # Rolled-up provenance — the legible 'why' of the whole run.
    rr = run.reject_reasons()
    L.append("\n  provenance summary (the auditable 'why'):")
    L.append(f"      designs rejected by the DRC: {run.n_rejected}"
             + (f"  ({', '.join(f'{k}×{v}' for k, v in rr.items())})" if rr else ""))
    L.append(f"      readouts distrusted by qualification: {run.n_flagged}")
    updated = sum(c.n_qualified for c in run.cycles)
    chosen = sum(len(c.chosen) for c in run.cycles)
    L.append(f"      model updated on {updated} trustworthy measurements (of {chosen} measured)")
    L.append("  Every rejection and distrusted readout above carries a human-readable reason — the "
             "auditable trail a black-box operator cannot produce.")
    return "\n".join(L)
