#!/usr/bin/env python3
"""repair_antibody — watch the agent self-repair loop fix antibody developability liabilities.

The reference `AntibodyRepairAgent` starts from a trastuzumab Fv with planted liabilities (an unpaired
cysteine + an N-glycosylation sequon in CDR-H3, plus chemistry hotspots); karyon qualifies it; the agent reads
each NAMED reason and applies the textbook residue-class-preserving developability fix — Cys→Ser, break the
sequon, Asn→Gln (deamidation), Asp→Glu (isomerization) — until the gate passes. Pure stdlib — no rdkit, no
network, no API key.

This is the loop made runnable as a *reference*. In real use the agent is your harness (e.g. Claude Code):
the model designs an Fv, runs `karyon qualify --modality antibody`, reads the reasons, edits, re-runs.

    python examples/agent_loop/repair_antibody.py
"""

from karyon import AntibodyRepairAgent, AntibodySpec, format_trajectory, qualify, repair_loop


def main() -> int:
    agent = AntibodyRepairAgent()
    print("spec  : a developable Fv — no unpaired cysteine, no CDR N-glyc sequon, chemistry hotspots cleared")
    print("agent : AntibodyRepairAgent (deterministic reference; each named reason → a conservative fix)\n")

    traj = repair_loop(AntibodySpec(), agent, "antibody",
                       clear_disclosures=AntibodyRepairAgent.DEMANDED, max_rounds=12)

    print(format_trajectory(traj))
    heavy = traj.final.split(":", 1)[0]
    v = qualify(traj.final, modality="antibody").items[0][1]
    print(f"\nfinal heavy chain ({len(heavy)} aa) — gate {'PASS' if v.ok else 'FAIL'}:\n{heavy}")
    return 0 if traj.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
