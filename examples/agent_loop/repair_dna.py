#!/usr/bin/env python3
"""repair_dna — watch the agent self-repair loop converge on a DNA design spec.

The reference `DnaRepairAgent` makes a rough first draft; karyon qualifies it; the agent reads each NAMED
reason (GC band, homopolymer, hairpin, restriction site) and makes the corresponding surgical edit, until
the gate passes. Pure stdlib — no rdkit, no network, no API key.

This is the loop made runnable as a *reference*. In real use the agent is your harness (e.g. Claude Code):
see README.md in this directory.

    python examples/agent_loop/repair_dna.py
"""

from karyon import DnaRepairAgent, DnaSpec, format_trajectory, repair_loop


def main() -> int:
    spec = DnaSpec(length=240, gc_target=0.50, avoid_enzymes=("EcoRI", "BsaI"), seed=1)
    print("spec  : a ~240 bp synthesizable insert — GC in band, free of EcoRI/BsaI sites and strong hairpins")
    print("agent : DnaRepairAgent (deterministic reference; maps each named reason → a targeted edit)\n")

    traj = repair_loop(spec, DnaRepairAgent(), "dna", clear_disclosures=("RESTRICTION_SITE",))

    print(format_trajectory(traj))
    print(f"\nfinal sequence ({len(traj.final)} nt):\n{traj.final}")
    return 0 if traj.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
