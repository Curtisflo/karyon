#!/usr/bin/env python3
"""repair_mol — the self-repair loop for molecules (reason-guided variant search).

The reference `MolRepairAgent` proposes a flawed molecule (extreme / unsynthesizable / invalid); karyon
qualifies it; the agent reads the named condemning reason and returns the most drug-like molecule from a
generated pool that PASSES the gate. This is *search*, not structural surgery — the honest deterministic
reference; a real harness (Claude Code) edits the structure directly (see README.md).

Needs rdkit:  pip install "karyon[chem]"

    python examples/agent_loop/repair_mol.py
"""

from karyon import MolRepairAgent, MolSpec, format_trajectory, repair_loop


def main() -> int:
    try:
        import rdkit  # noqa: F401
    except Exception:
        print("SKIP — molecule repair needs rdkit (pip install \"karyon[chem]\")")
        return 0

    print("agent : MolRepairAgent (deterministic reference; reads the named flaw → a passing analogue)\n")
    traj = repair_loop(MolSpec(seed=0), MolRepairAgent(), "mol")

    print(format_trajectory(traj))
    print(f"\nfinal molecule (SMILES):\n{traj.final}")
    return 0 if traj.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
