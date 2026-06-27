#!/usr/bin/env python3
"""Composition demo, sequence modality — core deps, no rdkit, no GPU, no network.

A generative DNA model (Evo2 / GenMol stand-in) proposes sequences; karyon's
sequence-DFM gate qualifies them for synthesis and names every rejection. Same
propose -> qualify -> act loop as demo.py, a different modality — proof the
contract pattern generalizes past cheminformatics.

    pip install karyon
    python examples/compose/seq_demo.py
"""
from __future__ import annotations

from karyon import crispr_qc

# "Generated" candidates (stand-in for Evo2/GenMol output): one clean guide, one
# carrying a Pol-III terminator (TTTT), one with a long homopolymer run.
CANDIDATES = [
    ("seq_1", "GACCTGCAGTACGTACGTAC"),   # clean
    ("seq_2", "GACCTTTTGCAGTACGTACG"),   # TTTT terminator
    ("seq_3", "GACCGGGGGGGGGGTACGTA"),   # 10-base homopolymer run
]


def main() -> int:
    print("model proposes 3 sequences; karyon qualifies them for synthesis:\n")
    survivors = []
    for name, seq in CANDIDATES:
        reasons = crispr_qc.hard_contracts(seq)
        if reasons:
            print(f"  [REJECT] {name}  {seq} — {'; '.join(reasons)}")
        else:
            print(f"  [valid ] {name}  {seq}")
            survivors.append(name)
    kept = ", ".join(survivors) or "none"
    print(f"\nagent acts — order {len(survivors)}/{len(CANDIDATES)} for synthesis: {kept}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
