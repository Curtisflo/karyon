#!/usr/bin/env python3
"""Synthesize three ranked candidate poses — a stand-in for a generative docking
model's output (DiffDock / Boltz-2 NIM) so the composition demo runs with no GPU,
no NIM, and no network.

We take one drug-like ligand, build a physically valid conformer, and plant two
defects with karyon's *own* decoy generators (the same ones its pose instrument
check uses). The point the demo makes: the model's TOP-confidence pick is the
broken one — physical validity and model confidence are different axes.

    pose_1.sdf  (model rank 1, conf 0.61)  steric clash    <- invalid
    pose_2.sdf  (model rank 2, conf 0.48)  stretched bond  <- invalid
    pose_3.sdf  (model rank 3, conf 0.31)  clean           <- valid

Needs: pip install "karyon[chem]"
"""
from __future__ import annotations

import os

from rdkit import Chem

from karyon import pose_validity as pv

# Ibuprofen — a small, unambiguous drug-like ligand that embeds reliably and has
# both an aromatic ring and terminal bonds for the decoy generators to perturb.
SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidates")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    clean = pv.clean_conformer(SMILES, seed=7)
    if clean is None:
        raise SystemExit("could not embed a conformer (rdkit/ETKDG failed)")

    candidates = [
        ("pose_1.sdf", "model rank 1 (conf 0.61)", pv.decoy_clash(clean, seed=1)),
        ("pose_2.sdf", "model rank 2 (conf 0.48)", pv.decoy_stretch(clean, seed=2)),
        ("pose_3.sdf", "model rank 3 (conf 0.31)", clean),
    ]
    for fname, note, mol in candidates:
        path = os.path.join(OUT, fname)
        mol.SetProp("_Name", note)
        writer = Chem.SDWriter(path)
        writer.write(mol)
        writer.close()
        print(f"   wrote {os.path.relpath(path, os.getcwd())}  ({note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
