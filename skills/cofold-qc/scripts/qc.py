#!/usr/bin/env python3
"""qc.py — the cofold-QC gate: a deterministic physical-validity verdict over a co-folding pose.

Input a co-folding model's output (protein + ligand in ONE frame; PDB or mmCIF, e.g. a Boltz-2 prediction)
and get a PASS/FAIL verdict with a legible reason per fired contract. The INTERmolecular DRC (ligand↔
protein clash / volume overlap / out-of-pocket) runs from coordinates alone — the owned contribution and
the headline gate. The INTRAmolecular ligand DRC (bond/angle/ring/strain) needs bond orders, so it runs
when a ligand SDF is supplied (or the structure is itself an SDF) and is otherwise skipped-with-disclosure.

    python qc.py --structure complex.cif                       # inter gate from a co-folding output
    python qc.py --structure complex.cif --ligand ligand.sdf   # + the full intramolecular ligand DRC
    python qc.py --structure complex.pdb --ligand-resname LIG --json

Exit code is non-zero on FAIL so it gates a pipeline directly. Reuses the validated karyon DRC (the
installed `karyon` package) — this wrapper only does I/O + presentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from karyon import structure_io as sio


def _fmt_of(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {"cif": "cif", "mmcif": "cif", "pdb": "pdb", "pdbqt": "pdbqt", "ent": "pdb"}.get(ext, "pdb")


def _ligand_mol_from_sdf(path: str):
    from karyon.cofold_data import ligand_mol
    return ligand_mol(Path(path).read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description="cofold-QC: physical-validity gate for a co-folding pose.")
    ap.add_argument("--structure", required=True, help="co-folding output (PDB/mmCIF), protein+ligand one frame")
    ap.add_argument("--ligand", help="ligand SDF (bond orders) → enables the intramolecular DRC")
    ap.add_argument("--ligand-resname", help="explicit ligand residue id (else auto: HETATM / non-standard)")
    ap.add_argument("--json", action="store_true", help="emit the verdict as JSON")
    cli = ap.parse_args()

    try:
        from karyon import cofold_validity as cv
    except Exception as e:                                       # noqa: BLE001
        print(f"ERROR — cofold_validity import failed (need numpy): {e}", file=sys.stderr)
        return 2

    text = Path(cli.structure).read_text()
    atoms = sio.read_atoms(text, fmt=_fmt_of(cli.structure))
    protein, ligand_atoms = sio.split_protein_ligand(atoms, ligand_resname=cli.ligand_resname)

    # the ligand for the intermolecular geometry: prefer the SDF (exact), else the atoms split from the frame
    lig_mol = _ligand_mol_from_sdf(cli.ligand) if cli.ligand else None
    if lig_mol is not None:
        inter_ligand = cv.ligand_atoms_from_mol(lig_mol)
    else:
        inter_ligand = ligand_atoms

    if not protein or not inter_ligand:
        msg = (f"could not resolve a protein+ligand pair in one frame "
               f"({len(protein)} protein / {len(inter_ligand)} ligand atoms)")
        print(json.dumps({"verdict": "ERROR", "reason": msg}) if cli.json else f"ERROR — {msg}",
              file=sys.stderr)
        return 2

    tol = cv.InterTol()
    inter_cs = cv.intermolecular_contracts()
    fx = cv.interface_features(protein, inter_ligand, tol)
    inter_v = inter_cs.evaluate(fx, tol)

    intra_v = None
    if lig_mol is not None:
        from karyon import pose_validity as pv
        fi = pv.featurize(lig_mol, pv.Tol())
        intra_v = pv.validity_contracts().evaluate(fi, pv.Tol())

    inter_bad = inter_v.score > 0.0
    intra_bad = bool(intra_v) and intra_v.score > 0.0
    ok = not (inter_bad or intra_bad)

    reasons = ([{"axis": "intermolecular", "contract": r.contract, "why": r.message} for r in inter_v.reasons]
               + ([{"axis": "intramolecular", "contract": r.contract, "why": r.message}
                   for r in intra_v.reasons] if intra_v else []))
    verdict = {
        "verdict": "PASS" if ok else "FAIL",
        "structure": cli.structure,
        "n_protein_atoms": fx.n_protein_atoms,
        "n_ligand_atoms": fx.n_ligand_atoms,
        "intermolecular": {
            "min_distance_A": round(fx.min_lig_prot_A, 3),
            "min_distance_vdw_frac": round(fx.min_lig_prot_rel, 3),
            "volume_overlap_frac": round(fx.vol_overlap_frac, 3),
            "n_clash_pairs": fx.n_clash_pairs,
            "severity": round(fx.severity(tol), 3),
        },
        "intramolecular_checked": intra_v is not None,
        "reasons": reasons,
    }

    if cli.json:
        print(json.dumps(verdict, indent=2))
    else:
        print(f"{verdict['verdict']}  —  {cli.structure}")
        print(f"  protein {fx.n_protein_atoms} atoms · ligand {fx.n_ligand_atoms} atoms · "
              f"closest approach {fx.min_lig_prot_A:.2f} Å ({fx.min_lig_prot_rel:.0%} vdW) · "
              f"volume overlap {fx.vol_overlap_frac:.0%}")
        if not verdict["intramolecular_checked"]:
            print("  (intramolecular ligand DRC skipped — pass --ligand <sdf> with bond orders to enable)")
        for r in reasons:
            print(f"  ✗ [{r['axis']}] {r['contract']}: {r['why']}")
        if ok:
            print("  ✓ physically valid (no contract fired)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
