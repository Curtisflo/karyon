#!/usr/bin/env python3
"""Qualify docking poses with karyon's physical-validity DRC.

The composition step for a generative docking model: after diffdock-nim (or
boltz2-nim / openfold3-nim) writes ranked poses as SDF, run this to keep only the
physically valid ones and report a named reason for every rejection.

    python qualify_poses.py poses/                  # a directory of *.sdf
    python qualify_poses.py 'pose_*.sdf'            # a glob
    python qualify_poses.py poses/ --json           # machine-readable, for an agent to branch on

A pose is INVALID iff a condemning contract fires (bond/angle/ring/clash/strain,
or unparseable / no-3D); the energy-uncheckable disclosure (weight 0) never
condemns. Needs: pip install "karyon[chem]"  (rdkit); stdlib otherwise.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _collect(paths: list[str]) -> list[str]:
    """Expand files, globs, and directories into a sorted list of .sdf paths."""
    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.sdf")))
        else:
            out += sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
    return out


def qualify(files: list[str]) -> list[dict]:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")            # rdkit parse chatter is not our verdict
    from karyon import pose_validity as pv

    cs, tol = pv.validity_contracts(), pv.Tol()
    results: list[dict] = []
    for path in files:
        mols = list(Chem.SDMolSupplier(path, sanitize=True)) or [None]   # garbage file -> 1 unparseable
        for k, mol in enumerate(mols):
            label = path if len(mols) == 1 else f"{path}[{k}]"
            verdict = cs.evaluate(pv.featurize(mol, tol), tol)
            results.append({
                "pose": label,
                "valid": verdict.score == 0.0,
                "score": verdict.score,
                "fired": list(verdict.fired),
                "reasons": list(verdict.messages),
            })
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="SDF files, globs, or directories of *.sdf")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of the human summary")
    cli = ap.parse_args()

    files = _collect(cli.paths)
    if not files:
        print("no .sdf poses found", file=sys.stderr)
        return 2
    results = qualify(files)

    if cli.json:
        print(json.dumps(results, indent=2))
        return 0

    for r in results:
        if r["valid"]:
            print(f"  VALID    {r['pose']}")
        else:
            print(f"  REJECTED {r['pose']} — {'; '.join(r['reasons'])}")
    n_valid = sum(r["valid"] for r in results)
    print(f"\n{n_valid}/{len(results)} poses physically valid "
          f"({len(results) - n_valid} rejected with named reasons).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
