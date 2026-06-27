#!/usr/bin/env python3
"""qc.py — the mol-QC gate: a deterministic validity/synthesizability verdict over generated molecules.

Input a SMILES (or a .smi file of them, e.g. NVIDIA GenMol / MolMIM output) and get a PASS/FAIL verdict with
a legible reason per fired contract. Condemning: invalid molecule, unsynthesizable (Ertl SA), extreme
property (MW/cLogP). Advisory disclosures: structural alerts (PAINS/Brenk), Lipinski Ro5, Veber.

    python qc.py --smiles "CC(=O)Oc1ccccc1C(=O)O"        # one generated molecule
    python qc.py --smi-file generated.smi                 # a batch (one SMILES per line; optional name)
    python qc.py --smi-file generated.smi --json          # machine-readable verdict

Exit code is non-zero on FAIL so it gates a pipeline directly (0 PASS / 1 FAIL / 2 input-or-setup error).
Reuses the validated karyon DRC (`karyon.mol_qc`) — this wrapper only does I/O + presentation. Requires
rdkit (the cheminformatics engine the gate composes over).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_smi(text: str) -> list[tuple[str, str]]:
    """(name, smiles) from a .smi file: one record per line, 'SMILES [name]'; blank/`#` lines skipped."""
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        smiles = parts[0]
        name = parts[1] if len(parts) > 1 else f"mol{len(out)}"
        out.append((name, smiles))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="mol-QC: validity/synthesizability gate for generated molecules.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", help="a single molecule SMILES")
    src.add_argument("--smi-file", help="a .smi file of generated molecules (one SMILES per line)")
    ap.add_argument("--json", action="store_true", help="emit the verdict as JSON")
    cli = ap.parse_args()

    try:
        from karyon import mol_qc as mq
    except Exception as e:                                       # noqa: BLE001
        print(f"ERROR — mol_qc import failed: {e}", file=sys.stderr)
        return 2
    if not mq._HAVE_RDKIT:
        print("ERROR — mol-qc requires rdkit (the cheminformatics engine the gate composes over)",
              file=sys.stderr)
        return 2

    if cli.smiles:
        records = [("input", cli.smiles.strip())]
    else:
        try:
            records = _read_smi(Path(cli.smi_file).read_text())
        except OSError as e:
            print(f"ERROR — cannot read {cli.smi_file}: {e}", file=sys.stderr)
            return 2
    if not records:
        print("ERROR — no molecules found in the input", file=sys.stderr)
        return 2

    tol = mq.MolTol()
    cs = mq.mol_contracts()

    per = []
    any_fail = False
    for name, smi in records:
        f = mq.featurize(smi, tol)
        v = cs.evaluate(f, tol)
        any_fail = any_fail or v.score > 0.0
        per.append({
            "name": name, "smiles": smi, "parsed": f.parsed,
            "mw": round(f.mw, 1), "logp": round(f.logp, 2), "sa": round(f.sa, 2), "qed": round(f.qed, 2),
            "fail": v.score > 0.0,
            "reasons": [{"contract": r.contract, "why": r.message, "condemns": r.weight > 0.0}
                        for r in v.reasons],
        })

    ok = not any_fail
    verdict = {"verdict": "PASS" if ok else "FAIL", "n_molecules": len(per), "molecules": per}

    if cli.json:
        print(json.dumps(verdict, indent=2))
    else:
        head = cli.smi_file if cli.smi_file else "molecule"
        print(f"{verdict['verdict']}  —  {head}  ({len(per)} molecule{'s' if len(per) != 1 else ''})")
        for r in per:
            if r["parsed"]:
                print(f"  {r['name']}  MW {r['mw']:.0f} · cLogP {r['logp']:.1f} · SA {r['sa']:.1f} · "
                      f"QED {r['qed']:.2f}" + ("" if r["reasons"] else "  ✓"))
            else:
                print(f"  {r['name']}  <invalid SMILES>")
            for x in r["reasons"]:
                print(f"      {'✗' if x['condemns'] else '·'} {x['contract']}: {x['why']}")
        if ok:
            print("  ✓ usable (no condemning contract fired)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
