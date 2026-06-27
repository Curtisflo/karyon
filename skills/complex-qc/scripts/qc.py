#!/usr/bin/env python3
"""qc.py — the complex-QC gate: a deterministic interface-validity verdict over a protein COMPLEX.

Input a predicted or designed protein-protein complex (two or more chains in ONE frame; PDB or mmCIF — e.g.
an AlphaFold-Multimer model, or an RFdiffusion + ProteinMPNN designed binder against its target) and get a
PASS/FAIL verdict with a legible reason per fired contract. The interface DRC (inter-chain clash / gross
interpenetration / out-of-contact) runs from coordinates alone — the owned contribution.

    python qc.py --structure complex.pdb                       # auto: the two largest chains
    python qc.py --structure complex.cif --chain-a A --chain-b B
    python qc.py --structure binder.pdb --chain-a A --chain-b B,C --json   # binder=A vs target=B+C

The verdict separates DISCLOSURE (every inter-chain clash is reported — the detection signal) from
CONDEMNATION (only physically-unphysical interpenetration / out-of-contact FAILS the structure): a deposited,
valid complex commonly carries a few shallow interface clashes, so they inform without failing it. Exit code
is non-zero on FAIL so it gates a pipeline directly. Reuses the validated karyon DRC (the installed `karyon`
package).
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


def _chain_set(arg: str | None) -> list | None:
    return [c.strip() for c in arg.split(",") if c.strip()] if arg else None


def main() -> int:
    ap = argparse.ArgumentParser(description="complex-QC: interface-validity gate for a protein complex.")
    ap.add_argument("--structure", required=True, help="complex (PDB/mmCIF), two or more chains in one frame")
    ap.add_argument("--chain-a", help="chain id(s) for partner A (comma-separated; default: largest chain)")
    ap.add_argument("--chain-b", help="chain id(s) for partner B (comma-separated; default: 2nd largest)")
    ap.add_argument("--json", action="store_true", help="emit the verdict as JSON")
    cli = ap.parse_args()

    try:
        from karyon import protein_interface_validity as piv
    except Exception as e:                                       # noqa: BLE001
        print(f"ERROR — protein_interface_validity import failed (need numpy): {e}", file=sys.stderr)
        return 2

    atoms = sio.read_atoms(Path(cli.structure).read_text(), fmt=_fmt_of(cli.structure))
    ca, cb = _chain_set(cli.chain_a), _chain_set(cli.chain_b)
    if ca and cb:
        group_a, group_b = sio.split_by_chain(atoms, ca, cb)
        a_label, b_label = ",".join(ca), ",".join(cb)
    else:
        groups = sio.group_by_chain(atoms)
        if len(groups) < 2:
            msg = f"need ≥2 chains for an interface; found {sorted(groups)}"
            print(json.dumps({"verdict": "ERROR", "reason": msg}) if cli.json else f"ERROR — {msg}",
                  file=sys.stderr)
            return 2
        ranked = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
        (a_label, group_a), (b_label, group_b) = ranked[0], ranked[1]

    if not group_a or not group_b:
        msg = f"could not resolve two chain groups (A={a_label!r}:{len(group_a)} / B={b_label!r}:{len(group_b)})"
        print(json.dumps({"verdict": "ERROR", "reason": msg}) if cli.json else f"ERROR — {msg}", file=sys.stderr)
        return 2

    tol = piv.IfaceTol()
    cs = piv.protein_interface_contracts()
    fx = piv.interface_features(group_a, group_b, tol)
    v = cs.evaluate(fx, tol)
    ok = not piv.is_interface_invalid(fx, cs, tol)

    reasons = [{"contract": r.contract, "why": r.message, "condemns": r.weight > 0.0} for r in v.reasons]
    verdict = {
        "verdict": "PASS" if ok else "FAIL",
        "structure": cli.structure,
        "chain_a": a_label, "chain_b": b_label,
        "n_a_atoms": fx.n_a_atoms, "n_b_atoms": fx.n_b_atoms,
        "interface": {
            "min_distance_A": round(fx.min_ab_A, 3),
            "n_inter_chain_clashes": fx.n_clash_pairs,
            "deepest_overlap_A": round(fx.max_overlap_A, 3),
            "burial_frac": round(fx.vol_overlap_frac, 3),
            "severity": round(fx.severity(tol), 3),
        },
        "reasons": reasons,
    }

    if cli.json:
        print(json.dumps(verdict, indent=2))
    else:
        print(f"{verdict['verdict']}  —  {cli.structure}  (chain {a_label} vs {b_label})")
        print(f"  A {fx.n_a_atoms} atoms · B {fx.n_b_atoms} atoms · closest approach {fx.min_ab_A:.2f} Å · "
              f"{fx.n_clash_pairs} inter-chain clashes (deepest {fx.max_overlap_A:.2f} Å)")
        for r in reasons:
            mark = "✗" if r["condemns"] else "·"
            print(f"  {mark} {r['contract']}: {r['why']}")
        if ok:
            print("  ✓ physically valid interface (no condemning contract fired)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
