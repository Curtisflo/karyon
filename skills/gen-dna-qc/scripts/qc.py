#!/usr/bin/env python3
"""qc.py — the gen-DNA-QC gate: a deterministic synthesizability/manufacturability verdict over generated DNA.

Input a DNA sequence (or a FASTA of them, e.g. NVIDIA Evo2 output) and get a PASS/FAIL verdict with a legible
reason per fired contract. Per-sequence: GC band, homopolymer / poly-G runs, length window, strong hairpin,
restriction sites. Per-batch (multi-record FASTA): cross-hybridization across the set — the design-level
invariant no single sequence owns. The gate is pure stdlib (no numpy, no rdkit, no network).

    python qc.py --sequence ACGT...                 # one generated sequence
    python qc.py --fasta evo2_designs.fasta         # a batch (adds the cross-hyb set check)
    python qc.py --fasta evo2_designs.fasta --json  # machine-readable verdict

Exit code is non-zero on FAIL so it gates a pipeline directly (0 PASS / 1 FAIL / 2 input-or-setup error).
Needs: pip install karyon  (stdlib only). This wrapper only does I/O + presentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_fasta(text: str) -> list[tuple[str, str]]:
    """(name, sequence) records from a FASTA string (stdlib; no Biopython)."""
    out: list[tuple[str, str]] = []
    name, chunks = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(chunks)))
            header = line[1:].strip()
            name = header.split()[0] if header else f"seq{len(out)}"
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if name is not None:
        out.append((name, "".join(chunks)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="gen-DNA-QC: synthesizability gate for generated DNA.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence", help="a single DNA sequence (ACGT)")
    src.add_argument("--fasta", help="a FASTA file of generated sequences (multi-record → batch cross-hyb)")
    ap.add_argument("--json", action="store_true", help="emit the verdict as JSON")
    cli = ap.parse_args()

    try:
        from karyon import gen_dna_validity as gv
        from karyon import seq_dfm
    except Exception as e:                                       # noqa: BLE001
        print(f"ERROR — gen_dna_validity import failed: {e}", file=sys.stderr)
        return 2

    if cli.sequence:
        records = [("input", cli.sequence.strip())]
    else:
        try:
            records = _parse_fasta(Path(cli.fasta).read_text())
        except OSError as e:
            print(f"ERROR — cannot read {cli.fasta}: {e}", file=sys.stderr)
            return 2
    if not records:
        print("ERROR — no sequences found in the input", file=sys.stderr)
        return 2

    # normalize + validate input alphabet (the gate assumes ACGT, as Evo2 emits)
    named: list[tuple[str, str]] = []
    for name, seq in records:
        s = seq.upper().replace(" ", "")
        bad = set(s) - set("ACGT")
        if not s or bad:
            print(f"ERROR — {name}: sequence empty or has non-ACGT characters {sorted(bad)}", file=sys.stderr)
            return 2
        named.append((name, s))

    tol = gv.GenDNATol()
    cs = gv.dna_contracts()

    per_records = []
    any_fail = False
    for name, s in named:
        f = gv.featurize(s, tol)
        v = cs.evaluate(f, tol)
        any_fail = any_fail or v.score > 0.0
        per_records.append({
            "name": name, "length": f.length, "gc_frac": round(f.gc_frac, 3),
            "longest_run": f.max_run, "hairpin_stem": f.hairpin.stem,
            "fail": v.score > 0.0,
            "reasons": [{"contract": r.contract, "why": r.message, "condemns": r.weight > 0.0}
                        for r in v.reasons],
        })

    # batch cross-hybridization (only meaningful for >1 sequence)
    batch_reasons = []
    batch_fail = False
    if len(named) > 1:
        bf = gv.featurize_set(named, tol)
        bv = gv.set_contracts().evaluate(bf, tol)
        batch_fail = bv.score > 0.0
        batch_reasons = [{"contract": r.contract, "why": r.message, "condemns": r.weight > 0.0}
                         for r in bv.reasons]

    ok = not (any_fail or batch_fail)
    verdict = {
        "verdict": "PASS" if ok else "FAIL",
        "n_sequences": len(named),
        "sequences": per_records,
        "batch": {"cross_hyb": batch_reasons} if len(named) > 1 else None,
    }

    if cli.json:
        print(json.dumps(verdict, indent=2))
    else:
        head = cli.fasta if cli.fasta else "sequence"
        print(f"{verdict['verdict']}  —  {head}  ({len(named)} sequence{'s' if len(named) != 1 else ''})")
        for r in per_records:
            print(f"  {r['name']}  len {r['length']} · GC {r['gc_frac']:.0%} · "
                  f"longest run {r['longest_run']} · hairpin {r['hairpin_stem']} bp"
                  + ("" if r["reasons"] else "  ✓"))
            for x in r["reasons"]:
                print(f"      {'✗' if x['condemns'] else '·'} {x['contract']}: {x['why']}")
        for x in batch_reasons:
            print(f"  {'✗' if x['condemns'] else '·'} [batch] {x['contract']}: {x['why']}")
        if ok:
            print("  ✓ manufacturable (no condemning contract fired)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
