#!/usr/bin/env python3
"""Reproduce the headline numbers from the karyon README.

Each claim maps to one ``python -m karyon.<module>`` entrypoint that fetches a
public benchmark (cached under ``$KARYON_CACHE`` or ``~/.cache/karyon``), runs a
deterministic, named-reason contract audit, and prints a pre-registered verdict.
This driver runs those entrypoints, pulls the headline line out of each, and
shows it next to the README claim — the printed value is the source of truth.

Usage:
    python examples/reproduce/run.py            # run all four
    python examples/reproduce/run.py --list     # show the claims and commands
    python examples/reproduce/run.py screen-qc  # run one (by id)

Requires ``pip install "karyon[chem]"`` for the three cheminformatics checks
(pose / retro / admet); ``screen-qc`` needs only the core install. Each check
downloads its dataset on first run; re-runs read the cache. Stdlib only.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class Claim:
    id: str
    readme: str                       # the claim as stated in the README
    module: str                       # python -m karyon.<module>
    args: list[str] = field(default_factory=list)
    needs: str = "karyon[chem]"       # install needed beyond core
    source: str = ""                  # where the dataset comes from
    # ordered (label, regex) pairs; the first capture group is the headline value
    extract: list[tuple[str, str]] = field(default_factory=list)


CLAIMS: list[Claim] = [
    Claim(
        id="pose-validity",
        readme="71% of DiffDock RMSD≤2 'successes' are physically invalid",
        module="pose_honesty",
        source="Zenodo (PoseBench / Buttenschoen et al. 2024)",
        extract=[("DiffDock successes invalid", r"successes invalid\s+(\d+%)")],
    ),
    Claim(
        id="retro-leakage",
        readme="retrosynthesis top-1 inflation from template memorization (USPTO-50k)",
        module="retro_template",
        source="retrosim USPTO-50k CSV (Coley et al.)",
        extract=[("top-1 inflation (standard − leakage-free)",
                  r"MEASURED inflation.*top-1\s*=\s*([+\-]\d+\.?\d*%)")],
    ),
    Claim(
        id="admet-leakage",
        readme="ADMET random-vs-scaffold inflation: +0.105 AUROC (bbbp) / +0.100 ρ (esol)",
        module="molnet_honesty",
        source="MoleculeNet (Wu et al. 2018), DeepChem S3",
        extract=[("INFLATION (random − scaffold)", r"INFLATION \(random − scaffold\)\s*=\s*([+\-]\d+\.\d+)")],
    ),
    Claim(
        id="screen-qc",
        readme="~53% of gold-standard CRISPR silent failures flagged at ~3% false-flag (the new check)",
        module="screen_qc",
        args=["--seeds", "50"],
        needs="core (no extras)",
        source="MAGeCK leukemia demo (Wang 2014) + hart-lab CEGv2/NEGv1",
        extract=[("Q1 recall (silent failures flagged)", r"Q1 recall.*:\s*(\d+\.?\d*%)"),
                 ("Q2 false-flag rate", r"Q2 false-flag.*:\s*(\d+\.?\d*%)")],
    ),
]


def run_claim(c: Claim) -> int:
    cmd = [sys.executable, "-m", f"karyon.{c.module}", *c.args]
    print(f"\n{'=' * 78}\n[{c.id}]  {c.readme}")
    print(f"  $ {' '.join(cmd)}")
    print(f"  dataset: {c.source}   (install: {c.needs})")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if "SKIP —" in out or "SKIP -" in out:
        skip = next((ln for ln in out.splitlines() if "SKIP" in ln), "(skipped)")
        print(f"  SKIPPED — {skip.strip()}")
        return proc.returncode
    found_any = False
    for label, pattern in c.extract:
        hits = re.findall(pattern, out)
        if hits:
            found_any = True
            print(f"  reproduced → {label}: {', '.join(hits)}")
        else:
            print(f"  reproduced → {label}: (line not found — see full output below)")
    if not found_any:
        print("  --- full output ---")
        print("\n".join("  " + ln for ln in out.splitlines()[-25:]))
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", nargs="?", help="run only this claim (default: all)")
    ap.add_argument("--list", action="store_true", help="list claims and commands, run nothing")
    cli = ap.parse_args()

    if cli.list:
        for c in CLAIMS:
            print(f"{c.id:16} python -m karyon.{c.module} {' '.join(c.args)}".rstrip())
            print(f"{'':16} {c.readme}")
        return 0

    claims = CLAIMS if cli.id is None else [c for c in CLAIMS if c.id == cli.id]
    if not claims:
        print(f"unknown claim id {cli.id!r}; known: {', '.join(c.id for c in CLAIMS)}")
        return 2
    rc = 0
    for c in claims:
        rc |= run_claim(c)
    print(f"\n{'=' * 78}\nNote: numbers are reproduced live from public data; small run-to-run drift")
    print("is expected where a check subsamples or seeds. See README.md for the canonical values.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
