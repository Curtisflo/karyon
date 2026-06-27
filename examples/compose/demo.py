#!/usr/bin/env python3
"""End-to-end composition demo: a generative docking model proposes, karyon
qualifies, the agent acts on the named-reason verdict.

Runs with no GPU / NIM / network. The candidate poses stand in for a DiffDock /
Boltz-2 NIM emission (see make_candidates.py); karyon's pose-validity skill is
the qualifier (skills/pose-validity/scripts/qualify_poses.py — the same script an
agent installs and calls); this script plays the agent that keeps the survivors.

    pip install "karyon[chem]"
    python examples/compose/demo.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
QUALIFY = os.path.join(ROOT, "skills", "pose-validity", "scripts", "qualify_poses.py")
CANDIDATES = os.path.join(HERE, "candidates")


def main() -> int:
    # 1. MODEL PROPOSES — stand in for a DiffDock/Boltz-2 NIM writing ranked SDF poses.
    print("1. model proposes — generating 3 ranked candidate poses (DiffDock/NIM stand-in):")
    subprocess.run([sys.executable, os.path.join(HERE, "make_candidates.py")], check=True)

    # 2. KARYON QUALIFIES — run the pose-validity skill's tool over the proposal.
    print("\n2. karyon qualifies — pose-validity DRC over the proposed poses:")
    proc = subprocess.run([sys.executable, QUALIFY, CANDIDATES, "--json"],
                          check=True, capture_output=True, text=True)
    results = sorted(json.loads(proc.stdout), key=lambda r: r["pose"])   # pose_1, pose_2, pose_3
    for r in results:
        tag = "valid " if r["valid"] else "REJECT"
        suffix = "" if r["valid"] else f" — {'; '.join(r['reasons'])}"
        print(f"   [{tag}] {os.path.basename(r['pose'])}{suffix}")

    # 3. AGENT ACTS — keep the physically valid poses; the model's rank-1 pick may be gone.
    survivors = [r for r in results if r["valid"]]
    print(f"\n3. agent acts — {len(survivors)}/{len(results)} poses survive qualification.")
    top = results[0]
    if not top["valid"]:
        kept = next((os.path.basename(r["pose"]) for r in survivors), None)
        print(f"   the model's #1-by-confidence pose ({os.path.basename(top['pose'])}) is physically "
              f"INVALID ({'; '.join(top['fired'])}).")
        print(f"   trusting confidence alone would have shipped it; the qualified pick is "
              f"{kept or 'NONE — all candidates rejected'}.")
    else:
        print(f"   the model's #1 pose ({os.path.basename(top['pose'])}) is physically valid and survives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
