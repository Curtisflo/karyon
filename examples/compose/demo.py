#!/usr/bin/env python3
"""End-to-end composition demo: a generative docking model proposes, karyon
qualifies, the agent acts on the named-reason verdict.

Runs with no GPU / NIM / network. The candidate poses stand in for a DiffDock /
Boltz-2 NIM emission (see make_candidates.py); karyon's qualify spine is the
qualifier (`karyon.qualify(dir, "pose")` — the same call the pose-validity skill's
`karyon qualify` CLI makes); this script plays the agent that keeps the survivors.

    pip install "karyon[chem]"
    python examples/compose/demo.py
"""
from __future__ import annotations

import os
import subprocess
import sys

from karyon import qualify

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = os.path.join(HERE, "candidates")


def main() -> int:
    # 1. MODEL PROPOSES — stand in for a DiffDock/Boltz-2 NIM writing ranked SDF poses.
    print("1. model proposes — generating 3 ranked candidate poses (DiffDock/NIM stand-in):")
    subprocess.run([sys.executable, os.path.join(HERE, "make_candidates.py")], check=True)

    # 2. KARYON QUALIFIES — the pose-validity gate over the proposal, one spine call.
    print("\n2. karyon qualifies — pose-validity DRC over the proposed poses:")
    result = qualify(CANDIDATES, modality="pose")
    items = sorted(result.items, key=lambda nv: nv[0])        # pose_1, pose_2, pose_3
    for name, v in items:
        valid = v.score == 0.0
        suffix = "" if valid else f" — {'; '.join(v.messages)}"
        print(f"   [{'valid ' if valid else 'REJECT'}] {os.path.basename(name)}{suffix}")

    # 3. AGENT ACTS — keep the physically valid poses; the model's rank-1 pick may be gone.
    survivors = [(n, v) for n, v in items if v.score == 0.0]
    print(f"\n3. agent acts — {len(survivors)}/{len(items)} poses survive qualification.")
    top_name, top_v = items[0]
    if top_v.score > 0.0:
        kept = os.path.basename(survivors[0][0]) if survivors else "NONE — all candidates rejected"
        print(f"   the model's #1-by-confidence pose ({os.path.basename(top_name)}) is physically "
              f"INVALID ({'; '.join(top_v.fired)}).")
        print(f"   trusting confidence alone would have shipped it; the qualified pick is {kept}.")
    else:
        print(f"   the model's #1 pose ({os.path.basename(top_name)}) is physically valid and survives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
