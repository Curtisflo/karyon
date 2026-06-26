---
name: pose-validity
description: >
  Run a deterministic physical-validity DRC over molecular docking poses with karyon. Use for pose validity, PoseBusters-style geometric checks, bond-length/angle/ring-planarity/steric-clash/strain screening, qualifying or filtering ranked docking poses (e.g. DiffDock output), or deciding whether a predicted binding pose is physically plausible before trusting it.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "karyon[chem] (rdkit)"
allowed-tools: Bash, Read, Write
---

# Pose Validity DRC

A **deterministic, legible physical-validity gate** for predicted ligand poses. It checks the geometry
a docking model can get wrong — unparseable molecule, no real 3D conformer, bond-length / bond-angle
outliers, non-planar aromatic rings, twisted double bonds, internal steric clashes, excessive strain
energy — and returns a **named reason for every violation**, not just a score.

This is the programmatic version of the "sanity check" docking tools tell you to do by eye. NVIDIA's
`diffdock-nim` skill, for example, instructs the agent to *"inspect poses in PyMOL … look for obvious
clashes, disconnected fragments."* This skill automates that into a falsifiable verdict. (On the
PoseBusters benchmark, karyon's DRC found **71% of DiffDock's RMSD≤2 "successes" are physically invalid**.)

## Install
```bash
pip install "karyon[chem]"      # pulls rdkit
```

## Usage
DiffDock writes ranked poses as `pose_<rank>_conf<score>.sdf`. Gate each one:

```python
import glob
from rdkit import Chem
from karyon import pose_validity as pv

cs  = pv.validity_contracts()          # the DRC: bond/angle/ring/clash/strain contracts
tol = pv.Tol()                         # tolerances (PoseBusters-style cutoffs; override per field)

for path in sorted(glob.glob("pose_*.sdf")):
    mol     = Chem.MolFromMolFile(path, sanitize=True)
    feats   = pv.featurize(mol, tol)
    verdict = cs.evaluate(feats, tol)              # -> contracts.Verdict
    if verdict.score > 0:                          # a condemning contract fired
        print(f"{path}: INVALID — {verdict.messages}")
    else:
        print(f"{path}: valid")
```

`verdict.messages` lists the human-readable reasons; `verdict.fired` lists the contract names
(`BOND_ANGLE_OUTLIER`, `INTERNAL_STERIC_CLASH`, …). `pv.is_invalid(feats, cs, tol)` is the boolean shortcut.

## Composition with NVIDIA BioNeMo
Install alongside `diffdock-nim` (or `boltz2-nim` / `openfold3-nim`): the model proposes poses, this skill
qualifies them. The agent keeps only poses that pass, and reports *why* the rest were rejected — turning
"top pose, confidence 0.42" into "top pose, confidence 0.42, physically valid (0 contracts fired)."

## Scope (honest)
This DRC owns the **intramolecular** axis end-to-end (the ligand's own geometry). **Intermolecular**
validity — clash with the receptor, ligand outside the pocket — needs the receptor structure and is *not*
covered by this single-molecule check; pair it with receptor-aware tooling for that axis.
