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
PoseBusters benchmark, karyon's DRC found **70% of DiffDock's RMSD≤2 "successes" are physically invalid**.)

## Install
```bash
pip install "karyon[chem]"      # pulls rdkit
```

## Usage
Gate a directory, a glob, or a single SDF of poses (DiffDock writes `pose_<rank>_conf<score>.sdf`):

```bash
karyon qualify diffdock_out/ --modality pose --json   # a directory, a glob, or one .sdf
karyon qualify pose_1.sdf --modality pose             # human summary; exit 1 if any pose is invalid
```
`.sdf` is unambiguous, so `--modality pose` may be omitted. `--json` emits the stable spine schema —
`{modality, ok, items:[{name, ok, score, reasons:[{contract, message, weight}]}], batch}` — where a pose
passes iff `score == 0` (weight-0 reasons are disclosures that inform but don't fail).

From Python:
```python
from karyon import qualify
r = qualify("diffdock_out/", modality="pose")         # or qualify("pose_1.sdf")
for name, v in r.items:
    print(name, "valid" if v.score == 0 else f"INVALID — {v.messages}")
```

## Composition with NVIDIA BioNeMo
Install alongside `diffdock-nim` (or `boltz2-nim` / `openfold3-nim`): the model proposes poses, this skill
qualifies them. The agent keeps only poses that pass, and reports *why* the rest were rejected — turning
"top pose, confidence 0.42" into "top pose, confidence 0.42, physically valid (0 contracts fired)."

Run it on the directory the model wrote:
```bash
karyon qualify diffdock_out/ --modality pose --json
```
A runnable end-to-end demo (model proposes → karyon qualifies → agent acts, no GPU/NIM) lives in the
karyon repo at [`examples/compose/`](https://github.com/Curtisflo/karyon/tree/main/examples/compose).

## Scope (honest)
This DRC owns the **intramolecular** axis end-to-end (the ligand's own geometry). **Intermolecular**
validity — clash with the receptor, ligand outside the pocket — needs the receptor structure and is *not*
covered by this single-molecule check; pair it with receptor-aware tooling for that axis.
