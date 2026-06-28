---
name: cofold-qc
description: >
  Deterministic physical-validity gate for co-folding model output (protein + ligand predicted in one
  frame, e.g. Boltz-2, AlphaFold-3, Chai-1) with karyon. Use AFTER predicting a protein-ligand complex and
  BEFORE trusting the pose — it answers "is this pose physically valid?" with a pass/fail verdict and
  legible per-reason explanations (ligand-protein clash, volume overlap, out-of-pocket placement, plus the
  ligand's own bond/angle/ring/strain geometry). No GPU required.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "karyon[chem] (rdkit)"
allowed-tools: Bash, Read, Write
---

# Co-folding Validity Gate

A **deterministic, legible physical-validity gate** for co-folding poses. Co-folding models (Boltz-2,
AlphaFold-3, Chai-1) predict a protein and a ligand **together in one coordinate frame** — powerful but not
self-checking: a pose can score well on confidence yet be **physically invalid**, the ligand clashing into
the protein, burying into its volume, or floating outside any pocket. This skill is the programmatic version
of the "inspect it in PyMOL for obvious clashes" sanity check: a design-rule check (DRC) that returns a
**named reason for every violation**, not just a score.

It owns the **intermolecular** axis end-to-end (where most co-folding placement failures live):

| contract | catches |
|---|---|
| `LIGAND_PROTEIN_CLASH` | ligand heavy atoms inside the protein's van-der-Waals shell |
| `LIGAND_PROTEIN_VOLUME_OVERLAP` | a fraction of the ligand's volume buried in the protein |
| `LIGAND_OUT_OF_POCKET` | the ligand making no contact — floated outside any pocket |

and reuses the karyon **intramolecular** ligand DRC (bond lengths/angles, ring/double-bond planarity,
internal steric clash, internal strain) for the whole physical-validity call. Thresholds are physical
constants / PoseBusters conventions — **zero parameters fitted to accuracy**.

## Install
```bash
pip install "karyon[chem]"      # pulls rdkit (numpy is a base dependency)
```

## Usage
Run the qualifier on a co-folding output (PDB or mmCIF) with protein + ligand in one frame. `--modality
cofold` is required (a `.cif`/`.pdb` could equally be a protein complex — see `complex-qc`):

```bash
# Intermolecular gate from coordinates alone:
karyon qualify complex.cif --modality cofold

# Add the ligand SDF (bond orders) → enables the full intramolecular ligand DRC:
karyon qualify complex.cif --modality cofold --ligand ligand.sdf

# Name the ligand residue explicitly if auto-detection is ambiguous:
karyon qualify complex.pdb --modality cofold --ligand-resname LIG

# JSON verdict for piping into an agent / pipeline:
karyon qualify complex.cif --modality cofold --json
```

Output is a `PASS` / `FAIL` verdict plus, on failure, one line per fired contract naming exactly what is
wrong (e.g. *"ligand clashes into the protein: closest atoms 1.4 Å apart (38% of their vdW sum); 6 clashing
pairs"*). Exit code is non-zero on `FAIL`, so it gates a pipeline directly. `--json` emits the stable spine
schema (`{modality, ok, items:[{name, ok, score, reasons}], batch}`); a pose passes iff `score == 0`.

From Python:

```python
from karyon import qualify
r = qualify("complex.pdb", modality="cofold")          # add ligand="ligand.sdf" for the intramolecular DRC
v = r.items[0][1]
if v.score > 0:
    print("INVALID —", v.messages)                     # named reasons per fired contract
```

## Composition with NVIDIA BioNeMo
Install alongside a co-folding NIM (`boltz2-nim` / `openfold3-nim`): the model proposes the protein-ligand
pose, this skill qualifies it. The agent keeps only poses that pass and reports *why* the rest were
rejected — turning "top pose, confidence 0.42" into "top pose, confidence 0.42, physically valid (0
contracts fired)." It is the **complement** to a co-folding model, not a competitor — it qualifies the
output, it does not predict structure.

## Validation
The gate is validated **faithful to the real PoseBusters package** across four co-folding methods (per-pose
intermolecular agreement, scoring the **raw** predicted pose — relaxation would hide the very violations the
DRC exists to catch):

| method | per-pose agreement vs PoseBusters | physically-invalid (raw) |
|---|---|---|
| Boltz-2 | **99%** | 4% (clean) |
| AlphaFold-3 | **89%** | 20% |
| RoseTTAFold-All-Atom | **97%** | 92% |
| NeuralPLexer | reference is FF-relaxed (not like-for-like)* | 94% |

Thresholds are physical constants / PoseBusters conventions, fixed **before** these runs — the agreement is
not fitted. The effect is method-specific: modern co-folding spans from clean (Boltz) to near-totally-
clashing (RFAA / NeuralPLexer), which is exactly what a gate is for. *(\*NeuralPLexer's deposited PoseBusters
reference scores a force-field-relaxed copy of each pose, so it isn't a like-for-like faithfulness
comparison; the gate flags the pre-relaxation clashes the relaxed reference hides.)*

## Scope (honest)
This DRC owns the **intermolecular** axis (ligand↔protein clash / volume overlap / out-of-pocket) from
coordinates alone, and reuses the karyon intramolecular ligand DRC when a ligand SDF with bond orders is
supplied. It is a *physical-validity* gate — qualification, not accuracy: it reports the honest
physically-valid verdict per pose; it does not make any model dock better.
