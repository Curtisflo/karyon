---
name: mol-qc
description: >
  Deterministic validity / synthesizability / drug-likeness gate over GENERATED molecules with karyon —
  reject invalid, unsynthesizable, or out-of-range candidates with named reasons before trusting or ordering
  them. Use for molecule validity, SMILES sanitization, Ertl synthetic-accessibility screening, extreme
  MW/cLogP rejection, PAINS/Brenk structural-alert and Lipinski/Veber drug-likeness disclosure, or qualifying
  the output of a generative-chemistry model (e.g. NVIDIA BioNeMo GenMol / MolMIM) before downstream use.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "karyon[chem] (rdkit)"
allowed-tools: Bash, Read, Write
---

# mol-qc — validity / synthesizability gate for generated molecules

A **deterministic, legible validity / synthesizability / drug-likeness DRC** for molecules a
generative-chemistry model proposes. It catches what such a model can get wrong — an invalid valence that
won't parse, a structure no chemist could synthesize, properties so extreme it isn't a small molecule — and
returns a **named reason for every flag**, not just a score. No GPU, no network.

This is the programmatic version of the "inspect the molecule" sanity check generative-chemistry tools tell
you to do by eye. The verdict separates **disclosure** from **condemnation**: the gate *fails* only broken or
unmakeable molecules; structural alerts and drug-likeness notes are **advisory disclosures**, because PAINS /
Rule-of-5 have well-known false positives (many marketed drugs hit them). Thresholds are medicinal-chemistry
conventions — **zero parameters fitted to accuracy**.

| contract | tier | catches |
|---|---|---|
| `INVALID_MOLECULE` | **fails** | does not parse / sanitize (bad valence or syntax) |
| `UNSYNTHESIZABLE` | **fails** | Ertl synthetic-accessibility score above the cap (not reasonably makeable) |
| `EXTREME_PROPERTY` | **fails** | egregiously out of small-molecule range (MW > 900 / cLogP > 7) |
| `STRUCTURAL_ALERT` | discloses | PAINS / Brenk hits (assay-interference / reactive — advisory) |
| `LIPINSKI_RO5` | discloses | ≥2 Rule-of-5 violations (drug-likeness note) |
| `VEBER` | discloses | rotatable bonds > 10 / TPSA > 140 (oral-bioavailability note) |

## Install
```bash
pip install "karyon[chem]"      # pulls rdkit
```

## Usage
Gate a single generated molecule, or a batch in a `.smi` file (one SMILES per line, optional name):
```bash
python scripts/qc.py --smiles "CC(=O)Oc1ccccc1C(=O)O"     # one molecule
python scripts/qc.py --smi-file generated.smi             # a batch
python scripts/qc.py --smi-file generated.smi --json      # machine-readable, for an agent to branch on
```
Output is a `PASS` / `FAIL` verdict plus, per molecule, one line per fired contract — `·` for a disclosed
advisory, `✗` for a condemning one. The exit code is non-zero on `FAIL`, so it gates a pipeline directly.

From Python:
```python
from karyon import mol_qc

tol = mol_qc.MolTol()
cs  = mol_qc.mol_contracts()                    # the DRC: validity/SA/property/alert/Ro5/Veber contracts
for smi in candidate_smiles:                    # e.g. GenMol / MolMIM output
    verdict = cs.evaluate(mol_qc.featurize(smi, tol), tol)
    if verdict.score > 0:                        # a condemning contract fired
        print(f"REJECT {smi} — {verdict.messages}")
```
`mol_qc.is_unusable(smi, tol)` is the boolean shortcut; `verdict.fired` lists the contract names
(`INVALID_MOLECULE`, `UNSYNTHESIZABLE`, …) and `verdict.messages` the human-readable reasons.

## Composition with NVIDIA BioNeMo
Install alongside **`genmol-nim`** (or the MolMIM generator): the model proposes molecules, this skill
qualifies the batch, so the agent only carries forward valid, makeable candidates and can explain every
rejection. It complements the generator's advisory validation with a programmatic verdict — it qualifies the
output, it does not generate molecules.

## Validation
Three pre-registered predictions (PI-1 and PI-2 PASS; PI-3 descriptive):

| prediction | result |
|---|---|
| **PI-1** instrument — real drugs pass, planted/invalid decoys flagged | AUROC **1.000**, flag-decoy 100%, real-drug pass **99%** (92 approved drugs) |
| **PI-2** faithful — the gate faithfully composes the canonical RDKit primitives | composition correctness **100%** (gate INVALID/ALERT == fresh canonical RDKit calls) + owned Rule-of-5 **100%** vs hand computation |
| **PI-3** effect — defect rates per generator (descriptive) | un-gated raw SMILES ~**100% invalid** (the validity gate's headline catch); a structure-aware generator (BRICS) is valid but carries structural alerts ~19% — a weak-condemn / high-disclose gate |

**Honest posture (disclosed):** **RDKit is the cheminformatics engine here** — so `mol-qc` *composes*
canonical primitives (sanitization, descriptors, the PAINS/Brenk `FilterCatalog`, the Ertl SA score) into a
legible deterministic gate rather than reimplementing them. Its faithfulness is *correct composition* +
*correct owned rules* (both 100%), not an independent reimplementation; an attempted independent complexity
corroboration (SA vs RDKit `BertzCT`) is weak (ρ≈0.36 — the two measure complexity differently) and is
reported, not claimed. Qualification, not accuracy.

## Scope (honest)
A fast, legible *validity / synthesizability / drug-likeness* gate (the cheap, certain checks), not a
binding-affinity, ADMET, or potency predictor. It owns the single-molecule axis; pair it with receptor-aware
or property-prediction tooling for the quantitative axes.
