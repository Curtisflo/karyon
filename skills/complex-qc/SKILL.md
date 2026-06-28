---
name: complex-qc
description: >
  Deterministic interface-validity gate for predicted or designed protein COMPLEXES (two or more protein
  chains in one frame, e.g. AlphaFold-Multimer models, or RFdiffusion + ProteinMPNN designed binders against
  a target) with karyon. Use AFTER predicting/designing a complex and BEFORE trusting the interface — it
  answers "is this interface physically valid?" with a pass/fail verdict and legible per-reason explanations
  (inter-chain clash, gross interpenetration, chains out of contact). No GPU required.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "karyon (numpy)"
allowed-tools: Bash, Read, Write
---

# Complex Interface Validity Gate

A **deterministic, legible interface-validity gate** for protein complexes. Complex predictors and binder
designers (AlphaFold-Multimer, RFdiffusion + ProteinMPNN) place two or more protein chains **together in one
coordinate frame** — powerful but not self-checking: a predicted or designed interface can score well on
confidence yet be **physically invalid**, chains driven into each other (steric clash), grossly
interpenetrating, or floated apart so they never actually touch. This skill is the programmatic version of
the "inspect it in PyMOL for obvious clashes" sanity check, and the protein↔protein sibling of `cofold-qc`
(which gates protein↔ligand co-folding poses), reusing the same intermolecular geometry.

It owns the **inter-chain interface** axis end-to-end:

| contract | catches | tier |
|---|---|---|
| `INTERFACE_CLASH` | inter-chain heavy atoms inside each other's van-der-Waals shell | discloses |
| `SEVERE_INTERFACE_CLASH` | a clash deep enough to be backbone-through-backbone interpenetration | **fails** |
| `INTERFACE_VOLUME_OVERLAP` | a chain's atoms buried inside the partner (gross interpenetration) | **fails** |
| `CHAINS_NOT_IN_CONTACT` | the chains make no contact — a failed placement, no interface | **fails** |

The verdict separates **disclosure** from **condemnation**: every inter-chain clash is *reported* (the
detection signal), but only physically-unphysical interpenetration or out-of-contact *fails* the structure —
because a real, deposited, experimentally-determined complex commonly carries a few shallow interface clashes
that inform without invalidating it. Thresholds are physical constants / MolProbity conventions (including
MolProbity's own vdW radii, its H-bond/salt-bridge allowance, and disulfide exclusion) — **zero parameters
fitted to accuracy**.

## Install
```bash
pip install karyon          # numpy only — no extras needed
```

## Usage
Run the qualifier on a predicted/designed complex (PDB or mmCIF), two or more chains in one frame.
`--modality complex` is required (a `.cif`/`.pdb` could equally be a protein-ligand co-folding pose — see
`cofold-qc`):

```bash
# Auto-pick the two largest chains:
karyon qualify complex.pdb --modality complex

# Name the partners explicitly (a designed binder = chain A against a two-chain target = chains B+C):
karyon qualify binder.pdb --modality complex --chain-a A --chain-b B,C

# JSON verdict for piping into an agent / pipeline:
karyon qualify complex.cif --modality complex --chain-a A --chain-b B --json
```

Output is a `PASS` / `FAIL` verdict plus one line per fired contract (a `·` for a disclosed clash, an `✗` for
a condemning one), e.g. *"chains interpenetrate: an inter-chain pair overlaps 2.10 Å (>0.90 Å — unphysically
deep…)"*. Exit code is non-zero on `FAIL`, so it gates a pipeline directly. `--json` emits the stable spine
schema (`{modality, ok, items:[{name, ok, score, reasons}], batch}`); the interface passes iff `score == 0`.

From Python:

```python
from karyon import qualify
r = qualify("complex.pdb", modality="complex")           # add chain_a="A", chain_b="B,C" to name partners
v = r.items[0][1]
if v.score > 0:
    print("INVALID —", v.messages)                       # named reasons per fired contract
```

## Composition with NVIDIA BioNeMo
Install alongside a multimer / binder-design NIM (e.g. RFdiffusion / ProteinMPNN): the tool gives you the
multimer model or the designed binder, this skill qualifies the interface. The agent keeps only complexes
that pass and reports *why* the rest were rejected. It is the **complement** to a complex predictor / binder
designer, not a competitor — it qualifies the output, it does not predict structure — as does its sibling
`cofold-qc` for protein-ligand poses.

## Validation
The gate is validated **faithful to the wwPDB validation reference** (MolProbity, the field's gold-standard
steric-clash validator), restricted to the same axis it owns — inter-chain heavy↔heavy clashes — across
deposited protein complexes. It is an instrument (deposited complexes pass; rigid-body interpenetration /
separation decoys are flagged) and its inter-chain clash detection tracks the deposited reference's clash
counts. The thresholds are physical constants / MolProbity conventions, fixed **before** the runs — the
agreement is not fitted. The *effect* (predicted multimer complexes carry far more interface clashes than
deposited natives) is reported on CASP15 predictions, where no deposited validity reference exists.

## Scope (honest)
This DRC owns the **inter-chain interface** axis (clash / interpenetration / out-of-contact) from coordinates
alone. It separates disclosure (every clash reported) from condemnation (only unphysical interpenetration /
out-of-contact fails), and is heavy-atom-only — it over-flags favorable polar/catalytic contacts vs an
H-aware reference, the safe direction for a QC gate (it misses no real clash). Qualification, not accuracy.
