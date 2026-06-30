# The qualify spine — one surface, one schema

karyon's gates all reach the same surface: **`karyon.qualify(artifact, modality)`** in Python, or
**`karyon qualify …`** on the command line. Both return the same JSON-serializable result, so an agent, a CI
job, or a pipeline can call one thing and branch on a uniform verdict no matter the modality.

> The model proposes, karyon qualifies — every rejection names its reason.

## The two verbs

| verb | what | returns |
|------|------|---------|
| `karyon qualify <artifact>` | a **per-artifact** gate (pose / cofold / complex / antibody / mol / dna / promoter) | a `QualifyResult` → the JSON below; exit **0** PASS / **1** FAIL / **2** usage error |
| `karyon audit <leakage\|screen>` | a **dataset-level** audit (benchmark leakage / CRISPR-screen power) | a serialized report; exit **0** on a report / **2** if the dataset is unavailable |

`karyon list` enumerates the modalities and audits.

## Modalities

| modality | artifact | inferred from | optional deps | options |
|----------|----------|---------------|---------------|---------|
| `pose` | docking pose(s) — `.sdf`, a glob, or a directory | `.sdf` | rdkit (`karyon[chem]`) | — |
| `mol` | generated molecule(s) — inline SMILES or `.smi` | `.smi` | rdkit (`karyon[chem]`) | — |
| `dna` | generated DNA — inline ACGT or `.fasta` | — (ambiguous with `promoter`) | — | — |
| `promoter` | σ70 promoter — inline ACGT or `.fasta` | — (ambiguous with `dna`) | — | — |
| `cofold` | co-folding pose — `.pdb` / `.cif` | — (ambiguous with `complex`) | numpy (core) | `--ligand SDF`, `--ligand-resname` |
| `complex` | protein complex — `.pdb` / `.cif` | — (ambiguous with `cofold`) | numpy (core) | `--chain-a`, `--chain-b` |
| `antibody` | antibody Fv developability — VH(+VL) as `HEAVY:LIGHT`, a 2-record `.fasta`, or a single VHH | — (ambiguous with `dna`/`promoter`) | — | — |

**Modality inference is conservative.** Only `.sdf` (→ pose) and `.smi` (→ mol) auto-resolve. Structure
files (`.pdb` / `.cif` — could be a co-folding pose *or* a protein complex) and FASTA (`.fasta` / a raw
sequence — could be generic DNA, a promoter, *or* an antibody Fv) are intentionally ambiguous, so `--modality`
is **required** for them; an omitted-but-ambiguous input raises an error naming the candidates. Inline strings
always need `--modality`.

## The result schema

`QualifyResult.to_dict()` (and `karyon qualify --json`) emit exactly:

```json
{
  "modality": "dna",
  "ok": false,
  "items": [
    {
      "name": "designA",
      "ok": false,
      "score": 2.0,
      "reasons": [
        {"contract": "GC_OUT_OF_BAND", "message": "GC 12% outside 25–65% …", "weight": 1.0},
        {"contract": "STRONG_HAIRPIN", "message": "a 24 bp self-complementary stem …", "weight": 1.0}
      ]
    }
  ],
  "batch": {"ok": true, "score": 0.0, "reasons": []}
}
```

- **`ok`** (top level) — PASS iff no *condemning* contract fired anywhere (every item and the batch score 0).
- **`items`** — one entry per artifact. For a directory/glob/FASTA, there are many; for an inline string, one.
- **`item.ok` / `item.score`** — an item **passes iff `score == 0.0`**. The score is the sum of fired
  contract weights.
- **`reasons[].weight`** — a contract is **condemning** when `weight > 0` and a **disclosure** when
  `weight == 0`. Disclosures (e.g. a PAINS structural alert, a restriction-site collision) are reported in
  `reasons` but do **not** fail the gate — they inform. This is the one pass semantic across all gates.
- **`batch`** — a set-level verdict (currently DNA cross-hybridization across a multi-record FASTA), or
  `null` when the modality has no set-level check or the input is a single artifact.

This shape is stable: consumers may depend on these keys.

## Python

```python
from karyon import qualify

r = qualify("evo2_designs.fasta", modality="dna")
print(r.ok)                                   # overall PASS/FAIL
for name, verdict in r.items:                 # verdict is a contracts.Verdict
    if verdict.score > 0:
        print(name, verdict.messages)         # the named reasons
import json; print(json.dumps(r.to_dict()))   # the stable schema
```

The underlying engine (`Verdict`, `Reason`, `Contract`, `ContractSet`) is re-exported from `karyon` for
callers who want to build their own gate; `karyon.GATES` is the modality registry and `karyon.modalities()`
lists it.
