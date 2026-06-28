---
name: gen-dna-qc
description: >
  Deterministic synthesizability/manufacturability gate for GENERATED DNA sequences (e.g. NVIDIA BioNeMo's Evo2, or any genomic sequence generator) with karyon. Use AFTER generating a DNA sequence and BEFORE ordering or cloning it — answers "can this actually be synthesized and cloned, and will it behave?" with a pass/fail verdict and legible per-reason explanations (GC out of the synthesis band, homopolymer / poly-G runs, out-of-range length, strong self-hairpin, restriction-site collisions, and — across a batch — cross-hybridization between sequences). Pure stdlib — no GPU, no network, no numpy.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "karyon (stdlib only)"
allowed-tools: Bash, Read, Write
---

# gen-dna-qc — a legible design-for-manufacture gate for generated DNA

Genomic sequence generators (Evo2, and other DNA language models) emit DNA that scores well on the model's
own confidence yet can be **unmanufacturable** — GC outside the synthesis window, a homopolymer run that
slips the synthesizer, a hairpin that won't anneal, a cloning-site collision, or (across a generated batch)
two sequences that hybridize to each other instead of their targets.

`gen-dna-qc` is the programmatic version of that check: a **deterministic design-rule check (DRC)** that
emits a pass/fail verdict with a human-readable reason for every flag — the "unroutable net" report for a
generative model's output. It is the **complement** to a generator, not a competitor — it qualifies the
output, it does not generate sequence.

It owns two ownership levels (a per-sequence fact and a design-level invariant no single sequence owns):

| contract | tier | catches |
|---|---|---|
| `GC_OUT_OF_BAND` | **fails** | GC fraction outside the synthesis envelope (default 25–65%) |
| `HOMOPOLYMER_RUN` | **fails** | a run of one base longer than the cap (synthesis slippage) |
| `LENGTH_OUT_OF_RANGE` | **fails** | below the oligo floor / above the single-fragment gene window |
| `STRONG_HAIRPIN` | **fails** | a long self-complementary stem — folds on itself, won't synthesize/anneal |
| `SEVERE_CROSS_HYBRIDIZATION` | **fails** | two sequences in a batch anneal to each other (not their targets) |
| `POLY_G_RUN` | discloses | GGGG+ — a G-quadruplex risk |
| `RESTRICTION_SITE` | discloses | recognition-site collisions — will be cut if cloned with those enzymes |
| `CROSS_HYBRIDIZATION` | discloses | a moderate complementary stretch between two batch sequences |

The verdict separates **disclosure** from **condemnation**: every hazard is *reported*, but only the
synthesis-breaking ones *fail* the structure — a restriction site or a poly-G run informs without condemning,
because they're cloning/risk notes, not synthesis failures. Thresholds are commercial-synthesis constants /
DnaChisel conventions — **zero parameters fitted to accuracy**.

## Install
```bash
pip install karyon          # no extras needed — pure stdlib
```

## Usage
`--modality dna` is required (a `.fasta`/sequence could equally be a σ70 promoter — see `promoter-design`):
```bash
# A single generated sequence (inline):
karyon qualify ACGTACGT... --modality dna

# A batch (multi-record FASTA, e.g. Evo2 output) — adds the cross-hybridization set check:
karyon qualify evo2_designs.fasta --modality dna

# JSON verdict for piping into an agent / pipeline:
karyon qualify evo2_designs.fasta --modality dna --json
```

Output is a `PASS` / `FAIL` verdict plus, per sequence, one line per fired contract — a `·` for a disclosed
hazard, an `✗` for a condemning one (e.g. *"strong hairpin: a 24 bp self-complementary stem (≥12) across a 6
nt loop — folds on itself, won't synthesize/anneal cleanly"*). Exit code is non-zero on `FAIL` so it gates a
pipeline directly. `--json` emits the stable spine schema — `{modality, ok, items:[...], batch}` — where the
set-level cross-hybridization verdict (multi-record only) rides in `batch`; a design passes iff `score == 0`.

From Python:
```python
from karyon import qualify
r = qualify("evo2_designs.fasta", modality="dna")     # or qualify("ACGT...", modality="dna")
for name, v in r.items:                               # per-sequence verdicts
    if v.score > 0:
        print(f"REJECT {name} — {v.messages}")
if r.batch and r.batch.score > 0:                     # cross-hybridizing pairs across the batch
    print("batch:", r.batch.messages)
```

## Composition with NVIDIA BioNeMo
Install alongside **`evo2-nim`** (DNA generation): the model generates the sequence; this skill gates the
batch, so the agent only carries forward synthesizable candidates and can explain every rejection. Generate
with the BioNeMo Evo2 skill, then pass the output to `gen-dna-qc` for a deterministic manufacturability
verdict — the packaging mirrors BioNeMo's `SKILL.md` convention so the two compose cleanly.

## Validation
The gate is validated as a real instrument and **faithful to two gold-standard packages**, with three
pre-registered predictions (all PASS):

| prediction | result |
|---|---|
| **PI-1** instrument — clean sequences pass, planted decoys flagged | AUROC **1.000**, flag-decoy 100%, pass-clean 100%, **real E. coli CDS pass 100%** |
| **PI-2** faithful — owned verdict vs **DnaChisel** (EnforceGCContent + AvoidPattern + AvoidHairpins) | per-sequence agreement **100%** (GC / homopolymer / hairpin all 100%); hairpin signal vs **ViennaRNA** ΔG AUROC **0.88** |
| **PI-3** effect — synthesis/cloning hazard rates per generator (descriptive) | honest weak-condemn (random DNA is mostly synthesizable under lenient vendor rules) but high **disclose** rates (restriction sites, poly-G) — the hazards a generator is blind to |

Thresholds are commercial-synthesis constants, calibrated to the DnaChisel reference and fixed **before** the
runs — the agreement is not fitted. Qualification, not accuracy: the gate reports what won't synthesize or
clone, it does not make the generator better.

## Scope (honest)
A fast, legible *manufacturability/usability* gate (the cheap, certain checks over string geometry: GC,
runs, hairpin, cross-hyb, restriction sites), not a folding or expression-strength predictor. Pair it with
the karyon expression predictors for the soft, quantitative axis.
