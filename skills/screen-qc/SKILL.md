---
name: screen-qc
description: >
  Qualify a pooled CRISPR screen's non-hits with karyon — flag genes called "no effect" that the screen was simply under-powered to detect. Use for CRISPR screen QC, under-power detection, silent-failure / false-negative qualification, guide-count reliability, NT/NEG-calibrated screen analysis, or deciding whether a non-hit is trustworthy or just low-powered.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "karyon (numpy)"
allowed-tools: Bash, Read, Write
---

# CRISPR Screen QC

A pooled-screen pipeline emits a per-gene q-value: hit / non-hit. That scalar throws away the within-gene
guide structure, so a gene the screen had **no power** to call looks identical to a gene that genuinely has
no effect. This skill reads the structure back — **purely from the counts, calibrated on the
non-targeting / negative controls, never from a gold standard** — and qualifies every non-hit with a named
reason (too few usable guides, dispersion too high, effect below the null band).

On the Wang-2014 leukemia dropout screen it flags **~53% of gold-standard silent failures at a 3%
false-flag rate**, and the flag is **non-redundant with the FDR** (|ρ|≈0.29) — it catches what the q-value
misses. (Honest: the value here is *legibility / qualification*, not a recovery-accuracy lift.) Full method,
pre-registered Q1–Q4 evaluation, and limits:
[docs/screen-power.md](https://github.com/Curtisflo/karyon/blob/main/docs/screen-power.md).

## Install
```bash
pip install karyon
```

## Usage
Run the reference analysis (the demo screen):
```python
from karyon import screen_qc
screen_qc.run()                 # default seeds=50; prints the qualification + the non-redundancy guard
```

Qualify your own screen's non-hits (counts → reasons):
```python
from karyon import screen_qc as sq
null    = sq.null_band(calib_lfcs, direction="deplete")   # null band from NT/NEG controls
verdicts = sq.qualify(non_hit_genes, gene_guides, null=null)   # GeneReliability per gene, with reasons
```
`reliability_contracts(...)` is the underlying `contracts.ContractSet`; each `GeneReliability` names why a
non-hit is (un)trustworthy.

## Composition with NVIDIA BioNeMo
Runs **downstream** of an accelerated genomics pipeline (e.g. **Parabricks** → guide counts → a MAGeCK-style
gene summary): the model/pipeline calls hits, this skill qualifies the *non*-hits so the agent reports
"gene X: non-hit, but under-powered (4 usable guides, dispersion above threshold)" instead of a bare
"no effect."

## Scope (honest)
Qualification, not accuracy: it tells you which non-hits to *distrust*, it does not re-rank hits. Calibrated
on controls, so it needs a screen with non-targeting / negative-control guides.
