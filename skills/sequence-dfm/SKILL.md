---
name: sequence-dfm
description: >
  Design-for-manufacturability gate over generated DNA sequences with karyon — reject unsynthesizable or unusable candidates with named reasons before ordering. Use for sequence synthesizability, DNA design rule check, GC-content band, homopolymer run, Pol-III terminator (TTTT) screening, or qualifying the output of a generative DNA model (e.g. Evo2, GenMol) before downstream use.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "karyon (stdlib only)"
allowed-tools: Bash, Read, Write
---

# Sequence DFM Gate

A **deterministic synthesizability / usability DRC** over DNA sequences. A generative model proposes
sequences; this gate rejects the ones that won't work — and names why — before anything is ordered or fed
downstream. Pure stdlib, no model, no network.

Contracts (each returns a human-readable reason):
- `TTTT` — Pol-III terminator that truncates an sgRNA;
- GC outside the **20–80%** band — poor loading / over-stable;
- homopolymer run ≥ 5 — synthesis and folding risk.

## Install
```bash
pip install karyon          # no extras needed
```

## Usage
```python
from karyon import crispr_qc

for seq in candidate_sequences:                  # e.g. Evo2 / GenMol output
    reasons = crispr_qc.hard_contracts(seq)      # [] == passes the rules
    if reasons:
        print(f"REJECT {seq[:24]}… — {reasons}")
    # e.g. ["TTTT: Pol-III terminator truncates the sgRNA", "GC 12% <20%: poor RISC loading"]
```

For feasibility-aware *construction* (derive sequences that pass by design rather than rejecting after the
fact), use the constructive core:
```python
from karyon import constructive_core as cc
ok = cc.is_feasible(seq)        # GC band + homopolymer feasibility
```

## Composition with NVIDIA BioNeMo
Install alongside **`evo2-nim`** (DNA generation) or **`genmol-nim`**: the model generates, this skill gates
the batch, so the agent only carries forward synthesizable candidates and can explain every rejection — the
"unroutable net" report for a generative model's output.

## Scope (honest)
A fast, legible *manufacturability/usability* gate (the cheap, certain checks), not a folding or
expression-strength predictor. Pair it with the karyon expression predictors for the soft, quantitative axis.
