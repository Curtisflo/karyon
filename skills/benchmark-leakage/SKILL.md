---
name: benchmark-leakage
description: >
  Audit an ML benchmark for data leakage with karyon — measure how much a reported accuracy is inflated by train/test contamination. Use for leakage detection, data-leakage audit, scaffold leakage, template memorization, random-vs-scaffold split inflation, retrosynthesis or molecular-property benchmark honesty, or deciding whether a model's headline number is real.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "karyon[chem] (rdkit, rdchiral)"
allowed-tools: Bash, Read, Write
---

# Benchmark Leakage Audit

A **legible, model-free audit** that asks whether a benchmark number is inflated by train/test leakage,
and names the mechanism. Two substrates ship:

- **Retrosynthesis** (`retro_template`) — flags test reactions whose transform template or near-duplicate
  was already seen in training. On USPTO-50k a faithful retrosim baseline scores top-1 **37.9%** on the
  standard split but **16.1%** on a leakage-free partition (93.8% of the test set carries leakage) — a
  measured **+21.8-point** inflation.
- **Molecular property / ADMET** (`molnet_honesty`) — flags test molecules sharing a Bemis-Murcko scaffold
  with training. Random-vs-scaffold split inflation is real on both a classifier (AUROC +0.105) and a
  regressor (ρ +0.100).

## Install
```bash
pip install "karyon[chem]"      # rdkit + rdchiral
```

## Usage
Molecular property benchmark (the quickest entry — runs the full audit + prints the verdict):
```python
from karyon import molnet_honesty
molnet_honesty.run()            # BBBP (clf) + ESOL (reg): inflation + leakage-contract fire rates
```

Retrosynthesis, programmatic over your own split:
```python
from karyon import retro_honesty as rh
from karyon.uspto_data import load_reactions, random_split

split = random_split(load_reactions(), seed=0)
audit = rh.audit_split(split)
print(rh.contract_fire_rates(audit))     # e.g. {"reactant_seen": 0.83, ...} — the leak rate, named
```

`leakage_contracts()` (in both modules) is the reusable `contracts.ContractSet`; `.evaluate(...)` returns
a `Verdict` whose `.messages` name each leak.

## Composition with NVIDIA BioNeMo
Point it at a property model such as **KERMT** (the toolkit's ADMET GNN) or any retrosynthesis model:
audit the split *before* trusting the reported metric, so the agent reports "AUROC 0.90 on a random split,
0.79 scaffold-disjoint — 61% of test shares a train scaffold" instead of the inflated headline.

## Scope (honest)
This is **qualification, not accuracy** — it reveals *where* and *how much* a benchmark leaks, not a better
model. Effect size is substrate-dependent (large for retrosynthesis templates, moderate for ADMET scaffolds).
