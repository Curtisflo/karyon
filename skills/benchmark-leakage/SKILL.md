---
name: benchmark-leakage
description: >
  Audit an ML benchmark for data leakage with karyon — measure how much a reported accuracy is inflated by train/test contamination. Use for leakage detection, data-leakage audit, scaffold leakage, template memorization, random-vs-scaffold split inflation, protein-protein interaction (PPI) node-identity leakage, retrosynthesis / molecular-property / PPI benchmark honesty, or deciding whether a model's headline number is real.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "karyon (ppi — stdlib); karyon[chem] (retro, admet — rdkit, rdchiral)"
allowed-tools: Bash, Read, Write
---

# Benchmark Leakage Audit

A **legible, model-free audit** that asks whether a benchmark number is inflated by train/test leakage,
and names the mechanism. Three substrates ship:

- **Retrosynthesis** (`retro_template`) — flags test reactions whose transform template or near-duplicate
  was already seen in training. On USPTO-50k a faithful retrosim baseline scores top-1 **37.9%** on the
  standard split but **16.1%** on a leakage-free partition (93.8% of the test set carries leakage) — a
  measured **+21.8-point** inflation.
- **Molecular property / ADMET** (`molnet_honesty`) — flags test molecules sharing a Bemis-Murcko scaffold
  with training. Random-vs-scaffold split inflation is real on both a classifier (AUROC +0.105) and a
  regressor (ρ +0.100).
- **Protein–protein interaction (PPI)** (`ppi_leakage`) — the pair-input case: flags test pairs whose
  proteins were already seen in training (Park–Marcotte C1/C2/C3 stratification). On the Guo-yeast benchmark
  a transparent node-degree-memorization baseline scores AUROC **0.77** on the reported (both-seen) eval and
  **0.50 — exactly chance** on the honest neither-seen eval: a **+0.27** node-identity inflation, ~85% of the
  test set leaking. **Pure stdlib (no rdkit).**

## Install
```bash
pip install karyon              # the ppi audit is core-only (stdlib)
pip install "karyon[chem]"      # + rdkit, rdchiral — for the retro / admet audits
```

## Usage
From the command line (`karyon audit leakage`):
```bash
karyon audit leakage --benchmark uspto50k --json   # retrosynthesis (template / near-duplicate leakage)
karyon audit leakage --benchmark bbbp --json       # MoleculeNet BBBP (scaffold leakage, classification)
karyon audit leakage --benchmark esol --json       # MoleculeNet ESOL (scaffold leakage, regression)
karyon audit leakage --benchmark ppi --json        # Guo-yeast PPI (node-identity leakage, pair-input; core)
```
The JSON report carries the inflation, leakage prevalence, and per-contract fire rates. (Public datasets are
fetched on first run.)

Programmatic — molecular property benchmark (runs the full audit + prints the verdict):
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
Point it at a property model such as **KERMT** (the toolkit's ADMET GNN), any retrosynthesis model, or a
sequence-based **PPI** predictor: audit the split *before* trusting the reported metric, so the agent reports
"AUROC 0.90 on a random split, 0.79 scaffold-disjoint — 61% of test shares a train scaffold" (or, for PPI,
"0.77 reported vs 0.50 on neither-seen pairs") instead of the inflated headline.

## Scope (honest)
This is **qualification, not accuracy** — it reveals *where* and *how much* a benchmark leaks, not a better
model. Effect size is substrate-dependent (large for retrosynthesis templates and PPI node-identity, moderate
for ADMET scaffolds).
