---
name: promoter-design
description: >
  Functional + synthesizability design-rule check over σ70 bacterial promoter sequences with karyon — qualify whether a designed or generated promoter carries a real −35/−10 architecture before ordering, each rejection named. Use for promoter design rules, −35/−10 box check, inter-box spacer, promoter GC band, regulatory-element QC, or qualifying the output of a DNA/regulatory sequence generator (e.g. Evo2) before synthesis.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "karyon (stdlib only)"
allowed-tools: Bash, Read, Write
---

# Promoter Design DRC

A **deterministic design-rule check over a σ70 promoter sequence.** A model (or a human) proposes a
promoter; this gate qualifies whether it carries the architecture that actually drives transcription — and
names every defect — before anything is ordered. Pure stdlib, no model, no network.

Two tiers of contract (each returns a human-readable reason):

- **C1–C4 — hard, mechanism-grounded.** A `−35` box (`TTGACA`) and `−10` box (`TATAAT`) located by minimum
  Hamming over the spacer window, the inter-box spacer (15–19 nt, 17 optimal), and a GC band. These are
  *measured-validated*: promoters that flag as weak-box express significantly lower on the real Urtecho set
  (AUROC 0.66, box-OK vs weak), so a flag predicts lower function — not just "looks wrong."
- **C5–C6 — calibrated, dormant-by-correctness.** Homopolymer-run and rare-forbidden-motif limits read from
  a reference pool of *buildable* promoters, so natural tracts / scaffold sites stay silent and only an
  out-of-distribution run or a genuinely-introduced rare site fires.

## Install
```bash
pip install karyon          # no extras needed
```

## Usage
```python
from karyon import promoter_contracts as pc

# Hard rules only (uncalibrated): C1–C4 evaluate; C5–C6 fall back to safe defaults.
v = pc.DESIGN.evaluate(seq)                  # -> Verdict(ok, reasons, score)
if not v.ok:
    print(v.fired, v.messages)
    # e.g. (['C1 −35 box'], ["weak −35 box: best 'TTGCCA' is 2/6 mismatches from TTGACA"])

# Calibrated to a pool of known-buildable promoters (C5/C6 become substrate-relative):
ctx = pc.calibrate_design(reference_promoters)   # list[str] of deposited, buildable sequences
for seq in generated_promoters:                  # e.g. Evo2 output
    v = pc.DESIGN.evaluate(seq, ctx)
    if not v.ok:
        print(f"REJECT — {v.fired}: {v.messages}")
```

## Composition with NVIDIA BioNeMo
Install alongside **`evo2-nim`** (genomic / regulatory sequence generation): Evo2 proposes promoters, this
skill qualifies which ones carry a real −35/−10 architecture and a buildable sequence before synthesis — the
"unroutable net" report for a generated regulatory element. For non-promoter synthesizability (sgRNA GC /
homopolymer / Pol-III terminator), pair with the **`sequence-dfm`** skill.

## Scope (honest)
The box model is a legible best-arrangement scan (no PWM training), so the operating regime is the real σ70
pool and mutated-from-real designs, where the boxes are genuine; on fully-random sequences, chance boxes
limit discrimination. This is a *design-rule* gate (does the regulatory architecture exist, is it buildable),
not an expression-strength predictor — pair it with the karyon expression predictors for the quantitative axis.
