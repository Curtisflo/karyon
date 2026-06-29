---
name: single-cell-screen-qc
description: >
  Qualify a single-cell Perturb-seq screen's no-phenotype calls with karyon — flag perturbations called "no effect" where the guide never actually knocked the target down (a silent failure), so the null is an artifact, not biology. Use for Perturb-seq QC, single-cell CRISPR screen silent-failure / false-negative qualification, on-target knockdown reliability, low-cell-count power checks, or deciding whether a "no-phenotype" perturbation is a true negative or just a failed knockdown.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "karyon (your own screen, CSV/TSV); karyon[singlecell] (the bundled Replogle reference)"
allowed-tools: Bash, Read, Write
---

# Single-cell Perturb-seq Screen QC

A genome-scale Perturb-seq screen calls each perturbation a transcriptomic **hit** or **no-phenotype** via a
calibrated test (e.g. an energy-distance / SCEPTRE-style p-value). The no-phenotype side is ambiguous: a
perturbation shows no phenotype either because the gene knockdown genuinely has no consequence (a **true
negative**) or because the guide **never knocked the target down** (a **silent failure** — the null is an
artifact). Bulk screens can only *infer* this; Perturb-seq **measures on-target knockdown directly**, so the
silent-failure label is real.

This skill is the legible QC layer over that signal. It reads each no-phenotype call's own knockdown and cell
count and qualifies it with a **named reason** — `WEAK_KNOCKDOWN` (the guide failed → the null is
untrustworthy), `KNOCKDOWN_UNMEASURED` (target undetected → the null is unqualifiable), `LOW_CELL_COUNT`
(too few cells to call a phenotype). The phenotype caller is *consumed* (deposited), not reimplemented —
karyon's owned part is the reliability gate over its output.

On the **Replogle et al. 2022** K562-essential Perturb-seq screen, the layer flags **34% of no-phenotype
essential-gene calls as untrustworthy** (the gene is essential, the call is "no effect," but the knockdown
failed), at **|ρ| = 0.003** with the deposited significance — i.e. **non-redundant** with the p-value, not a
softer restatement of it — and corrects roughly **a quarter** of the negative calls. A knockdown-shuffle
control collapses the effect (3.1× → 1.0×). This is the sharpest form of the screen-QC thesis: the value is
largest *in-domain*, where the screen directly measures the thing the QC layer reasons about. Full method and
the bulk-vs-single-cell comparison:
[docs/screen-power.md](https://github.com/Curtisflo/karyon/blob/main/docs/screen-power.md).

## Install
```bash
pip install karyon                 # qualify YOUR OWN screen (CSV/TSV) — core install, no extras
pip install "karyon[singlecell]"   # + h5py, to reproduce the bundled Replogle reference (.h5ad)
```

## Usage — qualify your own screen (the common case)
A screen summary, **one row per perturbation** (CSV or TSV). Required columns: a target gene and a phenotype
p-value; optional: residual on-target expression (`0` = fully knocked down, `1` = unchanged) and a cell count.
Column names are matched case-insensitively (`target`/`gene`, `residual_expression`/`fold_expr`/`knockdown`,
`energy_test_p_value`/`pvalue`, `num_cells`/`cells`, `is_control` or a `non-targeting` target).

```bash
karyon audit screen --single-cell --input my_screen.csv          # human summary
karyon audit screen --single-cell --input my_screen.csv --json   # the stable JSON report
```

The report's centerpiece is `flagged` — every no-phenotype call you should not trust, each with its named
reasons, worst-knockdown first:

```
single-cell screen QC · my_screen.csv
  no-phenotype calls (silent-failure denominator): 490
  flagged untrustworthy: 34.1%  ·  weak-KD enrichment 3.1×  ·  |ρ| vs significance 0.003
  partition: 113 untrustworthy · 332 trustworthy negatives · 45 unmeasurable
  no-phenotype calls you should NOT trust (top 12 of 167):
     TMEM240   214% residual   energy-p 0.30, 181 cells  [WEAK_KNOCKDOWN]
     SON       128% residual   energy-p 0.089, 17 cells  [WEAK_KNOCKDOWN, LOW_CELL_COUNT]
     ...
```

From Python:
```python
from karyon import perturbseq_qc as pq
perts = pq.load_user_screen("my_screen.csv")     # → list[Perturbation]; or build them yourself
rep   = pq.audit_report(perts=perts)             # JSON-safe; rep["flagged"] is the per-call payload
```

## Usage — reproduce the reference
```bash
pip install "karyon[singlecell]"
karyon audit screen --single-cell --json         # the bundled Replogle K562-essential screen (~80 MB, cached)
python -m karyon.perturbseq_qc                    # the pre-registered P1–P4 verdict + the shuffle control
```

## Composition with NVIDIA BioNeMo
Runs **downstream** of a single-cell pipeline (e.g. **Parabricks** alignment/counts → a Perturb-seq DE /
phenotype caller). The pipeline calls hits; this skill qualifies the *no-phenotype* calls so the agent reports
"perturbation X: no phenotype, but the knockdown failed (82% residual) — distrust this negative" instead of a
bare "no effect."

## Scope (honest)
Qualification, not accuracy: it tells you which no-phenotype calls to *distrust*, it does not re-rank hits or
replace the phenotype caller. It needs the screen's own knockdown / cell metadata to bite; a table with only
target + p-value still runs but can only flag what those columns reveal.
