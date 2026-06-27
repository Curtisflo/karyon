# Screen-power QC: flagging under-powered non-hits in a CRISPR screen

> Methods note for the `screen_qc` check and the `screen-qc` skill. This is the one
> contribution in karyon that is *not* a reproduction of a published critique — the
> incumbents don't compute it. Reproduce everything here with
> `python -m karyon.screen_qc --seeds 50`.

## The gap

A pooled CRISPR screen is called by a tool like **MAGeCK** (or BAGEL, drugZ, …),
which reduces each gene to a **gene-level q-value**: *hit* or *non-hit*. That
single scalar is an FDR over the gene's guides, and it throws away the *within-gene
guide structure*. As a result a "non-hit" is **two different things fused into one
label**:

1. a **true negative** — the gene's guides were abundant and consistent and showed
   no depletion, so "no phenotype" is trustworthy; or
2. an **under-powered silent failure** — the screen simply lacked the power to call
   this gene (guides at the read-count floor, too few usable guides, or a real
   signal diluted by dead guides), so "non-hit" should *not* be trusted.

No standard caller tells you which non-hits are which. Downstream, a silent failure
reads as a confident negative — a gene quietly dropped from a hit list. That is the
gap `screen_qc` fills: a deterministic, named-reason qualifier over the caller's
**non-hit pile**.

## The check

For every gene the baseline called a non-hit, read the guide-level structure back
**from the counts alone**, calibrated on control / non-essential guides and **never
on the gold standard**. Three deterministic contracts fire a named reason; an empty
reason set means *trustworthy true-negative*:

| Contract | Fires when | Reason emitted |
|---|---|---|
| **A — count-power floor** | ≥ 50% (`FLOOR_FRAC`) of the gene's guides are below a 30-read initial floor (`MIN_INITIAL`) — too sparse to show dropout | `"{n}/{N} guides below 30-read floor (can't show dropout)"` |
| **B — guide scarcity** | fewer than 3 usable guides (`MIN_USABLE_GUIDES`, ≥ 30 reads) | `"only {k} usable guides ≥30 reads (under-powered)"` |
| **C — strong-but-diluted** | usable guides disagree (LFC spread above the non-essential dispersion baseline) **and** ≥ 1 guide cleared the null band | `"guides disagree (sd={s}; {m} cleared null) — diluted signal"` |

Calibration is label-free:
- **Null band** = `mean − 3σ` (`NULL_K`) of the calibration (control / non-essential)
  guide LFCs — the threshold beyond which a guide clears the non-essential null.
- **Dispersion baseline** = `median + 3·MAD` (`DISP_K`) of the within-gene usable-LFC
  spread among calibration genes — "how much do healthy genes' guides normally
  disagree?"

The continuous **under-power score** = `floor_fraction + guide_scarcity +
dispersion_term`. Every threshold is a field-standard constant or derived from the
non-essential calibration half — none is tuned to recover more gold-standard
essentials.

The contracts measure **guide power**, which is *independent of essentiality*. This
is the load-bearing distinction: a gene-level "is this gene essential?" call is
confounded (you can't tell a non-essential gene from a powerless assay), but "did
the screen have power *here*?" is answerable from counts without knowing the answer.

## Evaluation (pre-registered)

Thresholds and guards were fixed **before** the end-to-end run. Data, all public
(see [`../DATASETS.md`](../DATASETS.md)):

- **Counts** — the MAGeCK demo screen, leukemia dropout (Wang et al. 2014).
- **Reference sets** — hart-lab **CEGv2** (core-essential genes; the silent-failure
  gold standard *among the baseline's non-hits*) and **NEGv1** (non-essential genes;
  split gene-disjoint into a calibration half and a held-out evaluation half — the
  double-use guard, so the false-flag rate is never measured on calibration data).

| Guard | Question | Result | Threshold |
|---|---|---|---|
| **Q1 recall** | of CEGv2 essentials the baseline missed (silent failures), how many are flagged under-powered? | **~53%** | > 50% |
| **Q2 false-flag** | on the held-out NEGv1 non-hits, how often does the flag fire wrongly? | **~3%** | < 20% |
| **Q3 non-redundancy** | \|ρ(under-power score, baseline −log10 q)\| — is this just a softer FDR? | **≈0.29** | < 0.60 |
| **Q4 lift (honest)** | does a parameter-free rank-combine of (baseline, QC) beat baseline-alone AUPRC for essential recovery in the non-hit pile? | **no lift** (bootstrap CIs overlap) | reported either way |

Q1 is the headline: **about half** of the screen's silent failures carry a legible
under-power reason, at a **~3%** false-flag rate, and Q3 confirms the flag is **not a
restatement of the q-value** (a low correlation with the baseline FDR — it adds
information the scalar discarded). Q4 is reported as a negative honestly: within this
single screen the QC score does not improve ranked essential recovery over the
baseline; the value is the *legible non-hit triage*, not a better ranker.

> **Seed note.** Q1/Q2/Q3 average over random NEGv1 calibration/evaluation splits.
> The Q1 recall is split-sensitive and needs ~25+ seeds to converge (~53%); too few
> (e.g. 3) is noisy (~48%) and can spuriously fail the >50% gate. The module default
> is `--seeds 50` so the headline reproduces at the default invocation.

## Reproduce

```bash
pip install karyon                       # core only — no rdkit needed
python -m karyon.screen_qc --seeds 50    # prints the Q1–Q4 verdict + example flagged genes
```

Datasets download once and cache under `$KARYON_CACHE` (`~/.cache/karyon`); set
`KARYON_NO_NETWORK=1` to force the offline path. The same Q1–Q4 machinery runs on
any bulk screen — pass your own counts and reference sets to
`karyon.screen_qc.run(sc=..., refs=...)`.

## Scope and limits (honest)

- It is a **QC layer over a caller's output**, not a replacement caller and not an
  essentiality predictor. It only qualifies the baseline's *non-hits*.
- Gene-level "under-powered" can correlate with non-essentiality. The check
  deliberately scores *guide power* rather than phenotype to avoid that confound —
  and we disclose the confound rather than paper over it: a sequence-only prediction
  of observed gene power is ≈ chance (AUROC ~0.54), precisely because most
  "under-powered" non-hits are simply non-essential genes with no phenotype to
  detect. Guide-power QC reads the *counts*, not the sequence, for exactly this
  reason.
- v0.1 validates on one public screen (Wang 2014). The contracts are screen-agnostic;
  broader validation across screens is future work.

## Relation to prior work

MAGeCK / BAGEL / drugZ and kin produce the gene-level hit/non-hit FDR this layer sits
*on top of*; it consumes a caller's output and qualifies the non-hits, it does not
re-call the screen. Replicate-concordance and guide-count QC exist as screen
diagnostics, but they are screen-wide health checks; `screen_qc` is **per-gene, from
counts, with a named reason per flag, and a pre-registered test (Q3) that the flag is
non-redundant with the FDR** — the "unroutable net" report, ported from EDA design-rule
checking to screen analysis.
