# Does qualifying compound? — a DBTL-loop demonstration

karyon's per-artifact gates answer "is this one output trustworthy?". The legible **operator**
(`dbtl_operator`) puts a gate *inside a closed design-build-test-learn loop*: it qualifies each readout
before folding it into its surrogate, so a measurement that fails the readout contracts (built /
replicate-CV / dynamic-range / controls — `promoter_contracts` QA–QD) is excluded from the model update,
with a named reason. A single round of that is "qualifying protects the model."

This doc asks the **loop-dynamics** question: does that protection *compound* over recursive cycles, or
merely establish itself early? The answer is **yes — but conditionally**, and the condition is itself the
useful finding.

> The headline: a quality edge (keep corrupt labels out of the model) **compounds**; a quantity edge
> (just spend less budget) saturates. Qualifying is the quality edge — but only once the underlying tool
> is unreliable enough to be worth the data it drops.

## The setup

- **Substrate.** The σ70 promoter pool (~8,700 designs, ~435 winners, ~2,200 held-out). It is
  *mid-learnability* and large enough that recall stays well below ceiling — so a gap between two arms has
  room to grow rather than saturating.
- **The single variable — the readout gate.** Two arms run the same operator on the same corrupted assay:
  **GATED** qualifies readouts (the QA–QD contracts), **UNGATED** ingests everything. The design DRC is off
  on both, so qualification is the only difference.
- **A model-degrading, imperfect failure mode** (`noisy_assay`). A corrupted readout gets a **wrong value**
  (sign-flipped about the pool mean — a strong design reads weak, so an ingested label *misleads* the
  surrogate) and, separately, the QC metadata the (label-blind) gate reads — with explicit **sensitivity**
  (only ~75% of corrupted wells are detectably flagged, so some poison slips through even the gate) and
  **specificity** (~5% of clean wells are falsely flagged, so qualifying also drops good data — a real cost).
  The gate is a realistic, imperfect filter, not an oracle.
- **The metric is held-out ρ on clean truth** — the gate's effect on model quality, unconfounded. (Recall
  is reported only as a diagnostic: it credits a design as "found" once *measured*, which rewards the ungated
  arm for corrupt-measured winners and penalizes the gate for dropping them — the wrong signal here.)

## The result (8 seeds)

**The reliability crossover** (uniform/bulk corruption). The gated−ungated held-out-ρ gap, by how
unreliable the assay is:

| corruption rate | ρ-gap slope / cycle | final ρ-gap | gated ρ | ungated ρ | read |
|---|---|---|---|---|---|
| 0% | +0.000 | +0.000 | +0.24 | +0.24 | clean anchor — the gate is inert (no-regression) |
| 30% | −0.003 | −0.034 | +0.21 | +0.24 | **net-negative** — the data dropped costs more than the poison avoided |
| 45% | +0.003 | +0.040 | +0.22 | +0.18 | break-even / slight win |
| 60% | **+0.015** | **+0.164** | +0.17 | **+0.00** | **net-positive and compounding** — the ungated model's ρ collapses |

In the model-degrading regime the gap **widens cycle-over-cycle** (positive slope; final gap ≫ early gap) —
the ungated surrogate, fed corrupt labels, makes worse acquisitions whose labels poison it further, while
the gated surrogate stays honest. That is the compounding (flywheel) signature: a quality edge runs away,
where merely spending less budget would have front-loaded and saturated.

**Mode contrast.** The corruption has to move the metric it's measured on. **Bulk** corruption degrades the
*global* model, so the ρ-protection compounds. **Top-biased** corruption (only high-expressers saturate the
reader) barely moves global ρ and so does not compound on it — its protective effect lands on top-discovery
instead.

**Controls.** The **shuffle** negative control places the gate's flags on a set *disjoint* from the actual
corruption (flags ⊥ poison); the compounding collapses — in fact qualifying becomes net-costly, since it
then only drops good data. The **clean** anchor (0% corruption) is exactly flat. So the compounding requires
the gate's flags to genuinely track the failure — it is not an artifact of dropping data.

## Honest bounds

- **Conditional, and on a high effective corruption fraction.** Compounding needs the degrading regime
  (here ≈45–60% bulk corruption under a *mild* sign-flip severity); a more severe corruption model would
  move the crossover lower. The **shape** — a crossover, then compounding — is the result, not the exact
  threshold, which is config- and severity-dependent.
- **Synthetic corruption, deposited oracle.** This is a demonstration of *loop dynamics* under a realistic
  corruption distribution, not a wet-lab result. The oracle is the deposited promoter strength.
- **Qualification, not accuracy.** Consistent with the rest of karyon: the value is keeping the model honest
  and legible, not beating a predictor — and the gate is *net-costly* where the tool is reliable enough,
  which the crossover makes explicit.

## Reproduce

```bash
python -m karyon.operator_compound_honesty --seeds 8     # the pre-registered test + the crossover sweep
python -m karyon.operator_compound --rate 0.60 --modes random,shuffle,clean   # a legible per-pair view
pytest tests/test_operator_compound.py                   # the six falsification proofs
```

Needs the σ70 promoter dataset (fetched once, then cached under `$KARYON_CACHE`); everything skips cleanly
offline.
