# karyon

[![CI](https://github.com/Curtisflo/karyon/actions/workflows/ci.yml/badge.svg)](https://github.com/Curtisflo/karyon/actions/workflows/ci.yml)

**A legible reliability / QC / qualification layer over commodity bio-AI tools.**

Modern bio-AI toolkits (structure prediction, docking, generative chemistry, genomics) are getting
powerful and cheap — NVIDIA's BioNeMo Agent Toolkit, for example, packages a decade of them as
ready-to-call agent skills. What they *don't* ship is a deterministic, independent gate that answers the
question that comes right after a model returns an answer:

> **Is this output trustworthy?** Is this docking pose physically valid? Is this benchmark number
> inflated by leakage? Is this "no-effect" screen result just under-powered? Is this generated sequence
> even synthesizable?

karyon is that gate. It is **not a model.** Every check is a legible, deterministic contract, and every
rejection **names its reason** — the "unroutable net" report, ported from EDA/CAD design-rule checking to
biology. It ships as a pip-installable Python library *and* as agent skills that compose alongside the
generative tools (install a karyon skill next to a BioNeMo skill; the model proposes, karyon qualifies).

## What the checks show

karyon's checks run on public benchmarks. None of these *problems* are discovered here — each is a known
reliability failure mode. karyon's contribution is to express each as a legible, named-reason contract,
cross-validate it against the reference tool where one exists, and make it agent-callable — plus one check
the incumbents skip. The headline numbers, with lineage:

- **70% of DiffDock's RMSD≤2 "successes" are physically invalid** — reproduces PoseBusters
  (Buttenschoen et al., *Chem. Sci.* 2024): deep-learning docking scores well on RMSD yet emits physically
  invalid poses (77% of DiffDock poses fail an inter-molecular check vs just 1% for classical Vina docking).
  karyon re-derives it as a deterministic geometric DRC (bond/angle/ring/clash/strain, zero fitted
  parameters) and agrees with the real PoseBusters package on 87% of poses (≥85% pre-registered).
- **Retrosynthesis "accuracy" is largely template memorization** — a known leakage concern in
  retrosynthesis benchmarking, quantified here on USPTO-50k: a faithful retrosim baseline scores top-1
  **37.9%** on the standard split but **16.1%** on a leakage-free partition (93.8% of the test set carries
  a near-duplicate or shared training template) — a measured **+21.8-point** inflation.
- **ADMET benchmark numbers inflate under random splits** — the reason MoleculeNet (Wu et al., *Chem. Sci.*
  2018) prescribes scaffold splits; karyon measures the gap directly: random-vs-scaffold lifts AUROC by
  **+0.105** (classification) and ρ by **+0.100** (regression).
- **CRISPR screens hide under-powered non-hits** *(the new check)* — incumbents (MAGeCK and kin) emit a
  gene-level hit/non-hit q-value and throw away the within-gene guide structure. karyon reads that structure
  back from counts alone, control-calibrated, and flags **~53%** of gold-standard silent failures at a
  **3%** false-flag rate — shown non-redundant with the FDR, not just a softer q-value. Full method +
  pre-registered evaluation: [docs/screen-power.md](docs/screen-power.md).

## Reproduce these numbers

Every figure above is printed by a `python -m karyon.<module>` entrypoint that fetches a public benchmark
and runs the audit — nothing is hand-entered, the printed value is the source of truth. Reproduce them all:

```bash
pip install "karyon[chem]"          # screen-qc needs only the core install
python examples/reproduce/run.py    # claim ↔ command ↔ reproduced value   (or: --list)
```

Per-claim commands, datasets, runtimes, and the offline (`KARYON_NO_NETWORK=1`) path are documented in
[`examples/reproduce/`](examples/reproduce).

## Install

```bash
pip install karyon                 # core (numpy, scikit-learn)
pip install "karyon[chem]"         # + rdkit, rdchiral  (pose validity, leakage audits)
pip install "karyon[seqdesign]"    # + dnachisel, ostir (sequence/expression predictors)
pip install "karyon[data]"         # + xlrd             (one Excel-backed dataset loader)
```

Installing karyon puts the **`karyon` CLI** on your PATH (`karyon qualify …`, `karyon audit …`,
`karyon list`).

Datasets are fetched on demand from public sources and cached under `~/.cache/karyon`
(override with `$KARYON_CACHE`). See [DATASETS.md](DATASETS.md).

## Quickstart

One surface — `karyon.qualify(artifact, modality)` — gates every modality and returns one stable result:

```python
from karyon import qualify

# Is this docking pose physically valid?
r = qualify("pose_1.sdf", modality="pose")          # .sdf infers "pose" (modality optional here)
print(r.ok, r.items[0][1].messages)

# Is this generated DNA sequence synthesizable?
r = qualify("GACCTTTTGCA...", modality="dna")
print("synthesizable" if r.ok else r.items[0][1].messages)
```

Same thing on the command line — exit 0 = PASS, 1 = FAIL, so it gates a pipeline directly:

```bash
karyon qualify pose_1.sdf --modality pose --json
karyon qualify diffdock_out/ --modality pose          # a whole directory of poses
karyon audit screen --json                            # a dataset-level audit (leakage / screen power)
```

Every verdict is JSON-serializable with named reasons (the stable schema — see
[docs/qualify.md](docs/qualify.md)):

```json
{"modality": "pose", "ok": false,
 "items": [{"name": "pose_1.sdf", "ok": false, "score": 1.5,
            "reasons": [{"contract": "INTERNAL_STERIC_CLASH", "message": "…", "weight": 1.5}]}],
 "batch": null}
```

## Agent skills

v0.3 ships skills spanning the major modalities a generative toolkit touches — docking and structure
prediction (poses, co-folding, complex interfaces), generative chemistry and DNA, functional-genomics
screens, and sequence/regulatory design. It's a cross-section that proves the contract pattern generalizes,
not exhaustive coverage; the library underneath carries more checks than the marquee skills, and the roadmap
wraps more of them over time.

Each skill is a `SKILL.md` (YAML frontmatter + instructions) installable into Claude Code, Codex, and
other harnesses via the [`skills` CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add Curtisflo/karyon --skill pose-validity --agent claude-code
```

| Skill | What it qualifies | Composes with (BioNeMo) |
|-------|-------------------|-------------------------|
| [`pose-validity`](skills/pose-validity) | physical validity of docking poses (single-molecule / intramolecular) | `diffdock-nim`, `boltz2-nim`, `openfold3-nim` |
| [`cofold-qc`](skills/cofold-qc) | physical validity of co-folding poses (protein↔ligand, intermolecular) | `boltz2-nim`, `diffdock-nim`, `openfold3-nim` |
| [`complex-qc`](skills/complex-qc) | interface validity of protein complexes / designed binders | `rfdiffusion`, `proteinmpnn`, AlphaFold-Multimer |
| [`mol-qc`](skills/mol-qc) | validity / synthesizability of generated molecules | `genmol-nim`, `molmim` |
| [`gen-dna-qc`](skills/gen-dna-qc) | synthesizability / manufacturability of generated DNA | `evo2-nim` |
| [`benchmark-leakage`](skills/benchmark-leakage) | train/test leakage in a model's benchmark | `kermt`, retrosynthesis models |
| [`screen-qc`](skills/screen-qc) | under-powered non-hits in a CRISPR screen | `parabricks` (downstream) |
| [`promoter-design`](skills/promoter-design) | σ70 promoter architecture (−35/−10 boxes, spacer, GC), reference-calibrated | `evo2-nim` |

## Agent self-repair loop

Because every rejection **names its reason**, a named reason is a *repair instruction* — an agent can read
it and make the corresponding edit, then re-check. A black-box pass/fail can't drive that loop; a legible
one can. `karyon.repair` closes it: **generate → qualify → fix-from-reasons → re-qualify → converge.**

```bash
python examples/agent_loop/repair_dna.py     # watch the loop converge (pure stdlib, no API)
karyon repair my_draft.fasta -m dna --json   # repair your own draft via the CLI
```

```
repair loop · dna · CONVERGED in 3 edit(s)
  round 0: FAIL  [GC_OUT_OF_BAND, HOMOPOLYMER_RUN, RESTRICTION_SITE]  ↳ broke a 14-base homopolymer run at 46
  round 1: FAIL  [GC_OUT_OF_BAND, RESTRICTION_SITE]                   ↳ rebalanced GC 22%→32% into the band
  round 2: PASS  [RESTRICTION_SITE]                                  ↳ removed the EcoRI site at 80
  round 3: PASS  [clean]
```

The bundled `DnaRepairAgent` / `MolRepairAgent` make the loop runnable and CI-tested with **no LLM**. In real
use the agent is *your harness* — e.g. **Claude Code in your terminal, no API key**: it writes a candidate,
runs `karyon qualify`, reads the named reasons, edits, re-runs until PASS. That's the whole thesis — *legible
QC is what makes agentic self-repair possible*. See [`examples/agent_loop/`](examples/agent_loop) and
[`docs/repair.md`](docs/repair.md).

## Does qualifying compound? (a DBTL-loop demonstration)

The self-repair loop above gates *one artifact*. The legible **operator** (`dbtl_operator`) gates a whole
**design-build-test-learn loop** — it qualifies each measured readout before folding it into its surrogate,
so a corrupt or under-powered measurement is excluded from the model update, with a reason. On a headroom
substrate with a model-degrading (synthetic) assay, that protection **compounds over recursive cycles**: the
gated arm's held-out-ρ advantage *widens* cycle-over-cycle — a quality edge (keep bad labels out) runs away
where merely spending less budget would saturate — **but only once the tool is unreliable enough.** Below a
reliability crossover the gate is net-costly (it drops good data too); above it, it pays and compounds. Full
method, the crossover table, and the negative controls: [docs/compounding.md](docs/compounding.md).

```bash
python -m karyon.operator_compound_honesty --seeds 8   # the pre-registered test + reliability-crossover sweep
```

## Library layout

```
src/karyon/
  spine.py            the qualify spine — qualify(artifact, modality) -> QualifyResult over every gate
  repair.py           the agent self-repair loop — generate -> qualify -> fix-from-reasons -> converge
  cli.py              the `karyon` command-line entry point (qualify / repair / audit / list)
  contracts.py        the legible verdict engine (named contracts -> Verdict with reasons)
  pose_validity.py    cofold_validity.py  protein_interface_validity.py   structural-validity DRCs (pose / co-fold / complex interface)
  mol_qc.py           gen_dna_validity.py   generated-output DRCs (molecule validity & SA / DNA synthesizability)
  retro_honesty.py    molnet_honesty.py   benchmark leakage audits
  screen_qc.py        crispr_qc.py        CRISPR screen / guide QC
  loop.py             dbtl_operator.py    a legible design-build-test-learn loop + operator
  operator_compound.py  noisy_assay.py    does readout-qualification compound over recursive cycles?
  *_data.py           on-demand loaders for public benchmark datasets
skills/               the SKILL.md agent skills
tests/                the test suite
```

## What this is not

karyon does not predict structures, dock ligands, or generate molecules — it **qualifies** the output of
tools that do. Its value is legibility and trust, not accuracy. Pair it with a generative toolkit
(e.g. BioNeMo) for the soft, quantitative axis; use karyon for the deterministic, auditable one.

## License

Dual-licensed: code under [Apache-2.0](LICENSE-APACHE-2.0), skills/docs under
[CC-BY-4.0](LICENSE-CC-BY-4.0). See [LICENSE](LICENSE).
