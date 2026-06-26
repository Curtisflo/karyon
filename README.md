# karyon

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

- **71% of DiffDock's RMSD≤2 "successes" are physically invalid** — reproduces PoseBusters
  (Buttenschoen et al., *Chem. Sci.* 2024): deep-learning docking scores well on RMSD yet emits physically
  invalid poses, while classical docking (Vina) stays valid. karyon re-derives it as a deterministic
  geometric DRC (bond/angle/ring/clash/strain, zero fitted parameters) and cross-checks ≥85% per-pose
  agreement against the real PoseBusters package.
- **Retrosynthesis "accuracy" is largely template memorization** — a known leakage concern in
  retrosynthesis benchmarking, quantified here on USPTO-50k: top-1 is 43.5% on seen templates vs 11.0% on
  novel ones — a measured **+25.4-point** inflation.
- **ADMET benchmark numbers inflate under random splits** — the reason MoleculeNet (Wu et al., *Chem. Sci.*
  2018) prescribes scaffold splits; karyon measures the gap directly: random-vs-scaffold lifts AUROC by
  **+0.105** (classification) and ρ by **+0.100** (regression).
- **CRISPR screens hide under-powered non-hits** *(the new check)* — incumbents (MAGeCK and kin) emit a
  gene-level hit/non-hit q-value and throw away the within-gene guide structure. karyon reads that structure
  back from counts alone, control-calibrated, and flags **~53%** of gold-standard silent failures at a
  **3%** false-flag rate — shown non-redundant with the FDR, not just a softer q-value.

## Install

```bash
pip install karyon                 # core (numpy, scikit-learn)
pip install "karyon[chem]"         # + rdkit, rdchiral  (pose validity, leakage audits)
pip install "karyon[seqdesign]"    # + dnachisel, ostir (sequence/expression predictors)
pip install "karyon[data]"         # + xlrd             (one Excel-backed dataset loader)
```

Datasets are fetched on demand from public sources and cached under `~/.cache/karyon`
(override with `$KARYON_CACHE`). See [DATASETS.md](DATASETS.md).

## Quickstart

```python
# Is this docking pose physically valid?
from rdkit import Chem
from karyon import pose_validity as pv
cs, tol = pv.validity_contracts(), pv.Tol()
verdict = cs.evaluate(pv.featurize(Chem.MolFromMolFile("pose_1.sdf"), tol), tol)
print("valid" if verdict.score == 0 else f"INVALID — {verdict.messages}")

# Is this generated DNA sequence synthesizable?
from karyon import crispr_qc
print(crispr_qc.hard_contracts("GACCTTTTGCA..."))   # [] == clean; else named reasons
```

## Agent skills

v0.1 ships skills spanning the major modalities a generative toolkit touches — docking, cheminformatics,
functional-genomics screens, and sequence/regulatory design. It's a cross-section that proves the contract
pattern generalizes, not exhaustive coverage; the library underneath carries more checks than the marquee
skills, and the roadmap wraps more of them over time.

Each skill is a `SKILL.md` (YAML frontmatter + instructions) installable into Claude Code, Codex, and
other harnesses via the [`skills` CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add Curtisflo/karyon --skill pose-validity --agent claude-code
```

| Skill | What it qualifies | Composes with (BioNeMo) |
|-------|-------------------|-------------------------|
| [`pose-validity`](skills/pose-validity) | physical validity of docking poses | `diffdock-nim`, `boltz2-nim`, `openfold3-nim` |
| [`benchmark-leakage`](skills/benchmark-leakage) | train/test leakage in a model's benchmark | `kermt`, retrosynthesis models |
| [`screen-qc`](skills/screen-qc) | under-powered non-hits in a CRISPR screen | `parabricks` (downstream) |
| [`sequence-dfm`](skills/sequence-dfm) | synthesizability of generated DNA sequences | `evo2-nim`, `genmol-nim` |
| [`promoter-design`](skills/promoter-design) | σ70 promoter architecture (−35/−10 boxes, spacer, GC), reference-calibrated | `evo2-nim` |

## Library layout

```
src/karyon/
  contracts.py        the legible verdict engine (named contracts -> Verdict with reasons)
  pose_validity.py    physical-validity DRC for docking poses
  retro_honesty.py    molnet_honesty.py   benchmark leakage audits
  screen_qc.py        crispr_qc.py        CRISPR screen / guide QC
  loop.py             dbtl_operator.py    a legible design-build-test-learn loop + operator
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
