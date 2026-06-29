# Reproduce the headline numbers

Every figure in the top-level README is produced by a `python -m karyon.<module>`
entrypoint that fetches a **public** benchmark, runs a deterministic named-reason
contract audit, and prints a pre-registered verdict. Nothing here is hand-entered —
the printed value is the source of truth.

```bash
pip install "karyon[chem]"            # pose / retro / admet need rdkit; screen-qc does not
python examples/reproduce/run.py      # run all four and print claim ↔ command ↔ reproduced value
python examples/reproduce/run.py --list
python examples/reproduce/run.py screen-qc      # run one by id
```

Datasets are downloaded on first run and cached under `$KARYON_CACHE`
(default `~/.cache/karyon`); re-runs read the cache. Dataset sources, citations,
and licenses are in [`../../DATASETS.md`](../../DATASETS.md). To prove the offline
path, set `KARYON_NO_NETWORK=1` — cached datasets still load, uncached ones raise
`DatasetUnavailable` instead of fetching (this is also what CI uses).

## The six claims

| id | command | install | dataset |
|----|---------|---------|---------|
| `pose-validity` | `python -m karyon.pose_honesty` | `karyon[chem]` | Zenodo / PoseBench (Buttenschoen 2024) |
| `retro-leakage` | `python -m karyon.retro_template` | `karyon[chem]` | retrosim USPTO-50k (Coley) |
| `admet-leakage` | `python -m karyon.molnet_honesty` | `karyon[chem]` | MoleculeNet / DeepChem (Wu 2018) |
| `screen-qc` | `python -m karyon.screen_qc --seeds 50` | core | MAGeCK demo (Wang 2014) + hart-lab CEGv2/NEGv1 |
| `single-cell-screen-qc` | `python -m karyon.perturbseq_qc` | `karyon[singlecell]` | Replogle 2022 K562-essential Perturb-seq (Figshare+) |
| `ppi-leakage` | `python -m karyon.ppi_leakage` | core | Guo-yeast sequence-based PPI (PIPR / seq_ppi) |

### `retro-leakage` — template-memorization inflation (deterministic, `seed=0`)
```
standard split        : top1=37.9% ...
leakage-free partition: top1=16.1% ...   (n=93, 6% survives)
MEASURED inflation (standard − leakage-free) top-1 = +21.8%
ANY (leaked)              93.8%
```
A faithful retrosim baseline (Morgan-Tanimoto NN + RDChiral templates) scores top-1
**37.9%** on the standard split but **16.1%** once near-duplicate / shared-template
test reactions are removed — **+21.8 points** of inflation, with **93.8%** of the
standard test set carrying some leakage. The leakage-free partition is small (n≈93),
but the run is seeded end-to-end, so the figure is stable. Template extraction takes
~2 min on the first run (cached afterwards).

### `admet-leakage` — random-vs-scaffold inflation (deterministic)
```
BBBP  INFLATION (random − scaffold) = +0.105   (AUROC)
ESOL  INFLATION (random − scaffold) = +0.100   (ρ)
```
The gap MoleculeNet's scaffold split exists to prevent, measured directly:
**+0.105** AUROC (classification, BBBP) and **+0.100** ρ (regression, ESOL).

### `screen-qc` — under-powered non-hits *(the new check)*
```
Q1 recall (CEGv2 silent failures flagged) : ~53%   PASS (>50%)
Q2 false-flag (held-out NEGv1 non-hits)   : ~3%    PASS (<20%)
Q3 |ρ(under-power, baseline −log10 q)|     : ~0.29  PASS (<0.60)  ← non-redundant with the FDR
```
Reads within-gene guide structure back from counts alone (control-calibrated) and
flags **~53%** of gold-standard silent failures (CEGv2 essentials the baseline
missed) at a **~3%** false-flag rate on a held-out negative set. Q1 averages over
NEGv1 calibration/eval splits and needs ~25+ seeds to converge; the module default
is 50, so the headline reproduces at the default invocation.

### `single-cell-screen-qc` — failed-knockdown nulls *(in-domain Perturb-seq)*
```
B1 incumbent calibration (deposited energy-test): targeting 78% hit vs controls 15%  (credible)
Q1 flagged among no-phenotype:        34.1%   (bar ≥15%)   <- P1
Q3 |ρ(knockdown, energy-p)| in no-hit: 0.003  (bar <0.30, non-redundancy)   <- P3
Q4 weak-KD enrichment no-hit vs hit:  3.1×   <- P4   (shuffle control → ~1.0×)
```
The sharpest form of the screen-QC thesis. On the Replogle K562-essential Perturb-seq
screen, the deposited energy-test caller is credible (78% vs 15%); within its
**no-phenotype** pile, karyon flags **34%** of essential-gene "no-effect" calls as
untrustworthy (the guide failed to knock the target down), at **|ρ| = 0.003** with the
deposited significance — i.e. **non-redundant**, not a softer restatement of the
p-value. A knockdown-shuffle control collapses the enrichment **3.1× → ~1.0×**. Needs
`karyon[singlecell]` (h5py) and downloads the ~80 MB pseudobulk on first run; the QC
layer itself is stdlib. (You can run the same gate on your own screen with
`karyon audit screen --single-cell --input your_screen.csv` — core install, no h5py.)

### `ppi-leakage` — node-identity inflation in PPI prediction *(pair-input)*
```
node-leakage prevalence (≥1 partner seen): 85.0%   <- P1
leakage class    n      pos%   AUROC(node)
  C1 both seen           AUROC 0.774
  C3 neither             AUROC 0.500   (exactly chance)
mean node inflation +0.275  (reported≈C1 0.774 → honest C3 0.500)
```
Sequence-based PPI benchmarks report on random *pair* splits, where the same
proteins straddle train and test (Park & Marcotte 2012). The audit assigns every
test pair its C1/C2/C3 leakage class and scores a transparent node-degree-memorization
baseline per class: it reaches AUROC **0.77** on the reported (both-proteins-seen)
eval and collapses to **0.50 — exactly chance** on the honest neither-seen eval, a
**+0.27** node-identity inflation with **~85%** of the test set leaking. Pure stdlib
(no rdkit); `--seeds 3` by default, stable across seeds. Shuffling the labels
collapses the inflation to ~0 (the harness can't manufacture it).

### `pose-validity` — physically-invalid docking "successes"
```
B1 INFLATION — of 54 RMSD≤2 'successes': 70% physically INVALID (intra 15% | inter 65%)
P2 faithful   min per-pose intra agreement vs PoseBusters 87% ≥ 85%
DL-vs-classical: physically-invalid (inter) rate DiffDock 77% vs Vina 1% (Δ +75%)
```
karyon re-derives the PoseBusters result as a deterministic geometric DRC
(bond/angle/ring/clash/strain, zero fitted parameters): **70%** of DiffDock's
RMSD≤2 "successes" are physically invalid, the failure localizes to placement
(inter 77% vs intra 5%) so classical Vina docking stays clean (1%), and the legible
DRC agrees with the reference PoseBusters package on **87%** of poses (≥85%
pre-registered). Defaults to a deterministic 150-pose head-slice per method;
downloads ~11 MB on the first run, and the cross-check against the reference
package is the slow step (several minutes).
