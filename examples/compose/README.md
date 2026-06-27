# Compose with a generative model

karyon is the **qualifier**, not the generator: a model proposes, karyon qualifies
with a deterministic named-reason verdict, and the agent acts on it. These demos
make that loop runnable — **no GPU, no NVIDIA NIM, no network** — with synthetic
stand-ins for the generative step.

```bash
pip install "karyon[chem]"          # demo.py needs rdkit; seq_demo.py is core-only
python examples/compose/demo.py     # pose validity  × diffdock-nim / boltz2-nim
python examples/compose/seq_demo.py # sequence DFM    × evo2-nim / genmol-nim
```

## `demo.py` — docking poses (× `diffdock-nim`)

`make_candidates.py` synthesizes three ranked candidate poses for one ligand,
standing in for a DiffDock/Boltz-2 NIM emission: it builds a valid conformer and
plants defects with karyon's own decoy generators. The model's **rank-1** pose
(highest confidence) is the broken one — confidence and physical validity are
different axes. The demo runs the pose-validity skill's real tool over them
([`skills/pose-validity/scripts/qualify_poses.py`](../../skills/pose-validity/scripts/qualify_poses.py))
and plays the agent that keeps the survivors:

```
2. karyon qualifies — pose-validity DRC over the proposed poses:
   [REJECT] pose_1.sdf — a bond is 499% of its reference length (>125%); an aromatic ring is
            non-planar (...); two non-bonded atoms clash (15% of their van-der-Waals sum, <60%); ...
   [REJECT] pose_2.sdf — a bond is 160% of its reference length (>125%)
   [valid ] pose_3.sdf

3. agent acts — 1/3 poses survive qualification.
   the model's #1-by-confidence pose (pose_1.sdf) is physically INVALID
   (BOND_LENGTH_OUTLIER; AROMATIC_RING_NONPLANAR; INTERNAL_STERIC_CLASH; INTERNAL_STRAIN_ENERGY).
   trusting confidence alone would have shipped it; the qualified pick is pose_3.sdf.
```

In production you delete `make_candidates.py` and point `qualify_poses.py` at the
directory the NIM actually wrote: `python qualify_poses.py diffdock_out/ --json`.

## `seq_demo.py` — generated DNA (× `evo2-nim` / `genmol-nim`)

The same loop in a second modality, core-deps only: a generative DNA model proposes
sequences, karyon's sequence-DFM gate (`crispr_qc.hard_contracts`) qualifies them
for synthesis and names every rejection.

```
  [valid ] seq_1  GACCTGCAGTACGTACGTAC
  [REJECT] seq_2  GACCTTTTGCAGTACGTACG — TTTT: Pol-III terminator truncates the sgRNA
  [REJECT] seq_3  GACCGGGGGGGGGGTACGTA — homopolymer run 10: synthesis/folding risk

agent acts — order 1/3 for synthesis: seq_1.
```

## Honest note

The candidate poses and sequences are **synthetic stand-ins** so the demo is
self-contained — in real use they come from the generative model. The qualifier is
not a stand-in: `qualify_poses.py` and `crispr_qc.hard_contracts` are the exact code
the skills install and run. The headline benchmark numbers (where karyon's checks
meet *real* model output on public data) are reproduced in
[`../reproduce/`](../reproduce).
