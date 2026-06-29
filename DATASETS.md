# Datasets

karyon's loaders **fetch public benchmark datasets on demand** and cache them under `~/.cache/karyon`
(override with `$KARYON_CACHE`). karyon **does not redistribute** any dataset — it stores a processed
subset locally after fetching from the original source. Loaders degrade to a typed `DatasetUnavailable`
when offline (tests skip rather than fail).

Each dataset carries its own license and citation. Verify the original source's terms before using a
dataset beyond its published scope.

| Loader | Dataset | Source | Reference |
|--------|---------|--------|-----------|
| `emopec_data` | EMOPEC ribosome-binding sites (~3.1k) | GitHub `smsaladi/EMOPEC` | Bonde et al., *Nat. Methods* 2015 |
| `promoter_data` | σ70 promoters, Urtecho (~10.9k) | Springer (supp. table) | Urtecho et al. / La Fleur–Salis, *Nat. Commun.* 2022 (CC-BY-4.0) |
| `hossain_data` | in-vivo σ70 promoters (~4.3k) | Springer (supp. table) | Hossain et al., *Nat. Biotechnol.* |
| `rbs_synbiomts_data` | SynBioMTS RBS (~394) | GitHub `hsalis/SalisLabCode` | Salis Lab (academic use) |
| `rbs_hollerer_data` | uASPIre RBS (~300k) | GitHub `JeschekLab/uASPIre` | Höllerer et al. |
| `ko_efficacy_data` | CRISPR-KO efficacy benchmark | GitHub `maximilianh/crisporPaper` | Haeussler et al. (CRISPOR) |
| `stability_data` | Tsuboyama/Rocklin MegaScale stability | Hugging Face `RosettaCommons/MegaScale` | Tsuboyama et al., *Nature* 2023 (Zenodo 7992926, CC0) |
| `utr5_data` | Optimus 5′UTR MRL | NCBI GEO `GSM3130435` | Sample et al. |
| `uspto_data` | USPTO-50k retrosynthesis | GitHub `connorcoley/retrosim` | Coley et al. |
| `molnet_data` | MoleculeNet BBBP + ESOL | DeepChem S3 | Wu et al., *Chem. Sci.* 2018 (DeepChem, Apache-2.0) |
| `crispr_qc_data` | Horlbeck CRISPRi guides + SD7 screen | eLife CDN (supp. data) | Horlbeck et al., *eLife* 2016 (CC-BY-4.0) |
| `screen_qc_data` | pooled CRISPR screen (Wang 2014, ~73k sgRNAs) | MAGeCK demo data | Wang et al., *Science* 2014 |
| `pose_data` | PoseBench docking poses | Zenodo `19138652` | PoseBusters / PoseBench |
| `cofold_data` | co-folding method outputs (Boltz/AF3/RFAA/NeuralPLexer) + PoseBusters refs | Zenodo `19138652` (PoseBench) | Buttenschoen et al., *Chem. Sci.* 2024 |
| `ppi_data` | deposited PDB complexes + wwPDB/MolProbity validation; CASP15 multimer outputs | RCSB / PDBe; CASP15 | wwPDB validation report / MolProbity |
| `gen_dna_data` | *E. coli* K-12 MG1655 coding sequences | NCBI RefSeq `NC_000913.3` | E. coli K-12 reference genome |
| `ppi_leakage_data` | Guo yeast sequence-based PPI benchmark (11,188 pairs / 2,497 proteins) | GitHub `muhaochen/seq_ppi` (PIPR) | Guo et al. 2008 / Chen et al., *Bioinformatics* 2019 |
| `perturbseq_data` | Replogle K562-essential Perturb-seq gemgroup-Z pseudobulk (~80 MB; needs `h5py`) | Figshare+ `20029387` | Replogle et al., *Cell* 2022 (CC-BY-4.0) |
