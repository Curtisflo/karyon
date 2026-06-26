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
