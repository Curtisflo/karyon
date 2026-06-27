"""mol_qc_data — corpora for the generated-molecule QC gate (mol-QC).

The gate (`mol_qc.py`) qualifies a generative chemistry model's output (e.g. NVIDIA BioNeMo's GenMol /
MolMIM). Producing real GenMol molecules needs a GPU (out of scope at QC time), so this module supplies the
molecules the honesty probe needs, all bundled/seeded and offline:

  * `reference_drugs`  — a curated, RDKit-validated list of real approved-drug SMILES (the clean class + the
                         BRICS fragment source). Bundled, not fetched: small, canonical, deterministic.
  * `brics_generated`  — BRICS fragment-recombination of the drug set: a real (if simple) molecular
                         *generator*, the GenMol stand-in (deterministic via `random.seed`).
  * `random_smiles`    — corrupted SMILES strings: the un-gated invalid baseline (~100% invalid).
  * `planted_decoys`   — molecules each carrying one CONDEMNING defect (extreme MW/logP alkanes; high-SA
                         caged structures) — the instrument's negative class.

Requires rdkit (the cheminformatics engine; see `mol_qc.py`'s honest-posture note).
"""

from __future__ import annotations

import random

from rdkit import Chem, RDLogger

from . import mol_qc as mq

RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------------------- #
# Curated reference drugs — RDKit-validated canonical small molecules (the clean class + BRICS source).
# --------------------------------------------------------------------------- #
_DRUGS: tuple[str, ...] = (
    "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "CC(=O)Nc1ccc(O)cc1", "COc1ccc2cc(C(C)C(=O)O)ccc2c1", "O=C(O)c1ccccc1O",
    "CCOC(=O)c1ccc(N)cc1", "CCN(CC)CC(=O)Nc1c(C)cccc1C", "CCN(CC)CCOC(=O)c1ccc(N)cc1",
    "CN(C)C(=N)N=C(N)N", "NCCc1ccc(O)c(O)c1", "NCCc1c[nH]c2ccc(O)cc12", "NCCc1c[nH]cn1",
    "CNCC(O)c1ccc(O)c(O)c1", "CC(N)Cc1ccccc1", "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",
    "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "Cn1c(=O)c2[nH]cnc2n(C)c1=O", "Cn1cnc2c1c(=O)[nH]c(=O)n2C", "CN1CCCC1c1cccnc1",
    "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
    "CC1(C)SC2C(NC(=O)C(N)c3ccccc3)C(=O)N2C1C(=O)O",
    "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1", "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC",
    "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O", "CN1CCC23C=CC(O)C2Oc2c(O)ccc(c23)C1",
    "COc1ccc2c3c1OC1C(O)C=CC4C(C2)N(C)CCC134",
    "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12",
    "CNC(=Cc1ccc(o1)CN(C)C)/[N+](=O)[O-]",
    "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1", "CN(C)CCOC(c1ccccc1)c1ccccc1",
    "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1",
    "OC(=O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1",
    "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
    "Fc1ccc(C2CCNCC2COc2ccc3c(c2)OCO3)cc1",
    "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(S(=O)(=O)N2CCN(C)CC2)ccc1OCC",
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl", "CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1",
    "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1", "CC(C)NCC(O)COc1cccc2ccccc12",
    "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "COCCc1ccc(OCC(O)CNC(C)C)cc1",
    "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl",
    "COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC",
    "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl", "NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O",
    "CC12CC(=O)C3C(CCC4=CC(=O)C=CC43C)C1CCC2(O)C(=O)CO",
    "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
    "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O", "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O",
    "CC(C)CCCC(C)C1CCC2C1(C)CCC1C2CC=C2CC(O)CCC12C", "OCC1OC(O)C(O)C(O)C1O",
    "CCCC(CCC)C(=O)O", "NCC1(CC(=O)O)CCCCC1", "CC(C)CC(CN)CC(=O)O",
    "NC(Cc1ccc(O)c(O)c1)C(=O)O", "NC(Cc1ccc(O)cc1)C(=O)O", "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "NC(Cc1ccccc1)C(=O)O", "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",
    "COc1ccc2nccc(C(O)C3CC4CCN3CC4C=C)c2c1", "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "Cc1ncc([N+](=O)[O-])n1CCO", "Nc1nc2c(c(=O)[nH]1)n(COCCO)cn2",
    "CC(C)Cc1ccc(C(C)C(N)=O)cc1", "c1ccc2ccccc2c1", "c1ccc(-c2ccccc2)cc1",
    "COc1cc(C=O)ccc1O", "COc1cc(CNC(=O)CCCCC=CC(C)C)ccc1O", "Cc1ccc(C(C)C)cc1O",
    "CC(C)C1CCC(C)CC1O", "CC1(C)C2CCC1(C)C(=O)C2", "NC(=O)c1cccnc1",
    "Cc1ncc(CO)c(CO)c1O", "O=C1NC2C(CCCCC(=O)O)SCC2N1",
    "Nc1nc2ncc(CNc3ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc3)nc2c(=O)[nH]1",
    "OCC(O)C1OC(=O)C(O)=C1O", "NCC(=O)O", "CC(N)C(=O)O", "O=C(O)c1ccccc1", "Oc1ccccc1",
    "Nc1ccccc1", "Cc1ccccc1", "C=Cc1ccccc1", "c1ccc2[nH]ccc2c1", "c1c[nH]cn1",
    "c1cncnc1", "c1ncc2[nH]cnc2n1",
)

# deliberately complex, high synthetic-accessibility caged scaffolds (SA > the default cap) — verified
# in __main__; used (filtered through is_unusable) as UNSYNTHESIZABLE instrument decoys.
_HIGH_SA = (
    "C1C2CC3CC1CC(C2)(C3)C1CC2CC3CC1CC(C2)C3C1CC2CC3CC1CC(C2)C3",
    "C1C2CC3CC1CC(C2)(C3)C1CC2CC3CC1CC(C2)(C3)C1CC2CC3CC1CC(C2)C3C1CC2CC3CC1CC(C2)C3",
)


def reference_drugs(limit: int | None = None) -> list[str]:
    """The curated approved-drug SMILES (each parses; the clean class + BRICS fragment source)."""
    out = [s for s in _DRUGS if Chem.MolFromSmiles(s) is not None]
    return out[:limit] if limit else out


def brics_generated(n: int = 120, seed: int = 0, source: list[str] | None = None) -> list[str]:
    """`n` molecules from BRICS recombination of the drug fragments — a real molecular generator (the GenMol
    stand-in). Deterministic for a given seed. Returns canonical SMILES of sanitized products."""
    from rdkit.Chem import BRICS
    drugs = source if source is not None else reference_drugs()
    frags: set[str] = set()
    for s in drugs:
        frags |= set(BRICS.BRICSDecompose(Chem.MolFromSmiles(s)))
    fragmols = [Chem.MolFromSmiles(f) for f in sorted(frags)]
    random.seed(seed)
    out: list[str] = []
    for i, prod in enumerate(BRICS.BRICSBuild(fragmols)):
        if len(out) >= n or i >= n * 50:               # cap the builder so it always terminates
            break
        try:
            prod.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(prod)
            out.append(Chem.MolToSmiles(prod))
        except Exception:
            continue
    return out


def random_smiles(n: int = 120, seed: int = 2, length_range: tuple[int, int] = (10, 30)) -> list[str]:
    """`n` random strings from a SMILES-ish alphabet — the un-gated invalid baseline (~100% invalid)."""
    rng = random.Random(seed)
    alpha = "CCCCNNOO()[]=#@+-12345cccnos"
    return ["".join(rng.choice(alpha) for _ in range(rng.randint(*length_range))) for _ in range(n)]


def planted_decoys(n: int = 120, seed: int = 1, tol: "mq.MolTol" = mq.MolTol()) -> list[str]:
    """`n` molecules each carrying one CONDEMNING defect: an extreme-MW/logP long alkane, or a high-SA caged
    scaffold. Each is verified `is_unusable` (so the instrument's negative class is guaranteed-flagged)."""
    rng = random.Random(seed)
    out: list[str] = []
    i = 0
    while len(out) < n:
        kind = i % 2
        i += 1
        s = ("C" * rng.randint(75, 130)) if kind == 0 else rng.choice(_HIGH_SA)
        if mq.is_unusable(s, tol):
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(__import__("rdkit").RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    drugs = reference_drugs()
    print(f"reference_drugs: {len(drugs)} valid")
    sa = [sascorer.calculateScore(Chem.MolFromSmiles(s)) for s in drugs]
    print(f"  drug SA range {min(sa):.1f}–{max(sa):.1f}; pass-gate {sum(not mq.is_unusable(s) for s in drugs)}/{len(drugs)}")
    for s in _HIGH_SA:
        m = Chem.MolFromSmiles(s)
        print(f"  high-SA decoy SA={sascorer.calculateScore(m):.2f} unusable={mq.is_unusable(s)}")
    gen = brics_generated(8, seed=0)
    print(f"brics_generated (8): {gen}")
    rnd = random_smiles(50, seed=2)
    print(f"random_smiles invalid: {sum(mq.is_unusable(s) for s in rnd)}/50")
    dec = planted_decoys(8, seed=1)
    print(f"planted_decoys all unusable: {all(mq.is_unusable(s) for s in dec)} (n={len(dec)})")
