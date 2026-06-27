"""mol_qc — a legible validity/synthesizability/drug-likeness DRC for generated molecules (mol-QC).

A deterministic pass/fail design-rule check over a generative chemistry model's output (e.g. NVIDIA
BioNeMo's GenMol / MolMIM), where the generator ships only advisory validation. Thin
`contracts.Contract`s read a precomputed `MolFeatures` scalar, with disclose-vs-condemn tiering keyed off
the verdict's continuous *score* — the same shape as karyon's other generative-output QC gates.

**Honest posture:** for molecules **RDKit *is* the cheminformatics engine** — you cannot parse a SMILES,
compute a descriptor, or match a structural alert without it. So this gate **composes** canonical RDKit
primitives (sanitization, descriptors, the PAINS/Brenk `FilterCatalog`, the Ertl SA score) into a legible
deterministic DRC — the gate a generator's "eyeball it" validation doesn't ship — rather than
reimplementing them to check against an *independent* package. What is genuinely **owned & independently
checkable** is the *rules*: the property-window thresholds (Lipinski Ro5 / Veber) over RDKit descriptors,
and the contracts composition + verdict. The validity / structural-alert / SA axes are composed
(faithfulness = correct composition, disclosed); the SA "hard-to-make" flag is independently corroborated
against RDKit's `BertzCT` molecular-complexity in `mol_qc_honesty.py`. Requires rdkit at runtime.
Qualification, not accuracy.

    python -m karyon.mol_qc     # smoke: a real drug passes; planted/invalid decoys fail
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from . import contracts

try:
    from rdkit import Chem, RDConfig, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False


# --------------------------------------------------------------------------- #
# Calibration — disclosed constants (commercial/medicinal-chemistry conventions). No fit-to-accuracy.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MolTol:
    sa_max: float = 6.0          # Ertl synthetic-accessibility score > this ⇒ unsynthesizable (1 easy … 10 hard)
    mw_extreme: float = 900.0    # MW > this ⇒ not a usable small molecule (condemn)
    logp_extreme: float = 7.0    # cLogP > this ⇒ condemn (insoluble / non-drug-like extreme)
    # Lipinski rule-of-5 (DISCLOSE at ≥2 violations — a drug-likeness note, not a validity failure)
    ro5_mw: float = 500.0
    ro5_logp: float = 5.0
    ro5_hbd: int = 5
    ro5_hba: int = 10
    # Veber oral-bioavailability rules (DISCLOSE)
    veber_rotb: int = 10
    veber_tpsa: float = 140.0
    alert_catalogs: tuple[str, ...] = ("PAINS", "BRENK")   # structural-alert catalogs to run


# --------------------------------------------------------------------------- #
# FilterCatalog cache — building a catalog is non-trivial; build once per catalog-set.
# --------------------------------------------------------------------------- #
_CATALOG_CACHE: dict[tuple[str, ...], "FilterCatalog"] = {}


def _catalog(names: tuple[str, ...]) -> "FilterCatalog":
    if names not in _CATALOG_CACHE:
        params = FilterCatalogParams()
        for n in names:
            params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, n))
        _CATALOG_CACHE[names] = FilterCatalog(params)
    return _CATALOG_CACHE[names]


def _alert_hits(mol, names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """(catalog-prop, description) for every structural-alert match — the legible 'why' of each alert."""
    cat = _catalog(names)
    out = []
    for m in cat.GetMatches(mol):
        try:
            scope = m.GetProp("FilterSet")
        except KeyError:
            scope = "ALERT"
        out.append((scope, m.GetDescription()))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Precomputed per-molecule features the contracts read.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MolFeatures:
    smiles: str = ""
    parsed: bool = True          # False ⇒ SMILES did not parse/sanitize (the "framed" idiom)
    mw: float = 0.0
    logp: float = 0.0
    hbd: int = 0
    hba: int = 0
    tpsa: float = 0.0
    rotb: int = 0
    sa: float = 1.0              # Ertl synthetic accessibility
    bertz: float = 0.0          # BertzCT molecular complexity (the independent SA corroborator)
    qed: float = 1.0
    alerts: tuple[tuple[str, str], ...] = ()

    def ro5_violations(self, tol: MolTol) -> int:
        return int(self.mw > tol.ro5_mw) + int(self.logp > tol.ro5_logp) + \
            int(self.hbd > tol.ro5_hbd) + int(self.hba > tol.ro5_hba)

    def severity(self, tol: MolTol) -> float:
        """Continuous severity (the ranking statistic for the instrument AUROC). 0 for a clean molecule;
        an unparseable SMILES is maximally severe. Disclosed-only axes (alerts/Ro5/Veber) don't contribute."""
        if not self.parsed:
            return 10.0

        def relu(x: float) -> float:
            return x if x > 0 else 0.0
        s = relu(self.sa / tol.sa_max - 1.0)
        s += relu(self.mw / tol.mw_extreme - 1.0)
        s += relu(self.logp / tol.logp_extreme - 1.0)
        return s


def featurize(smiles: str, tol: MolTol = MolTol()) -> MolFeatures:
    """Parse + featurize a SMILES via RDKit. `parsed=False` ⇒ invalid molecule (the headline condemn)."""
    if not _HAVE_RDKIT:
        raise RuntimeError("mol_qc.featurize needs rdkit")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return MolFeatures(smiles=smiles, parsed=False)
    return MolFeatures(
        smiles=smiles, parsed=True,
        mw=Descriptors.MolWt(mol), logp=Crippen.MolLogP(mol),
        hbd=Lipinski.NumHDonors(mol), hba=Lipinski.NumHAcceptors(mol),
        tpsa=rdMolDescriptors.CalcTPSA(mol), rotb=rdMolDescriptors.CalcNumRotatableBonds(mol),
        sa=sascorer.calculateScore(mol), bertz=Descriptors.BertzCT(mol), qed=QED.qed(mol),
        alerts=_alert_hits(mol, tol.alert_catalogs))


# --------------------------------------------------------------------------- #
# The DRC — each check a legible contract reading a precomputed MolFeatures scalar.
# --------------------------------------------------------------------------- #
def mol_contracts() -> contracts.ContractSet:
    cs = contracts.ContractSet("mol-validity")

    cs.add(contracts.Contract("INVALID_MOLECULE",
        lambda f, t: "SMILES does not parse/sanitize into a valid molecule (bad valence or syntax)"
        if not f.parsed else None, weight=2.0))

    cs.add(contracts.Contract("UNSYNTHESIZABLE",
        lambda f, t: (f"synthetic accessibility {f.sa:.1f} > {t.sa_max:.1f} (1 easy … 10 hard) — "
                      f"not reasonably synthesizable")
        if f.parsed and f.sa > t.sa_max else None, weight=1.5))

    cs.add(contracts.Contract("EXTREME_PROPERTY",
        lambda f, t: (f"out of small-molecule range: "
                      + ", ".join(([f"MW {f.mw:.0f}>{t.mw_extreme:.0f}"] if f.mw > t.mw_extreme else [])
                                  + ([f"cLogP {f.logp:.1f}>{t.logp_extreme:.1f}"] if f.logp > t.logp_extreme else [])))
        if f.parsed and (f.mw > t.mw_extreme or f.logp > t.logp_extreme) else None, weight=1.5))

    # Disclose (weight 0.0 — advisory): structural alerts (PAINS/Brenk have known false positives).
    cs.add(contracts.Contract("STRUCTURAL_ALERT",
        lambda f, t: (f"{len(f.alerts)} structural alert{'s' if len(f.alerts) != 1 else ''} "
                      f"({'; '.join(sorted({d for _, d in f.alerts})[:3])}"
                      f"{' …' if len({d for _, d in f.alerts}) > 3 else ''}) — assay-interference / reactive")
        if f.parsed and f.alerts else None, weight=0.0))

    cs.add(contracts.Contract("LIPINSKI_RO5",
        lambda f, t: (f"{f.ro5_violations(t)} Rule-of-5 violations "
                      f"(MW {f.mw:.0f} / cLogP {f.logp:.1f} / HBD {f.hbd} / HBA {f.hba}) — drug-likeness")
        if f.parsed and f.ro5_violations(t) >= 2 else None, weight=0.0))

    cs.add(contracts.Contract("VEBER",
        lambda f, t: (f"Veber: rotatable bonds {f.rotb} (>{t.veber_rotb}) / TPSA {f.tpsa:.0f} "
                      f"(>{t.veber_tpsa:.0f}) — oral-bioavailability concern")
        if f.parsed and (f.rotb > t.veber_rotb or f.tpsa > t.veber_tpsa) else None, weight=0.0))
    return cs


def validate(smiles: str, tol: MolTol = MolTol()) -> contracts.Verdict:
    """The per-molecule verdict: featurize then evaluate the DRC."""
    return mol_contracts().evaluate(featurize(smiles, tol), tol)


def is_unusable(smiles: str, tol: MolTol = MolTol()) -> bool:
    """A molecule is unusable iff a CONDEMNING contract fired (score > 0; disclosed flags are weight 0)."""
    return validate(smiles, tol).score > 0.0


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not _HAVE_RDKIT:
        print("SKIP — mol_qc smoke needs rdkit.")
        raise SystemExit(0)
    tol = MolTol()
    cases = {
        "aspirin (clean)":      "CC(=O)Oc1ccccc1C(=O)O",
        "invalid (pentavalent)": "c1ccccc1C(C)(C)(C)(C)",
        "extreme MW":           "C" * 80,                         # a huge alkane → MW well over 900
        "PAINS/Brenk alert":    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",  # diclofenac (Brenk-flagged)
    }
    for label, smi in cases.items():
        v = validate(smi, tol)
        f = featurize(smi, tol)
        print(f"\n{label:22} → {'PASS' if v.score == 0 else 'FAIL'}  "
              f"(parsed {f.parsed}, MW {f.mw:.0f}, SA {f.sa:.1f}, QED {f.qed:.2f})")
        for r in v.reasons:
            print(f"    {'✗' if r.weight > 0 else '·'} {r.contract}: {r.message}")
    print("\nmol_qc smoke OK.")
