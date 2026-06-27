"""test_mol_qc — falsification proofs for the generated-molecule QC gate (mol-QC).

Falsification proofs in the karyon probe-suite style. Shows:
  1. each owned contract fires on a planted case and stays silent on a real drug;
  2. the disclose-vs-condemn tiering is by SCORE (an alert/Ro5 disclosure does not fail the gate);
  3. the kicker — a *valid*, confidently-emitted molecule is still unusable (extreme property): validity ≠
     usability, the gap a generator's confidence is blind to;
  4. composition correctness — the gate's INVALID / ALERT flags equal a fresh, independent invocation of the
     canonical RDKit calls it composes (the honest faithfulness posture: RDKit is the engine).

Runnable as a script (`python test_mol_qc.py`) or under pytest. Skips cleanly if rdkit is absent.
"""

from __future__ import annotations

from karyon import mol_qc as mq

_HAVE = mq._HAVE_RDKIT
_TOL = mq.MolTol()
_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"     # a real drug — passes (carries a disclosed Brenk alert)
_CLEAN = "NC(Cc1ccccc1)C(=O)O"          # phenylalanine — clean, no alerts


def _skip() -> bool:
    if not _HAVE:
        print("SKIP — mol_qc tests need rdkit.")
    return not _HAVE


# --------------------------------------------------------------------------- #
# 1) contracts fire on planted / silent on clean
# --------------------------------------------------------------------------- #
def test_real_drug_passes() -> None:
    if _skip():
        return
    assert not mq.is_unusable(_CLEAN, _TOL), mq.validate(_CLEAN, _TOL).messages


def test_invalid_molecule_condemns() -> None:
    if _skip():
        return
    v = mq.validate("c1ccccc1C(C)(C)(C)(C)", _TOL)     # pentavalent carbon → unparseable
    assert "INVALID_MOLECULE" in v.fired and v.score > 0


def test_extreme_property_condemns() -> None:
    if _skip():
        return
    v = mq.validate("C" * 100, _TOL)                   # huge alkane → MW/logP extreme
    assert "EXTREME_PROPERTY" in v.fired and v.score > 0


def test_unsynthesizable_condemns() -> None:
    if _skip():
        return
    # a deliberately complex fused/caged scaffold (Ertl SA > the cap)
    caged = "C1C2CC3CC1CC(C2)(C3)C1CC2CC3CC1CC(C2)C3C1CC2CC3CC1CC(C2)C3"
    v = mq.validate(caged, _TOL)
    assert "UNSYNTHESIZABLE" in v.fired and v.score > 0, mq.featurize(caged, _TOL).sa


def test_structural_alert_discloses_without_condemning() -> None:
    if _skip():
        return
    v = mq.validate(_ASPIRIN, _TOL)                    # aspirin hits Brenk (phenol_ester), else fine
    assert "STRUCTURAL_ALERT" in v.fired
    assert v.score == 0.0 and not mq.is_unusable(_ASPIRIN, _TOL)


def test_lipinski_discloses_without_condemning() -> None:
    if _skip():
        return
    # ≥2 Rule-of-5 violations (MW>500, logP>5) on an otherwise-makeable molecule ⇒ DISCLOSED (weight 0):
    # reported in `fired` but the gate still PASSES. Tested at the feature level so a fixture SMILES's
    # incidental alert/SA can't confound the tiering assertion (the gen-dna poly-G style).
    f = mq.MolFeatures(smiles="x", parsed=True, mw=520.0, logp=5.5, hbd=2, hba=4, sa=3.0, qed=0.3, alerts=())
    assert f.ro5_violations(_TOL) == 2
    v = mq.mol_contracts().evaluate(f, _TOL)
    assert "LIPINSKI_RO5" in v.fired and v.score == 0.0, v.messages


def test_severity_orders_clean_below_decoy() -> None:
    if _skip():
        return
    assert mq.featurize(_CLEAN, _TOL).severity(_TOL) == 0.0 < mq.featurize("C" * 100, _TOL).severity(_TOL)


# --------------------------------------------------------------------------- #
# 2) the kicker — valid but unusable (validity ≠ usability)
# --------------------------------------------------------------------------- #
def test_valid_molecule_can_be_unusable() -> None:
    if _skip():
        return
    huge = "C" * 100                                   # a perfectly valid SMILES a generator could emit
    from rdkit import Chem
    assert Chem.MolFromSmiles(huge) is not None and mq.is_unusable(huge, _TOL)


# --------------------------------------------------------------------------- #
# 3) composition correctness vs fresh, independent canonical RDKit calls
# --------------------------------------------------------------------------- #
def test_composition_matches_fresh_rdkit() -> None:
    if _skip():
        return
    from rdkit import Chem
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    params = FilterCatalogParams()
    for n in _TOL.alert_catalogs:
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, n))
    cat = FilterCatalog(params)
    for smi in (_ASPIRIN, _CLEAN, "c1ccccc1C(C)(C)(C)(C)", "C" * 100):
        fired = set(mq.validate(smi, _TOL).fired)
        assert ("INVALID_MOLECULE" in fired) == (Chem.MolFromSmiles(smi) is None)
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            assert ("STRUCTURAL_ALERT" in fired) == cat.HasMatch(mol)


def main() -> None:
    if _skip():
        return
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
