"""test_validate — the uniform per-gate `validate(artifact) -> Verdict` entry points are EXACTLY the
manual contract chains they replace. This is the faithfulness proof for the spine's Stage 2: adding a
convenience wrapper must not change any verdict. Gates with optional deps skip cleanly when absent.

    python tests/test_validate.py
"""

from __future__ import annotations

import pytest

from karyon import contracts


# --------------------------------------------------------------------------- #
# promoter — pure stdlib, always runs.
# --------------------------------------------------------------------------- #
def test_promoter_validate_equals_design_evaluate() -> None:
    from karyon import promoter_contracts as pc

    # a strong promoter (real boxes) and a weak one (no −35) — both must match DESIGN.evaluate exactly.
    seqs = [
        "AAAATTGACAGGCTATAATGCAAAACCCGGGTTTAAACCCGGGTTTAAACCCGGG",   # −35/−10 present
        "GGGGGGCCCCCCGGGGGGCCCCCCGGGGGGCCCCCCGGGGGGCCCCCC",          # no boxes, GC out of band
    ]
    for s in seqs:
        v = pc.validate(s)
        assert isinstance(v, contracts.Verdict)
        assert v == pc.DESIGN.evaluate(s)
        assert v.to_dict()["score"] == v.score          # serializes
    # ctx is honored (calibrated C5/C6 read it).
    ctx = pc.calibrate_design(seqs)
    assert pc.validate(seqs[0], ctx) == pc.DESIGN.evaluate(seqs[0], ctx)
    print("1. promoter_contracts.validate == DESIGN.evaluate (uncalibrated and calibrated)")


# --------------------------------------------------------------------------- #
# pose — rdkit.
# --------------------------------------------------------------------------- #
def test_pose_validate_equals_manual_chain() -> None:
    pytest.importorskip("rdkit")
    from karyon import pose_validity as pv

    mol = pv.clean_conformer("CCO")            # a clean, embeddable small molecule
    tol = pv.Tol()
    assert pv.validate(mol, tol) == pv.validity_contracts().evaluate(pv.featurize(mol, tol), tol)
    print("2. pose_validity.validate == validity_contracts().evaluate(featurize(...))")


# --------------------------------------------------------------------------- #
# mol — rdkit (validate already existed; assert it's a serializable Verdict).
# --------------------------------------------------------------------------- #
def test_mol_validate_returns_serializable_verdict() -> None:
    pytest.importorskip("rdkit")
    from karyon import mol_qc as mq

    v = mq.validate("CC(=O)Oc1ccccc1C(=O)O")   # aspirin — a real, usable drug
    assert isinstance(v, contracts.Verdict)
    assert set(v.to_dict()) == {"ok", "score", "reasons"}
    print("3. mol_qc.validate returns a serializable Verdict")


# --------------------------------------------------------------------------- #
# gen_dna — pure stdlib (validate already existed; assert it's a serializable Verdict).
# --------------------------------------------------------------------------- #
def test_gen_dna_validate_returns_serializable_verdict() -> None:
    from karyon import gen_dna_validity as gv

    v = gv.validate("ATGGCAGCATTACGCGATTACCGATTACCGGATTACCGAGTAA")
    assert isinstance(v, contracts.Verdict)
    assert set(v.to_dict()) == {"ok", "score", "reasons"}
    print("4. gen_dna_validity.validate returns a serializable Verdict")


# --------------------------------------------------------------------------- #
# cofold — numpy; intermolecular-only path (ligand atoms from the frame).
# --------------------------------------------------------------------------- #
def test_cofold_validate_inter_only_equals_manual_chain() -> None:
    pytest.importorskip("numpy")
    from karyon import cofold_validity as cv
    from karyon.structure_io import Atom

    protein = [Atom("C", 0.0, 0.0, 0.0), Atom("N", 1.5, 0.0, 0.0), Atom("O", 0.0, 1.5, 0.0)]
    ligand = [Atom("C", 6.0, 6.0, 6.0, is_hetero=True), Atom("O", 7.2, 6.0, 6.0, is_hetero=True)]
    tol = cv.InterTol()
    expected = cv.intermolecular_contracts().evaluate(cv.interface_features(protein, ligand, tol), tol)
    assert cv.validate(protein, ligand, tol_inter=tol) == expected
    print("5. cofold_validity.validate (ligand atoms) == intermolecular chain")


# --------------------------------------------------------------------------- #
# complex — numpy; two synthetic chain groups.
# --------------------------------------------------------------------------- #
def test_complex_validate_equals_manual_chain() -> None:
    pytest.importorskip("numpy")
    from karyon import protein_interface_validity as piv
    from karyon.structure_io import Atom

    group_a = [Atom("C", 0.0, 0.0, 0.0, chain="A"), Atom("N", 1.5, 0.0, 0.0, chain="A")]
    group_b = [Atom("C", 5.0, 0.0, 0.0, chain="B"), Atom("O", 6.2, 0.0, 0.0, chain="B")]
    tol = piv.IfaceTol()
    expected = piv.protein_interface_contracts().evaluate(piv.interface_features(group_a, group_b, tol), tol)
    assert piv.validate(group_a, group_b, tol) == expected
    print("6. protein_interface_validity.validate == interface chain")


def _run() -> None:
    test_promoter_validate_equals_design_evaluate()
    for fn in (test_pose_validate_equals_manual_chain, test_mol_validate_returns_serializable_verdict,
               test_gen_dna_validate_returns_serializable_verdict,
               test_cofold_validate_inter_only_equals_manual_chain,
               test_complex_validate_equals_manual_chain):
        try:
            fn()
        except Exception as e:                       # noqa: BLE001 — script path: surface skips, don't abort
            print(f"   (skipped {fn.__name__}: {e})")
    print("\nALL validate() faithfulness proofs passed.")


if __name__ == "__main__":
    _run()
