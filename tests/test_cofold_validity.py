"""test_cofold_validity — falsification proofs for the co-folding intermolecular DRC (cofold-QC).

The teeth: each owned intermolecular contract fires on its planted violation with an auditable reason; a
clean interface stays clean; NOT_FRAMED discloses without condemning; severity is monotone; the stdlib
structure readers parse PDB + mmCIF and split protein/ligand correctly; and — numpy/rdkit/data-gated —
the geometry detects a planted clash / ejection and a real crystal complex passes. The pure-logic + parser
groups run with no rdkit/numpy; the geometry + e2e SKIP if numpy/rdkit/data are absent. Dual pytest /
__main__, mirroring test_pose_honesty.py.
"""

from __future__ import annotations

import os

from karyon import cofold_validity as cv
from karyon import structure_io as sio


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


# --------------------------------------------------------------------------- #
# Pure-logic proofs (no rdkit/numpy) — the contracts over planted InterFeatures.
# --------------------------------------------------------------------------- #
_TOL = cv.InterTol()
_CS = cv.intermolecular_contracts()

_PLANTED = [
    ("LIGAND_PROTEIN_CLASH", dict(min_lig_prot_rel=0.4, min_lig_prot_A=1.3, n_clash_pairs=7), "clashes"),
    ("LIGAND_PROTEIN_VOLUME_OVERLAP", dict(vol_overlap_frac=0.4), "buried"),
    ("LIGAND_OUT_OF_POCKET", dict(min_lig_prot_A=12.0), "no contact"),
]


def test_each_contract_fires_on_its_planted_violation() -> None:
    for name, fields, substr in _PLANTED:
        f = cv.InterFeatures(framed=True, **fields)
        v = _CS.evaluate(f, _TOL)
        assert name in v.fired, f"{name} did not fire on {fields} (fired {v.fired})"
        msg = next(r.message for r in v.reasons if r.contract == name)
        assert substr in msg, f"{name} reason {msg!r} missing {substr!r}"
    print(f"1. each of {len(_PLANTED)} intermolecular contracts fires with the right reason")


def test_clean_interface_stays_clean() -> None:
    v = _CS.evaluate(cv.InterFeatures(), _TOL)               # defaults = a within-tolerance interface
    assert v.ok and v.reasons == (), f"a clean interface must fire nothing: {v.fired}"
    print("2. a within-tolerance interface fires nothing (Verdict.ok)")


def test_not_framed_discloses_not_condemns() -> None:
    f = cv.InterFeatures(framed=False)
    v = _CS.evaluate(f, _TOL)
    assert "NOT_FRAMED" in v.fired and v.score == 0.0, "NOT_FRAMED must disclose at weight 0"
    assert not cv.is_inter_invalid(f, _CS, _TOL), "an unframed pose must not be condemned"
    print("3. NOT_FRAMED discloses (weight 0) without condemning (the disclosure philosophy)")


def test_severity_monotone() -> None:
    base = cv.InterFeatures().severity(_TOL)
    worse = cv.InterFeatures(framed=True, min_lig_prot_rel=0.3, vol_overlap_frac=0.5,
                             min_lig_prot_A=12.0).severity(_TOL)
    assert base == 0.0 and worse > base, f"severity not monotone ({base} -> {worse})"
    print("4. severity is 0 for a clean interface and grows with violations")


def test_vdw_lookup() -> None:
    assert cv.vdw("C") == 1.70 and cv.vdw("S") == 1.80
    assert cv.vdw("Xx") == cv._VDW_DEFAULT, "unknown element must fall back to the default radius"
    print("5. vdw() returns Bondi radii with a sane default")


# --------------------------------------------------------------------------- #
# Structure-reader proofs (stdlib only).
# --------------------------------------------------------------------------- #
def _pdb_line(rec, serial, name, res, chain, resseq, x, y, z, elem) -> str:
    s = [" "] * 80

    def put(start, text):                                    # 1-indexed PDB column
        for i, ch in enumerate(text):
            s[start - 1 + i] = ch
    put(1, rec); put(7, f"{serial:>5}"); put(13, f"{name:<4}"); put(18, f"{res:>3}")
    put(22, chain); put(23, f"{resseq:>4}")
    put(31, f"{x:>8.3f}"); put(39, f"{y:>8.3f}"); put(47, f"{z:>8.3f}"); put(77, f"{elem:>2}")
    return "".join(s)


def test_pdb_reader_and_split() -> None:
    pdb = "\n".join([
        _pdb_line("ATOM", 1, " N", "ALA", "A", 1, 1.0, 2.0, 3.0, "N"),
        _pdb_line("ATOM", 2, " CA", "ALA", "A", 1, 1.5, 2.5, 3.5, "C"),
        _pdb_line("ATOM", 3, " H", "ALA", "A", 1, 0.0, 0.0, 0.0, "H"),     # H dropped (heavy-only)
        _pdb_line("HETATM", 4, " O", "LIG", "A", 101, 5.0, 5.0, 5.0, "O"),
        _pdb_line("HETATM", 5, "O", "HOH", "A", 201, 9.0, 9.0, 9.0, "O"),  # solvent dropped on split
    ])
    atoms = sio.read_pdb_atoms(pdb)
    assert len(atoms) == 4, f"expected 4 heavy atoms (H dropped), got {len(atoms)}"
    assert atoms[0].element == "N" and abs(atoms[0].x - 1.0) < 1e-6
    protein, ligand = sio.split_protein_ligand(atoms)
    assert len(protein) == 2 and len(ligand) == 1, f"split protein/ligand wrong: {len(protein)}/{len(ligand)}"
    assert ligand[0].resname == "LIG"
    print("6. PDB reader: heavy-atom parse + protein/ligand/solvent split")


def test_cif_reader_and_split() -> None:
    cif = "\n".join([
        "data_test", "loop_",
        "_atom_site.group_PDB", "_atom_site.type_symbol", "_atom_site.label_comp_id",
        "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
        "ATOM C ALA 1.0 2.0 3.0",
        "ATOM N ALA 1.5 2.5 3.5",
        "ATOM H ALA 0.0 0.0 0.0",                            # H dropped
        "HETATM O LIG 5.0 5.0 5.0",
        "#",
    ])
    atoms = sio.read_cif_atoms(cif)
    assert len(atoms) == 3, f"expected 3 heavy atoms, got {len(atoms)}"
    protein, ligand = sio.split_protein_ligand(atoms)
    assert len(protein) == 2 and len(ligand) == 1
    assert ligand[0].resname == "LIG" and ligand[0].is_hetero
    print("7. mmCIF _atom_site reader: heavy-atom parse + split")


# --------------------------------------------------------------------------- #
# Geometry proofs (numpy-gated) — planted protein/ligand coordinate sets.
# --------------------------------------------------------------------------- #
def _shell(center, r, n=40):
    import math
    out = []
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for k in range(n):
        y = 1.0 - 2.0 * (k + 0.5) / n
        rad = math.sqrt(max(0.0, 1.0 - y * y))
        out.append(sio.Atom("C", center[0] + r * math.cos(ga * k) * rad,
                            center[1] + r * y, center[2] + r * math.sin(ga * k) * rad))
    return out


def test_geometry_detects_clash_and_ejection() -> None:
    if not cv._HAVE_NUMPY:
        _skip("numpy absent")
        return
    ligand = [sio.Atom("C", 0.5 * i, 0.0, 0.0) for i in range(5)]    # a little linear ligand at the origin
    seated = _shell((1.0, 0.0, 0.0), 4.0)                            # a surrounding pocket → clean
    f = cv.interface_features(seated, ligand, _TOL)
    assert f.framed and not cv.is_inter_invalid(f, _CS, _TOL), \
        f"a seated ligand must pass: {_CS.evaluate(f, _TOL).fired}"

    buried = cv.decoy_bury_into_protein(seated, ligand)
    fb = cv.interface_features(seated, buried, _TOL)
    assert "LIGAND_PROTEIN_CLASH" in _CS.evaluate(fb, _TOL).fired, "burying must fire the clash contract"

    ejected = cv.decoy_eject_from_pocket(seated, ligand)
    fe = cv.interface_features(seated, ejected, _TOL)
    assert "LIGAND_OUT_OF_POCKET" in _CS.evaluate(fe, _TOL).fired, "ejection must fire the out-of-pocket contract"
    print("8. interface geometry: seated passes, burying clashes, ejection is out-of-pocket")


def test_features_deterministic() -> None:
    if not cv._HAVE_NUMPY:
        _skip("numpy absent")
        return
    lig = [sio.Atom("C", 0.4 * i, 0.1, 0.0) for i in range(6)]
    prot = _shell((1.0, 0.0, 0.0), 4.2)
    a = cv.interface_features(prot, lig, _TOL)
    b = cv.interface_features(prot, lig, _TOL)
    assert (a.min_lig_prot_A, a.vol_overlap_frac) == (b.min_lig_prot_A, b.vol_overlap_frac)
    print("9. interface_features is deterministic")


# --------------------------------------------------------------------------- #
# e2e proof (skips offline) — a real crystal complex passes the owned DRC.
# --------------------------------------------------------------------------- #
def test_e2e_crystal_complex_passes() -> None:
    if not (cv._HAVE_NUMPY and cv._HAVE_RDKIT):
        _skip("numpy/rdkit absent")
        return
    from karyon.cofold_data import PoseUnavailable, ligand_mol, load_crystal_complexes
    try:
        complexes = load_crystal_complexes(limit=10)
    except PoseUnavailable as e:
        _skip(f"crystal complexes unavailable: {e}")
        return
    passed = total = 0
    for c in complexes:
        m = ligand_mol(c.ligand_sdf)
        if m is None:
            continue
        v = cv.full_verdict(c.protein, m)
        total += 1
        passed += 1 if (v.inter.score == 0.0) else 0
    assert total and passed / total >= 0.9, f"native complexes should pass the inter DRC ({passed}/{total})"
    print(f"10. e2e: {passed}/{total} native crystal complexes pass the owned intermolecular DRC")


# --------------------------------------------------------------------------- #
# Multi-method loader proofs (stdlib only) — registry + the dual-layout pairing discovery.
# --------------------------------------------------------------------------- #
def test_method_registry_consistent() -> None:
    from karyon import cofold_data as cd
    for key, m in cd._METHODS.items():
        assert m.key == key, f"registry key {key!r} != CofoldMethod.key {m.key!r}"
        assert m.tarball.endswith(".tar.gz"), f"{key}: tarball must be a .tar.gz"
        assert m.pairing in ("suffix", "ranked"), f"{key}: unknown pairing {m.pairing!r}"
    dirs = [m.extract_dir for m in cd._METHODS.values()]
    assert len(dirs) == len(set(dirs)), "extract_dirs must be distinct (else methods overwrite each other)"
    assert cd.cofold_methods() == list(cd._METHODS), "cofold_methods() must list the registry keys"
    print(f"11. method registry consistent ({', '.join(cd._METHODS)})")


def test_pairing_discovers_both_layouts() -> None:
    import tempfile
    from pathlib import Path

    from karyon import cofold_data as cd

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # "suffix" layout: Boltz flat + RFAA subdir + AF3 `_model` subdir, with aligned/fragment decoys.
        sx = root / "boltz_posebusters_benchmark_outputs_1"
        sx.mkdir()
        (sx / "7AAA_model_0_protein.pdb").touch()
        (sx / "7AAA_model_0_ligand.sdf").touch()
        (sx / "7AAA_model_0_protein_aligned.pdb").touch()        # decoy: aligned, no pair
        (sx / "8BBB").mkdir()
        (sx / "8BBB" / "8BBB_protein.pdb").touch()
        (sx / "8BBB" / "8BBB_ligand.sdf").touch()
        (sx / "8BBB" / "8BBB_ligand_LG1_1.sdf").touch()          # decoy: fragment, not the pair
        (sx / "9CCC").mkdir()
        (sx / "9CCC" / "9CCC_model_protein.pdb").touch()         # AF3 `_model` (no digit)
        (sx / "9CCC" / "9CCC_model_ligand.sdf").touch()
        got = {t for t, _, _ in cd._pairs_suffix(sx)}
        assert got == {"7AAA", "8BBB", "9CCC"}, f"suffix pairing wrong: {got}"

        # "ranked" layout: NeuralPLexer rank1+plddt, with rank2 / ref / aligned decoys.
        rk = root / "neuralplexer_posebusters_benchmark_outputs_1"
        (rk / "7K0V").mkdir(parents=True)
        (rk / "7K0V" / "prot_rank1_plddt0.64.pdb").touch()
        (rk / "7K0V" / "lig_rank1_plddt0.64.sdf").touch()
        (rk / "7K0V" / "prot_rank2_plddt0.63.pdb").touch()       # decoy: not top rank
        (rk / "7K0V" / "lig_ref.sdf").touch()                    # decoy: reference
        (rk / "7K0V" / "prot_rank1_plddt0.64_aligned.pdb").touch()  # decoy: aligned
        pairs = list(cd._pairs_ranked(rk))
        assert {t for t, _, _ in pairs} == {"7K0V"}, f"ranked pairing wrong targets: {pairs}"
        _, pf, lf = pairs[0]
        assert "rank1" in pf.name and "aligned" not in pf.name and "rank1" in lf.name, \
            f"ranked pairing picked the wrong files: {pf.name} / {lf.name}"
    print("12. loader pairing discovers both the suffix (Boltz/RFAA/AF3) and ranked (NeuralPLexer) layouts")


def test_raw_faithful_guard() -> None:
    # the like-for-like guard: a method whose deposited reference describes a RELAXED copy (low
    # ref_struct_match) is not a faithfulness test of the raw-pose gate and is excluded from PI-4.
    from karyon import cofold_honesty as ch
    mk = lambda rsm: ch.FaithResult("m", "M", 100, 0.7, 0.9, 0.6, 0.5, {}, rsm)
    assert mk(1.00).raw_faithful and mk(0.90).raw_faithful, "≥0.90 struct-match is raw-faithful"
    assert not mk(0.69).raw_faithful, "0.69 (NeuralPLexer) is a relaxed reference, not like-for-like"
    print("13. raw-faithful guard excludes relaxed-reference methods (NeuralPLexer) from PI-4")


if __name__ == "__main__":
    test_each_contract_fires_on_its_planted_violation()
    test_clean_interface_stays_clean()
    test_not_framed_discloses_not_condemns()
    test_severity_monotone()
    test_vdw_lookup()
    test_pdb_reader_and_split()
    test_cif_reader_and_split()
    test_geometry_detects_clash_and_ejection()
    test_features_deterministic()
    test_e2e_crystal_complex_passes()
    test_method_registry_consistent()
    test_pairing_discovers_both_layouts()
    test_raw_faithful_guard()
    print("\ncofold_validity proofs pass.")
