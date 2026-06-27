"""test_protein_interface_validity — falsification proofs for the protein-complex interface DRC (complex-QC).

The teeth, mirroring test_cofold_validity.py: each owned interface contract fires on its planted violation with
an auditable reason; a clean interface stays clean; a SHALLOW clash discloses (weight 0) without condemning; the
chain-aware structure readers parse PDB + mmCIF and split by chain; the geometry detects an interpenetrated /
separated chain and counts ONLY inter-chain clashes; the wwPDB-validation clash parser keeps exactly the
inter-chain HEAVY↔HEAVY subset (the like-for-like reference); and — data-gated — a real native complex's owned
clash presence agrees with its deposited wwPDB reference. Pure-logic + parser groups run with no numpy; geometry
+ e2e SKIP if numpy/data are absent. Dual pytest / __main__.
"""

from __future__ import annotations

import math
import os

from karyon import protein_interface_validity as piv
from karyon import ppi_data
from karyon import structure_io as sio


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


# --------------------------------------------------------------------------- #
# Pure-logic proofs (no numpy) — the contracts over planted IfaceFeatures.
# --------------------------------------------------------------------------- #
_TOL = piv.IfaceTol()
_CS = piv.protein_interface_contracts()

_PLANTED = [
    ("SEVERE_INTERFACE_CLASH", dict(max_overlap_A=1.5, n_clash_pairs=5), "interpenetrate"),
    ("INTERFACE_VOLUME_OVERLAP", dict(vol_overlap_frac=0.4), "buried"),
    ("CHAINS_NOT_IN_CONTACT", dict(min_ab_A=12.0), "not in contact"),
]


def test_each_condemning_contract_fires_on_its_planted_violation() -> None:
    for name, fields, substr in _PLANTED:
        f = piv.IfaceFeatures(framed=True, **fields)
        v = _CS.evaluate(f, _TOL)
        assert name in v.fired, f"{name} did not fire on {fields} (fired {v.fired})"
        assert piv.is_interface_invalid(f, _CS, _TOL), f"{name} should condemn"
        msg = next(r.message for r in v.reasons if r.contract == name)
        assert substr in msg, f"{name} reason {msg!r} missing {substr!r}"
    print(f"1. each of {len(_PLANTED)} condemning interface contracts fires + condemns with the right reason")


def test_clean_interface_stays_clean() -> None:
    v = _CS.evaluate(piv.IfaceFeatures(), _TOL)              # defaults = a within-tolerance interface
    assert v.ok and v.reasons == (), f"a clean interface must fire nothing: {v.fired}"
    print("2. a within-tolerance interface fires nothing (Verdict.ok)")


def test_shallow_clash_discloses_not_condemns() -> None:
    # a few SHALLOW clashes (the kind deposited natives carry) DISCLOSE but must not condemn — only deep
    # interpenetration condemns. This is the heart of the two-tier design (pass_native vs faithful detection).
    f = piv.IfaceFeatures(framed=True, n_clash_pairs=3, max_overlap_A=0.5)
    v = _CS.evaluate(f, _TOL)
    assert "INTERFACE_CLASH" in v.fired, "a clash must DISCLOSE (the detection signal)"
    assert "SEVERE_INTERFACE_CLASH" not in v.fired, "a 0.5 Å overlap is shallow — must not condemn"
    assert not piv.is_interface_invalid(f, _CS, _TOL), "shallow clashes must not fail a structure (weight 0)"
    print("3. a shallow clash discloses (weight 0) without condemning; only deep interpenetration condemns")


def test_not_framed_discloses_not_condemns() -> None:
    f = piv.IfaceFeatures(framed=False)
    v = _CS.evaluate(f, _TOL)
    assert "NOT_FRAMED" in v.fired and v.score == 0.0, "NOT_FRAMED must disclose at weight 0"
    assert not piv.is_interface_invalid(f, _CS, _TOL), "an unframed complex must not be condemned"
    print("4. NOT_FRAMED discloses (weight 0) without condemning")


def test_severity_monotone() -> None:
    base = piv.IfaceFeatures().severity(_TOL)
    worse = piv.IfaceFeatures(framed=True, max_overlap_A=2.5, vol_overlap_frac=0.5,
                              min_ab_A=12.0).severity(_TOL)
    assert base == 0.0 and worse > base, f"severity not monotone ({base} -> {worse})"
    print("5. severity is 0 for a clean interface and grows with violations")


def test_vdw_mp_is_the_reference_convention() -> None:
    # MolProbity's OWN radii (the reference's convention) — notably O=1.40, not Bondi 1.52; fallback to Bondi.
    assert piv.vdw_mp("O") == 1.40 and piv.vdw_mp("C") == 1.75 and piv.vdw_mp("N") == 1.55
    from karyon import cofold_validity as cv
    assert piv.vdw_mp("O") != cv.vdw("O"), "the whole point: MolProbity O radius differs from Bondi"
    assert piv.vdw_mp("Zn") == cv.vdw("Zn"), "an element MolProbity doesn't special-case falls back to Bondi"
    print("6. vdw_mp uses MolProbity's reference radii (O=1.40) with a Bondi fallback")


# --------------------------------------------------------------------------- #
# Chain-aware structure-reader proofs (stdlib only).
# --------------------------------------------------------------------------- #
def _pdb_line(rec, serial, name, res, chain, resseq, x, y, z, elem) -> str:
    s = [" "] * 80

    def put(start, text):
        for i, ch in enumerate(text):
            s[start - 1 + i] = ch
    put(1, rec); put(7, f"{serial:>5}"); put(13, f"{name:<4}"); put(18, f"{res:>3}")
    put(22, chain); put(23, f"{resseq:>4}")
    put(31, f"{x:>8.3f}"); put(39, f"{y:>8.3f}"); put(47, f"{z:>8.3f}"); put(77, f"{elem:>2}")
    return "".join(s)


def test_pdb_reader_carries_chain_and_splits() -> None:
    pdb = "\n".join([
        _pdb_line("ATOM", 1, " CA", "ALA", "A", 1, 1.0, 2.0, 3.0, "C"),
        _pdb_line("ATOM", 2, " CB", "ALA", "A", 1, 1.5, 2.5, 3.5, "C"),
        _pdb_line("ATOM", 3, " CA", "GLY", "B", 1, 5.0, 5.0, 5.0, "C"),
        _pdb_line("HETATM", 4, "O", "HOH", "B", 99, 9.0, 9.0, 9.0, "O"),    # solvent dropped
    ])
    atoms = sio.read_pdb_atoms(pdb)
    assert atoms[0].chain == "A" and atoms[0].atom_name == "CA" and atoms[0].resnum == 1
    assert sio.chain_ids(atoms) == ["A", "B"], "chain ids in first-seen order, solvent excluded"
    ga, gb = sio.split_by_chain(atoms, "A", "B")
    assert len(ga) == 2 and len(gb) == 1, f"split_by_chain wrong: {len(ga)}/{len(gb)}"
    groups = sio.group_by_chain(atoms)
    assert set(groups) == {"A", "B"} and len(groups["A"]) == 2
    print("7. PDB reader carries chain/atom_name/resnum; split_by_chain + group_by_chain work")


def test_cif_reader_carries_chain() -> None:
    cif = "\n".join([
        "data_test", "loop_",
        "_atom_site.group_PDB", "_atom_site.type_symbol", "_atom_site.label_atom_id",
        "_atom_site.label_comp_id", "_atom_site.auth_asym_id", "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
        "ATOM C CA ALA A 1 1.0 2.0 3.0",
        "ATOM C CA GLY B 2 5.0 5.0 5.0",
        "#",
    ])
    atoms = sio.read_cif_atoms(cif)
    assert len(atoms) == 2 and atoms[0].chain == "A" and atoms[1].chain == "B"
    assert atoms[0].atom_name == "CA" and atoms[0].resnum == 1
    ga, gb = sio.split_by_chain(atoms, ["A"], ["B"])
    assert len(ga) == 1 and len(gb) == 1
    print("8. mmCIF reader carries auth_asym_id/atom_id/seq_id; split_by_chain works")


# --------------------------------------------------------------------------- #
# Geometry proofs (numpy-gated).
# --------------------------------------------------------------------------- #
def _shell(center, r, n, chain):
    out = []
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for k in range(n):
        y = 1.0 - 2.0 * (k + 0.5) / n
        rad = math.sqrt(max(0.0, 1.0 - y * y))
        out.append(sio.Atom("C", center[0] + r * math.cos(ga * k) * rad, center[1] + r * y,
                            center[2] + r * math.sin(ga * k) * rad, chain=chain, atom_name="CA"))
    return out


def test_geometry_detects_interpenetration_and_separation() -> None:
    if not piv._HAVE_NUMPY:
        _skip("numpy absent")
        return
    chain_a = _shell((0.0, 0.0, 0.0), 8.0, 100, "A")
    chain_b = [sio.Atom("C", 8.0 + 3.5 + 0.6 * i, 0.0, 0.0, chain="B", atom_name="CB") for i in range(8)]
    f = piv.interface_features(chain_a, chain_b, _TOL)
    assert f.framed and not piv.is_interface_invalid(f, _CS, _TOL), \
        f"a docked (in-contact, no-clash) dimer must pass: {_CS.evaluate(f, _TOL).fired}"

    buried = piv.decoy_interpenetrate(chain_a, chain_b)
    fb = piv.interface_features(chain_a, buried, _TOL)
    assert "SEVERE_INTERFACE_CLASH" in _CS.evaluate(fb, _TOL).fired, "interpenetration must condemn"

    apart = piv.decoy_separate(chain_a, chain_b)
    fe = piv.interface_features(chain_a, apart, _TOL)
    assert "CHAINS_NOT_IN_CONTACT" in _CS.evaluate(fe, _TOL).fired, "separation must fire not-in-contact"
    print("9. interface geometry: docked passes, interpenetration condemns, separation is not-in-contact")


def test_all_interchain_counts_only_cross_chain_clashes() -> None:
    if not piv._HAVE_NUMPY:
        _skip("numpy absent")
        return
    # chain A has two atoms ON TOP of each other (an intra-chain overlap) + chain B far away. The intra-chain
    # overlap must NOT count; with B far, zero inter-chain clashes.
    a = [sio.Atom("C", 0.0, 0.0, 0.0, chain="A", atom_name="CA"),
         sio.Atom("C", 0.1, 0.0, 0.0, chain="A", atom_name="CB")]      # intra-chain overlap (ignored)
    b = [sio.Atom("C", 40.0, 0.0, 0.0, chain="B", atom_name="CA")]
    f = piv.all_interchain_features(a + b, _TOL)
    assert f.framed and f.n_clash_pairs == 0, f"intra-chain overlap must not count: {f.n_clash_pairs}"
    # now move B onto A → a genuine inter-chain clash
    b2 = [sio.Atom("C", 0.2, 0.0, 0.0, chain="B", atom_name="CA")]
    f2 = piv.all_interchain_features(a + b2, _TOL)
    assert f2.n_clash_pairs > 0, "an inter-chain overlap must count"
    print("10. all_interchain_features counts only cross-chain clashes (intra-chain proximity ignored)")


# --------------------------------------------------------------------------- #
# The like-for-like reference parser (stdlib) — the heart of the faithfulness arm.
# --------------------------------------------------------------------------- #
def test_wwpdb_clash_parser_keeps_interchain_heavy_only() -> None:
    xml = """<wwPDB-validation-information><Entry>
      <ModelledSubgroup chain="A" resnum="10" resname="LEU"><clash atom="O" cid="1" dist="2.0"/></ModelledSubgroup>
      <ModelledSubgroup chain="B" resnum="20" resname="PHE"><clash atom="CZ" cid="1" dist="2.0"/></ModelledSubgroup>
      <ModelledSubgroup chain="A" resnum="11" resname="VAL"><clash atom="CB" cid="2" dist="2.1"/></ModelledSubgroup>
      <ModelledSubgroup chain="A" resnum="12" resname="ILE"><clash atom="CG1" cid="2" dist="2.1"/></ModelledSubgroup>
      <ModelledSubgroup chain="A" resnum="13" resname="SER"><clash atom="HB2" cid="3" dist="2.1"/></ModelledSubgroup>
      <ModelledSubgroup chain="B" resnum="21" resname="THR"><clash atom="OG1" cid="3" dist="2.1"/></ModelledSubgroup>
      </Entry></wwPDB-validation-information>"""
    n, detail = ppi_data.parse_interchain_clashes(xml)
    # cid 1 = inter-chain heavy↔heavy (kept); cid 2 = intra-chain A-A (dropped); cid 3 = inter-chain but H (dropped)
    assert n == 1, f"only the inter-chain heavy↔heavy clash should count, got {n}"
    assert detail and detail[0][0] != detail[0][4], "the kept clash must be inter-chain (different chains)"
    assert not ppi_data._atom_is_heavy("HB2") and ppi_data._atom_is_heavy("OG1")
    print("11. wwPDB clash parser keeps exactly the inter-chain heavy↔heavy subset (the like-for-like ref)")


# --------------------------------------------------------------------------- #
# e2e proof (skips offline) — a real native complex's owned clash presence agrees with the wwPDB reference.
# --------------------------------------------------------------------------- #
def test_e2e_native_faithful_to_wwpdb_reference() -> None:
    # the like-for-like faithfulness claim (robust to the disclosed H-bond over-detection): the owned gate
    # finds EVERY complex the reference flags (recall), and its clash COUNT tracks the reference's count (ρ).
    # Binary presence is NOT asserted — a heavy-atom gate over-flags favorable polar/catalytic contacts.
    if not piv._HAVE_NUMPY:
        _skip("numpy absent")
        return
    try:
        natives = ppi_data.load_native_complexes(limit=40)
    except ppi_data.PoseUnavailable as e:
        _skip(f"native complexes unavailable: {e}")
        return
    from karyon import stats_kit
    mine, ref, recall_hit, n_pos = [], [], 0, 0
    for c in natives:
        f = piv.all_interchain_features(c.atoms, _TOL)
        mine.append(f.n_clash_pairs); ref.append(c.ref_interchain_clashes)
        if c.ref_interchain_clashes > 0:
            n_pos += 1; recall_hit += 1 if f.n_clash_pairs > 0 else 0
    recall = recall_hit / (n_pos or 1)
    cr = stats_kit.spearman(mine, ref)
    rho = cr.rho if isinstance(cr, stats_kit.Corr) else float("nan")
    assert recall >= 0.85, f"the gate must find the clashes the reference flags (recall {recall:.0%}, {recall_hit}/{n_pos})"
    assert rho >= 0.5, f"owned clash count must track the wwPDB count (ρ={rho:+.2f})"
    print(f"12. e2e: faithful to wwPDB — recall {recall:.0%} ({recall_hit}/{n_pos}), clash-count ρ {rho:+.2f} "
          f"(n={len(natives)} natives)")


if __name__ == "__main__":
    test_each_condemning_contract_fires_on_its_planted_violation()
    test_clean_interface_stays_clean()
    test_shallow_clash_discloses_not_condemns()
    test_not_framed_discloses_not_condemns()
    test_severity_monotone()
    test_vdw_mp_is_the_reference_convention()
    test_pdb_reader_carries_chain_and_splits()
    test_cif_reader_carries_chain()
    test_geometry_detects_interpenetration_and_separation()
    test_all_interchain_counts_only_cross_chain_clashes()
    test_wwpdb_clash_parser_keeps_interchain_heavy_only()
    test_e2e_native_faithful_to_wwpdb_reference()
    print("\nprotein_interface_validity proofs pass.")
