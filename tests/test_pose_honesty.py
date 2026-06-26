"""test_pose_honesty — falsification proofs for the docking physical-validity probe (avenue 7).

The teeth: each legible contract fires on its planted violation with an auditable reason and a clean pose
stays clean (the legibility core); ENERGY_UNCHECKABLE discloses without condemning; severity is monotone;
and — rdkit/data-gated — a clean ETKDG conformer passes (the relative-bounds regression guard), the decoys
are invalid + deterministic, and Arm A discriminates. The pure-logic group runs without rdkit (contracts
over planted PoseFeatures); the e2e SKIPs if rdkit/data are absent. Dual pytest / __main__, mirroring
test_molnet_honesty.py.
"""

from __future__ import annotations

import os

from karyon import pose_validity as pv


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


# --------------------------------------------------------------------------- #
# Pure-logic proofs (no rdkit) — the contracts over planted PoseFeatures.
# --------------------------------------------------------------------------- #
_TOL = pv.Tol()
_CS = pv.validity_contracts()

# one planted-bad PoseFeatures per condemning contract, + the reason substring it must name
_PLANTED = [
    ("POSE_UNPARSEABLE", dict(parsed=False, has_3d=False), "sanitize"),
    ("NO_3D_CONFORMER", dict(parsed=True, has_3d=False), "3D conformer"),
    ("BOND_LENGTH_OUTLIER", dict(worst_bond_rel_hi=1.6), "reference length"),
    ("BOND_ANGLE_OUTLIER", dict(worst_angle_rel=1.5), "bond angle"),
    ("AROMATIC_RING_NONPLANAR", dict(max_ring_dev_A=0.5), "out of plane"),
    ("DOUBLE_BOND_NONPLANAR", dict(max_double_bond_deg=60.0), "twisted"),
    ("INTERNAL_STERIC_CLASH", dict(min_clash_rel=0.5), "van-der-Waals"),
    ("INTERNAL_STRAIN_ENERGY", dict(energy_ratio=500.0), "internal energy"),
]


def test_each_contract_fires_on_its_planted_violation() -> None:
    for name, fields, substr in _PLANTED:
        f = pv.PoseFeatures(**fields)
        v = _CS.evaluate(f, _TOL)
        assert name in v.fired, f"{name} did not fire on {fields} (fired {v.fired})"
        msg = next(r.message for r in v.reasons if r.contract == name)
        assert substr in msg, f"{name} reason {msg!r} missing {substr!r}"
    print(f"1. each of {len(_PLANTED)} contracts fires on its planted violation with the right reason")


def test_clean_pose_stays_clean() -> None:
    v = _CS.evaluate(pv.PoseFeatures(), _TOL)               # all defaults = a within-tolerance pose
    assert v.ok and v.reasons == (), f"a clean pose must not fire: {v.fired}"
    print("2. a within-tolerance pose fires nothing (Verdict.ok)")


def test_energy_uncheckable_discloses_not_condemns() -> None:
    f = pv.PoseFeatures(ref_ok=True, uff_ok=False, energy_ratio=1.0)
    v = _CS.evaluate(f, _TOL)
    assert "ENERGY_UNCHECKABLE" in v.fired and "INTERNAL_STRAIN_ENERGY" not in v.fired
    assert v.score == 0.0, "ENERGY_UNCHECKABLE must not raise the severity (weight 0)"
    assert not pv.is_invalid(f, _CS, _TOL), "a disclosure must not condemn the pose"
    print("3. ENERGY_UNCHECKABLE discloses (weight 0) without condemning — the stats_kit.Degenerate philosophy")


def test_severity_monotone() -> None:
    base = pv.PoseFeatures().severity(_TOL)
    worse = pv.PoseFeatures(min_clash_rel=0.4, energy_ratio=1000.0).severity(_TOL)
    assert base == 0.0 and worse > base, f"severity not monotone ({base} -> {worse})"
    print("4. severity is 0 for a clean pose and grows with violations (the ranking statistic)")


# --------------------------------------------------------------------------- #
# rdkit-gated proofs — real geometry.
# --------------------------------------------------------------------------- #
def test_clean_conformer_passes() -> None:
    if not pv._HAVE_RDKIT:
        _skip("rdkit absent")
        return
    clean = pv.clean_conformer("CC(C)Cc1ccc(cc1)C(C)C(=O)O")   # ibuprofen
    assert clean is not None
    f = pv.featurize(clean, _TOL)
    assert f.ref_ok, "the ETKDG reference must build for a drug-like molecule"
    assert not pv.is_invalid(f, _CS, _TOL), \
        f"a clean ETKDG+UFF conformer must pass (the relative-bounds guard): {_CS.evaluate(f, _TOL).fired}"
    print("5. a clean ETKDG conformer passes the DRC (relative-bounds regression guard)")


def test_clash_decoy_flags_clash() -> None:
    if not pv._HAVE_RDKIT:
        _skip("rdkit absent")
        return
    clean = pv.clean_conformer("Cn1cnc2c1c(=O)n(C)c(=O)n2C")   # caffeine
    decoy = pv.decoy_clash(clean, seed=1)
    f = pv.featurize(decoy, _TOL)
    assert "INTERNAL_STERIC_CLASH" in _CS.evaluate(f, _TOL).fired, "a clash decoy must fire the clash contract"
    assert pv.is_invalid(f, _CS, _TOL)
    print("6. an injected steric clash fires INTERNAL_STERIC_CLASH on real geometry")


def test_decoys_are_deterministic() -> None:
    if not pv._HAVE_RDKIT:
        _skip("rdkit absent")
        return
    clean = pv.clean_conformer("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    s1 = pv.featurize(pv.decoy_jitter(clean, seed=7), _TOL).severity(_TOL)
    s2 = pv.featurize(pv.decoy_jitter(clean, seed=7), _TOL).severity(_TOL)
    assert abs(s1 - s2) < 1e-9, f"same seed must give identical severity ({s1} != {s2})"
    print("7. decoys + featurize are deterministic (same seed -> identical severity)")


def test_e2e_arm_a_discriminates() -> None:
    if not pv._HAVE_RDKIT:
        _skip("rdkit absent")
        return
    from karyon import pose_honesty as ph
    try:
        from karyon.molnet_data import DatasetUnavailable, load_dataset
        load_dataset("esol")
    except Exception as e:  # noqa: BLE001 — DatasetUnavailable or import
        _skip(f"ESOL unavailable: {e}")
        return
    r = ph.run_arm_a(30, _TOL)
    assert r, "Arm A produced no result"
    assert r["auroc"] >= 0.85, f"instrument AUROC implausibly low ({r['auroc']:.3f})"
    assert r["pass_valid"] >= 0.8 and r["flag_decoy"] >= 0.8
    print(f"8. e2e Arm A discriminates valid vs decoy (AUROC {r['auroc']:.2f})")


def test_e2e_arm_b_faithfulness_or_skip() -> None:
    if not pv._HAVE_RDKIT:
        _skip("rdkit absent")
        return
    from karyon import pose_honesty as ph
    from karyon.pose_data import PoseUnavailable, load_poses
    try:
        load_poses("diffdock", limit=5)
    except PoseUnavailable as e:
        _skip(f"deposited poses unavailable: {e}")
        return
    r = ph.run_method("diffdock", _TOL, limit=15)
    assert r is not None
    assert 0.0 <= r.intra_agreement <= 1.0, "faithfulness agreement out of range"
    print(f"9. e2e Arm B faithfulness computable (intra agreement {r.intra_agreement:.0%} on a 15-pose sample)")


if __name__ == "__main__":
    test_each_contract_fires_on_its_planted_violation()
    test_clean_pose_stays_clean()
    test_energy_uncheckable_discloses_not_condemns()
    test_severity_monotone()
    test_clean_conformer_passes()
    test_clash_decoy_flags_clash()
    test_decoys_are_deterministic()
    test_e2e_arm_a_discriminates()
    test_e2e_arm_b_faithfulness_or_skip()
    print("\npose_honesty proofs pass.")
