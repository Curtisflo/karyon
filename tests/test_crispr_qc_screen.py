"""test_crispr_qc_screen — proofs for the real-screen (SD7) confirmation (dual script / pytest).

Offline, always run: the SD7 sgId parser splits non-targeting controls from targeting guides and builds
the right (gene, strand, coord) join key; the NT null floor is mean − K·sd; `essential_subset` keeps only
real growth hits with enough guides; and the fixed-test gene-disjoint evaluation RECOVERS a planted
activity→phenotype rule while shuffled training collapses to chance. Online (skips offline): the real
join runs end-to-end, the activity↔gamma link is strongly negative, and the sequence-only score predicts
real phenotype above the shuffled baseline with monotonic terciles.

    cd karyon/probe && python test_crispr_qc_screen.py
"""

from __future__ import annotations

import os
import random

from karyon import crispr_qc as qc
from karyon import crispr_qc_data as cd
from karyon import crispr_qc_screen as scr


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def _rand_seq(rng: random.Random, n: int = 20) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_sgid_parsing() -> None:
    assert cd._sgid_is_ntc("non-targeting_00123") and cd._sgid_is_ntc("negative_control_5")
    assert not cd._sgid_is_ntc("AARS_+_70323441.23-P1")
    assert cd._sgid_key("AARS_+_70323441.23-P1") == ("AARS", "+", "70323441")
    assert cd._sgid_key("weird") is None
    print("1. sgId parser: NT controls split off; targeting sgId -> (gene, strand, coord) join key")


def test_null_floor() -> None:
    ntc = [-0.02, 0.02] * 5                                  # mean 0, pstdev 0.02 -> floor = -3·0.02
    assert abs(scr.null_floor(ntc) - (-0.06)) < 1e-9
    print(f"2. NT null floor = mean − {scr.NULL_K:.0f}·sd (Contract 3 calibration)")


def test_essential_subset() -> None:
    recs = ([cd.ScreenRecord("ESS", "s", 0.9, g) for g in (-0.4, -0.1, -0.05, 0.0)]      # a real hit
            + [cd.ScreenRecord("WEAK", "s", 0.1, g) for g in (-0.1, -0.05, 0.0, 0.02)]    # never strong
            + [cd.ScreenRecord("FEW", "s", 0.9, g) for g in (-0.6, -0.5)])                # too few guides
    ess = scr.essential_subset(recs)
    assert set(ess) == {"ESS"}, f"expected only ESS, got {set(ess)}"
    print("3. essential_subset keeps real hits (best gamma < threshold, ≥MIN_GUIDES), drops the rest")


def test_evaluate_recovers_planted_signal() -> None:
    """Plant a clean rule: activity = GC; real gamma = −1.2·activity + 0.3 (strong guide ⇒ negative gamma,
    weak guide ⇒ ~0 = silent). The gene-disjoint evaluation must recover it (AUROC well above chance) and
    the shuffled-training baseline must collapse to ~0.5."""
    rng = random.Random(11)
    # noisy on purpose — a razor-clean rule lets finite-sample shuffle residue fake a non-0.5 baseline;
    # real data is noisy (its shuffle baseline lands at ~0.47), so the synthetic must be too.
    full = [cd.Record(f"T{g}", s := _rand_seq(rng), qc.gc(s) + rng.gauss(0, 0.05))
            for g in range(100) for _ in range(6)]
    srecs = [cd.ScreenRecord(f"E{g}", s := _rand_seq(rng), a := qc.gc(s) + rng.gauss(0, 0.05),
                             -1.2 * a + 0.3 + rng.gauss(0, 0.05))
             for g in range(60) for _ in range(6)]
    ntc = [rng.gauss(0, 0.02) for _ in range(300)]
    floor = scr.null_floor(ntc)
    test, has_pheno, aurocs, shuf = scr.evaluate(full, srecs, floor, seeds=3)
    assert 0 < sum(has_pheno) < len(has_pheno), "planted test set is degenerate (all/none phenotype)"
    m_auroc = sum(aurocs) / len(aurocs)
    m_shuf = sum(shuf) / len(shuf)
    assert m_auroc > 0.70, f"failed to recover the planted activity→phenotype rule (AUROC={m_auroc:.3f})"
    assert abs(m_shuf - 0.5) < 0.15, f"shuffled-training baseline not ~0.5 (AUROC={m_shuf:.3f})"
    print(f"4. evaluate recovers a planted activity→phenotype rule (AUROC={m_auroc:.2f}) vs shuffled "
          f"({m_shuf:.2f}); test n={len(test)}")


def test_e2e_real_screen() -> None:
    try:
        srecs, ntc = cd.load_screen()
    except cd.DatasetUnavailable as e:
        _skip(f"SD7 real screen unreachable and not cached: {e}")
        return
    assert len(srecs) > 2000 and len(ntc) > 1000
    out = scr.run(seeds=2)
    assert out["within_ess_rho"] < -0.5, f"activity↔gamma link not strong-negative: {out['within_ess_rho']}"
    assert out["auroc_mean"] > 0.55, f"seq-only real-phenotype AUROC too low: {out['auroc_mean']:.3f}"
    assert out["auroc_mean"] > out["shuf_auroc_mean"] + 0.08, "barely beats the shuffled baseline"
    assert out["tercile_gammas"][0] > out["tercile_gammas"][-1], (
        "median gamma should grow MORE negative from the weakest- to the strongest-predicted tercile")
    print(f"5. e2e: {out['n_matched']} matched guides; within-essential ρ={out['within_ess_rho']:+.3f}; "
          f"seq-only AUROC {out['auroc_mean']:.3f} > shuffled {out['shuf_auroc_mean']:.3f}; terciles "
          f"{[round(g, 3) for g in out['tercile_gammas']]}")


if __name__ == "__main__":
    test_sgid_parsing()
    test_null_floor()
    test_essential_subset()
    test_evaluate_recovers_planted_signal()
    test_e2e_real_screen()
    print("\nall crispr_qc_screen tests pass.")
