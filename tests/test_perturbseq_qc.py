"""test_perturbseq_qc — proofs for the single-cell Perturb-seq silent-failure QC (dual pytest / __main__).

The teeth: the contracts fire on weak knockdown / low cells / unmeasured and a clean strong-KD perturbation
stays clean; the incumbent calibrates (controls < targeting); the layer reads a PLANTED weak-KD→no-phenotype
silent-failure signal (enrichment, with knockdown orthogonal to significance); and the decisive control —
**shuffling on-target knockdown collapses the enrichment to ≈1×**, so the harness reports silent failures
only when knockdown genuinely tracks the no-phenotype pile. The first four run without h5py/data (synthetic
Perturbation records); the e2e SKIPs if the pseudobulk is unreachable + uncached.
"""

from __future__ import annotations

import os
import random

from karyon import perturbseq_qc as pq
from karyon.perturbseq_data import DatasetUnavailable, Perturbation, load_perturbations


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def _p(target="GENE", control=False, kd=0.1, ep=1e-4, cells=150) -> Perturbation:
    return Perturbation(f"id_{target}_P1P2_ENSG", target, control, kd, ep,
                        50 if ep < 0.05 else 0, cells, float(cells))


def _synthetic(seed: int = 0, n: int = 600) -> list[Perturbation]:
    """Plant the silent-failure structure: failed knockdown (resid>0.5) → no phenotype; good knockdown →
    usually a phenotype. So weak-KD concentrates in the no-phenotype pile (enrichment), while within that
    pile knockdown varies independently of the (compressed) significance."""
    rng = random.Random(seed)
    perts: list[Perturbation] = []
    for i in range(60):                                            # non-targeting controls
        ep = 1e-3 if rng.random() < 0.15 else rng.uniform(0.05, 1.0)
        perts.append(Perturbation(f"c{i}_non-targeting_x_y", "non-targeting", True,
                                  float("nan"), ep, 0, rng.randint(80, 300), float("nan")))
    for i in range(n):                                             # targeting
        if rng.random() < 0.30:                                   # knockdown failed → mostly no phenotype
            resid = rng.uniform(0.6, 1.5)
            ep = 1e-3 if rng.random() < 0.10 else rng.uniform(0.05, 1.0)   # rare spurious hit (weak_hit > 0)
        else:                                                     # knockdown worked → usually a phenotype
            resid = rng.uniform(0.0, 0.4)
            ep = 1e-4 if rng.random() < 0.80 else rng.uniform(0.05, 1.0)   # significance ⟂ resid in no-hit pile
        cells = rng.randint(40, 400)
        if rng.random() < 0.05:
            resid = float("nan")                                  # unmeasured
        perts.append(Perturbation(f"t{i}_GENE{i}_P1P2_ENSG{i}", f"GENE{i}", False,
                                   resid, ep, 50 if ep < 0.05 else 0, cells, float(cells)))
    return perts


def test_contracts_fire_and_clean_stays_clean() -> None:
    cs, view = pq.qc_contracts(), pq.QCView(pq.WEAK_KD_FLOOR, pq.MIN_CELLS)
    assert "WEAK_KNOCKDOWN" in cs.evaluate(_p(kd=0.8), view).fired
    assert "LOW_CELL_COUNT" in cs.evaluate(_p(cells=5), view).fired
    assert "KNOCKDOWN_UNMEASURED" in cs.evaluate(_p(kd=float("nan")), view).fired
    assert cs.evaluate(_p(kd=0.1, cells=200), view).ok, "a strong-KD, well-powered perturbation must stay clean"
    print("1. contracts fire on weak-KD / low-cells / unmeasured; a clean perturbation stays clean")


def test_incumbent_calibrates_and_planted_signal_recovered() -> None:
    r = pq.run_one(perts=_synthetic(seed=1))
    assert r is not None
    assert r["cal_t"] > 0.5 > r["cal_c"], f"incumbent should separate targeting from controls ({r['cal_t']:.2f}/{r['cal_c']:.2f})"
    assert r["enrich"] >= 2.0, f"planted weak-KD→no-phenotype enrichment should be ≥2× ({r['enrich']:.1f})"
    assert r["q2"] <= 0.20, f"clear hits should rarely be flagged weak-KD ({r['q2']:.2f})"
    print(f"2. incumbent calibrates ({r['cal_t']:.0%}/{r['cal_c']:.0%}); planted silent-failure recovered "
          f"(enrich {r['enrich']:.1f}×, precision {r['q2']:.0%})")


def test_non_redundancy_guard() -> None:
    r = pq.run_one(perts=_synthetic(seed=2))
    assert r["rho"] < 0.30, f"knockdown must be non-redundant with significance in the no-phenotype pile ({r['rho']:.3f})"
    print(f"3. non-redundancy: |ρ(knockdown, energy-p)| = {r['rho']:.3f} in the no-phenotype pile (< 0.30)")


def test_knockdown_shuffle_collapses_enrichment() -> None:
    perts = _synthetic(seed=3)
    real = pq.run_one(perts=perts)
    shuf = pq.run_one(perts=pq._shuffle_knockdown(perts, seed=9))
    assert real["enrich"] >= 2.0, f"un-shuffled must show the planted silent-failure signal ({real['enrich']:.1f}×)"
    assert shuf["enrich"] < 1.6, f"shuffling knockdown must collapse the enrichment toward 1× ({shuf['enrich']:.1f}×)"
    print(f"4. shuffle control: enrichment {real['enrich']:.1f}× (real) → {shuf['enrich']:.1f}× (knockdown shuffled)")


def test_e2e_real_screen() -> None:
    try:
        load_perturbations()
    except DatasetUnavailable as e:
        _skip(f"Perturb-seq pseudobulk unreachable / h5py absent: {e}")
        return
    r = pq.run_one()
    assert r is not None
    assert r["cal_t"] > 0.5 > r["cal_c"], "deposited caller should be calibrated"
    assert r["q1"] >= 0.15, f"silent-failure flag rate implausibly low ({r['q1']:.2f})"
    assert r["q2"] <= 0.20, f"precision guard failed ({r['q2']:.2f})"
    assert r["rho"] < 0.30, f"non-redundancy guard failed (|ρ|={r['rho']:.3f})"
    assert r["enrich"] >= 2.0, f"weak-KD enrichment in no-phenotype implausibly low ({r['enrich']:.1f}×)"
    print(f"5. e2e: real Perturb-seq — flagged {r['q1']:.0%}, precision {r['q2']:.0%}, |ρ|={r['rho']:.3f}, "
          f"enrichment {r['enrich']:.1f}×")


if __name__ == "__main__":
    test_contracts_fire_and_clean_stays_clean()
    test_incumbent_calibrates_and_planted_signal_recovered()
    test_non_redundancy_guard()
    test_knockdown_shuffle_collapses_enrichment()
    test_e2e_real_screen()
    print("\nperturbseq_qc proofs pass.")
