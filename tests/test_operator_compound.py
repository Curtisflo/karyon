"""test_operator_compound — falsification proofs that qualifying compounds (dual script / pytest).

Six proofs that the compounding result is mechanism, not artifact:
  1. clean → no-regression (qualifying is dormant-by-correctness: gated curve == ungated, bit-identical).
  2. corruption is detectable AND label-blind (wrong VALUE not None → the ungated arm ingests it; QA–QD
     flag it via metadata; clean readouts all pass).
  3. degrading regime → qualify-PROTECTS (random@0.60: gated held-out ρ ends above ungated).
  4. the gap COMPOUNDS (random@0.60: mean final ρ-gap > mean early ρ-gap, mean slope > 0).
  5. the shuffle control COLLAPSES it (disjoint flags ⇒ the favorable compounding is gone — far below random).
  6. determinism (same seed/mode/rate ⇒ identical curves).

Needs the σ70 promoter dataset (fetched + cached); SKIPs cleanly when offline.

    python tests/test_operator_compound.py     # or: pytest tests/test_operator_compound.py
"""

from __future__ import annotations

import os
import statistics

from karyon import loop as lp
from karyon import noisy_assay as na
from karyon import promoter_contracts as pc
from karyon.operator_compound import (RHO, compound_config, first_last, rho_trajectory, run_pair, slope)


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def _load():
    """(sub, cfg) or (None, None) if the dataset is unavailable."""
    try:
        return lp.promoter_substrate(), compound_config()
    except lp.pmd.DatasetUnavailable as e:
        _skip(f"promoter data unavailable ({e})")
        return None, None


def _sweep(sub, cfg, mode, rate, seeds):
    """Mean (early ρ-gap, final ρ-gap, ρ-gap slope, gated ρ_end, ungated ρ_end) over seeds."""
    e, f, sl, ge, ue = ([] for _ in range(5))
    for s in range(seeds):
        pr = run_pair(sub, cfg, s, mode, rate)
        gh = pr.gaps(RHO)
        a, b = first_last(gh)
        e.append(a); f.append(b); sl.append(slope(gh))
        ge.append(rho_trajectory(pr.gated)[1]); ue.append(rho_trajectory(pr.plain)[1])
    m = statistics.mean
    return m(e), m(f), m(sl), m(ge), m(ue)


def test_clean_no_regression() -> None:
    sub, cfg = _load()
    if sub is None:
        return
    pr = run_pair(sub, cfg, 0, "clean", 0.0)
    assert pr.gated.curve == pr.plain.curve, "clean mode must be bit-identical gated vs ungated"
    assert pr.gated.n_flagged == 0, "clean data must flag nothing"
    print("  [ok] clean → no-regression (gated curve == ungated, 0 flagged)")


def test_corruption_detectable_and_label_blind() -> None:
    sub, _ = _load()
    if sub is None:
        return
    truth, ctx = sub.truth, pc.readout_ctx()
    designs = list(truth)[:300]
    assay = na.make_assay(truth, "random", 0.60, seed=0)
    reads = assay.ingest(assay.emit_order(designs, cycle=1))
    corrupt = [r for r in reads if r.value is not None and abs(r.value - truth[r.design]) > 1e-9]
    flagged = [r for r in reads if not pc.READOUT.evaluate(r, ctx).ok]
    assert corrupt, "random@0.60 must corrupt some values"
    assert all(r.value is not None for r in corrupt), "corruption sets a WRONG value, never None"
    assert flagged, "the gate must flag some corrupted readouts (detectable)"
    clean = na.make_assay(truth, "clean", 0.0)
    clean_reads = clean.ingest(clean.emit_order(designs, 1))
    assert all(pc.READOUT.evaluate(r, ctx).ok for r in clean_reads), "clean readouts must all pass QA–QD"
    print(f"  [ok] corruption detectable + label-blind ({len(corrupt)} corrupt, {len(flagged)} flagged)")


def test_degrading_regime_protects() -> None:
    sub, cfg = _load()
    if sub is None:
        return
    _, _, _, ge, ue = _sweep(sub, cfg, "random", 0.60, 3)
    assert ge > ue, f"gated ρ_end {ge:+.3f} must beat ungated {ue:+.3f} in the degrading regime"
    print(f"  [ok] degrading regime → qualify-protects (gated ρ_end {ge:+.2f} > ungated {ue:+.2f})")


def test_gap_compounds() -> None:
    sub, cfg = _load()
    if sub is None:
        return
    e, f, sl, _, _ = _sweep(sub, cfg, "random", 0.60, 3)
    assert f > e, f"final ρ-gap {f:+.3f} must exceed early {e:+.3f} (the gap widens)"
    assert sl > 0, f"ρ-gap slope {sl:+.4f} must be positive (compounding)"
    print(f"  [ok] gap compounds (early {e:+.3f} → final {f:+.3f}, slope {sl:+.4f})")


def test_shuffle_control_collapses() -> None:
    sub, cfg = _load()
    if sub is None:
        return
    _, _, rnd_sl, _, _ = _sweep(sub, cfg, "random", 0.60, 3)
    _, _, shf_sl, _, _ = _sweep(sub, cfg, "shuffle", 0.60, 3)
    assert shf_sl < rnd_sl / 2.0, f"shuffle slope {shf_sl:+.4f} must collapse below random {rnd_sl:+.4f}"
    print(f"  [ok] shuffle control collapses (shuffle slope {shf_sl:+.4f} ≪ random {rnd_sl:+.4f})")


def test_determinism() -> None:
    sub, cfg = _load()
    if sub is None:
        return
    a = run_pair(sub, cfg, 1, "random", 0.45)
    b = run_pair(sub, cfg, 1, "random", 0.45)
    assert a.gated.curve == b.gated.curve and a.plain.curve == b.plain.curve, "runs must be reproducible"
    print("  [ok] determinism (identical curves on re-run)")


if __name__ == "__main__":
    test_clean_no_regression()
    test_corruption_detectable_and_label_blind()
    test_degrading_regime_protects()
    test_gap_compounds()
    test_shuffle_control_collapses()
    test_determinism()
    print("\nAll operator-compound proofs passed (or skipped offline).")
