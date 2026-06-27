"""test_crispr_qc_benchmark — proofs for the CRISPRi QC ownership benchmark (dual script / pytest).

Offline, always run: the shared wide feature vector is fixed-width; a PLANTED linear rule is recovered by
every baseline (legible, rich-linear, and the GBM ceiling) while the shuffled control collapses to chance —
so a small ceiling−legible gap means "the sequence has little signal," not "the harness is broken." Online
(skips offline): on real Horlbeck data the legible layer sits within a hair of the non-legible ceiling
(the headline finding) and both clear the shuffled floor.

    python tests/test_crispr_qc_benchmark.py
"""

from __future__ import annotations

import os
import random

from karyon import crispr_qc as qc
from karyon import crispr_qc_data as cd
from karyon import crispr_qc_benchmark as bench


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def _rand_seq(rng: random.Random, n: int = 20) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_wide_features_fixed_width() -> None:
    """The shared feature vector must be the same width for any guide (k-mer freqs are length-invariant)."""
    rng = random.Random(0)
    widths = {len(bench.wide_features(_rand_seq(rng, n))) for n in (18, 20, 23, 25)}
    assert len(widths) == 1, f"wide_features width varies with sequence length: {widths}"
    print(f"1. wide_features fixed width across lengths 18–25 ({widths.pop()} dims)")


def test_planted_rule_recovered_by_all_and_noise_rejected() -> None:
    """Plant activity = GC + small noise. Every baseline should recover it (held-out ρ high); the shuffled
    control must collapse. This isolates 'is there signal' from 'is the model legible'."""
    rng = random.Random(3)
    recs = [cd.Record(f"G{i}", s := _rand_seq(rng), qc.gc(s) + rng.uniform(-0.05, 0.05)) for i in range(1200)]
    r = bench.evaluate_seed(recs, seed=0)
    assert r["legible"] > 0.7, f"legible failed to recover the planted rule (ρ={r['legible']:.3f})"
    assert r["rich_linear"] > 0.7, f"rich-linear failed (ρ={r['rich_linear']:.3f})"
    if r["ceiling_gbm"] is not None:
        assert r["ceiling_gbm"] > 0.6, f"ceiling GBM failed to recover the planted rule (ρ={r['ceiling_gbm']:.3f})"
    assert abs(r["shuffled"]) < 0.2, f"shuffled control did not collapse (ρ={r['shuffled']:.3f})"
    print(f"2. planted GC→activity recovered by all (legible {r['legible']:.2f}, rich {r['rich_linear']:.2f}, "
          f"ceiling {r['ceiling_gbm'] and round(r['ceiling_gbm'],2)}); shuffled {r['shuffled']:+.2f}")


def test_e2e_legibility_is_cheap_on_real_data() -> None:
    """The headline: on the real Horlbeck set the non-legible ceiling does NOT meaningfully beat the legible
    layer — legibility is ~free because the sequence-only ceiling itself is low."""
    try:
        cd.load_records()
    except cd.DatasetUnavailable as e:
        _skip(f"Horlbeck activity set unreachable and not cached: {e}")
        return
    out = bench.run_guide(seeds=2)
    m = out["means"]
    assert m["legible"] > 0.25, f"legible ρ unexpectedly low: {m['legible']:+.3f}"
    assert m["shuffled"] < 0.10, f"shuffled floor too high: {m['shuffled']:+.3f}"
    if m["ceiling_gbm"] is not None:
        gap = m["ceiling_gbm"] - m["legible"]
        assert gap < 0.10, f"ceiling beats legible by {gap:+.3f} — legibility is NOT cheap here (revisit the claim)"
        print(f"3. e2e: legible {m['legible']:+.3f} ≈ ceiling {m['ceiling_gbm']:+.3f} "
              f"(gap {gap:+.3f}); shuffled {m['shuffled']:+.3f} — legibility is ~free, ceiling is low")
    else:
        print(f"3. e2e: legible {m['legible']:+.3f}; shuffled {m['shuffled']:+.3f} (sklearn absent → no ceiling)")


if __name__ == "__main__":
    test_wide_features_fixed_width()
    test_planted_rule_recovered_by_all_and_noise_rejected()
    test_e2e_legibility_is_cheap_on_real_data()
    print("\nall crispr_qc_benchmark tests pass.")
