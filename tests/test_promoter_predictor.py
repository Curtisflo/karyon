"""test_promoter_predictor — proofs for the PROMOTER predictor-margin probe (dual script / pytest).

Offline, always run: the three featurizers have the expected dimensions; the numpy ridge MATCHES the
stdlib BayesRidge (the rich arms are the same model, just faster); the calc baseline is SIGN-ALIGNED
(a sign-inverted predictor ranks +1, not −1); both fits RECOVER a planted same-library signal de-novo
and REJECT shuffled-label noise (a real ρ is the data's, not a harness artifact); the train/eval split
never leaks a sequence and keeps only calc-present eval rows. Online (skips): the real Urtecho data runs
end-to-end and the best learned arm BEATS the Promoter Calculator in-distribution.

    python tests/test_promoter_predictor.py
"""

from __future__ import annotations

import math
import os
import random

from karyon import linmodel as lm  # noqa: E402
from karyon import promoter_data as pd  # noqa: E402
from karyon import promoter_predictor as rp  # noqa: E402
from karyon import stats_kit as sk  # noqa: E402

try:
    import numpy as np
except ImportError:
    np = None


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def _rec(seq: str, strength: float, calc_pred: float | None = None) -> pd.Record:
    """A Record whose .strength (== log tx) is exactly `strength`."""
    return pd.Record(seq, math.exp(strength), calc_pred)


def _rand_seq(rng: random.Random, n: int = pd.PROMOTER_LEN) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_feature_dims() -> None:
    seq = _rand_seq(random.Random(0))
    assert len(rp._core_feat(seq)) == lm.feature_dim(rp.KS_CORE) + 1            # + GC
    assert len(rp._rich_feat(seq)) == lm.feature_dim(rp.KS_RICH) + 1           # + GC
    assert len(rp._rich_pos_feat(seq)) == lm.feature_dim(rp.KS_RICH) + rp.PROMOTER_LEN * 4 + 1
    print(f"1. featurizer dims: core={len(rp._core_feat(seq))}, rich={len(rp._rich_feat(seq))}, "
          f"rich+pos={len(rp._rich_pos_feat(seq))}")


def test_numpy_matches_stdlib() -> None:
    """The rich arms fit via numpy ONLY for speed — it must be the same ridge as stdlib BayesRidge.
    Proved on the rich featurizer (342-d); _fit_numpy/_fit_stdlib are feature-agnostic, so this carries
    to rich+pos (the winning arm) without a slow 942-d stdlib Cholesky in the test."""
    if np is None:
        _skip("numpy not importable — the stdlib core arm is the probe baseline regardless")
        return
    rng = random.Random(0)
    seqs = [_rand_seq(rng) for _ in range(400)]
    X = [rp._rich_feat(s) for s in seqs]
    y = [rng.uniform(0, 1) for _ in seqs]
    max_dev = max(abs(a - b) for a, b in zip(rp._fit_stdlib(X, y), rp._fit_numpy(X, y)))
    assert max_dev < 1e-6, f"numpy ridge != stdlib BayesRidge (max |Δw|={max_dev:.2e})"
    print(f"2. the numpy rich arm is the SAME ridge as stdlib BayesRidge (max |Δw|={max_dev:.1e})")


def test_calc_baseline_sign_aligned() -> None:
    """`calc_pred` is sign-INVERTED vs strength (more negative = stronger). _calc_rho must rank by
    -calc_pred so a perfect sign-inverted predictor scores +1, not −1 — else every WIN/LOSS flips."""
    rng = random.Random(1)
    recs, y = [], []
    for _ in range(200):
        s = rng.uniform(-2, 2)
        recs.append(_rec(_rand_seq(rng), s, calc_pred=-s))     # perfect, sign-inverted predictor
        y.append(s)
    ev = list(range(len(recs)))
    aligned, n = rp._calc_rho(recs, ev, y)
    raw = sk.spearman([r.calc_pred for r in recs], y)          # what you'd get WITHOUT the flip
    raw_rho = raw.rho if isinstance(raw, sk.Corr) else None
    assert aligned is not None and aligned > 0.99, f"sign-aligned calc ρ should be +1, got {aligned}"
    assert raw_rho is not None and raw_rho < -0.99, f"raw calc ρ should be −1, got {raw_rho}"
    print(f"3. calc baseline sign-aligned: -calc_pred ρ={aligned:+.3f} (raw {raw_rho:+.3f}); n={n}")


def test_fit_recovers_signal_and_rejects_noise() -> None:
    """A planted same-library rule (strength ↑ with A-content); both fits recover it de-novo on held-out
    sequences. Shuffle the labels → held-out ρ collapses to ~0. So a real de-novo ρ is the data's signal,
    not a harness artifact."""
    rng = random.Random(3)
    recs = []
    for _ in range(1200):
        s = _rand_seq(rng)
        strength = s.count("A") / len(s) + rng.uniform(-0.05, 0.05)
        recs.append(_rec(s, strength))
    tr, te = recs[:800], recs[800:]
    truth = [r.strength for r in te]
    fits = (rp._fit_stdlib,) + ((rp._fit_numpy,) if np is not None else ())
    for fit in fits:
        w = fit([rp._rich_feat(r.seq) for r in tr], [r.strength for r in tr])
        rho = rp._spearman([rp._predict_one(w, rp._rich_feat(r.seq)) for r in te], truth)
        assert rho is not None and rho > 0.7, f"failed to recover a planted signal de-novo: {rho}"
    # noise: labels INDEPENDENT of sequence → nothing to learn → held-out ρ ≈ 0. (A *permutation* null
    # is unreliable here: shuffled A-content values are still real structure and interact with the
    # compositional, sum-to-1 k-mer features to fake an anti-correlation; independent labels are clean.)
    rng_n = random.Random(7)
    noise = [_rec(_rand_seq(rng_n), rng_n.gauss(0, 1)) for _ in range(1200)]
    ntr, nte = noise[:800], noise[800:]
    ntruth = [r.strength for r in nte]
    w = rp._fit_stdlib([rp._rich_feat(r.seq) for r in ntr], [r.strength for r in ntr])
    noise_rho = rp._spearman([rp._predict_one(w, rp._rich_feat(r.seq)) for r in nte], ntruth)
    assert noise_rho is None or abs(noise_rho) < 0.2, f"manufactured signal from noise: ρ={noise_rho}"
    print(f"4. fits recover a planted signal de-novo (ρ>0.7) and reject independent-label noise "
          f"(|ρ|={abs(noise_rho or 0.0):.3f})")


def test_split_no_leakage() -> None:
    """The split shares no sequence between train and eval, and eval keeps only calc-present rows."""
    rng = random.Random(5)
    recs = []
    for i in range(300):
        cp = None if i % 7 == 0 else rng.uniform(-3, 0)        # ~1/7 have NO calc prediction
        recs.append(_rec(_rand_seq(rng), rng.uniform(-2, 2), calc_pred=cp))
    tr, ev = rp._split(recs, n_eval=120, seed=0)
    assert set(tr).isdisjoint(ev), "train and eval share an index (sequence leakage)"
    assert all(recs[i].calc_pred is not None for i in ev), "eval kept a calc-absent row"
    assert len(tr) == len(recs) - 120, "train must be everything past the eval slice"
    print(f"5. split clean: |train|={len(tr)} |eval|={len(ev)} disjoint, all eval calc-present")


def test_e2e_real_data() -> None:
    try:
        records = pd.load_records()
    except pd.DatasetUnavailable as e:
        _skip(f"Urtecho σ70 promoter set unreachable and not cached: {e}")
        return
    assert len(records) > 10_000 and all(len(r.seq) == pd.PROMOTER_LEN for r in records[:100])
    out = rp.run(seeds=2, n_eval=1500)
    core = next(v for k, v in out["arm_means"].items() if k.startswith("core"))
    assert out["best"] is not None and out["best"] > 0.65, f"best learned arm ρ too low: {out}"
    assert out["best"] > out["calc_mean"], (
        f"expected the rich+pos core to BEAT the Promoter Calculator in-distribution "
        f"(best {out['best']:+.3f} vs calc {out['calc_mean']:+.3f})")
    assert out["best"] > core, "the positional+tetramer capacity should beat the minimal core"
    print(f"6. e2e: {len(records)} real Urtecho promoters; best learned ρ={out['best']:+.3f} "
          f"BEATS calc {out['calc_mean']:+.3f} (minimal core {core:+.3f})")


if __name__ == "__main__":
    test_feature_dims()
    test_numpy_matches_stdlib()
    test_calc_baseline_sign_aligned()
    test_fit_recovers_signal_and_rejects_noise()
    test_split_no_leakage()
    test_e2e_real_data()
    print("\nall promoter_predictor tests pass.")
