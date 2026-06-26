"""test_hossain_predictor — proofs for the IN-VIVO promoter predictor-margin probe (dual script / pytest).

Offline, always run: the loader's row→Record mapping picks the right columns and filters junk; the three
featurizers have the expected dims; the numpy ridge MATCHES stdlib BayesRidge; the calc baseline is
SIGN-ALIGNED; both fits recover a planted signal de-novo and reject independent-label noise; the split
never leaks a sequence. Online (skips): the real Hossain in-vivo set runs end-to-end and the best learned
arm BEATS the (held-out) Promoter Calculator.

    cd bio/probe && python test_hossain_predictor.py
"""

from __future__ import annotations

import math
import os
import random

from karyon import hossain_data as hd  # noqa: E402
from karyon import hossain_predictor as hp  # noqa: E402
from karyon import linmodel as lm  # noqa: E402
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


def _rec(seq: str, strength: float, calc_pred: float | None = None) -> hd.Record:
    return hd.Record(seq, math.exp(strength), calc_pred)


def _rand_seq(rng: random.Random, n: int = 78) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_loader_row_mapping() -> None:
    """_row_to_record reads seq=col C, tx=col E, calc=col G; filters non-ACGT / non-positive / missing tx;
    keeps a row with a blank calc (calc_pred=None); normalizes U->T."""
    good = hd._row_to_record({"A": "id", "B": "AAAA", "C": "acgtACGT", "D": "TTTT", "E": "12.5", "G": "-3.2"})
    assert good is not None and good.seq == "ACGTACGT" and good.tx == 12.5 and good.calc_pred == -3.2
    assert hd._row_to_record({"C": "ACGT", "G": "-3.2"}) is None              # missing TX
    assert hd._row_to_record({"C": "ACGTN", "E": "5.0"}) is None             # non-ACGT
    assert hd._row_to_record({"C": "ACGT", "E": "0"}) is None                # TX must be > 0
    blank = hd._row_to_record({"C": "ACGT", "E": "5.0", "G": ""})
    assert blank is not None and blank.calc_pred is None                     # kept, calc absent
    assert hd._row_to_record({"C": "ACGU", "E": "5.0", "G": "-1.0"}).seq == "ACGT"  # U->T
    print("1. loader row→Record maps cols C/E/G, filters junk, keeps blank-calc rows, normalizes U->T")


def test_feature_dims() -> None:
    seq = _rand_seq(random.Random(0))
    win = 78
    assert len(hp._core_feat(seq)) == lm.feature_dim(hp.KS_CORE) + 1
    assert len(hp._rich_feat(seq)) == lm.feature_dim(hp.KS_RICH) + 1
    assert len(hp._rich_pos_feat(seq, win)) == lm.feature_dim(hp.KS_RICH) + win * 4 + 1
    print(f"2. featurizer dims: core={len(hp._core_feat(seq))}, rich={len(hp._rich_feat(seq))}, "
          f"rich+pos[{win}]={len(hp._rich_pos_feat(seq, win))}")


def test_numpy_matches_stdlib() -> None:
    if np is None:
        _skip("numpy not importable — the stdlib core arm is the probe baseline regardless")
        return
    rng = random.Random(0)
    seqs = [_rand_seq(rng) for _ in range(400)]
    X = [hp._rich_feat(s) for s in seqs]
    y = [rng.uniform(0, 1) for _ in seqs]
    max_dev = max(abs(a - b) for a, b in zip(hp._fit_stdlib(X, y), hp._fit_numpy(X, y)))
    assert max_dev < 1e-6, f"numpy ridge != stdlib BayesRidge (max |Δw|={max_dev:.2e})"
    print(f"3. the numpy rich arm is the SAME ridge as stdlib BayesRidge (max |Δw|={max_dev:.1e})")


def test_calc_baseline_sign_aligned() -> None:
    """`calc_pred` is sign-INVERTED vs strength; _calc_rho ranks by -calc_pred so a perfect sign-inverted
    predictor scores +1, not −1."""
    rng = random.Random(1)
    recs, y = [], []
    for _ in range(200):
        s = rng.uniform(-2, 2)
        recs.append(_rec(_rand_seq(rng), s, calc_pred=-s))
        y.append(s)
    ev = list(range(len(recs)))
    aligned, n = hp._calc_rho(recs, ev, y)
    raw = sk.spearman([r.calc_pred for r in recs], y)
    raw_rho = raw.rho if isinstance(raw, sk.Corr) else None
    assert aligned is not None and aligned > 0.99, f"sign-aligned calc ρ should be +1, got {aligned}"
    assert raw_rho is not None and raw_rho < -0.99, f"raw calc ρ should be −1, got {raw_rho}"
    print(f"4. calc baseline sign-aligned: -calc_pred ρ={aligned:+.3f} (raw {raw_rho:+.3f}); n={n}")


def test_fit_recovers_signal_and_rejects_noise() -> None:
    """Planted same-library rule (strength ↑ with A-content) recovered de-novo (ρ>0.7); labels INDEPENDENT
    of sequence → held-out ρ ≈ 0 (the clean null — not a permutation null, which fakes anti-correlation
    through the compositional features)."""
    rng = random.Random(3)
    recs = []
    for _ in range(1200):
        s = _rand_seq(rng)
        recs.append(_rec(s, s.count("A") / len(s) + rng.uniform(-0.05, 0.05)))
    tr, te = recs[:800], recs[800:]
    truth = [r.strength for r in te]
    fits = (hp._fit_stdlib,) + ((hp._fit_numpy,) if np is not None else ())
    for fit in fits:
        w = fit([hp._rich_feat(r.seq) for r in tr], [r.strength for r in tr])
        rho = hp._spearman([hp._predict_one(w, hp._rich_feat(r.seq)) for r in te], truth)
        assert rho is not None and rho > 0.7, f"failed to recover a planted signal de-novo: {rho}"
    rng_n = random.Random(7)
    noise = [_rec(_rand_seq(rng_n), rng_n.gauss(0, 1)) for _ in range(1200)]
    ntr, nte = noise[:800], noise[800:]
    ntruth = [r.strength for r in nte]
    w = hp._fit_stdlib([hp._rich_feat(r.seq) for r in ntr], [r.strength for r in ntr])
    noise_rho = hp._spearman([hp._predict_one(w, hp._rich_feat(r.seq)) for r in nte], ntruth)
    assert noise_rho is None or abs(noise_rho) < 0.2, f"manufactured signal from noise: ρ={noise_rho}"
    print(f"5. fits recover a planted signal de-novo (ρ>0.7) and reject independent-label noise "
          f"(|ρ|={abs(noise_rho or 0.0):.3f})")


def test_split_no_leakage() -> None:
    rng = random.Random(5)
    recs = []
    for i in range(300):
        cp = None if i % 7 == 0 else rng.uniform(-3, 0)
        recs.append(_rec(_rand_seq(rng), rng.uniform(-2, 2), calc_pred=cp))
    tr, ev = hp._split(recs, n_eval=120, seed=0)
    assert set(tr).isdisjoint(ev), "train and eval share an index (sequence leakage)"
    assert all(recs[i].calc_pred is not None for i in ev), "eval kept a calc-absent row"
    assert len(tr) == len(recs) - 120
    print(f"6. split clean: |train|={len(tr)} |eval|={len(ev)} disjoint, all eval calc-present")


def test_e2e_real_data() -> None:
    try:
        records = hd.load_records()
    except hd.DatasetUnavailable as e:
        _skip(f"Hossain in-vivo set unreachable and not cached: {e}")
        return
    assert len(records) > 3000 and len(set(len(r.seq) for r in records)) == 1
    out = hp.run(seeds=2, n_eval=800)
    core = next(v for k, v in out["arm_means"].items() if k.startswith("core"))
    assert out["best"] is not None and out["best"] > 0.7, f"best learned arm ρ too low: {out}"
    assert out["best"] > out["calc_mean"], (
        f"expected rich+pos to BEAT the (held-out) Promoter Calculator in-vivo "
        f"(best {out['best']:+.3f} vs calc {out['calc_mean']:+.3f})")
    assert out["best"] > core, "positional+tetramer capacity should beat the minimal core"
    print(f"7. e2e: {len(records)} real Hossain in-vivo promoters; best learned ρ={out['best']:+.3f} "
          f"BEATS calc {out['calc_mean']:+.3f} (minimal core {core:+.3f})")


if __name__ == "__main__":
    test_loader_row_mapping()
    test_feature_dims()
    test_numpy_matches_stdlib()
    test_calc_baseline_sign_aligned()
    test_fit_recovers_signal_and_rejects_noise()
    test_split_no_leakage()
    test_e2e_real_data()
    print("\nall hossain_predictor tests pass.")
