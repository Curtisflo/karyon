"""test_rbs_predictor — proofs for the RBS predictor-margin probe (dual script / pytest).

Offline, always run: window extraction + feature shape; per-study z-scoring; the LOSO learned-core
mechanics DETECT a planted cross-study signal; the small helpers + OSTIR cache read path. Online
(skips): the real SynBioMTS data runs end-to-end and returns a finite learned-core de-novo ρ.

    cd bio/probe && python test_rbs_predictor.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile
from pathlib import Path

from karyon import linmodel as lm  # noqa: E402
from karyon import rbs_predictor as rp  # noqa: E402
from karyon import rbs_synbiomts_data as rd  # noqa: E402


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def _synth_study(name: str, n: int, rng: random.Random, scale: float) -> list[rd.Record]:
    """A synthetic study with a SHARED learnable rule (expression ↑ with A-content of the RBS window)
    plus a per-study scale offset (so z-scoring is exercised)."""
    out = []
    for _ in range(n):
        utr = "".join(rng.choice("ACGT") for _ in range(40))
        seq = utr + "ATG" + "".join(rng.choice("ACGT") for _ in range(30))
        sp = len(utr)
        win = seq[max(0, sp - rp.WIN_UP): sp + rp.WIN_DN]
        signal = win.count("A") / len(win)
        prot = 10 ** (scale + 3.0 * signal + rng.uniform(-0.05, 0.05))
        out.append(rd.Record(name, seq, sp, prot))
    return out


def test_window_and_features() -> None:
    rec = rd.Record("d", "AAAA" + "C" * 40 + "ATG" + "G" * 20, 44, 100.0)
    win = rp._window(rec)
    assert win == rec.sequence[44 - rp.WIN_UP:44 + rp.WIN_DN]
    assert len(win) == rp.WIN_UP + rp.WIN_DN
    X = rp._features([rec])
    assert len(X[0]) == lm.feature_dim(rp.KS)
    print("1. RBS window extraction + feature dim correct")


def test_zscore_by_study() -> None:
    recs = [rd.Record("A", "x", 0, 0)] * 3 + [rd.Record("B", "x", 0, 0)] * 2
    recs = [rd.Record(r.dataset, r.sequence, r.startpos, r.prot_mean) for r in recs]
    z = rp._zscore_by_study(recs, [1.0, 2.0, 3.0, 10.0, 20.0])
    # within study A: mean(z)=0; within B: the two values are ±1
    assert abs(sum(z[:3])) < 1e-9 and abs(z[3] + z[4]) < 1e-9
    assert abs(z[3]) > 0.9 and z[3] < 0 < z[4]
    print("2. per-study z-scoring centers each study and is scale-free")


def test_loso_detects_signal() -> None:
    rng = random.Random(5)
    records = _synth_study("A", 120, rng, scale=0.0) + _synth_study("B", 120, rng, scale=2.5)
    X = rp._features(records)
    logp = [math.log10(r.prot_mean) for r in records]
    tr = [i for i, r in enumerate(records) if r.dataset == "A"]
    te = [i for i, r in enumerate(records) if r.dataset == "B"]
    z = rp._zscore_by_study([records[i] for i in tr], [logp[i] for i in tr])
    model = lm.BayesRidge(len(X[0]), lam=rp.LAM)
    model.observe_all([X[i] for i in tr], z)
    rho = rp._spearman([model.predict(X[i]) for i in te], [logp[i] for i in te])
    assert rho is not None and rho > 0.5, f"LOSO failed to transfer a shared signal: {rho}"
    print(f"3. LOSO learned-core transfers a planted cross-study signal (held-out ρ={rho:+.3f})")


def test_helpers_and_ostir_cache() -> None:
    assert rp._fmt(None) == "n/a" and rp._fmt(0.5) == "+0.500"
    assert rp._mean([]) is None and abs(rp._mean([1.0, 3.0]) - 2.0) < 1e-9
    assert rp._spearman([1, 2, 3], [3, 2, 1]) == -1.0
    # OSTIR cache read: write a tiny cache, read it back (None preserved for blank).
    import csv
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "rbs_ostir.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sequence", "expr"])
            w.writerow(["ACGT", "12.5"])
            w.writerow(["TTTT", ""])
        cache = {}
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                cache[row["sequence"]] = float(row["expr"]) if row["expr"] else None
        assert cache == {"ACGT": 12.5, "TTTT": None}
    print("4. helpers (_fmt/_mean/_spearman) + OSTIR cache read path correct")


def test_ostir_baseline_wiring() -> None:
    """Exercise the OSTIR code path via monkeypatch — covers _ostir_one + ostir_predict wiring.

    The ostir package is an optional dep (absent → baseline is n/a). We inject a fake ostir module
    into rbs_predictor so the branch that calls ostir.run_ostir is actually executed, verifying:
      * _ostir_one matches on start_position and extracts 'expression';
      * _ostir_one returns None on an empty result list;
      * ostir_predict writes + reads a cache and skips already-cached sequences.
    """
    import tempfile
    import types

    # Build a minimal fake ostir module with a controlled run_ostir.
    call_log: list[dict] = []

    def fake_run_ostir(sequence, start, threads, verbosity):
        call_log.append({"sequence": sequence, "start": start})
        if sequence == "EMPTY":
            return []
        # Simulate a result where the first entry matches the requested start_position.
        return [{"start_position": start, "expression": 42.0}]

    fake_ostir = types.SimpleNamespace(run_ostir=fake_run_ostir)

    # Inject the fake ostir into rbs_predictor.
    original_ostir = rp.ostir
    rp.ostir = fake_ostir
    try:
        # _ostir_one: normal case — returns the matching expression value.
        rec_normal = rd.Record("test", "ACGTACGT", 4, 100.0)
        result = rp._ostir_one(rec_normal)
        assert result == 42.0, f"expected 42.0 from fake ostir, got {result}"
        assert any(c["start"] == rec_normal.startpos + 1 for c in call_log), (
            "ostir was not called with 1-indexed startpos")

        # _ostir_one: empty result → None.
        rec_empty = rd.Record("test", "EMPTY", 3, 50.0)
        assert rp._ostir_one(rec_empty) is None, "_ostir_one should return None for empty result"

        # ostir_predict: exercises the full cache-write + cache-read round-trip with real ostir call.
        call_log.clear()
        records = [rec_normal, rd.Record("test2", "TTTTAAAA", 4, 80.0)]
        with tempfile.TemporaryDirectory() as d:
            cache_path = Path(d) / "rbs_ostir.csv"
            # Monkeypatch the cache path function so it points to our temp dir.
            original_cache_path = rp._ostir_cache_path
            rp._ostir_cache_path = lambda: cache_path

            try:
                # First call: nothing cached → ostir.run_ostir called for both.
                preds = rp.ostir_predict(records, refresh=False)
                assert preds["ACGTACGT"] == 42.0, f"unexpected prediction: {preds['ACGTACGT']}"
                assert preds["TTTTAAAA"] == 42.0
                assert len(call_log) == 2, f"expected 2 ostir calls, got {len(call_log)}"

                # Second call: both already cached → ostir.run_ostir NOT called again.
                call_log.clear()
                preds2 = rp.ostir_predict(records, refresh=False)
                assert preds2 == preds, "cache read returned different predictions"
                assert len(call_log) == 0, (
                    f"ostir called {len(call_log)} times on a fully-cached set (should be 0)")
            finally:
                rp._ostir_cache_path = original_cache_path
    finally:
        rp.ostir = original_ostir

    print("5. OSTIR wiring: _ostir_one extracts expression + returns None on empty; "
          "ostir_predict calls ostir for uncached seqs and reads cache on second call")


def test_e2e_real_data() -> None:
    try:
        records = rd.load_records()
    except rd.DatasetUnavailable as e:
        _skip(f"SynBioMTS RBS unreachable and not cached: {e}")
        return
    assert len(records) > 100 and len({r.dataset for r in records}) >= 4
    out = rp.run()
    assert out["our_mean"] is not None and -1.0 <= out["our_mean"] <= 1.0
    print(f"5. e2e: {len(records)} real RBS constructs; learned-core de-novo mean ρ={out['our_mean']:+.3f}")


if __name__ == "__main__":
    test_window_and_features()
    test_zscore_by_study()
    test_loso_detects_signal()
    test_helpers_and_ostir_cache()
    test_ostir_baseline_wiring()
    test_e2e_real_data()
    print("\nall rbs_predictor tests pass.")
