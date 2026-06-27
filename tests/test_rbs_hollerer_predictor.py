"""test_rbs_hollerer_predictor — proofs for the BIG-DATA RBS predictor-margin probe (dual script / pytest).

Offline, always run: the gz parser keeps only valid 17-mers; the loader cache round-trips; the rich
featurizer has the expected dimension; the numpy ridge MATCHES the stdlib BayesRidge (the strong arm is
the same model, just faster); a planted same-library signal is RECOVERED de-novo by both fits; the OSTIR
construct reconstruction folds at the right start codon. Online (skips): the real Höllerer data runs
end-to-end and the rich learned arm clears a sane de-novo ρ.

    python tests/test_rbs_hollerer_predictor.py
"""

from __future__ import annotations

import csv
import gzip
import os
import random
import tempfile
from pathlib import Path

from karyon import linmodel as lm  # noqa: E402
from karyon import rbs_hollerer_data as hd  # noqa: E402
from karyon import rbs_hollerer_predictor as rp  # noqa: E402
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


def test_parser_keeps_valid_17mers() -> None:
    rows = [
        "RBS\t0\t480\tIFP480\tIFP480_fittor1\ttotal_reads",
        "TAAGGATACTTACGCAC\t0.04\t0.79\t0.591\t0.62\t13430",   # good
        "ACTCTGGATGTAATGTG\t0.00\t0.31\t0.147\t0.16\t12822",   # good
        "ACGTACGT\t0\t0\t0.5\t0.5\t100",                       # too short -> dropped
        "ACGTNCGTACGTACGTA\t0\t0\t0.5\t0.5\t100",              # has N -> dropped
        "ACGTACGTACGTACGTA\t0\t0\t\t0.5\t100",                 # blank IFP -> dropped
    ]
    raw = gzip.compress(("\n".join(rows) + "\n").encode())
    recs = hd._parse_gz(raw)
    assert len(recs) == 2 and all(len(r.sequence) == 17 for r in recs)
    assert recs[0].sequence == "TAAGGATACTTACGCAC" and abs(recs[0].ifp - 0.591) < 1e-9
    assert recs[0].reads == 13430
    print("1. gz parser keeps only valid 17-mer ACGT rows with a finite IFP480")


def test_cache_roundtrip() -> None:
    recs = [hd.Record("ACGTACGTACGTACGTA", 0.5, 200), hd.Record("TTTTTTTTTTTTTTTTT", 0.1, 300)]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "c.csv"
        hd._write_cache(path, recs)
        back = hd._read_cache(path)
    assert back == recs
    print("2. loader cache write/read round-trips records exactly")


def test_rich_feature_dim() -> None:
    seq = "ACGT" * 4 + "A"                  # 17 nt
    f = rp._rich_feat(seq)
    assert len(f) == lm.feature_dim(rp.KS_RICH) + rp.WIN * 4
    # core is the predecessor's exact featurizer
    assert len(rp._core_feat(seq)) == lm.feature_dim(rp.KS_CORE)
    print(f"3. rich featurizer dim = k-mer(1-4)+positional = {len(f)} (core dim {len(rp._core_feat(seq))})")


def test_numpy_matches_stdlib() -> None:
    if np is None:
        _skip("numpy not importable — the stdlib arm is the probe baseline regardless")
        return
    rng = random.Random(0)
    seqs = ["".join(rng.choice("ACGT") for _ in range(17)) for _ in range(400)]
    X = [rp._rich_feat(s) for s in seqs]
    y = [rng.uniform(0, 1) for _ in seqs]
    w_std = rp._fit_stdlib(X, y)
    w_np = rp._fit_numpy(X, y)
    max_dev = max(abs(a - b) for a, b in zip(w_std, w_np))
    assert max_dev < 1e-6, f"numpy ridge != stdlib BayesRidge (max |Δw|={max_dev:.2e})"
    print(f"4. the numpy strong arm is the SAME ridge as stdlib BayesRidge (max |Δw|={max_dev:.1e})")


def test_fit_recovers_planted_signal() -> None:
    """A planted same-library rule (IFP ↑ with A-content of the 17-mer); both fits recover it de-novo
    on held-out sequences — so a real de-novo ρ is the data's, not a harness artifact."""
    rng = random.Random(3)
    recs = []
    for _ in range(1500):
        s = "".join(rng.choice("ACGT") for _ in range(17))
        ifp = max(0.0, min(1.0, s.count("A") / 17 + rng.uniform(-0.05, 0.05)))
        recs.append(hd.Record(s, ifp, 500))
    tr, te = recs[:1000], recs[1000:]
    truth = [r.ifp for r in te]
    for fit in (rp._fit_stdlib,) + ((rp._fit_numpy,) if np is not None else ()):
        w = fit([rp._rich_feat(r.sequence) for r in tr], [r.ifp for r in tr])
        rho = rp._spearman([rp._predict(w, rp._rich_feat(r.sequence)) for r in te], truth)
        assert rho is not None and rho > 0.8, f"failed to recover a planted signal de-novo: {rho}"
    print(f"5. both fits recover a planted same-library signal de-novo (held-out ρ>0.8)")


def test_ostir_reconstruction_starts_at_atg() -> None:
    seq17 = "ACGTACGTACGTACGTA"
    full = rp._LEADER + seq17 + rp._BXB1
    start = len(rp._LEADER) + len(seq17) + 1        # 1-based ATG of bxb1
    assert full[start - 1:start + 2] == "ATG", "reconstruction must place bxb1 ATG right after the 17-mer"
    # the real pre-RBS constant abuts the variable region
    assert full[len(rp._LEADER) - 10:len(rp._LEADER)] == "GAGCTCGCAT"
    print("6. OSTIR construct reconstruction places the bxb1 ATG immediately 3' of the 17-mer")


def test_e2e_real_data() -> None:
    try:
        records = hd.load_records()                 # default rep r3
    except hd.DatasetUnavailable as e:
        _skip(f"Höllerer 300k RBS unreachable and not cached: {e}")
        return
    assert len(records) > 100_000 and all(len(r.sequence) == 17 for r in records[:100])
    out = rp.run(n_eval=1500)                        # small eval keeps the test quick; OSTIR cached if warm
    rich = [v for k, v in out["learned"].items() if k.startswith("rich") and v is not None]
    assert rich and max(rich) > 0.7, f"rich learned arm de-novo ρ too low: {out['learned']}"
    print(f"7. e2e: {len(records)} real Höllerer RBS; rich learned-arm de-novo ρ={max(rich):+.3f} "
          f"(OSTIR {rp._fmt(out['ostir'])})")


if __name__ == "__main__":
    test_parser_keeps_valid_17mers()
    test_cache_roundtrip()
    test_rich_feature_dim()
    test_numpy_matches_stdlib()
    test_fit_recovers_planted_signal()
    test_ostir_reconstruction_starts_at_atg()
    test_e2e_real_data()
    print("\nall rbs_hollerer_predictor tests pass.")
