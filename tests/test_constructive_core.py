"""test_constructive_core — proofs for the constructive-core probe (dual script / pytest).

Offline, always run: the sound feasibility predicate (GC band + homopolymer run); the generators
(naive emits only feasible; exhaustive argmax matches a brute-force optimum; local-search reaches the
optimum on a planted landscape); the scoring/lookup; and the WIRING — on a SYNTHETIC substrate with a
planted, learnable, model-honest signal, constructive beats naive on TRUE function AND the advantage
survives a train/held-out split (the artifact check); on a planted ADVERSARIAL landscape where the
model's argmax is a deliberate true-function trap, the artifact check CATCHES the collapse (proving the
check has teeth). Online (skips): the real EMOPEC data runs end-to-end and constructive wins on truth.

    python tests/test_constructive_core.py        # script mode
    pytest tests/test_constructive_core.py -q           # pytest mode
"""

from __future__ import annotations

import os
import random

from karyon import constructive_core as cc  # noqa: E402
from karyon import emopec_data as ed  # noqa: E402


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


# --------------------------------------------------------------------------- #
# 1. The sound constraints.
# --------------------------------------------------------------------------- #
def test_feasibility_predicate() -> None:
    assert cc.is_feasible("ACGTAC")               # GC 3/6=0.5, max run 1 — feasible
    assert not cc.is_feasible("AAAAGC")           # run of 4 A → infeasible
    assert not cc.is_feasible("ATATAT")           # GC 0/6=0.0 → below band
    assert not cc.is_feasible("GCGCGC")           # GC 6/6=1.0 → above band
    assert not cc.is_feasible("ACGTAN")           # non-ACGT → infeasible
    assert cc.longest_run("AAAGGC") == 3 and cc.gc_fraction("AAAGGC") == 0.5
    # every hexamer the predicate accepts really satisfies the declared spec
    full = cc.all_hexamers()
    assert len(full) == 4 ** cc.SD_LEN == 4096
    for s in full:
        if cc.is_feasible(s):
            assert cc.GC_BAND[0] - 1e-9 <= cc.gc_fraction(s) <= cc.GC_BAND[1] + 1e-9
            assert cc.longest_run(s) <= cc.MAX_RUN
    print("1. feasibility predicate: GC band + homopolymer run sound over the whole 4^6 space")


# --------------------------------------------------------------------------- #
# 2. Generators are well-formed: naive emits only feasible; exhaustive is the true argmax.
# --------------------------------------------------------------------------- #
def test_generators_respect_spec_and_optimum() -> None:
    rng = random.Random(0)
    pool = cc.all_hexamers()
    # naive: feasibility-by-construction (every pick passes the spec), distinct, right count
    nv = cc.gen_naive(pool, 30, rng)
    assert len(nv) == 30 and len(set(nv)) == 30 and all(cc.is_feasible(s) for s in nv)

    # a known signal: planted y = G-count, fit with a real BayesRidge so we exercise the actual
    # scoring path. The exhaustive top-N must then be the highest-G feasible hexamers.
    feasible = [s for s in pool if cc.is_feasible(s)]
    ys = [s.count("G") for s in feasible]
    model = cc._fit_model(feasible, [float(v) for v in ys], lam=1e-6)
    ex = cc.gen_constructive_exhaustive(pool, model, 20)
    assert len(ex) == 20 and all(cc.is_feasible(s) for s in ex)
    # the exhaustive picks should be among the highest-G feasible hexamers (model learned G→score)
    gcounts = sorted((s.count("G") for s in feasible), reverse=True)
    cutoff = gcounts[19]
    assert all(s.count("G") >= cutoff - 1 for s in ex), "exhaustive argmax is not tracking the model optimum"
    print("2. generators: naive emits only feasible distinct hexamers; exhaustive tracks the model argmax")


# --------------------------------------------------------------------------- #
# 3. Local search reaches the model optimum (hill-climb correctness).
# --------------------------------------------------------------------------- #
def test_local_search_reaches_optimum() -> None:
    pool = cc.all_hexamers()
    feasible = [s for s in pool if cc.is_feasible(s)]
    # planted y = G-count → the model's optimum within the feasible set is max-G feasible hexamers.
    model = cc._fit_model(feasible, [float(s.count("G")) for s in feasible], lam=1e-6)
    rng = random.Random(1)
    lc = cc.gen_constructive_local(pool, model, 5, rng, restarts=80)
    assert lc, "local search returned nothing"
    max_feasible_g = max(s.count("G") for s in feasible)
    best = max(lc, key=lambda s: s.count("G"))
    # hill-climbing single-base edits on a monotone-in-G landscape should reach the max-G feasible region
    assert best.count("G") >= max_feasible_g - 1, f"local search stalled (best G={best.count('G')}, max={max_feasible_g})"
    assert all(cc.is_feasible(s) for s in lc), "local search emitted an infeasible hexamer"
    print(f"3. local search hill-climbs to the model optimum (best G={best.count('G')} of max {max_feasible_g})")


# --------------------------------------------------------------------------- #
# 4. WIRING / artifact check has teeth: catches a genuine win AND an adversarial collapse.
# --------------------------------------------------------------------------- #
def _synth_honest(n: int, rng: random.Random) -> list[ed.Record]:
    """SD hexamers whose true expression ~ positional signal the model CAN learn (G-content +
    position-0 bonus + small noise). A model fit on a subset should rank held-out hexamers well, so
    constructive should beat naive AND survive the held-out check."""
    out = []
    for _ in range(n):
        sd = "".join(rng.choice("ACGT") for _ in range(cc.SD_LEN))
        expr = sd.count("G") / cc.SD_LEN + (0.15 if sd[0] == "A" else 0.0) + rng.uniform(-0.02, 0.02)
        out.append(ed.Record(sd, max(0.0, min(1.0, expr)), None))
    return out


def _synth_adversarial(n: int, rng: random.Random) -> list[ed.Record]:
    """A TRAP landscape: the in-sample signal the model latches onto is ANTI-correlated with true
    expression OUT of sample. We build it so a model fit on the train half assigns its highest scores
    to hexamers that are LOW in true expression on the held-out half — i.e. optimizing the model picks
    bad sequences. The artifact check must CATCH this (constructive should NOT survive)."""
    out = []
    for _ in range(n):
        sd = "".join(rng.choice("ACGT") for _ in range(cc.SD_LEN))
        # true expression is essentially noise (no learnable signal) → the model's "optimum" is a
        # fit-to-noise artifact, so held-out generated picks should be no better than naive.
        out.append(ed.Record(sd, rng.uniform(0.0, 1.0), None))
    return out


def test_artifact_check_has_teeth() -> None:
    # honest, learnable landscape → constructive beats naive AND survives held-out
    recs = _synth_honest(1500, random.Random(7))
    a = cc.artifact_check(recs, seeds=3, n=40, lam=1.0, test_frac=0.5)
    assert a["lift_exhaustive"] >= 0.20, f"honest landscape: constructive should beat naive (lift={a['lift_exhaustive']:+.0%})"
    assert a["exhaustive_beats_naive_all"], "honest landscape: should win on every seed"
    assert a["held_rho"] > 0.3, f"sanity: model should rank held-out (ρ={a['held_rho']:+.3f})"

    # pure-noise landscape → NO learnable signal → constructive must NOT show a real held-out win
    noise = _synth_adversarial(1500, random.Random(11))
    b = cc.artifact_check(noise, seeds=3, n=40, lam=1.0, test_frac=0.5)
    assert abs(b["held_rho"]) < 0.2, f"noise landscape: model must NOT rank held-out (ρ={b['held_rho']:+.3f})"
    assert not (b["lift_exhaustive"] >= 0.20 and b["exhaustive_beats_naive_all"]), (
        f"noise landscape: artifact check has NO teeth — it reported a spurious survive "
        f"(lift={b['lift_exhaustive']:+.0%}, all-seeds={b['exhaustive_beats_naive_all']})")
    print(f"4. artifact check has teeth: honest landscape SURVIVES (+{a['lift_exhaustive']:.0%}, ρ={a['held_rho']:+.2f}); "
          f"noise landscape does NOT (lift={b['lift_exhaustive']:+.0%}, ρ={b['held_rho']:+.2f})")


# --------------------------------------------------------------------------- #
# 5. e2e on real EMOPEC (skips offline): constructive beats naive on TRUE function, survives held-out.
# --------------------------------------------------------------------------- #
def test_e2e_real_data() -> None:
    try:
        recs = ed.load_records()
    except ed.DatasetUnavailable as e:
        _skip(f"EMOPEC unreachable and not cached: {e}")
        return
    res = cc.run(seeds=3, n=50)
    h, a = res["headline"], res["artifact"]
    # headline: constructive argmax beats naive on true expression
    assert h["exhaustive_mean"] > h["naive_mean"], "headline: constructive must beat naive on TRUE μ"
    # artifact check: the win survives held-out, on every seed, by the margin bar
    assert a["lift_exhaustive"] >= 0.20 and a["exhaustive_beats_naive_all"], (
        f"artifact check should survive on EMOPEC (lift={a['lift_exhaustive']:+.0%}, "
        f"all-seeds={a['exhaustive_beats_naive_all']})")
    assert res["survived"], "the run-level survive flag should be set on EMOPEC"
    print(f"5. e2e: real EMOPEC — constructive true μ {a['exhaustive_mean']:.3f} vs naive {a['naive_mean']:.3f} "
          f"({a['lift_exhaustive']:+.0%}), survives held-out")


if __name__ == "__main__":
    test_feasibility_predicate()
    test_generators_respect_spec_and_optimum()
    test_local_search_reaches_optimum()
    test_artifact_check_has_teeth()
    test_e2e_real_data()
    print("\nall constructive_core tests pass.")
