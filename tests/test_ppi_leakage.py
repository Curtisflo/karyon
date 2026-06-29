"""test_ppi_leakage — proofs for the PPI node-/sequence-leakage honesty audit (dual pytest / __main__).

The teeth: the holdout split populates all three leakage classes and never leaks a train pair into test;
the leakage contracts fire on the right signals and a novel/clean pair stays clean; the node baseline reads
a PLANTED node-leakage signal (C1 ≫ C3); and the decisive control — **shuffling the labels collapses the
measured inflation to ≈0**, so the harness reports leakage only when node identity genuinely tracks the label
(it cannot manufacture an inflation from the split mechanics). The first four run without network (synthetic);
the e2e SKIPs if the benchmark is unreachable + uncached.
"""

from __future__ import annotations

import os
import random

from karyon import ppi_leakage as pl
from karyon.ppi_leakage_data import (
    DatasetUnavailable,
    Pair,
    Protein,
    holdout_split,
    load_pairs,
    proteins_in,
)

_AA = "ACDEFGHIKLMNPQRSTVWY"


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def _synthetic(seed: int = 0, n_prot: int = 220, n_pairs: int = 2600) -> list[Pair]:
    """A PPI set with PLANTED node-leakage: each protein has a latent 'hubness'; a pair's label is driven by
    the two hubnesses (+ noise). So a protein's training positive-degree predicts its test-pair labels — the
    exact structure a random split leaks — but ONLY while labels track hubness (shuffling must destroy it)."""
    rng = random.Random(seed)
    prots = [Protein(f"P{i:04d}", "".join(rng.choice(_AA) for _ in range(60))) for i in range(n_prot)]
    hub = {p.pid: rng.random() for p in prots}
    pairs, seen = [], set()
    while len(pairs) < n_pairs:
        a, b = rng.choice(prots), rng.choice(prots)
        if a.pid == b.pid or (a.pid, b.pid) in seen or (b.pid, a.pid) in seen:
            continue
        seen.add((a.pid, b.pid))
        score = hub[a.pid] + hub[b.pid] + rng.gauss(0, 0.15)
        pairs.append(Pair(a, b, 1 if score > 1.0 else 0))
    return pairs


def _shuffle_labels(pairs: list[Pair], seed: int = 0) -> list[Pair]:
    labels = [p.label for p in pairs]
    random.Random(seed).shuffle(labels)
    return [Pair(p.a, p.b, lab) for p, lab in zip(pairs, labels)]


def _outcome(a_seen: bool, b_seen: bool, near_dup: bool, label: int = 0) -> pl.Outcome:
    return pl.Outcome(Pair(Protein("A", "AAA"), Protein("B", "CCC"), label),
                      a_seen, b_seen, near_dup, 0.9 if near_dup else 0.1, 0.0, 0.0)


def test_holdout_split_populates_classes_and_train_pair_disjoint() -> None:
    pairs = _synthetic()
    s = holdout_split(pairs, seed=0)
    train_prots = proteins_in(s.train)
    train_keys = {frozenset((p.a.pid, p.b.pid)) for p in s.train}
    cls = {"C1": 0, "C2": 0, "C3": 0}
    for p in s.test:
        assert frozenset((p.a.pid, p.b.pid)) not in train_keys, "a train pair leaked into test"
        a, b = p.a.pid in train_prots, p.b.pid in train_prots
        cls["C1" if a and b else "C2" if a or b else "C3"] += 1
    assert all(cls.values()), f"a leakage class is empty: {cls}"
    assert proteins_in(s.train) and not (
        {p.a.pid for p in s.test if p.a.pid not in train_prots} & train_prots)
    print(f"1. holdout split: train/test {s.sizes}; classes {cls}; no train pair in test")


def test_leakage_contracts_fire_and_clean_stays_clean() -> None:
    cs, tv = pl.leakage_contracts(), pl.TrainView(tau=0.7)
    leaked = cs.evaluate(_outcome(a_seen=True, b_seen=True, near_dup=True), tv)
    clean = cs.evaluate(_outcome(a_seen=False, b_seen=False, near_dup=False), tv)
    assert {"BOTH_PARTNERS_SEEN", "PARTNER_SEEN_IN_TRAIN", "NEAR_DUP_PROTEIN"} <= set(leaked.fired), leaked.fired
    assert clean.ok, f"a novel, dissimilar pair must not leak: {clean.fired}"
    assert _outcome(True, True, False).leak_class == "C1"
    assert _outcome(True, False, False).leak_class == "C2"
    assert _outcome(False, False, False).leak_class == "C3"
    print(f"2. leakage contracts fire {leaked.fired}; novel pair clean; C1/C2/C3 classification correct")


def test_node_baseline_reads_planted_leakage() -> None:
    r = pl.run_one(seed=0, pairs=_synthetic(seed=1))
    assert r is not None
    assert r["au_c1_node"] > 0.70, f"planted node-leakage should make C1 strong ({r['au_c1_node']:.3f})"
    assert abs(r["au_c3_node"] - 0.5) < 1e-6, f"degree memorization must be chance on novel C3 ({r['au_c3_node']:.3f})"
    assert r["inflation"] > 0.10, f"node-identity inflation should be large ({r['inflation']:+.3f})"
    print(f"3. node baseline reads planted leakage: C1 {r['au_c1_node']:.3f} vs C3 {r['au_c3_node']:.3f} "
          f"(inflation {r['inflation']:+.3f})")


def test_shuffle_control_kills_inflation() -> None:
    pairs = _synthetic(seed=2)
    real = pl.run_one(seed=0, pairs=pairs)
    shuf = pl.run_one(seed=0, pairs=_shuffle_labels(pairs, seed=7))
    assert real["inflation"] > 0.10, f"un-shuffled must show real leakage ({real['inflation']:+.3f})"
    assert shuf["inflation"] < 0.05, f"shuffling labels must collapse the inflation ({shuf['inflation']:+.3f})"
    print(f"4. shuffle control: inflation {real['inflation']:+.3f} (real) → {shuf['inflation']:+.3f} (labels "
          f"shuffled) — the harness reports leakage only when node identity tracks the label")


def test_e2e_real_benchmark() -> None:
    try:
        load_pairs()
    except DatasetUnavailable as e:
        _skip(f"PPI benchmark unreachable and not cached: {e}")
        return
    r = pl.run_one(seed=0)
    assert r is not None
    assert r["prevalence"] > 0.50, f"node-leakage implausibly rare ({r['prevalence']:.1%})"
    assert r["inflation"] > 0.10, f"real node-identity inflation should be large ({r['inflation']:+.3f})"
    assert abs(r["au_c3_node"] - 0.5) < 0.05, f"honest C3 should sit near chance ({r['au_c3_node']:.3f})"
    print(f"5. e2e: real PPI leakage large (inflation {r['inflation']:+.3f}, C1 {r['au_c1_node']:.3f} → "
          f"C3 {r['au_c3_node']:.3f}); prevalence {r['prevalence']:.0%}")


if __name__ == "__main__":
    test_holdout_split_populates_classes_and_train_pair_disjoint()
    test_leakage_contracts_fire_and_clean_stays_clean()
    test_node_baseline_reads_planted_leakage()
    test_shuffle_control_kills_inflation()
    test_e2e_real_benchmark()
    print("\nppi_leakage proofs pass.")
