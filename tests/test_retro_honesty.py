"""test_retro_honesty — proofs for the retrosynthesis benchmark-honesty layer (dual pytest / __main__).

The teeth: the similarity index finds an exact duplicate; planted leakage fires the right contracts (and
a clean reaction stays clean); a no-leakage split shows ~0 residual; the stdlib retriever recovers a planted
exact-duplicate reaction (the near-dup-detector instrument check); the AUROC metric separates a planted
sim↔correctness signal and is null under shuffle; the patent-disjoint split never lets a patent straddle.
The real-data e2e SKIPs offline.
"""

from __future__ import annotations

import os
import random

from karyon import retro_honesty as rh
from karyon.retro_baseline import Outcome, ProductIndex, class_accuracy, kmers
from karyon.stats_kit import MannWhitney
from karyon.uspto_data import (
    DatasetUnavailable,
    Reaction,
    Split,
    load_reactions,
    patent_disjoint_split,
    random_split,
)


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def _rxn(rid: str, klass: int, product: str, reactants: str) -> Reaction:
    return Reaction(rid=rid, klass=klass, product=product, reactant_sig=reactants)


# Distinctive synthetic products (k-mer-disjoint enough that nn_sim is meaningful on tiny sets).
_P = {
    "a": "CCCCCCCCCCBr",
    "b": "c1ccccc1OCNNN",
    "c": "FC(F)(F)c2ccncc2SS",
    "d": "O=C(N)C3CC3PPPP",
    "novel": "IIIIWWWWQQQQzzzz",          # shares no k-mers with any of the above
}


def test_kmers_and_index_similarity() -> None:
    assert kmers("ABCDE", 3) == {"ABC", "BCD", "CDE"}
    a, b = kmers("CCCCCCCCCC"), kmers("CCCCCCCCCC")
    assert len(a & b) / len(a | b) == 1.0                      # identical → Jaccard 1.0
    assert kmers("AAAAA") & kmers("GGGGG") == set()            # disjoint → 0
    idx = ProductIndex([_P["a"], _P["b"], _P["c"]], df_cap_frac=1.0)
    nn = idx.query(_P["a"], top_n=3)
    assert nn and nn[0][0] == 0 and nn[0][1] == 1.0, f"exact product not found at sim 1.0: {nn}"
    assert idx.exact_count(_P["a"]) == 1 and idx.exact_count(_P["novel"]) == 0
    print("1. similarity index: k-mers + Jaccard + exact-dup retrieval at sim 1.0")


def test_planted_leakage_fires_right_contracts() -> None:
    train = [_rxn("PAT1", 1, _P["a"], "x.y"), _rxn("PAT2", 2, _P["b"], "p.q")]
    leaked = _rxn("PAT1", 1, _P["a"], "x.y")          # same product AND same patent as a train row
    clean = _rxn("PAT9", 3, _P["novel"], "z.w")       # unique product, unique patent, dissimilar
    audit = rh.audit_split(Split("synthetic", train, [leaked, clean]))
    v_leaked, v_clean = audit.verdicts
    assert "EXACT_PRODUCT_IN_TRAIN" in v_leaked.fired, v_leaked.fired
    assert "SAME_PATENT_IN_TRAIN" in v_leaked.fired, v_leaked.fired
    assert v_clean.ok, f"a unique reaction should not leak: {v_clean.fired}"
    print(f"2. planted leakage flagged with reasons {v_leaked.fired}; clean reaction stays clean")


def test_no_leakage_control() -> None:
    train = [_rxn(f"T{i}", 1, p, "x.y") for i, p in enumerate([_P["a"], _P["b"], _P["c"]])]
    test = [_rxn("U0", 1, _P["novel"], "z.w"), _rxn("U1", 2, _P["d"], "m.n")]
    audit = rh.audit_split(Split("synthetic", train, test))
    assert sum(audit.leaked) == 0, f"no-leakage control leaked {sum(audit.leaked)}"
    print("3. no-leakage control: 0 residual flags on unique products / patents")


def test_retriever_recovers_exact_duplicate_reaction() -> None:
    train = [_rxn("PAT1", 1, _P["a"], "bb.cc"), _rxn("PAT2", 2, _P["b"], "dd.ee")]
    dup = _rxn("PAT7", 1, _P["a"], "bb.cc")           # identical reaction (different patent)
    _, outs = __import__("karyon.retro_baseline", fromlist=["run_baseline"]).run_baseline(Split("synthetic", train, [dup]))
    o = outs[0]
    assert o.reactant_rank == 1, f"exact-duplicate reactants not recovered at rank 1: {o.reactant_rank}"
    assert o.reactant_overlap == 1.0
    audit = rh.audit_split(Split("synthetic", train, [dup]))
    assert "REACTION_DUP" in audit.verdicts[0].fired
    print("4. stdlib retriever recovers an exact-duplicate reaction (rank 1) + REACTION_DUP fires")


def test_auroc_separates_signal_and_is_null_under_shuffle() -> None:
    rng = random.Random(0)
    dummy = _rxn("X", 1, "Z", "z")
    # Planted: high similarity ⇒ correct, low ⇒ wrong. The metric must read it.
    signal = [Outcome(dummy, sim, False, None, 0.0, sim > 0.5)
              for sim in (rng.random() for _ in range(400))]
    au = rh.auroc_sim_explains(signal, "class")
    assert isinstance(au, MannWhitney) and au.auroc > 0.90, f"metric missed a planted signal: {au}"
    # Shuffle: similarity and correctness independent ⇒ AUROC ≈ 0.5.
    shuffled = [Outcome(dummy, rng.random(), False, None, 0.0, rng.random() > 0.5) for _ in range(400)]
    au0 = rh.auroc_sim_explains(shuffled, "class")
    assert isinstance(au0, MannWhitney) and abs(au0.auroc - 0.5) < 0.12, f"shuffle not null: {au0}"
    print(f"5. AUROC metric: planted sim↔correctness {au.auroc:.2f}; shuffle null {au0.auroc:.2f}")


def test_patent_disjoint_split_invariant() -> None:
    rxns = [_rxn(f"PAT{i % 7}", 1 + i % 10, f"prod{i}", "x.y") for i in range(200)]
    ps = patent_disjoint_split(rxns, seed=1, test_frac=0.25)
    straddle = {r.rid for r in ps.train} & {r.rid for r in ps.test}
    assert not straddle, f"patents straddle the disjoint split: {straddle}"
    assert ps.train and ps.test
    print(f"6. patent-disjoint split: 0 straddling patents, sizes {ps.sizes}")


def test_e2e_real_audit() -> None:
    try:
        rxns = load_reactions(limit=4000)
    except DatasetUnavailable as e:
        _skip(f"USPTO-50k unreachable and not cached: {e}")
        return
    audit = rh.audit_split(random_split(rxns, seed=0))
    prevalence = sum(audit.leaked) / len(audit.outcomes)
    full, floor = class_accuracy(audit.outcomes), rh.majority_class_floor(audit.outcomes)
    assert prevalence > 0.10, f"leakage prevalence implausibly low ({prevalence:.1%})"     # P1 direction
    assert floor < full < 0.95, f"class accuracy out of range (floor={floor:.2f}, full={full:.2f})"
    assert len(audit.clean) < len(audit.outcomes), "clean partition should be a strict subset"
    au = rh.auroc_sim_explains(audit.outcomes, "reactant")
    assert isinstance(au, MannWhitney) and au.auroc > 0.7, "reactant recovery should track similarity"
    print(f"7. e2e: prevalence {prevalence:.1%}, class {full:.1%} (floor {floor:.1%}), "
          f"reactant↔sim AUROC {au.auroc:.2f}")


if __name__ == "__main__":
    test_kmers_and_index_similarity()
    test_planted_leakage_fires_right_contracts()
    test_no_leakage_control()
    test_retriever_recovers_exact_duplicate_reaction()
    test_auroc_separates_signal_and_is_null_under_shuffle()
    test_patent_disjoint_split_invariant()
    test_e2e_real_audit()
    print("\nretro_honesty proofs pass.")
