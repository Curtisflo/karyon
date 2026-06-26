"""test_molnet_honesty — proofs for the scaffold-leakage honesty audit (dual pytest / __main__).

The teeth: the scaffold split never lets a scaffold straddle; the leakage contracts fire on the right
signals and a clean molecule stays clean; the metric reads a planted signal (AUROC for clf, ρ for reg); and
the e2e shows the random-vs-scaffold inflation is real on BBBP + ESOL. The first three run without rdkit
(pure split/contract/metric logic); the e2e SKIPs if rdkit/data are absent.
"""

from __future__ import annotations

import os

from karyon import molnet_honesty as mh
from karyon.molnet_data import DATASETS, DatasetUnavailable, Molecule, Split, load_dataset, scaffold_split


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def _mol(smiles: str, label: float, scaffold: str) -> Molecule:
    return Molecule(smiles, label, scaffold)


def _outcome(nn_sim: float, scaffold_seen: bool, pred: float, label: float = 0.0) -> mh.MolOutcome:
    return mh.MolOutcome(_mol("X", label, "s"), nn_sim, scaffold_seen, pred)


def test_scaffold_split_never_straddles() -> None:
    mols = [_mol(f"m{i}", i % 2, f"scaf{i % 8}") for i in range(160)] + \
           [_mol(f"a{i}", 0, "") for i in range(20)]            # acyclic bucket
    s = scaffold_split(mols, test_frac=0.25)
    straddle = {m.scaffold for m in s.train if m.scaffold} & {m.scaffold for m in s.test if m.scaffold}
    assert not straddle, f"scaffolds straddle: {straddle}"
    assert s.train and s.test
    print(f"1. scaffold split: 0 straddling scaffolds, sizes {s.sizes}")


def test_leakage_contracts_fire_and_clean_stays_clean() -> None:
    cs, tv = mh.leakage_contracts(), mh.TrainView(tau=0.7)
    leaked = cs.evaluate(_outcome(nn_sim=0.92, scaffold_seen=True, pred=1.0), tv)
    clean = cs.evaluate(_outcome(nn_sim=0.40, scaffold_seen=False, pred=1.0), tv)
    assert "NEAR_DUP_MOLECULE" in leaked.fired and "SCAFFOLD_SEEN_IN_TRAIN" in leaked.fired, leaked.fired
    assert clean.ok, f"a novel, dissimilar molecule must not leak: {clean.fired}"
    print(f"2. leakage contracts fire {leaked.fired}; novel molecule stays clean")


def test_metric_reads_planted_signal() -> None:
    # classification: predictions perfectly separate the classes → AUROC 1.0
    clf = [_outcome(0.5, False, pred=0.9, label=1.0) for _ in range(10)] + \
          [_outcome(0.5, False, pred=0.1, label=0.0) for _ in range(10)]
    assert abs(mh.metric(clf, classification=True) - 1.0) < 1e-9
    # regression: predictions monotone in the label → ρ ≈ 1.0
    reg = [_outcome(0.5, False, pred=float(i), label=float(i)) for i in range(12)]
    assert mh.metric(reg, classification=False) > 0.99
    print("3. metric reads a planted signal (AUROC=1.0 clf, ρ≈1.0 reg)")


def test_e2e_inflation_is_real() -> None:
    if not mh._HAVE_RDKIT:
        _skip("rdkit/numpy absent")
        return
    try:
        load_dataset("bbbp")
    except DatasetUnavailable as e:
        _skip(f"MoleculeNet unreachable and not cached: {e}")
        return
    for name in ("bbbp", "esol"):
        r = mh.run_one(name)
        assert r is not None
        assert r["m_rnd"] > (0.70 if r["clf"] else 0.55), f"{name} baseline implausibly weak ({r['m_rnd']:.3f})"
        assert r["inflation"] > 0.0, f"{name} random split should beat scaffold split ({r['inflation']:+.3f})"
        assert r["prevalence"] > 0.40, f"{name} leakage prevalence implausibly low ({r['prevalence']:.1%})"
    print("4. e2e: random>scaffold inflation real on bbbp + esol; leakage prevalent; baseline competent")


if __name__ == "__main__":
    test_scaffold_split_never_straddles()
    test_leakage_contracts_fire_and_clean_stays_clean()
    test_metric_reads_planted_signal()
    test_e2e_inflation_is_real()
    print("\nmolnet_honesty proofs pass.")
