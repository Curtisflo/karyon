"""test_crispr_qc — proofs for the CRISPRi screen-QC probe (dual script / pytest).

Offline, always run: the named feature vector lines up with FEATURE_NAMES; the legible helpers
(`gc`/`max_run`/`hairpin_score`/`revcomp`) match hand-computed values; the HARD contracts fire on the
exact structural failures they name and pass a clean guide; the flag is **sequence-only** (mutating a
guide's measured activity never changes its verdict — the non-circularity the whole claim rests on); the
gene split is gene-DISJOINT; the fit RECOVERS a planted activity rule de-novo and REJECTS
independent-label noise; and the QC-score AUROC separates a planted silent-failure tail while the
shuffled baseline collapses to chance. Online (skips offline): the real Horlbeck data runs end-to-end
and the legible layer clears the pre-registered bar.

    cd karyon/probe && python test_crispr_qc.py
"""

from __future__ import annotations

import os
import random

from karyon import crispr_qc as qc
from karyon import crispr_qc_data as cd


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def _rand_seq(rng: random.Random, n: int = 20) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_feature_dims() -> None:
    """Every feature is named — the vector and FEATURE_NAMES must stay aligned (legibility invariant)."""
    seq = _rand_seq(random.Random(0))
    assert len(qc.features(seq)) == len(qc.FEATURE_NAMES), "feature vector / names drifted out of sync"
    assert qc.FEATURE_NAMES[0] == "intercept"
    print(f"1. features aligned with names ({len(qc.FEATURE_NAMES)} named features incl. intercept)")


def test_legible_helpers() -> None:
    """The named features are hand-checkable — that is the point of them being legible."""
    assert qc.gc("GCGC") == 1.0 and qc.gc("ATAT") == 0.0
    assert qc.max_run("AAAGGGGTC") == 4 and qc.max_run("ACGT") == 1
    assert qc.revcomp("AAAATTTT") == "AAAATTTT"                 # a palindrome
    assert qc.revcomp("ACCG") == "CGGT"
    assert qc.hairpin_score("AAAATTTT") == 1.0                 # every 4-mer's rev-comp is present
    assert qc.hairpin_score("AAAAAAAA") == 0.0                 # no self-complementarity
    print("2. legible helpers (gc / max_run / revcomp / hairpin) match hand-computed values")


def test_hard_contracts_fire_and_pass() -> None:
    """The DRC spine: deterministic rules flag the exact structural failures they name, and a clean
    guide passes. No fitting, no data — these are provable exactly."""
    assert any("Pol-III" in r for r in qc.hard_contracts("GGAACTTTTGGCCAAGGTTCA")), "missed TTTT"
    assert any("homopolymer" in r for r in qc.hard_contracts("GGGGGAACCTTGGAACCTTGG")), "missed run≥5"
    assert any("GC" in r for r in qc.hard_contracts("GCGCGCGCGCGCGCGCGCGC")), "missed extreme GC"
    assert qc.hard_contracts("GCAACTTGGACCTTGAACCT") == [], "false-flagged a clean guide"
    print("3. hard contracts fire on TTTT / homopolymer / extreme-GC and pass a clean guide")


def test_flag_is_sequence_only() -> None:
    """Non-circularity: the verdict is built from sequence ALONE — mutating the measured activity must
    not move a single flag. (If it did, validating the flag against measured activity would be circular.)"""
    rng = random.Random(2)
    recs = [cd.Record(f"G{i}", _rand_seq(rng), rng.uniform(-0.5, 1.5)) for i in range(200)]
    model = qc.fit_model(recs)
    before = [(v.predicted, v.flagged) for v in qc.check_guides(model, recs)]
    bumped = [cd.Record(r.gene, r.seq, r.activity + 100.0) for r in recs]   # wreck the labels
    after = [(v.predicted, v.flagged) for v in qc.check_guides(model, bumped)]
    assert before == after, "the flag depends on measured activity — it is not sequence-only"
    print("4. flag is sequence-only: perturbing measured activity leaves every verdict unchanged")


def test_gene_disjoint_split() -> None:
    rng = random.Random(4)
    recs = [cd.Record(f"G{i % 50}", _rand_seq(rng), rng.uniform(0, 1)) for i in range(600)]
    train, test = qc.split_by_gene(recs, seed=0, test_frac=0.30)
    tr_g = {r.gene for r in train}
    te_g = {r.gene for r in test}
    assert tr_g.isdisjoint(te_g), "a gene leaked across the train/test split"
    assert len(train) + len(test) == len(recs)
    print(f"5. gene-disjoint split: {len(tr_g)} train / {len(te_g)} test genes, no overlap")


def test_fit_recovers_signal_and_rejects_noise() -> None:
    """Plant a real rule (activity rises with GC); the transparent ridge recovers it de-novo on held-out
    sequences. Independent-label noise → held-out ρ ≈ 0. So a real ρ is the data's, not a harness artifact."""
    rng = random.Random(3)
    recs = [cd.Record(f"G{i}", s := _rand_seq(rng), qc.gc(s) + rng.uniform(-0.05, 0.05))
            for i in range(1000)]
    tr, te = recs[:700], recs[700:]
    model = qc.fit_model(tr)
    rho = qc._rho([model.predict(qc.features(r.seq)) for r in te], [r.activity for r in te])
    assert rho > 0.7, f"failed to recover a planted GC→activity rule de-novo (ρ={rho:.3f})"

    noise = [cd.Record(f"G{i}", _rand_seq(rng), rng.gauss(0, 1)) for i in range(1000)]
    ntr, nte = noise[:700], noise[700:]
    nmodel = qc.fit_model(ntr)
    nrho = qc._rho([nmodel.predict(qc.features(r.seq)) for r in nte], [r.activity for r in nte])
    assert abs(nrho) < 0.2, f"manufactured signal from independent-label noise (ρ={nrho:.3f})"
    print(f"6. fit recovers a planted rule de-novo (ρ={rho:.2f}) and rejects noise (|ρ|={abs(nrho):.2f})")


def test_qc_auroc_separates_planted_tail() -> None:
    """End-to-end on synthetic: plant a silent-failure tail (low-GC guides are weak); the QC-score AUROC
    must separate it, and the shuffled-label baseline must collapse to ~0.5."""
    rng = random.Random(8)
    recs = []
    for g in range(80):                                        # 80 genes × 8 guides
        for _ in range(8):
            s = _rand_seq(rng)
            recs.append(cd.Record(f"G{g}", s, 2.0 * qc.gc(s) - 0.8 + rng.uniform(-0.1, 0.1)))
    res = qc.evaluate_seed(recs, seed=0)
    assert res.qc_auroc > 0.70, f"QC score failed to separate a planted tail (AUROC={res.qc_auroc:.3f})"
    assert abs(res.shuf_auroc - 0.5) < 0.1, f"shuffled baseline not ~0.5 (AUROC={res.shuf_auroc:.3f})"
    print(f"7. QC-score AUROC separates a planted tail ({res.qc_auroc:.2f}) vs shuffled "
          f"baseline ({res.shuf_auroc:.2f})")


def test_e2e_real_data() -> None:
    try:
        recs = cd.load_records()
    except cd.DatasetUnavailable as e:
        _skip(f"Horlbeck CRISPRi activity set unreachable and not cached: {e}")
        return
    assert len(recs) > 10_000 and all(set(r.seq) <= set("ACGT") for r in recs[:200])
    out = qc.run(seeds=2)
    assert out["rho_mean"] > 0.25, f"held-out ρ too low: {out['rho_mean']:+.3f}"
    assert out["qc_auroc_mean"] > 0.60, f"QC-score AUROC too low: {out['qc_auroc_mean']:.3f}"
    assert out["qc_auroc_mean"] > out["shuf_auroc_mean"] + 0.10, "QC score barely beats the noise baseline"
    assert out["hard_recall_mean"] > 0.0, "the hard DRC spine caught none of the silent-failure tail"
    print(f"8. e2e: {out['n_guides']} real guides; QC-AUROC {out['qc_auroc_mean']:.3f} "
          f"(ρ {out['rho_mean']:+.3f}) ≫ shuffled {out['shuf_auroc_mean']:.3f}")


if __name__ == "__main__":
    test_feature_dims()
    test_legible_helpers()
    test_hard_contracts_fire_and_pass()
    test_flag_is_sequence_only()
    test_gene_disjoint_split()
    test_fit_recovers_signal_and_rejects_noise()
    test_qc_auroc_separates_planted_tail()
    test_e2e_real_data()
    print("\nall crispr_qc tests pass.")
