"""test_promoter_contracts — proofs for the σ70 design DRC + readout qualification (dual script / pytest).

Offline (always): the box scan locates a textbook −35/−10 arrangement; each design contract fires on a
TARGETED planted break and a clean promoter passes; C5/C6 are CALIBRATED (an out-of-distribution run /
a reference-rare motif fires, an in-distribution one is dormant); the readout QA–QD contracts fire on a
dropout / noisy / floored / saturated / control-failed readout and pass a clean one; calibration reads
the reference distribution; evaluation is deterministic.

Online (skips offline): the MEASURED-VALIDATION grounding — on the real Urtecho σ70 set the calibrated
DRC passes well-formed promoters and the ones it rejects express significantly WEAKER (AUROC of
passed>rejected measured strength), so a flag predicts lower function rather than just "looks wrong";
and the calibrated real-pass rate is high and clearly separates real from composition-scrambled.

    cd karyon/probe && python test_promoter_contracts.py
"""

from __future__ import annotations

import os
import random
import statistics

from karyon import promoter_contracts as pc
from karyon.assay import Readout

# A textbook σ70 promoter: TTGACA + 17-nt spacer + TATAAT, GC in band, no long run, no forbidden motif.
CLEAN = "GGCAT" + "TTGACA" + "ATCGATGCATCGATGCA" + "TATAAT" + "GCATGCATGCATGGCA"


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def test_find_boxes_textbook() -> None:
    """The box scan locates the canonical arrangement exactly (h35=h10=0, spacer 17)."""
    b = pc.find_boxes(CLEAN)
    assert b is not None and b.h35 == 0 and b.h10 == 0 and b.spacer == 17
    assert CLEAN[b.i35:b.i35 + 6] == "TTGACA" and CLEAN[b.i10:b.i10 + 6] == "TATAAT"
    print("1. find_boxes locates the textbook −35/−10 arrangement (h=0, spacer 17)")


def test_clean_passes() -> None:
    """A well-formed promoter passes all six contracts (uncalibrated: no forbidden motif, short runs)."""
    v = pc.DESIGN.evaluate(CLEAN)
    assert v.ok, f"clean promoter false-flagged: {v.messages}"
    print("2. a clean promoter passes C1–C6")


def test_box_and_spacer_contracts_fire() -> None:
    """C1/C2/C3 fire on a targeted box/spacer break (the right named contract is among the reasons)."""
    bad35 = CLEAN.replace("TTGACA", "CCAGTC", 1)           # −35 destroyed (6 mismatches)
    assert "C1 −35 box" in pc.DESIGN.evaluate(bad35).fired
    bad10 = CLEAN.replace("TATAAT", "GCCGGC", 1)           # −10 destroyed
    assert "C2 −10 box" in pc.DESIGN.evaluate(bad10).fired
    badsp = "GGCAT" + "TTGACA" + "ATCGATGCATCGATGCATCGA" + "TATAAT" + "GCATGGCA"  # spacer 21
    fired = pc.DESIGN.evaluate(badsp).fired
    assert "C3 spacer" in fired, f"spacer break not caught: {pc.DESIGN.evaluate(badsp).messages}"
    print("3. C1/C2/C3 fire on targeted −35 / −10 / spacer breaks")


def test_gc_contract_fires() -> None:
    """C4 fires outside the GC band and passes inside it."""
    gc_rich = "GGCGC" + "TTGACA" + "GCGCGCGCGCGCGCGCG" + "TATAAT" + "GCGCGCGCGGCGCGC"
    assert "C4 GC band" in pc.DESIGN.evaluate(gc_rich).fired
    print("4. C4 GC-band fires on an extreme-GC promoter")


def test_c5_homopolymer_calibrated() -> None:
    """C5 is calibrated: a run beyond the reference max fires; one within is dormant."""
    ref = ["ACGT" * 10, "TGCA" * 10]                       # reference longest run = 1
    ctx = pc.calibrate_design(ref)
    assert ctx["max_run"] == 1
    runny = CLEAN[:20] + "AAAAA" + CLEAN[25:]              # introduce a run of 5 > ref max 1
    assert "C5 homopolymer" in pc.DESIGN.evaluate(runny, ctx).fired
    # With a reference that itself carries the run, the SAME design is in-distribution → dormant.
    ctx2 = pc.calibrate_design(ref + [runny])
    assert "C5 homopolymer" not in pc.DESIGN.evaluate(runny, ctx2).fired
    print("5. C5 homopolymer is calibrated (fires only out-of-distribution)")


def test_c6_forbidden_motif_calibrated() -> None:
    """C6 fires on a motif RARE in the reference and is dormant on one the reference commonly carries."""
    eco = CLEAN[:10] + "GAATTC" + CLEAN[16:]               # introduce an EcoRI site
    ctx_rare = pc.calibrate_design([CLEAN] * 50)           # reference never carries GAATTC ⇒ rare ⇒ fires
    assert "C6 forbidden motif" in pc.DESIGN.evaluate(eco, ctx_rare).fired
    ctx_common = pc.calibrate_design([eco] * 50)           # reference carries it on every member ⇒ dormant
    assert "C6 forbidden motif" not in pc.DESIGN.evaluate(eco, ctx_common).fired
    print("6. C6 forbidden-motif is calibrated (rare-in-reference fires, scaffold motif dormant)")


def test_readout_contracts() -> None:
    """QA–QD pass a clean readout and each fires on its own failure mode."""
    ctx = pc.readout_ctx()
    assert pc.READOUT.evaluate(Readout("s", 1.2, built=True, replicate_cv=0.05, signal=1.0), ctx).ok
    assert "QA built/measured" in pc.READOUT.evaluate(Readout("s", None, built=False), ctx).fired
    assert "QB replicate CV" in pc.READOUT.evaluate(Readout("s", 1.0, replicate_cv=0.9, signal=1.0), ctx).fired
    assert "QC dynamic range" in pc.READOUT.evaluate(Readout("s", 0.0, signal=0.0), ctx).fired
    assert "QC dynamic range" in pc.READOUT.evaluate(Readout("s", 99, signal=99.0), ctx).fired
    assert "QD controls" in pc.READOUT.evaluate(Readout("s", 1.0, signal=1.0, controls_ok=False), ctx).fired
    print("7. readout QA–QD pass a clean readout and fire on dropout / CV / range / controls")


def test_calibration_reads_reference() -> None:
    """calibrate_design takes max_run from the reference and marks reference-absent forbidden motifs rare."""
    ctx = pc.calibrate_design(["AAATTT", "ACGTACGT"])
    assert ctx["max_run"] == 3 and "GAATTC" in ctx["rare_motifs"]
    print("8. calibrate_design reads the reference distribution")


def test_deterministic() -> None:
    rng = random.Random(3)
    seqs = ["".join(rng.choice("ACGT") for _ in range(60)) for _ in range(200)]
    ctx = pc.calibrate_design(seqs)
    assert all(pc.DESIGN.evaluate(s, ctx) == pc.DESIGN.evaluate(s, ctx) for s in seqs)
    print("9. evaluation is deterministic")


def test_grounded_in_measured_strength() -> None:
    """ONLINE: the legible DRC predicts measured function — passed promoters express stronger than the
    ones it rejects, and the calibrated gate passes well-formed promoters (and beats a scramble)."""
    try:
        from karyon import promoter_data as pmd
        recs = pmd.load_records()
    except Exception as e:                                  # noqa: BLE001 — offline → skip
        return _skip(f"promoter data unavailable ({e})")
    from karyon import stats_kit as sk
    ctx = pc.calibrate_design([r.seq for r in recs])
    rng = random.Random(0)
    sample = rng.sample(recs, min(800, len(recs)))
    ok = [r.strength for r in sample if pc.DESIGN.evaluate(r.seq, ctx).ok]
    bad = [r.strength for r in sample if not pc.DESIGN.evaluate(r.seq, ctx).ok]
    auroc = getattr(sk.mann_whitney(ok, bad), "auroc", 0.5)
    pass_rate = len(ok) / len(sample)
    scr_pass = sum(pc.DESIGN.evaluate("".join(rng.sample(r.seq, len(r.seq))), ctx).ok for r in sample) / len(sample)
    assert auroc > 0.55, f"DRC not grounded in strength (AUROC {auroc:.3f})"
    assert pass_rate > 0.40 and pass_rate > scr_pass, f"pass {pass_rate:.2f} vs scramble {scr_pass:.2f}"
    print(f"10. grounded: passed promoters stronger than rejected (AUROC {auroc:.3f}); "
          f"calibrated pass {pass_rate:.0%} vs scramble {scr_pass:.0%}  [μ pass {statistics.mean(ok):+.2f} "
          f"vs reject {statistics.mean(bad):+.2f}]")


def _run() -> None:
    test_find_boxes_textbook()
    test_clean_passes()
    test_box_and_spacer_contracts_fire()
    test_gc_contract_fires()
    test_c5_homopolymer_calibrated()
    test_c6_forbidden_motif_calibrated()
    test_readout_contracts()
    test_calibration_reads_reference()
    test_deterministic()
    test_grounded_in_measured_strength()
    print("\nALL promoter-contract proofs passed.")


if __name__ == "__main__":
    _run()
