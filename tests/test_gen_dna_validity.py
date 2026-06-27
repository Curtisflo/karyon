"""test_gen_dna_validity — falsification proofs for the generated-DNA synthesizability gate (gen-DNA-QC).

Shows:
  1. the pure-stdlib primitives (`seq_dfm`) compute GC / runs / reverse-complement / cross-hyb / restriction
     sites / hairpins correctly;
  2. each owned contract fires on a planted violator and stays silent on a clean sequence, and the
     disclose-vs-condemn tiering is by SCORE (a disclosed flag does not fail the gate);
  3. the batch (set-level) cross-hybridization invariant catches a complementary pair;
  4. the kicker — a valid, normal-looking ACGT sequence (the kind a generator emits confidently) is still
     condemned for GC / homopolymer: validity ≠ manufacturability, the gap the gate owns;
  5. faithfulness — owned per-axis flags match the DnaChisel reference on planted + clean cases.

Runnable as a script (`python tests/test_gen_dna_validity.py`) or under pytest. DnaChisel checks skip when
absent. No network required.
"""

from __future__ import annotations

from pathlib import Path

from karyon import seq_dfm
from karyon import gen_dna_validity as gv

try:
    import dnachisel  # noqa: F401
    from karyon.gen_dna_honesty import dnachisel_axis_flags, owned_axis_flags
    _HAVE_DC = True
except Exception:
    _HAVE_DC = False

_TOL = gv.GenDNATol()
_CLEAN = "ACGTACGTGGTTACCAGTCAGTACTGACTAGTCAGTGCATGCATGGTACAACGTACGTAGT"  # ~48% GC, no long run/hairpin


# --------------------------------------------------------------------------- #
# 1) seq_dfm primitives
# --------------------------------------------------------------------------- #
def test_gc_fraction() -> None:
    assert seq_dfm.gc_fraction("GGCC") == 1.0
    assert seq_dfm.gc_fraction("ATAT") == 0.0
    assert abs(seq_dfm.gc_fraction("ACGT") - 0.5) < 1e-9
    assert seq_dfm.gc_fraction("") == 0.0


def test_longest_run() -> None:
    assert seq_dfm.longest_run("AAACGGGGT") == 4            # GGGG
    assert seq_dfm.longest_run("AAACGGGGT", "A") == 3
    assert seq_dfm.longest_run("ACGT") == 1


def test_reverse_complement() -> None:
    assert seq_dfm.reverse_complement("AAAACGT") == "ACGTTTT"
    assert seq_dfm.reverse_complement("GAATTC") == "GAATTC"  # palindrome


def test_anneal_stretch_finds_complementary_run() -> None:
    a = "AAAAGGGGCCGATTTT"
    b = "CCCCC" + seq_dfm.reverse_complement("GGGGCCGA") + "AAAA"
    assert "GGGGCCGA" in seq_dfm.anneal_stretch(a, b)
    # a non-complementary pair shares no long anneal stretch
    assert len(seq_dfm.anneal_stretch("ACGTACGTACGTACGT", "ACACACACACACACAC")) < 8


def test_restriction_sites() -> None:
    hits = seq_dfm.restriction_sites("AAAGAATTCAAAGGATCCAAA")     # EcoRI + BamHI
    enzymes = {h.enzyme for h in hits}
    assert "EcoRI" in enzymes and "BamHI" in enzymes, enzymes
    assert seq_dfm.restriction_sites("AAAAAAAAAAAAAAAAAA") == []


def test_hairpin_stem() -> None:
    arm = "GGGTTACCAGTC"
    hp = arm + "TTTAT" + seq_dfm.reverse_complement(arm)       # 12 bp stem, 5 nt loop
    h = seq_dfm.hairpin_stem(hp)
    assert h.stem >= 12, h
    assert seq_dfm.hairpin_stem("ACACACACATATATATGTGTGT").stem < 12  # no strong perfect stem


# --------------------------------------------------------------------------- #
# 2) per-sequence contracts — fire on planted, silent on clean; disclose vs condemn
# --------------------------------------------------------------------------- #
def test_clean_passes() -> None:
    v = gv.validate(_CLEAN, _TOL)
    assert v.score == 0.0, v.messages
    assert not gv.is_unsynthesizable(_CLEAN, _TOL)


def test_gc_out_of_band_condemns() -> None:
    v = gv.validate("GC" * 40, _TOL)
    assert "GC_OUT_OF_BAND" in v.fired and v.score > 0


def test_homopolymer_condemns() -> None:
    seq = "ACGTACGTAC" + "A" * (_TOL.max_homopolymer_run + 3) + "GTCAGTCAGT"
    assert "HOMOPOLYMER_RUN" in gv.validate(seq, _TOL).fired


def test_strong_hairpin_condemns() -> None:
    arm = "GGTTACCAGTCA"
    seq = "ACGTACGT" + arm + "TTGAT" + seq_dfm.reverse_complement(arm) + "ACGTACGT"
    v = gv.validate(seq, _TOL)
    assert "STRONG_HAIRPIN" in v.fired and v.score > 0


def test_length_condemns() -> None:
    assert "LENGTH_OUT_OF_RANGE" in gv.validate("ACGTACGT", _TOL).fired           # too short
    assert "LENGTH_OUT_OF_RANGE" in gv.validate("AC" * 2000, _TOL).fired           # too long


def test_poly_g_discloses_without_condemning() -> None:
    # A poly-G run (GGGG, >max_g_run) is a DISCLOSED risk (weight 0): reported in `fired`, but it does not
    # raise the verdict score, so the gate still PASSES. Tested at the feature level (cofold planted-feature
    # style) so a fixture sequence's incidental hairpin/site can't confound the tiering assertion.
    f = gv.DNAFeatures(length=60, gc_frac=0.5, max_run=4, max_g_run=4, hairpin=seq_dfm.Hairpin(0), sites=())
    v = gv.dna_contracts().evaluate(f, _TOL)
    assert "POLY_G_RUN" in v.fired and v.score == 0.0, v.messages


def test_restriction_site_discloses_without_condemning() -> None:
    seq = "ACGTACGTAGAATTCAGTACTGACTAGTCAGTACTGACGTACAACGTACGTAGTACGTAC"  # carries EcoRI, else clean
    v = gv.validate(seq, _TOL)
    assert "RESTRICTION_SITE" in v.fired
    assert v.score == 0.0 and not gv.is_unsynthesizable(seq, _TOL)


def test_severity_orders_clean_below_decoy() -> None:
    clean_sev = gv.featurize(_CLEAN, _TOL).severity(_TOL)
    decoy_sev = gv.featurize("G" * 60, _TOL).severity(_TOL)
    assert clean_sev == 0.0 < decoy_sev


# --------------------------------------------------------------------------- #
# 3) batch cross-hybridization (the design-level invariant no single sequence owns)
# --------------------------------------------------------------------------- #
def test_batch_cross_hyb_condemns() -> None:
    a = "ACGTAGCTAGGTCATGCATTGCAACGATCGATCGTAGCATGCATCGATCGTAGCATGCAT"
    b = "TTTTTT" + seq_dfm.reverse_complement(a[10:40]) + "TTTTTT"   # b carries rc(a[10:40]) → 30 nt anneal
    sv = gv.validate_set([("a", a), ("b", b)], _TOL)
    assert not sv.ok
    assert any(r.contract == "SEVERE_CROSS_HYBRIDIZATION" for _, r in sv.reasons)


def test_batch_clean_passes() -> None:
    from karyon.gen_dna_data import synthetic_clean
    a, b = synthetic_clean(2, seed=5)                          # two guaranteed-clean, independent sequences
    sv = gv.validate_set([("a", a), ("b", b)], _TOL)
    assert sv.ok, [(_n, r.contract) for _n, r in sv.reasons]


# --------------------------------------------------------------------------- #
# 4) the kicker — valid ACGT, normal length, yet unmanufacturable (validity ≠ manufacturability)
# --------------------------------------------------------------------------- #
def test_valid_dna_can_be_unmanufacturable() -> None:
    """A confidently-emitted, perfectly-valid ACGT sequence of normal length can still be unsynthesizable —
    the gap a generator's confidence is blind to and this gate owns."""
    looks_fine = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC"   # valid, 56 nt, but 100% GC
    assert set(looks_fine) <= set("ACGT") and gv.is_unsynthesizable(looks_fine, _TOL)


# --------------------------------------------------------------------------- #
# 5) faithfulness vs DnaChisel (gold-standard DFM package); skip when unavailable
# --------------------------------------------------------------------------- #
def test_owned_matches_dnachisel_on_planted_and_clean() -> None:
    if not _HAVE_DC:
        return  # skip — dnachisel not installed
    cases = {
        "clean":        _CLEAN,
        "gc":           "GC" * 40,
        "homopolymer":  "ACGTACGT" + "T" * 12 + "ACGTACGTACGTACGT",
        "hairpin":      ("ACGT" + "GGTTACCAGTCA" + "TTGAT"
                         + seq_dfm.reverse_complement("GGTTACCAGTCA") + "ACGT"),
    }
    for name, seq in cases.items():
        ow = owned_axis_flags(seq, _TOL)
        rf = dnachisel_axis_flags(seq, _TOL)
        assert ow == rf, f"{name}: owned {ow} vs DnaChisel {rf}"


# --------------------------------------------------------------------------- #
# 6) stdlib purity — the gate imports neither numpy nor rdkit (the lightest karyon QC layer)
# --------------------------------------------------------------------------- #
def test_gate_is_pure_stdlib() -> None:
    for mod in (seq_dfm, gv):
        src = Path(mod.__file__).read_text()
        assert "import numpy" not in src and "import rdkit" not in src, mod.__file__


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL ASSERTIONS PASSED" + ("" if _HAVE_DC else "  (DnaChisel faithfulness test skipped)"))


if __name__ == "__main__":
    main()
