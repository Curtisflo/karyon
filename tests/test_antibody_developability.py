"""test_antibody_developability — the antibody developability gate.

Real approved therapeutics pass the gate (low false-positive, like mol_qc's "real drugs pass"); planted
liabilities fire the exact named reasons; CDR anchoring round-trips on known Fv sequences; the disclose-vs-condemn
tiering holds; and the gate is pure stdlib. No network, fully deterministic.
"""

from __future__ import annotations

import karyon.antibody_developability as ab
from karyon import qualify
from karyon.antibody_developability import (ADALIMUMAB_VH, ADALIMUMAB_VL, AntibodyTol,
                                            TRASTUZUMAB_VH, TRASTUZUMAB_VL)

_TOL = AntibodyTol()
_REFERENCES = ((TRASTUZUMAB_VH, TRASTUZUMAB_VL), (ADALIMUMAB_VH, ADALIMUMAB_VL))


def _fired(h, l=None):
    return set(ab.validate(h, l, _TOL).fired)


# --------------------------------------------------------------------------- #
# Real therapeutics pass the gate (the false-positive control).
# --------------------------------------------------------------------------- #
def test_reference_therapeutics_pass():
    for h, l in _REFERENCES:
        v = ab.validate(h, l, _TOL)
        condemning = [r.contract for r in v.reasons if r.weight > 0]
        assert v.score == 0.0, f"approved therapeutic condemned by {condemning}"
        assert not ab.is_undevelopable(h, l, _TOL)


def test_reference_pI_in_plausible_band():
    for h, l in _REFERENCES:
        f = ab.featurize(h, l, _TOL)
        assert _TOL.pi_min <= f.pI <= _TOL.pi_max          # a real antibody Fv sits inside the charge band


# --------------------------------------------------------------------------- #
# CDR anchoring round-trips on known Fv sequences (the canonical CDRs are recovered exactly).
# --------------------------------------------------------------------------- #
def test_cdr_detection_locates_heavy_h3():
    cdrs = ab.find_cdrs(TRASTUZUMAB_VH, heavy=True, tol=_TOL)
    assert cdrs is not None and {r.name for r in cdrs} == {"H1", "H2", "H3"}
    h3 = next(r for r in cdrs if r.name == "H3")
    assert TRASTUZUMAB_VH[h3.start:h3.end] == "WGGDGFYAMDY"     # canonical trastuzumab CDR-H3
    f = ab.featurize(TRASTUZUMAB_VH, TRASTUZUMAB_VL, _TOL)
    assert f.cdr_ok and f.h3_len == 11


def test_cdr_detection_locates_light_l1():
    cdrs = ab.find_cdrs(TRASTUZUMAB_VL, heavy=False, tol=_TOL)
    assert cdrs is not None
    l1 = next(r for r in cdrs if r.name == "L1")
    assert TRASTUZUMAB_VL[l1.start:l1.end] == "RASQDVNTAVA"     # canonical trastuzumab CDR-L1


def test_single_domain_vhh_qualifies():
    # a heavy-only VHH/nanobody: the gate runs, CDRs resolve, and the VH/VL-asymmetry disclosure stands down.
    f = ab.featurize(TRASTUZUMAB_VH, None, _TOL)
    assert f.cdr_ok and f.light is None
    assert "CHARGE_ASYMMETRY" not in ab.validate(TRASTUZUMAB_VH, None, _TOL).fired


# --------------------------------------------------------------------------- #
# Planted liabilities fire the exact named reasons (the condemning DRC).
# --------------------------------------------------------------------------- #
def test_unpaired_cysteine_condemns():
    bad = TRASTUZUMAB_VH.replace("WGGDGFYAMDY", "WGGDGFYACMDY")     # an extra Cys in CDR-H3 (odd count)
    v = ab.validate(bad, TRASTUZUMAB_VL, _TOL)
    assert "UNPAIRED_CYSTEINE" in v.fired and v.score > 0


def test_cdr_n_glycosylation_sequon_condemns():
    bad = TRASTUZUMAB_VH.replace("WGGDGFYAMDY", "WGGDNISYAMDY")     # an N-I-S sequon planted in CDR-H3
    v = ab.validate(bad, TRASTUZUMAB_VL, _TOL)
    assert "N_GLYCOSYLATION_SEQUON_CDR" in v.fired and v.score >= 1.5
    assert any(h.region == "H3" for h in ab.featurize(bad, TRASTUZUMAB_VL, _TOL).cdr_sequons)


def test_extreme_h3_length_condemns():
    bad = TRASTUZUMAB_VH.replace("WGGDGFYAMDY", "W" + "A" * 35)     # CDR-H3 length 36 > h3_max 32
    v = ab.validate(bad, TRASTUZUMAB_VL, _TOL)
    assert "CDR_LENGTH_OUT_OF_RANGE" in v.fired and ab.featurize(bad, TRASTUZUMAB_VL, _TOL).h3_len == 36


def test_extreme_charge_condemns():
    bad = TRASTUZUMAB_VH + "K" * 12                                  # a poly-Lys tail pushes the Fv pI out of band
    v = ab.validate(bad, TRASTUZUMAB_VL, _TOL)
    assert "EXTREME_FV_CHARGE" in v.fired and ab.featurize(bad, TRASTUZUMAB_VL, _TOL).pI > _TOL.pi_max


# --------------------------------------------------------------------------- #
# Disclose-vs-condemn: the common chemistry flags inform without failing the gate.
# --------------------------------------------------------------------------- #
def test_chemistry_hotspots_disclose_not_condemn():
    # trastuzumab carries CDR deamidation/isomerization hotspots — they ride in `reasons` at weight 0 and do
    # NOT raise the score (the gate still passes). This is the antibody analogue of the DNA RESTRICTION_SITE note.
    v = ab.validate(TRASTUZUMAB_VH, TRASTUZUMAB_VL, _TOL)
    disclosed = {r.contract for r in v.reasons if r.weight == 0}
    assert "DEAMIDATION_HOTSPOT_CDR" in disclosed
    assert v.score == 0.0 and not v.clean                          # passes the gate, but not strictly silent


def test_unresolved_cdrs_disclose_and_stand_down():
    # a non-variable-domain input (no framework anchors): CDR-scoped checks stand down with a disclosure.
    v = ab.validate("MKTAYIAKQR" * 8, None, _TOL)
    assert "CDR_DETECTION_UNCERTAIN" in v.fired
    assert "DEAMIDATION_HOTSPOT_CDR" not in v.fired                 # no false CDR hits without resolved CDRs


# --------------------------------------------------------------------------- #
# Spine integration — the public `qualify` surface routes to the gate over every input form.
# --------------------------------------------------------------------------- #
def test_qualify_inline_and_vhh():
    assert qualify(f"{TRASTUZUMAB_VH}:{TRASTUZUMAB_VL}", modality="antibody").ok
    assert qualify(TRASTUZUMAB_VH, modality="antibody").ok          # single-chain VHH


def test_qualify_fasta_file(tmp_path):
    p = tmp_path / "fv.fasta"
    p.write_text(f">heavy\n{TRASTUZUMAB_VH}\n>light\n{TRASTUZUMAB_VL}\n")
    r = qualify(str(p), modality="antibody")
    assert r.ok and r.items[0][0] == "fv.fasta"


def test_fasta_extension_is_ambiguous_and_lists_antibody():
    from karyon import QualifyError
    try:
        qualify("designs.fasta")                                    # no modality → must refuse, naming antibody
        assert False, "expected QualifyError"
    except QualifyError as e:
        assert "antibody" in str(e)


def test_antibody_is_a_registered_modality():
    from karyon import modalities
    assert "antibody" in modalities()


# --------------------------------------------------------------------------- #
# The gate is pure stdlib (no numpy / rdkit), like gen-dna-qc.
# --------------------------------------------------------------------------- #
def test_gate_is_pure_stdlib():
    src = ab.__file__
    text = open(src).read()
    assert "import numpy" not in text and "import rdkit" not in text
