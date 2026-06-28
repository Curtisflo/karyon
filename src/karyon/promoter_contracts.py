"""promoter_contracts — the σ70 substrate's legible contracts (design DRC C1–C6 + readout QA–QD).

The substrate-specific content the operator gates on, expressed in the substrate-agnostic `contracts`
engine. Two registries:

  - `DESIGN` (C1–C6) — design-time DRC over a promoter SEQUENCE, live in v0.
      C1–C4 are HARD, mechanism-grounded rules (a −35/−10 box by minimum Hamming to consensus, the
        inter-box spacer, the GC band). They are *measured-validated*: promoters C1/C2 flag as
        weak-box express significantly lower on the real Urtecho set (AUROC 0.66 box-OK vs weak — on
        par with the CRISPRi-QC AUROC), so a flag predicts lower function, not just "looks wrong."
      C5–C6 are CALIBRATED against the known-good reference pool (the deposited, buildable promoters):
        the homopolymer-run limit and the "rare forbidden motif" set are read from the reference's own
        distribution, so natural tracts and scaffold motifs are *dormant-by-correctness* and only an
        OUT-OF-DISTRIBUTION run / a genuinely-introduced rare site fires. This is the `screen_qc`
        calibration pattern (a contract fires on deviation BEYOND the buildable reference, not on a
        guessed universal threshold) — the honest fix for the fact that hard `run>5` / restriction-site
        rules false-fire on most real 150-nt members (natural runs reach 8; scaffold carries sites).

  - `READOUT` (QA–QD) — readout-time qualification over an `assay.Readout`. Calibrated bands;
    dormant-by-correctness on clean retrospective data, exercised by a synthetic-dropout stress. A build dropout is no data (not a true zero); an under-powered/saturated
    well is not a real negative.

Reuses `constructive_core` (`gc_fraction`, `longest_run`) and `assay.Readout`. The −35/−10 box model is
a legible best-arrangement scan (no PWM training) — register-agnostic, so on full-random sequences
chance boxes limit discrimination; the operator's regime is the real σ70 pool / mutated-from-real
designs, where the boxes are real.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from . import constructive_core as cc
from .assay import Readout
from .contracts import CALIBRATED, ContractSet, Verdict

# σ70 consensus elements + legible tolerances (named so a verdict reads against meaning).
BOX35, BOX10 = "TTGACA", "TATAAT"          # canonical −35 / −10 hexamers
SPACER_OK = (15, 19)                       # functional inter-box spacer (17 optimal)
SPACER_SEARCH = (12, 22)                   # wider window the box scan searches (so a bad spacer is visible)
TOL35, TOL10 = 2, 2                        # max Hamming from consensus before a box "isn't there"
GC_BAND = (0.30, 0.70)                     # same band as promoter_design.is_feasible (real set sits inside)
MAX_RUN_DEFAULT = 8                        # fallback homopolymer limit (= the Urtecho reference max) if uncalibrated
RARE_FRAC = 0.02                           # a forbidden motif present in < this fraction of the reference is avoidable
# Forbidden sub-sequences: common assembly restriction sites (+ rev-comp) and a poly-T terminator/Pol-III
# tract. Whether one is a real defect is SUBSTRATE-RELATIVE — a site carried by the scaffold of every
# library member is normal; one introduced into a fresh design is avoidable. C6 resolves this by
# calibration (only motifs RARE in the reference fire).
FORBIDDEN = {
    "GGTCTC": "BsaI site", "GAGACC": "BsaI site (rc)",
    "CGTCTC": "BsmBI/Esp3I site", "GAGACG": "BsmBI/Esp3I site (rc)",
    "GAATTC": "EcoRI site", "GGATCC": "BamHI site", "TTTTTTT": "poly-T terminator/Pol-III tract",
}


@dataclass(frozen=True)
class BoxFit:
    """The best −35/−10 arrangement found in a promoter (min combined Hamming over the spacer window)."""

    h35: int            # Hamming(−35 hexamer, TTGACA)
    i35: int            # start index of the −35 hexamer
    h10: int            # Hamming(−10 hexamer, TATAAT)
    i10: int            # start index of the −10 hexamer
    spacer: int         # nt between the boxes (i10 − (i35 + 6))


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


@functools.lru_cache(maxsize=8192)
def find_boxes(seq: str) -> BoxFit | None:
    """Locate the −35/−10 boxes as the upstream/downstream hexamer pair (spacer in SPACER_SEARCH) that
    minimizes combined Hamming distance to the consensus. Legible and deterministic; None if the seq is
    too short to hold a pair. The wider search window means a *suboptimal* spacer is found and reported
    (so C3 is a real contract, not vacuously satisfied)."""
    n = len(seq)
    best: BoxFit | None = None
    last_i35 = n - 6 - SPACER_SEARCH[0] - 6
    for i in range(0, last_i35 + 1):
        h35 = _hamming(seq[i:i + 6], BOX35)
        for sp in range(SPACER_SEARCH[0], SPACER_SEARCH[1] + 1):
            j = i + 6 + sp
            if j + 6 > n:
                break
            h10 = _hamming(seq[j:j + 6], BOX10)
            if best is None or (h35 + h10) < (best.h35 + best.h10):
                best = BoxFit(h35, i, h10, j, sp)
    return best


# --------------------------------------------------------------------------- #
# Calibration — the design ctx, learned from the known-good (buildable, deposited) reference pool.
# --------------------------------------------------------------------------- #
def calibrate_design(ref_seqs: list[str]) -> dict:
    """Derive the substrate-relative thresholds C5/C6 read, from a reference of buildable promoters.

    `max_run` = the longest homopolymer run any reference member carries (so a natural/scaffold tract is
    in-distribution and dormant; only a LONGER run fires). `rare_motifs` = the forbidden motifs that are
    rare in the reference (a motif the scaffold carries on most members is normal-for-substrate and must
    not fire). This is the honest alternative to a guessed universal threshold."""
    if not ref_seqs:
        return {"max_run": MAX_RUN_DEFAULT, "rare_motifs": set(FORBIDDEN)}
    max_run = max(cc.longest_run(s) for s in ref_seqs)
    n = len(ref_seqs)
    freq = {m: sum(m in s for s in ref_seqs) / n for m in FORBIDDEN}
    rare = {m for m, f in freq.items() if f < RARE_FRAC}
    return {"max_run": max_run, "rare_motifs": rare, "motif_freq": freq}


# --------------------------------------------------------------------------- #
# C1–C6 — design-time DRC. C1–C4 hard (mechanism-grounded); C5–C6 calibrated (dormant-by-correctness).
# --------------------------------------------------------------------------- #
def design_contracts() -> ContractSet:
    cs = ContractSet("σ70-promoter-design")

    @cs.rule("C1 −35 box")
    def _c1(seq, ctx):
        b = find_boxes(seq)
        if b is None or b.h35 > TOL35:
            got = seq[b.i35:b.i35 + 6] if b else "?"
            h = b.h35 if b else 6
            return f"weak −35 box: best '{got}' is {h}/6 mismatches from {BOX35}"
        return None

    @cs.rule("C2 −10 box")
    def _c2(seq, ctx):
        b = find_boxes(seq)
        if b is None or b.h10 > TOL10:
            got = seq[b.i10:b.i10 + 6] if b else "?"
            h = b.h10 if b else 6
            return f"weak −10 box: best '{got}' is {h}/6 mismatches from {BOX10}"
        return None

    @cs.rule("C3 spacer")
    def _c3(seq, ctx):
        b = find_boxes(seq)
        # Only meaningful once both boxes exist (else C1/C2 carry it); fire on a spacer outside 15–19.
        if b is not None and b.h35 <= TOL35 and b.h10 <= TOL10 and not (SPACER_OK[0] <= b.spacer <= SPACER_OK[1]):
            return f"spacer {b.spacer} nt outside {SPACER_OK[0]}–{SPACER_OK[1]} (17 optimal)"
        return None

    @cs.rule("C4 GC band")
    def _c4(seq, ctx):
        g = cc.gc_fraction(seq)
        if not (GC_BAND[0] <= g <= GC_BAND[1]):
            return f"GC {g:.0%} outside {GC_BAND[0]:.0%}–{GC_BAND[1]:.0%} (synthesis/expression risk)"
        return None

    @cs.rule("C5 homopolymer", kind=CALIBRATED)
    def _c5(seq, ctx):
        limit = (ctx or {}).get("max_run", MAX_RUN_DEFAULT)
        r = cc.longest_run(seq)
        if r > limit:
            return f"homopolymer run {r} > reference max {limit} (out-of-distribution; synthesis slippage)"
        return None

    @cs.rule("C6 forbidden motif", kind=CALIBRATED)
    def _c6(seq, ctx):
        rare = (ctx or {}).get("rare_motifs", set(FORBIDDEN))   # uncalibrated default: any forbidden motif
        hits = [f"{m} ({FORBIDDEN[m]})" for m in FORBIDDEN if m in seq and m in rare]
        if hits:
            return "forbidden motif (rare in reference ⇒ avoidable): " + "; ".join(hits)
        return None

    return cs


# --------------------------------------------------------------------------- #
# QA–QD — readout-time qualification (over assay.Readout; calibrated; dormant-by-correctness in v0).
# --------------------------------------------------------------------------- #
def readout_ctx(cv_max: float = 0.30, floor: float = 0.05, saturation: float = 10.0) -> dict:
    """Calibration bands for the readout contracts (a wet run derives these from its controls; the desk
    path uses these defaults). `floor` = autofluorescence; `saturation` = reader ceiling."""
    return {"cv_max": cv_max, "floor": floor, "saturation": saturation}


def readout_contracts() -> ContractSet:
    cs = ContractSet("σ70-promoter-readout")

    @cs.rule("QA built/measured", kind=CALIBRATED)
    def _qa(r: Readout, ctx):
        if not r.built or r.value is None:
            return "construct not built/sequenced — dropout is no-data, not a true negative"
        return None

    @cs.rule("QB replicate CV", kind=CALIBRATED)
    def _qb(r: Readout, ctx):
        if r.built and r.replicate_cv > ctx["cv_max"]:
            return f"replicate CV {r.replicate_cv:.2f} > {ctx['cv_max']:.2f} — unreliable"
        return None

    @cs.rule("QC dynamic range", kind=CALIBRATED)
    def _qc(r: Readout, ctx):
        if r.built and r.signal < ctx["floor"]:
            return f"signal {r.signal:.2f} below autofluorescence floor {ctx['floor']:.2f}"
        if r.signal > ctx["saturation"]:
            return f"signal {r.signal:.2f} above saturation {ctx['saturation']:.2f}"
        return None

    @cs.rule("QD controls", kind=CALIBRATED)
    def _qd(r: Readout, ctx):
        if not r.controls_ok:
            return "run controls out of band — measurement not validated"
        return None

    return cs


# Module singletons (built once; the operator holds these).
DESIGN = design_contracts()
READOUT = readout_contracts()


def validate(seq: str, ctx: dict | None = None) -> Verdict:
    """Qualify a σ70 promoter sequence → a Verdict (the uniform per-artifact entry point, mirroring the
    other gates' `validate`). `ctx` is the optional design calibration from
    `calibrate_design(reference_promoters)`; with `ctx=None` the calibrated C5/C6 fall back to safe
    defaults (the skill's uncalibrated path). Equals `DESIGN.evaluate(seq, ctx)`."""
    return DESIGN.evaluate(seq, ctx)
