"""noisy_assay — model-degrading readout corruption, to demonstrate that qualifying PROTECTS a loop.

The legible operator (`dbtl_operator`) qualifies each readout before folding it into its surrogate: a
measurement that fails the readout contracts (`promoter_contracts` QA–QD — built / replicate-CV / dynamic-
range / controls) is *flagged and excluded from the model update*, with a named reason. This module supplies
the failure mode that makes that qualification matter: `StressFn` factories for `assay.RetrospectiveAssay`
that corrupt a controlled fraction of readouts in ways the (label-blind) QA–QD contracts detect
**imperfectly**.

A corrupted readout gets a **wrong value** (so an *ingested* label poisons the surrogate) and, separately,
the QC metadata QA–QD read (`signal` / `replicate_cv`) — with an explicit **sensitivity** (a corrupted well
is only detectably flagged with probability `sens`, so some poison slips through even the gate) and
**specificity** (a clean well is falsely flagged with probability `1 - spec`, so qualifying also drops good
data — a real cost). The gate is therefore not an oracle; it is a realistic, imperfect filter.

Three modes, to show *when* qualifying compounds over recursive cycles:

  - `saturation_stress` — reader saturation **biased to high-expressers**. A strong design saturates the
    reader: its value is sign-flipped about the pool mean (reads weak) and its raw signal pushed above the QC
    ceiling. Acquisition-interacting (the poison concentrates on the winners the loop hunts).
  - `random_stress` — the same flip+flag on a **uniform-random** fraction. Model-degrading but not
    acquisition-biased — it degrades the *global* model, which is where the ρ-protection compounds.
  - `shuffle_stress` — poison one set, flag a **disjoint** set (only clean readouts are flagged). The
    negative control: the gate's flags carry zero information about the poison, so qualifying protects
    nothing and any compounding collapses.

Determinism: corruption is a pure function of `(mode, seed, design)` via SHA-256, so a construct's assay
behaviour is fixed (whether a given design saturates the reader is a property of the construct, not the run)
and the demonstration reproduces without a stateful RNG threaded through `ingest`.

stdlib-only.
"""

from __future__ import annotations

import hashlib
import statistics
from typing import Callable

from .assay import RetrospectiveAssay

# QC bands the corruption must push past to be *detectable* by promoter_contracts QA–QD — kept consistent
# with promoter_contracts.readout_ctx() defaults (saturation=10.0, cv_max=0.30) so a flagged well genuinely
# trips QC and an unflagged one genuinely passes.
_SAT_CEILING = 10.0          # QC fires on signal > 10.0  → set 15.0 to flag
_HI_CV = 0.40                # QC fires on replicate_cv > 0.30 → set 0.40 to flag

StressFn = Callable[[str, "float | None"], "dict | None"]


def _draw(tag: str, seed: int, design: str) -> float:
    """A deterministic uniform [0,1) keyed by (tag, seed, design) — process-stable (SHA-256, not the
    hash-randomized builtin). Distinct `tag`s give independent draws for independent decisions
    (is-corrupted vs is-detected vs false-flag) on the same design."""
    h = hashlib.sha256(f"{tag}|{seed}|{design}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0**64


def _flip_about_mean(v: float, mean: float) -> float:
    """Sign-flip a value about the pool mean: a strong (high-v) design reads as equally weak. A label that
    actively MISLEADS the surrogate, not just adds variance."""
    return 2.0 * mean - v


def saturation_stress(truth: dict[str, float], *, rate: float = 0.30, sat_quantile: float = 0.80,
                      sens: float = 0.75, spec: float = 0.05, seed: int = 0) -> StressFn:
    """Reader-saturation corruption biased to high-expressers (the acquisition-interacting failure mode).

    A construct above the `sat_quantile` of true expression saturates the reader with probability `rate`:
    its value is flipped about the pool mean (reads weak) and — with probability `sens` — its signal is
    pushed above the QC ceiling so the gate can catch it (the other `1-sens` slip through). A clean
    (below-threshold) well is falsely flagged with probability `spec` (its TRUE value kept, so the gate
    needlessly drops good data — qualify's cost)."""
    vals = list(truth.values())
    mean = statistics.mean(vals)
    sat_thr = sorted(vals)[min(len(vals) - 1, int(len(vals) * sat_quantile))]

    def stress(s: str, v: "float | None") -> "dict | None":
        if v is None:
            return None
        if v < sat_thr:                                         # clean low/mid expresser
            return {"replicate_cv": _HI_CV} if _draw("ff", seed, s) < spec else None
        if _draw("sat", seed, s) >= rate:                       # this winner didn't happen to saturate
            return None
        ov: dict = {"value": _flip_about_mean(v, mean)}         # poison: reads weak
        if _draw("det", seed, s) < sens:                        # detectable saturation flag
            ov["signal"] = _SAT_CEILING + 5.0
        return ov                                               # else: undetected, slips the gate

    return stress


def random_stress(truth: dict[str, float], *, rate: float = 0.30,
                  sens: float = 0.75, spec: float = 0.05, seed: int = 0) -> StressFn:
    """Uniform-random sign-flip corruption — model-degrading but NOT acquisition-biased. Same imperfect
    detection (sens/spec). The contrast against `saturation_stress` isolates whether compounding needs the
    failure mode to interact with acquisition or merely to degrade the global model."""
    vals = list(truth.values())
    mean = statistics.mean(vals)

    def stress(s: str, v: "float | None") -> "dict | None":
        if v is None:
            return None
        if _draw("rnd", seed, s) >= rate:                       # clean
            return {"replicate_cv": _HI_CV} if _draw("ff", seed, s) < spec else None
        ov: dict = {"value": _flip_about_mean(v, mean)}
        if _draw("det", seed, s) < sens:
            ov["replicate_cv"] = _HI_CV
        return ov

    return stress


def shuffle_stress(truth: dict[str, float], *, rate: float = 0.30, seed: int = 0) -> StressFn:
    """NEGATIVE CONTROL: poison one set, flag a DISJOINT set (only clean readouts are flagged, never
    corrupted ones). The gate's flags then carry zero information about the poison — qualifying drops only
    good data and protects nothing, so compounding must collapse. (Disjoint, not merely independent: at a
    high marginal rate an *independent* flag set overlaps the poison by ~rate² and leaks incidental
    protection; disjoint placement removes it.)"""
    vals = list(truth.values())
    mean = statistics.mean(vals)

    def stress(s: str, v: "float | None") -> "dict | None":
        if v is None:
            return None
        if _draw("val", seed, s) < rate:                        # poisoned — and NEVER flagged (flows to both arms)
            return {"value": _flip_about_mean(v, mean)}
        if _draw("flag", seed, s) < rate:                       # flagged but CLEAN (true value) → gate drops good data
            return {"signal": _SAT_CEILING + 5.0}
        return None

    return stress


# Mode registry — the harness selects by name; `clean` is a no-op stress (the rate-0 anchor).
_MODES: dict[str, "Callable[..., StressFn] | None"] = {
    "clean": None,
    "saturation": saturation_stress,
    "random": random_stress,
    "shuffle": shuffle_stress,
}


def make_assay(truth: dict[str, float], mode: str, rate: float, seed: int = 0,
               *, sens: float = 0.75, spec: float = 0.05) -> RetrospectiveAssay:
    """Build a RetrospectiveAssay whose `ingest` applies the named corruption. `clean` or `rate<=0` ⇒ the
    stock clean assay (the no-regression / negative anchor)."""
    if mode not in _MODES:
        raise ValueError(f"unknown corruption mode {mode!r}; choose from {sorted(_MODES)}")
    factory = _MODES[mode]
    if factory is None or rate <= 0.0:
        return RetrospectiveAssay(truth)
    if mode == "shuffle":
        stress = factory(truth, rate=rate, seed=seed)           # shuffle takes no sens/spec
    else:
        stress = factory(truth, rate=rate, sens=sens, spec=spec, seed=seed)
    return RetrospectiveAssay(truth, stress=stress)
