"""contracts — a substrate-agnostic LEGIBLE contract / verdict engine (the karyon centerpiece, Layer 2).

The thesis: AI authors a *legible* reliability / QC layer — the DRC
DRC + contracts + `realize` doctrine ported to biology — not a black-box predictor. Two single-substrate
instances already exist and are the SAME latent shape:

  - `crispr_qc.hard_contracts(seq) -> list[str]`     — deterministic sequence rules (the DRC spine)
  - `screen_qc.reliability_contracts(guides, null, disp) -> (reasons, score, ...)` — calibrated count rules

Each is "a named rule maps (design, calibration) -> a fired reason with a severity weight, and a verdict
aggregates the fired reasons for one item." This module lifts that shape into ONE engine so a NEW
substrate is a set of `Contract` objects + a calibration, not a re-implemented verdict type. That
generalization — two instances → one engine — is where the sophistication legitimately lives; the cores (predict/choose/construct) are commodity, the legible contract engine
is not.

Design:
  - A `Contract` is a named predicate `check(design, ctx) -> (fired? + why + weight)`. `ctx` is an
    arbitrary *calibration* object (a fitted threshold, a model, or None for a pure rule). HARD contracts
    ignore `ctx`; CALIBRATED contracts read it. Keeping `ctx` opaque is deliberate: the engine never
    needs to know what a substrate calibrates on.
  - `ContractSet.evaluate(design, ctx) -> Verdict(ok, reasons, score)`: `ok` iff nothing fired,
    `score` = Σ weights (a continuous severity, NOT a restatement of any single baseline), `reasons` =
    the human-readable why. Legibility is the product — every verdict names which rule fired and why.
  - `hard_only()` is the DRC-spine subset (the contracts that need no calibration) — the design-time gate
    the operator can run before any data exists.

stdlib-only; no probe imports. A substrate's contracts live in its own module (e.g. promoter_contracts)
and import THIS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

HARD = "hard"               # a deterministic rule; needs no calibration (ctx ignored)
CALIBRATED = "calibrated"   # reads a fitted threshold / model from ctx


@dataclass(frozen=True)
class Reason:
    """One fired contract — the atom of legibility."""

    contract: str            # which named contract fired
    message: str             # human-readable why (the auditable reason)
    weight: float = 1.0      # contribution to the verdict's continuous severity score


@dataclass(frozen=True)
class Verdict:
    """The aggregate QC call on one design. `ok` ⇒ no contract fired (trustworthy)."""

    ok: bool
    reasons: tuple[Reason, ...]
    score: float             # Σ fired weights — 0.0 when clean

    @property
    def messages(self) -> list[str]:
        """The reasons as plain strings (the shape the legacy instances returned)."""
        return [r.message for r in self.reasons]

    @property
    def fired(self) -> list[str]:
        """Names of the contracts that fired."""
        return [r.contract for r in self.reasons]


# A contract's check may return any of:
#   None / False           -> clean (did not fire)
#   True                   -> fired; message defaults to the contract name
#   a str                  -> fired with that message
#   a Reason               -> used verbatim (lets a contract override message/weight)
CheckResult = "None | bool | str | Reason"
Check = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class Contract:
    """A named, legible rule. `check(design, ctx)` decides if it fires (see CheckResult)."""

    name: str
    check: Check
    kind: str = HARD
    weight: float = 1.0      # default severity if the check doesn't return a Reason of its own


def _normalize(contract: Contract, result: Any) -> Reason | None:
    """Coerce a check's loose return into a Reason (or None when it didn't fire)."""
    if result is None or result is False:
        return None
    if result is True:
        return Reason(contract.name, contract.name, contract.weight)
    if isinstance(result, Reason):
        return result
    if isinstance(result, str):
        return Reason(contract.name, result, contract.weight)
    raise TypeError(f"contract {contract.name!r} returned {type(result).__name__}; "
                    "expected None/bool/str/Reason")


@dataclass
class ContractSet:
    """An ordered, named registry of contracts over one substrate. Order is preserved in the verdict's
    reasons so the legible output reads in a stable, designed sequence."""

    name: str
    contracts: list[Contract] = field(default_factory=list)

    def add(self, contract: Contract) -> "ContractSet":
        self.contracts.append(contract)
        return self

    def rule(self, name: str, *, kind: str = HARD, weight: float = 1.0) -> Callable[[Check], Check]:
        """Decorator sugar: register `check` as a contract and return it unchanged.

            @cs.rule("C5 homopolymer")
            def _(seq, ctx): return "long run" if max_run(seq) >= 5 else None
        """
        def deco(check: Check) -> Check:
            self.add(Contract(name, check, kind, weight))
            return check
        return deco

    def evaluate(self, design: Any, ctx: Any = None) -> Verdict:
        """Run every contract over `design` (with calibration `ctx`) and aggregate the fired reasons."""
        reasons: list[Reason] = []
        for c in self.contracts:
            r = _normalize(c, c.check(design, ctx))
            if r is not None:
                reasons.append(r)
        return Verdict(ok=not reasons, reasons=tuple(reasons), score=sum(r.weight for r in reasons))

    def hard_only(self) -> "ContractSet":
        """The DRC-spine subset — contracts that need no calibration (design-time gate)."""
        return ContractSet(f"{self.name}[hard]", [c for c in self.contracts if c.kind == HARD])

    def names(self) -> list[str]:
        return [c.name for c in self.contracts]
