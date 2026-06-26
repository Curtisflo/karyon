"""assay — the Assay boundary (Layer 3): the operator's swap-point between desk and wet operation.

This is the single seam that makes the operator "architected for live operation" while validated
retrospectively today. The operator never touches a dataset or a wet lab
directly — it speaks to an `Assay`:

    emit_order(designs) -> Order          # the manifest of what to build / measure
    ingest(order)       -> list[Readout]  # the measurements back, WITH power/QC metadata

Two implementations:
  - `RetrospectiveAssay(truth)` — `emit_order` is a manifest no-op; `ingest` looks the design up in a
    truth dict and attaches (synthetic) power metadata. THE DESK PATH v0 ships and is validated on.
  - `WetAssay` — `emit_order` writes a real oligo-order manifest + plate map (pure formatting, live now);
    `ingest` parses a sequencing file + a plate-reader file into `Readout`s (the part the $50k era
    implements — stubbed, with the file formats documented). Swapping Retrospective→Wet is the ONLY
    change that flips the operator from desk to a live autonomous loop.

The `Readout` carries power metadata (`built`, `replicate_cv`, `signal`, `controls_ok`) precisely so the
operator's qualification step (promoter_contracts QA–QD) can decide whether a measurement is trustworthy
— a build dropout is *no data*, not a true zero, and an under-powered well is not a real negative.

stdlib-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class Order:
    """A manifest of designs to build + measure — what `emit_order` produces. `meta` carries cycle /
    substrate / template context for the audit trail."""

    designs: tuple[str, ...]
    cycle: int = 0
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Readout:
    """One measurement back from the assay, with the power metadata the qualify step judges.

    `value is None` ⇒ the construct was never built/measured (no data) — distinct from a measured 0.0.
    The remaining fields let the readout-qualification contracts (QA–QD) flag an untrustworthy well
    without ever seeing the design's true label (that is the whole point — qualification is label-blind)."""

    design: str
    value: float | None            # measured function; None ⇒ build/measure failure (no data)
    built: bool = True             # sequencing confirmed the INTENDED construct (QA)
    replicate_cv: float = 0.0      # within-design replicate coefficient of variation (QB)
    signal: float = 1.0            # raw signal level, for dynamic-range floor/saturation checks (QC)
    controls_ok: bool = True       # run-level positive/negative controls validated (QD)
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Assay(Protocol):
    """The operator↔world boundary. Retrospective today, wet later — same two methods."""

    def emit_order(self, designs: list[str], cycle: int = 0, meta: dict | None = None) -> Order: ...

    def ingest(self, order: Order) -> list[Readout]: ...


# --------------------------------------------------------------------------- #
# RetrospectiveAssay — the desk path (v0 ships + validates on this).
# --------------------------------------------------------------------------- #
# A stress hook lets a test inject synthetic dropouts / noise so the dormant-by-correctness QA–QD
# contracts can be exercised — for design `s` with looked-up value `v`, return a
# dict of Readout-field overrides (e.g. {"built": False, "value": None}); return None to leave it clean.
StressFn = Callable[[str, "float | None"], "dict | None"]


@dataclass
class RetrospectiveAssay:
    """`ingest` is a truth-dict lookup + synthetic power metadata. Designs absent from `truth` come back
    as un-built readouts (no data) — exactly how a wet build dropout would read."""

    truth: dict[str, float]
    stress: StressFn | None = None

    def emit_order(self, designs: list[str], cycle: int = 0, meta: dict | None = None) -> Order:
        return Order(tuple(designs), cycle, dict(meta or {}, mode="retrospective"))

    def ingest(self, order: Order) -> list[Readout]:
        out: list[Readout] = []
        for s in order.designs:
            v = self.truth.get(s)
            # Clean synthetic power metadata: a present truth value ⇒ a well-built, in-band readout.
            r = Readout(design=s, value=v, built=v is not None,
                        replicate_cv=0.05 if v is not None else 0.0,
                        signal=1.0 if v is not None else 0.0,
                        controls_ok=True, meta={"cycle": order.cycle})
            if self.stress is not None:
                ov = self.stress(s, v)
                if ov:
                    r = Readout(**{**r.__dict__, **ov})
            out.append(r)
        return out


# --------------------------------------------------------------------------- #
# WetAssay — the live path (order emission is real now; ingest is the $50k step).
# --------------------------------------------------------------------------- #
# File formats (documented contracts the wet era reads/writes):
#   order manifest  (<run>.order.json) : {"cycle": int, "constructs": [{"id","promoter_seq","well"}...]}
#   sequencing file (<run>.seq.tsv)    : columns  id<TAB>built(0/1)   — which constructs assembled
#   plate-reader    (<run>.plate.tsv)  : columns  well<TAB>gfp<TAB>od<TAB>replicate — raw deGFP/OD
class WetAssay:
    """The live boundary. `emit_order` writes a real, orderable manifest + plate map; `ingest` parses the
    sequencing + plate-reader files the wet run returns. v0 ships the interface + the documented file
    formats; `ingest` raises until the wet rig exists (the Layer-3 / $50k step, the design notes §5)."""

    def __init__(self, run_dir: str | Path, template: str = "") -> None:
        self.run_dir = Path(run_dir)
        self.template = template

    def emit_order(self, designs: list[str], cycle: int = 0, meta: dict | None = None) -> Order:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        constructs = [{"id": f"C{cycle:02d}-{i:03d}", "promoter_seq": s,
                       "well": f"{'ABCDEFGH'[i // 12]}{i % 12 + 1}"} for i, s in enumerate(designs)]
        manifest = {"cycle": cycle, "template": self.template, "constructs": constructs}
        (self.run_dir / f"run{cycle:02d}.order.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return Order(tuple(designs), cycle, dict(meta or {}, mode="wet", manifest=str(self.run_dir)))

    def ingest(self, order: Order) -> list[Readout]:
        raise NotImplementedError(
            "WetAssay.ingest parses real sequencing (<run>.seq.tsv) + plate-reader (<run>.plate.tsv) "
            "files — the live wet step (Layer 3 / the $50k rig, the design notes §5/§6.4). Use "
            "RetrospectiveAssay at the desk.")
