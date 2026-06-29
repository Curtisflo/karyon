"""perturbseq_qc — a legible silent-failure QC layer over a real single-cell Perturb-seq screen.

The in-domain heavyweight the bulk screen-QC probe named but did not take (SCREEN_QC_RESULT.md § "natural
next probes"): the **SCEPTRE/Replogle single-cell analog.** A genome-scale Perturb-seq screen calls each
perturbation a transcriptomic **hit** or **no-phenotype** via a calibrated test (here Replogle's deposited
energy-distance p-value — the role SCEPTRE plays for single-cell screens). That call is ambiguous on the
no-phenotype side: a perturbation shows no phenotype either because the gene knockdown genuinely has no
transcriptomic consequence (a **true negative**) or because the guide **never knocked the target down** (a
**silent failure** — the null is an artifact). Bulk screens can only *infer* this; Perturb-seq **measures it
directly** via on-target knockdown — the cleanest possible silent-failure ground truth, and the reason the
in-domain single-cell substrate is the sharpest test of the screen-QC thesis.

The legible QC layer (the choane DRC spine, ported to Perturb-seq; `contracts` engine reused verbatim):
  * `WEAK_KNOCKDOWN`     — residual on-target expression above a floor (the guide failed → null untrustworthy);
  * `KNOCKDOWN_UNMEASURED` — on-target knockdown not measurable (target undetected → null unqualifiable);
  * `LOW_CELL_COUNT`    — too few cells to call a phenotype (the single-cell power floor).
Each verdict names which rule fired and why. The incumbent caller is *consumed* (deposited), not reimplemented
— karyon's owned, legible part is this qualification layer over the tool's output.

Pre-registered (mechanism-based, thresholds fixed before the verdict; mirrors the bulk Q1–Q4):
  P1  silent-failures flagged: ≥ 15% of no-phenotype targeting perturbations carry a QC flag.
  P2  precision: among clear hits (the perturbation demonstrably worked), weak-knockdown flag ≤ 20%.
  P3  non-redundancy (make-or-break): |ρ(knockdown, phenotype-p)| within the no-phenotype pile < 0.30 —
      the flag is NOT a softer restatement of the deposited significance.
  P4  decision-relevant: weak-knockdown is ≥ 2× enriched in the no-phenotype pile vs the hit pile.

stdlib + `contracts` + `stats_kit`; the substrate needs the optional h5py reader (SKIP-if-absent).

    python -m karyon.perturbseq_qc
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

from . import contracts
from . import stats_kit
from .perturbseq_data import (
    DatasetUnavailable,
    Perturbation,
    controls,
    load_perturbations,
    targeting,
)

# A perturbation whose cell count is unknown (a user table without a cells column) must not trip the
# under-power rule — sentinel it above any real screen's cell count, and render it as "unknown" downstream.
_NO_CELL_SENTINEL = 1_000_000_000

WEAK_KD_FLOOR = 0.5      # residual target expression above which on-target knockdown is deemed failed (<50% KD)
MIN_CELLS = 25           # cells below which a perturbation is under-powered to call a phenotype
HIT_Q = 0.05             # the deposited energy-test p-value threshold for "has a transcriptomic phenotype"
STRONG_HIT_Q = 1e-3      # "the perturbation demonstrably worked" — the precision guard's clear-hit set


@dataclass(frozen=True)
class QCView:
    weak_kd_floor: float
    min_cells: int


# --------------------------------------------------------------------------- #
# The legible QC layer (reused contracts engine).
# --------------------------------------------------------------------------- #
def qc_contracts() -> contracts.ContractSet:
    cs = contracts.ContractSet("perturbseq-reliability")
    cs.add(contracts.Contract(
        "WEAK_KNOCKDOWN",
        lambda p, ctx: (f"on-target knockdown failed — {p.knockdown_resid:.0%} of {p.target} still expressed "
                        f"(> {ctx.weak_kd_floor:.0%} floor); a 'no phenotype' here is untrustworthy")
        if (p.knockdown_measured and p.knockdown_resid > ctx.weak_kd_floor) else None,
        weight=2.0))
    cs.add(contracts.Contract(
        "KNOCKDOWN_UNMEASURED",
        lambda p, ctx: f"on-target knockdown of {p.target} not measurable (target undetected) — null unqualifiable"
        if not p.knockdown_measured else None))
    cs.add(contracts.Contract(
        "LOW_CELL_COUNT",
        lambda p, ctx: f"only {p.n_cells} cells — under-powered to call a phenotype"
        if p.n_cells < ctx.min_cells else None))
    return cs


def qualify(p: Perturbation, view: QCView, cs: contracts.ContractSet | None = None) -> contracts.Verdict:
    return (cs or qc_contracts()).evaluate(p, view)


# --------------------------------------------------------------------------- #
# Analysis helpers.
# --------------------------------------------------------------------------- #
def _weak_kd(p: Perturbation, floor: float = WEAK_KD_FLOOR) -> bool:
    return p.knockdown_measured and p.knockdown_resid > floor


def _frac(items: list, pred) -> float:
    return sum(1 for x in items if pred(x)) / (len(items) or 1)


def _compute(perts: list[Perturbation], view: QCView, cs: contracts.ContractSet) -> dict:
    """The shared numeric core — split hits / no-phenotype / clear-hits, the B1 calibration, the Q1–Q4
    scalars, and the no-phenotype partition. Pure (no printing) so both the reproduction `run_one` and the
    JSON-safe `audit_report` read from one source of truth. `floor` follows the view, so a caller's custom
    `weak_kd_floor` governs the enrichment/partition exactly as it governs the contract."""
    floor = view.weak_kd_floor
    tgt, ctl = targeting(perts), controls(perts)
    hits = [p for p in tgt if p.energy_p < HIT_Q]
    nohit = [p for p in tgt if p.energy_p >= HIT_Q]
    strong = [p for p in tgt if p.energy_p < STRONG_HIT_Q]

    cal_t = _frac(tgt, lambda p: p.energy_p < HIT_Q)              # B1 — deposited caller, calibrated on controls
    cal_c = _frac(ctl, lambda p: p.energy_p < HIT_Q)

    q1 = _frac(nohit, lambda p: not cs.evaluate(p, view).ok)     # Q1 — flagged among no-phenotype
    weak_nohit = _frac(nohit, lambda p: _weak_kd(p, floor))
    weak_hit = _frac(hits, lambda p: _weak_kd(p, floor))
    enrich = weak_nohit / (weak_hit or 1e-9)                      # P4
    q2 = _frac(strong, lambda p: _weak_kd(p, floor))             # Q2 — precision among clear hits

    sub = [p for p in nohit if p.knockdown_measured]             # Q3 — non-redundancy vs the deposited p-value
    r = stats_kit.spearman([p.knockdown_resid for p in sub], [p.energy_p for p in sub])
    rho = abs(r.rho) if isinstance(r, stats_kit.Corr) else float("nan")

    untrust = [p for p in nohit if _weak_kd(p, floor)]           # Q4 — the partition significance can't make
    trust = [p for p in nohit if p.knockdown_measured and not _weak_kd(p, floor)]
    unmeasured = [p for p in nohit if not p.knockdown_measured]

    return {"tgt": tgt, "ctl": ctl, "hits": hits, "nohit": nohit, "strong": strong,
            "cal_t": cal_t, "cal_c": cal_c, "q1": q1, "weak_nohit": weak_nohit, "weak_hit": weak_hit,
            "enrich": enrich, "q2": q2, "rho": rho,
            "untrust": untrust, "trust": trust, "unmeasured": unmeasured}


def run_one(*, perts: list[Perturbation] | None = None, view: QCView | None = None) -> dict | None:
    if perts is None:
        try:
            perts = load_perturbations()
        except DatasetUnavailable as e:
            print(f"  SKIP — {e}")
            return None
    view = view or QCView(WEAK_KD_FLOOR, MIN_CELLS)
    cs = qc_contracts()
    c = _compute(perts, view, cs)
    tgt, ctl, nohit = c["tgt"], c["ctl"], c["nohit"]
    untrust, trust, unmeasured = c["untrust"], c["trust"], c["unmeasured"]

    print(f"\n=== REPLOGLE K562-ESSENTIAL PERTURB-SEQ (single-cell screen QC) ===")
    print(f"  perturbations: {len(tgt)} targeting / {len(ctl)} controls")
    print(f"  B1 incumbent calibration (deposited energy-test): targeting {c['cal_t']:.0%} hit vs controls {c['cal_c']:.0%}"
          f"  ({'credible' if c['cal_t'] > 0.5 > c['cal_c'] else 'weak'})")
    print(f"  no-phenotype targeting perturbations (the silent-failure denominator): {len(nohit)}")
    print(f"\n  Q1 flagged among no-phenotype:        {c['q1']:.1%}   (bar ≥15%)   <- P1")
    for con in cs.contracts:
        f = _frac(nohit, lambda p, n=con.name: n in cs.evaluate(p, view).fired)
        print(f"       {con.name:<22} {f:6.1%}")
    print(f"  Q2 weak-KD flag among clear hits:     {c['q2']:.1%}   (bar ≤20%, precision)   <- P2")
    print(f"  Q3 |ρ(knockdown, energy-p)| in no-hit:{c['rho']:.3f}  (bar <0.30, non-redundancy)   <- P3")
    print(f"  Q4 weak-KD enrichment no-hit vs hit:  {c['enrich']:.1f}×   ({c['weak_nohit']:.1%} vs {c['weak_hit']:.1%})   <- P4")
    print(f"     → the no-phenotype pile partitions: {len(untrust)} untrustworthy (weak-KD silent failures) · "
          f"{len(trust)} trustworthy negatives (strong-KD) · {len(unmeasured)} unmeasurable")

    examples = sorted(untrust, key=lambda p: -p.knockdown_resid)[:6]
    if examples:
        print(f"  example silent failures (essential gene, 'no phenotype', but KD failed):")
        for p in examples:
            print(f"       {p.target:<10} residual expr {p.knockdown_resid:.0%}, energy-p {p.energy_p:.2f}, {p.n_cells} cells")

    return {"cal_t": c["cal_t"], "cal_c": c["cal_c"], "n_nohit": len(nohit), "q1": c["q1"], "q2": c["q2"],
            "rho": c["rho"], "enrich": c["enrich"], "weak_nohit": c["weak_nohit"], "weak_hit": c["weak_hit"],
            "n_untrust": len(untrust), "n_trust": len(trust)}


# --------------------------------------------------------------------------- #
# The shippable surface: qualify a screen (the bundled reference OR a user's own) into a JSON-safe report
# whose centerpiece is the per-call `flagged` list — the no-phenotype calls a user should not trust, each
# with its named reason. This is the "bring your own screen" power: the same legible QC over your data.
# --------------------------------------------------------------------------- #
def _finite(x):
    """JSON-safe scalar: a non-finite float (NaN from an undefined Spearman ρ on a tiny pile, ±inf) becomes
    `None` (null) so the report parses under STRICT JSON — `json.dumps(rep, allow_nan=False)` must not raise."""
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def _flag_item(p: Perturbation, v: contracts.Verdict) -> dict:
    return {"target": p.target, "pid": p.pid,
            "knockdown_residual": None if not p.knockdown_measured else round(p.knockdown_resid, 4),
            "energy_p": p.energy_p,
            "n_cells": None if p.n_cells >= _NO_CELL_SENTINEL else p.n_cells,
            "reasons": [r.to_dict() for r in v.reasons]}


def audit_report(*, perts: list[Perturbation], view: QCView | None = None, max_flagged: int = 50) -> dict:
    """Qualify a single-cell screen and return a stable, JSON-safe report. The Q1–Q4 scalars summarize the
    screen; `flagged` is the decision-relevant payload — every no-phenotype call whose verdict condemns
    (weak / unmeasured knockdown, too few cells), each with its named reasons, highest residual first."""
    view = view or QCView(WEAK_KD_FLOOR, MIN_CELLS)
    cs = qc_contracts()
    c = _compute(perts, view, cs)
    nohit = c["nohit"]

    flagged = [_flag_item(p, v) for p in nohit if not (v := cs.evaluate(p, view)).ok]
    flagged.sort(key=lambda d: -1.0 if d["knockdown_residual"] is None else -d["knockdown_residual"])
    by_contract = {name: _frac(nohit, lambda p, n=name: n in cs.evaluate(p, view).fired) for name in cs.names()}

    return {
        "screen": "single-cell",
        "n_targeting": len(c["tgt"]),
        "n_controls": len(c["ctl"]),
        "calibration": {"targeting_hit_rate": c["cal_t"], "control_hit_rate": c["cal_c"],
                        "credible": bool(c["cal_t"] > 0.5 > c["cal_c"])},
        "n_nophenotype": len(nohit),
        "flagged_rate": c["q1"],
        "by_contract": by_contract,
        "precision_weak_kd_in_hits": c["q2"],
        "rho_knockdown_vs_significance": _finite(c["rho"]),       # null when the no-phenotype pile is too small
        "weak_kd_enrichment": _finite(c["enrich"]) if c["weak_hit"] else None,   # null when no weak-KD hits (undefined)
        "partition": {"untrustworthy": len(c["untrust"]),
                      "trustworthy_negative": len(c["trust"]),
                      "unmeasurable": len(c["unmeasured"])},
        "n_flagged": len(flagged),
        "flagged": flagged[:max_flagged],
    }


# --------------------------------------------------------------------------- #
# Bring-your-own-screen ingestion: a user's screen summary (CSV/TSV, one row per perturbation) → Perturbation
# records the QC layer qualifies. Pure stdlib — NO h5py and NO network: the reference Replogle reader is only
# for reproducing the headline; qualifying your own screen needs only `pip install karyon`.
# --------------------------------------------------------------------------- #
# Column names are matched case-insensitively against these aliases. `target` and a phenotype p-value are
# required; knockdown residual and cell count are optional (their absence is itself a verdict — an unmeasured
# knockdown condemns the null as unqualifiable; a missing cell count simply can't trip the power floor).
_COL_ALIASES: dict[str, tuple[str, ...]] = {
    "target":    ("target", "gene", "target_gene", "gene_symbol", "symbol"),
    "knockdown": ("knockdown_residual", "residual_expression", "residual_expr", "residual",
                  "fold_expr", "fold_change", "knockdown"),
    "energy_p":  ("energy_p", "energy_test_p_value", "phenotype_pvalue", "phenotype_p",
                  "pvalue", "p_value", "pval", "p"),
    "n_cells":   ("n_cells", "num_cells", "ncells", "cells", "num_cells_unfiltered"),
    "control":   ("is_control", "control", "core_control", "non_targeting", "nt"),
}
_REQUIRED_COLS = ("target", "energy_p")
# A target whose symbol is itself a non-targeting marker is treated as a control even without a control column.
_NT_TARGETS = frozenset({"non-targeting", "non_targeting", "nontargeting", "nt",
                         "control", "neg", "negative", "safe-harbor", "safe_harbor"})
_TRUTHY = frozenset({"1", "true", "yes", "y", "t"})


def _to_float(s: str | None) -> float:
    s = (s or "").strip()
    if s == "" or s.lower() in {"na", "nan", "none", "null"}:
        return float("nan")
    return float(s)


def _resolve_columns(header) -> dict[str, str]:
    """Map each known field to the actual header name present (case-insensitive, first alias wins)."""
    lower = {h.lower().strip(): h for h in header if h}
    resolved: dict[str, str] = {}
    for field_name, aliases in _COL_ALIASES.items():
        for a in aliases:
            if a in lower:
                resolved[field_name] = lower[a]
                break
    return resolved


def _row_is_control(row: dict, cols: dict[str, str], target: str) -> bool:
    if "control" in cols:
        return (row.get(cols["control"]) or "").strip().lower() in _TRUTHY
    return target.lower() in _NT_TARGETS


def load_user_screen(path) -> list[Perturbation]:
    """Parse a user's own single-cell screen summary into `Perturbation` records the QC layer can qualify.

    The table is one row per perturbation, CSV or TSV (delimiter inferred from the extension / header). The
    expected knockdown value is **residual on-target expression** (0 = fully knocked down, 1 = unchanged);
    rows without it become unmeasured nulls. `target` and a phenotype p-value column are required — see
    `_COL_ALIASES` for accepted header names. Raises `QualifyError` with the accepted aliases on a bad table.
    """
    from .spine import QualifyError                              # local: avoid import cost on plain `import karyon`
    p = Path(path)
    if not p.is_file():
        raise QualifyError(f"no such screen file: {path!r}")
    text = p.read_text()
    if not text.strip():
        raise QualifyError(f"screen file {path!r} is empty")
    first = text.splitlines()[0]
    delim = "\t" if (p.suffix.lower() in (".tsv", ".tab") or ("\t" in first and "," not in first)) else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    if not rows:
        raise QualifyError(f"screen file {path!r} has a header but no data rows")

    cols = _resolve_columns(rows[0].keys())
    missing = [k for k in _REQUIRED_COLS if k not in cols]
    if missing:
        raise QualifyError(
            "screen table is missing required column(s): "
            + "; ".join(f"{k} (any of: {', '.join(_COL_ALIASES[k])})" for k in missing)
            + f". Found columns: {list(rows[0].keys())}")

    out: list[Perturbation] = []
    for i, row in enumerate(rows):
        target = (row.get(cols["target"]) or "").strip() or f"row{i}"
        energy_p = _to_float(row.get(cols["energy_p"]))
        if energy_p != energy_p:                                 # NaN — the incumbent phenotype call is the input
            raise QualifyError(f"row {i} ({target}): no phenotype p-value — every perturbation needs one to "
                               f"qualify its 'no-phenotype' call")
        kd = _to_float(row.get(cols["knockdown"])) if "knockdown" in cols else float("nan")
        if "n_cells" in cols:
            nc = _to_float(row.get(cols["n_cells"]))
            n_cells = _NO_CELL_SENTINEL if nc != nc else int(nc)
        else:
            n_cells = _NO_CELL_SENTINEL
        out.append(Perturbation(
            pid=f"{i}_{target}", target=target, is_control=_row_is_control(row, cols, target),
            knockdown_resid=kd, energy_p=energy_p, de_count=0,
            n_cells=n_cells, n_cells_filtered=float("nan")))
    return out


def _shuffle_knockdown(perts: list[Perturbation], seed: int) -> list[Perturbation]:
    """Permute on-target knockdown across targeting perturbations — the negative control: it must destroy the
    weak-KD ↔ no-phenotype relationship (enrichment → ~1×)."""
    tgt_idx = [i for i, p in enumerate(perts) if not p.is_control]
    folds = [perts[i].knockdown_resid for i in tgt_idx]
    random.Random(seed).shuffle(folds)
    out = list(perts)
    for j, i in enumerate(tgt_idx):
        out[i] = replace(perts[i], knockdown_resid=folds[j])
    return out


def run() -> None:
    print("Silent-failure QC over a real single-cell Perturb-seq screen (Replogle K562-essential)\n")
    try:
        perts = load_perturbations()
    except DatasetUnavailable as e:
        print(f"  SKIP — {e}")
        return
    real = run_one(perts=perts)
    shuf = run_one(perts=_shuffle_knockdown(perts, seed=0))

    p1 = real["q1"] >= 0.15
    p2 = real["q2"] <= 0.20
    p3 = real["rho"] < 0.30
    p4 = real["enrich"] >= 2.0
    print("\n=== PRE-REGISTERED VERDICT ===")
    print(f"  P1 flagged≥15%        {'PASS' if p1 else 'FAIL'} ({real['q1']:.0%})")
    print(f"  P2 precision≤20%      {'PASS' if p2 else 'FAIL'} ({real['q2']:.0%})")
    print(f"  P3 non-redundant<0.30 {'PASS' if p3 else 'FAIL'} (|ρ|={real['rho']:.3f})")
    print(f"  P4 enrichment≥2×      {'PASS' if p4 else 'FAIL'} ({real['enrich']:.1f}×)")
    print(f"\n  NEGATIVE CONTROL (knockdown shuffled): enrichment {real['enrich']:.1f}× → {shuf['enrich']:.1f}× "
          f"(must collapse toward 1× — confirms the silent-failure signal is real, not mechanical)")
    print("  Read: does the legible screen-QC layer add value in the IN-DOMAIN single-cell regime? Perturb-seq's")
    print("  direct on-target knockdown gives the ground-truth silent-failure label bulk screens lacked; the QC")
    print("  layer flags untrustworthy nulls NON-REDUNDANTLY with the deposited significance — qualification,")
    print("  not accuracy.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Silent-failure QC over the Replogle K562-essential Perturb-seq screen.")
    ap.add_argument("--weak-kd-floor", type=float, default=WEAK_KD_FLOOR)
    ap.add_argument("--min-cells", type=int, default=MIN_CELLS)
    cli = ap.parse_args()
    if cli.weak_kd_floor != WEAK_KD_FLOOR or cli.min_cells != MIN_CELLS:
        run_one(view=QCView(cli.weak_kd_floor, cli.min_cells))
    else:
        run()
