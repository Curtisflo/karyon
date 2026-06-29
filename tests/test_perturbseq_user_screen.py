"""test_perturbseq_user_screen — the *bring-your-own-screen* surface: parse a user's single-cell screen
summary into Perturbations, qualify it into the JSON-safe `audit_report` (per-call `flagged` + named
reasons), and drive it through the `karyon audit screen --single-cell` CLI. No h5py, no network — this is the
core-install path, so the whole file runs offline.

The CLI cases use pytest's capsys/monkeypatch; the parse/report cases run under plain `python` too.

    python -m pytest tests/test_perturbseq_user_screen.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from karyon import cli
from karyon import perturbseq_qc as pq
from karyon.perturbseq_data import Perturbation
from karyon.spine import QualifyError


def _write(text: str, suffix: str = ".csv") -> str:
    fh = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    fh.write(text)
    fh.close()
    return fh.name


# One row per kind of no-phenotype call, plus a clear hit and two controls (one a chance hit, for calibration).
def _synthetic() -> list[Perturbation]:
    return [
        Perturbation("0_WEAK", "WEAK", False, 0.80, 0.90, 0, 200, 200.0),   # weak KD, no phenotype → WEAK_KNOCKDOWN
        Perturbation("1_GOOD", "GOOD", False, 0.08, 0.90, 0, 200, 200.0),   # strong KD, no phenotype → trustworthy
        Perturbation("2_UNM", "UNM", False, float("nan"), 0.70, 0, 200, 200.0),  # unmeasured → KNOCKDOWN_UNMEASURED
        Perturbation("3_FEW", "FEW", False, 0.10, 0.80, 0, 5, 5.0),         # strong KD but → LOW_CELL_COUNT
        Perturbation("4_HIT", "HIT", False, 0.05, 1e-5, 50, 300, 300.0),    # a clear hit
        Perturbation("c0_non-targeting", "non-targeting", True, float("nan"), 0.80, 0, 300, float("nan")),
        Perturbation("c1_non-targeting", "non-targeting", True, float("nan"), 0.70, 0, 300, float("nan")),
    ]


# --------------------------------------------------------------------------- #
# audit_report — structure, the per-call flagged payload, JSON-safety.
# --------------------------------------------------------------------------- #
def test_audit_report_structure_and_flagged() -> None:
    rep = pq.audit_report(perts=_synthetic())
    assert set(rep) >= {"screen", "n_targeting", "n_controls", "calibration", "n_nophenotype",
                        "flagged_rate", "by_contract", "partition", "n_flagged", "flagged"}
    assert rep["screen"] == "single-cell"
    assert rep["n_targeting"] == 5 and rep["n_controls"] == 2
    assert rep["n_nophenotype"] == 4                                # everything but the clear hit
    # each of the three contracts fires exactly once among the four no-phenotype calls.
    flagged_by = {f["target"]: [r["contract"] for r in f["reasons"]] for f in rep["flagged"]}
    assert flagged_by["WEAK"] == ["WEAK_KNOCKDOWN"]
    assert flagged_by["UNM"] == ["KNOCKDOWN_UNMEASURED"]
    assert "LOW_CELL_COUNT" in flagged_by["FEW"]
    assert "GOOD" not in flagged_by                                 # a trustworthy negative is not flagged
    assert rep["n_flagged"] == 3 and abs(rep["flagged_rate"] - 0.75) < 1e-9
    json.dumps(rep, allow_nan=False)                               # STRICT JSON — no NaN/inf may leak in
    print(f"1. audit_report flags {rep['n_flagged']}/{rep['n_nophenotype']} no-phenotype calls, each named")


def test_audit_report_is_strict_json_on_degenerate_pile() -> None:
    # one measured no-phenotype call → Spearman ρ is undefined (NaN) and enrichment has no weak-KD-hit
    # denominator; both must serialize as null, not NaN/inf, so a strict JSON consumer can parse it.
    perts = [
        Perturbation("0_X", "X", False, 0.80, 0.90, 0, 200, 200.0),
        Perturbation("c_non-targeting", "non-targeting", True, float("nan"), 0.50, 0, 200, float("nan")),
    ]
    rep = pq.audit_report(perts=perts)
    assert rep["rho_knockdown_vs_significance"] is None
    assert rep["weak_kd_enrichment"] is None
    json.dumps(rep, allow_nan=False)                               # raises if any NaN/inf survives
    print("1b. a degenerate pile yields null (not NaN/inf) → the report stays strict-JSON valid")


def test_audit_report_flagged_sorted_residual_desc() -> None:
    rep = pq.audit_report(perts=_synthetic())
    resid = [f["knockdown_residual"] for f in rep["flagged"] if f["knockdown_residual"] is not None]
    assert resid == sorted(resid, reverse=True), "flagged should be highest-residual-first"
    print("2. flagged calls are ordered worst-knockdown first")


# --------------------------------------------------------------------------- #
# load_user_screen — column aliases, control inference, missing values, errors.
# --------------------------------------------------------------------------- #
def test_load_user_screen_aliases_and_controls() -> None:
    path = _write(
        "gene,residual_expression,energy_test_p_value,num_cells\n"
        "WEAK,0.82,0.71,180\n"
        "GOOD,0.07,0.92,210\n"
        "UNM,,0.65,150\n"            # blank knockdown → unmeasured
        "non-targeting,,0.80,300\n"  # control inferred from the target name (no control column)
    )
    perts = pq.load_user_screen(path)
    assert len(perts) == 4
    assert sum(p.is_control for p in perts) == 1
    unm = next(p for p in perts if p.target == "UNM")
    assert not unm.knockdown_measured                              # blank parsed to NaN → unmeasured
    weak = next(p for p in perts if p.target == "WEAK")
    assert weak.knockdown_resid == 0.82 and weak.n_cells == 180
    print("3. load_user_screen resolves aliases, infers controls, maps blank knockdown → unmeasured")


def test_load_user_screen_tsv_and_explicit_control_column() -> None:
    path = _write(
        "target\tknockdown_residual\tpvalue\tis_control\n"
        "FOO\t0.9\t0.5\tfalse\n"
        "BAR\t0.1\t0.5\ttrue\n",
        suffix=".tsv",
    )
    perts = pq.load_user_screen(path)
    assert [p.is_control for p in perts] == [False, True]
    # no cells column at all → the power floor cannot fire (sentinel, rendered as unknown downstream).
    rep = pq.audit_report(perts=perts)
    assert all("LOW_CELL_COUNT" not in [r["contract"] for r in f["reasons"]] for f in rep["flagged"])
    print("4. TSV + explicit control column parse; a missing cells column never trips LOW_CELL_COUNT")


def test_load_user_screen_missing_required_column() -> None:
    path = _write("foo,bar\n1,2\n")
    with pytest.raises(QualifyError) as e:
        pq.load_user_screen(path)
    assert "target" in str(e.value) and "energy_p" in str(e.value)
    print("5. a table missing target / p-value raises QualifyError naming the accepted aliases")


def test_load_user_screen_row_without_pvalue() -> None:
    path = _write("target,pvalue\nGENE,\n")                        # the phenotype call is the required input
    with pytest.raises(QualifyError):
        pq.load_user_screen(path)
    print("6. a row with no phenotype p-value raises (can't qualify a null without the call)")


# --------------------------------------------------------------------------- #
# the CLI surface.
# --------------------------------------------------------------------------- #
def test_cli_single_cell_user_screen_json(capsys) -> None:
    path = _write(
        "target,residual_expression,energy_test_p_value,num_cells\n"
        "WEAK,0.82,0.71,180\n"
        "GOOD,0.07,0.92,210\n"
        "HIT,0.05,0.00001,260\n"
        "non-targeting,,0.80,300\n"
    )
    rc = cli.main(["audit", "screen", "--single-cell", "--input", path, "--json"])
    out = capsys.readouterr().out
    d = json.loads(out)                                            # stdout is clean JSON (human summary sunk)
    assert rc == 0
    assert d["audit"] == "screen" and d["screen"] == "single-cell" and d["source"] == path
    assert set(d["by_contract"]) == {"WEAK_KNOCKDOWN", "KNOCKDOWN_UNMEASURED", "LOW_CELL_COUNT"}
    assert any(f["target"] == "WEAK" for f in d["flagged"])
    assert "NaN" not in out and "Infinity" not in out              # the wire output is strict JSON
    print("7. `karyon audit screen --single-cell --input F --json` → clean JSON report, exit 0")


def test_cli_single_cell_human_lists_untrusted(capsys) -> None:
    path = _write("target,residual_expression,pvalue,num_cells\nWEAK,0.82,0.71,180\nnon-targeting,,0.8,300\n")
    rc = cli.main(["audit", "screen", "--single-cell", "--input", path])
    out = capsys.readouterr().out
    assert rc == 0 and "should NOT trust" in out and "WEAK_KNOCKDOWN" in out
    print("8. human output names the no-phenotype calls you should not trust")


def test_cli_single_cell_reference_unavailable(monkeypatch, capsys) -> None:
    from karyon import perturbseq_data
    def _boom(*a, **k):
        raise perturbseq_data.DatasetUnavailable("h5py not importable (the approved .h5ad reader)")
    monkeypatch.setattr(perturbseq_data, "load_perturbations", _boom)
    rc = cli.main(["audit", "screen", "--single-cell"])           # no --input → the bundled reference
    err = capsys.readouterr().err
    assert rc == 2 and "singlecell" in err and "--input" in err
    print("9. the reference path with no reader exits 2 with an actionable install hint")


def test_cli_single_cell_flags_apply_only_to_screen(capsys) -> None:
    rc = cli.main(["audit", "leakage", "--single-cell"])
    assert rc == 2 and "only" in capsys.readouterr().err
    print("10. --single-cell on a non-screen audit exits 2 with a clear message")


def _run() -> None:
    test_audit_report_structure_and_flagged()
    test_audit_report_is_strict_json_on_degenerate_pile()
    test_audit_report_flagged_sorted_residual_desc()
    test_load_user_screen_aliases_and_controls()
    test_load_user_screen_tsv_and_explicit_control_column()
    test_load_user_screen_missing_required_column()
    test_load_user_screen_row_without_pvalue()
    print("   (CLI cases use pytest capsys/monkeypatch — run: python -m pytest tests/test_perturbseq_user_screen.py)")


if __name__ == "__main__":
    _run()
    print("\nperturbseq user-screen proofs pass.")
