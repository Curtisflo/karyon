"""test_ppi_leakage_cli — the `karyon audit leakage --benchmark ppi` surface: the report() aggregator
(strict-JSON, NaN-safe) and CLI routing. run_one / report are monkeypatched, so the whole file is offline —
the real audit's math is covered by test_ppi_leakage.py.

    python -m pytest tests/test_ppi_leakage_cli.py
"""

from __future__ import annotations

import json

from karyon import cli
from karyon import ppi_leakage as pl


def _seed_dict(seed, *, c1, infl=0.27, prev=0.85):
    return {"seed": seed, "prevalence": prev, "au_full_node": 0.70, "au_c1_node": c1,
            "au_c3_node": 0.50, "au_c3_seq": 0.50, "inflation": infl,
            "n_test": 1000, "n_c1": 600, "n_c2": 300, "n_c3": 100}


def test_report_aggregates_and_is_strict_json(monkeypatch) -> None:
    # second seed carries a NaN AUROC (an empty C1 stratum) — it must drop from the mean and serialize as null.
    outs = iter([_seed_dict(0, c1=0.77, infl=0.27), _seed_dict(1, c1=float("nan"), infl=0.25)])
    monkeypatch.setattr(pl, "run_one", lambda **k: next(outs))
    rep = pl.report(seeds=2)
    assert rep["benchmark"] == "ppi" and rep["seeds"] == 2
    assert abs(rep["node_identity_inflation"] - 0.26) < 1e-9      # mean(0.27, 0.25)
    assert rep["reported_auroc_c1"] == 0.77                       # the NaN seed dropped from the mean
    assert rep["honest_auroc_c3"] == 0.50
    assert rep["p1_prevalence_pass"] is True and rep["p2_inflation_pass"] is True
    json.dumps(rep, allow_nan=False)                             # STRICT JSON — no NaN/inf may survive
    print("1. report() aggregates across seeds, drops NaN AUROCs, stays strict-JSON")


def test_report_none_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(pl, "run_one", lambda **k: None)         # every seed skips (offline / uncached)
    assert pl.report(seeds=3) is None
    print("2. report() → None when the benchmark is unreachable")


def test_cli_audit_leakage_ppi_routing(capsys, monkeypatch) -> None:
    monkeypatch.setattr(pl, "report",
                        lambda **k: print("human noise") or {"benchmark": "ppi",
                                                             "node_identity_inflation": 0.275,
                                                             "p2_inflation_pass": True})
    rc = cli.main(["audit", "leakage", "--benchmark", "ppi", "--json"])
    out = capsys.readouterr().out
    d = json.loads(out)                                          # clean JSON (human noise sunk under --json)
    assert rc == 0 and d["audit"] == "leakage" and d["benchmark"] == "ppi"
    assert "human noise" not in out and "NaN" not in out
    print("3. `karyon audit leakage --benchmark ppi --json` routes + serializes")


def test_cli_audit_ppi_unavailable(capsys, monkeypatch) -> None:
    monkeypatch.setattr(pl, "report", lambda **k: None)
    rc = cli.main(["audit", "leakage", "--benchmark", "ppi"])
    assert rc == 2 and "unavailable" in capsys.readouterr().err
    print("4. an unavailable ppi dataset exits 2 with a clear message")


if __name__ == "__main__":
    print("CLI cases use pytest capsys/monkeypatch — run: python -m pytest tests/test_ppi_leakage_cli.py")
