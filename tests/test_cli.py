"""test_cli — the `karyon` command-line spine: qualify (exit codes + JSON schema), modality errors, the
list verb, and audit routing/serialization (audit compute is monkeypatched — no network in the test).

    python tests/test_cli.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from karyon import cli

_REPO = Path(__file__).resolve().parents[1]
_POSE = _REPO / "examples" / "compose" / "candidates" / "pose_1.sdf"

# a clean σ70 promoter: real −35/−10 boxes, 17-nt spacer, in-band GC → passes the design DRC.
_GOOD_PROMOTER = "GCATCGCTTGACAGCTAGCTAGCTAGCATTATAATGCATGCATGCATGCAT"
# no boxes + GC out of band → fails.
_BAD_PROMOTER = "GGGGGGCCCCCCGGGGGGCCCCCCGGGGGGCCCCCCGGGGGGCCCCCC"


def test_list_runs() -> None:
    assert cli.main(["list"]) == 0
    print("1. `karyon list` exits 0")


def test_qualify_json_schema_and_exit_codes(capsys) -> None:
    # a FAIL case (bad promoter) → exit 1 + valid JSON.
    rc = cli.main(["qualify", _BAD_PROMOTER, "-m", "promoter", "--json"])
    out = capsys.readouterr().out
    d = json.loads(out)
    assert rc == 1
    assert set(d) == {"modality", "ok", "items", "batch"} and d["ok"] is False
    assert set(d["items"][0]) == {"name", "ok", "score", "reasons"}
    # a PASS case (clean promoter) → exit 0.
    rc2 = cli.main(["qualify", _GOOD_PROMOTER, "-m", "promoter", "--json"])
    d2 = json.loads(capsys.readouterr().out)
    assert rc2 == 0 and d2["ok"] is True and d2["items"][0]["score"] == 0.0
    print("2. qualify --json: stable schema; exit 1 on FAIL, 0 on PASS")


def test_qualify_modality_errors(capsys) -> None:
    assert cli.main(["qualify", "CCO"]) == 2                        # inline → needs modality
    assert "modality" in capsys.readouterr().err
    assert cli.main(["qualify", "model.cif"]) == 2                  # ambiguous structure file
    assert "ambiguous" in capsys.readouterr().err
    print("3. qualify usage errors (inline-needs-modality, ambiguous ext) exit 2 with a message")


def test_qualify_human_output(capsys) -> None:
    rc = cli.main(["qualify", _GOOD_PROMOTER, "-m", "promoter"])
    out = capsys.readouterr().out
    assert rc == 0 and "PASS" in out and "overall (promoter)" in out
    print("4. qualify human output prints a PASS/FAIL summary")


def test_qualify_pose_sample(capsys) -> None:
    pytest.importorskip("rdkit")
    if not _POSE.exists():
        pytest.skip("bundled pose sample not present")
    rc = cli.main(["qualify", str(_POSE), "-m", "pose", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert rc in (0, 1) and d["modality"] == "pose" and len(d["items"]) >= 1
    print("5. qualify a bundled pose sample → valid JSON")


def test_audit_screen_routing(capsys, monkeypatch) -> None:
    from karyon import screen_qc
    monkeypatch.setattr(screen_qc, "run",
                        lambda **k: print("human noise") or {"q1": 0.53, "q2": 0.03, "thesis": True})
    rc = cli.main(["audit", "screen", "--json"])
    out = capsys.readouterr().out
    d = json.loads(out)                                             # stdout is clean JSON (human noise sunk)
    assert rc == 0 and d["audit"] == "screen" and d["q1"] == 0.53
    assert "human noise" not in out
    print("6. `karyon audit screen --json` routes + serializes (human prints suppressed)")


def test_audit_unknown_benchmark(capsys) -> None:
    assert cli.main(["audit", "leakage", "--benchmark", "nope", "--json"]) == 2
    assert "unknown leakage benchmark" in capsys.readouterr().err
    print("7. an unknown leakage benchmark exits 2 with a clear message")


def _run() -> None:
    test_list_runs()
    print("   (the rest use pytest's capsys/monkeypatch — run via: python -m pytest tests/test_cli.py)")


if __name__ == "__main__":
    _run()
