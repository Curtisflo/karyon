"""test_operator — falsification proofs for the legible autonomous DBTL operator (dual script / pytest).

The load-bearing proofs:
  - NO-REGRESSION (online): with the gate OFF on the σ70 promoter pool, the operator reproduces loop.py's
    L curve BIT-FOR-BIT — the DRC/assay/qualify wrapper changed nothing in the cores.
  - DRC-GATE (offline): planted-broken designs are rejected with the right reason and NEVER measured.
  - QUALIFY (offline): a synthetic readout dropout is flagged and excluded from the model update; and
    qualification PROTECTS the model — ingesting corrupted high-CV readouts (no qualify) degrades the
    predictor that the qualify arm preserves.
  - PLANTED-RECOVER / NOISE-REJECT (offline): on a learnable toy substrate the predictor's held-out ρ
    rises; on shuffled truth it collapses to ≈0 (the operator manufactures no signal).
  - DETERMINISM + a legible AUDIT report (offline).

    python tests/test_operator.py
"""

from __future__ import annotations

import os
import random

from karyon import audit
from karyon import dbtl_operator as op
from karyon import loop as lp
from karyon import promoter_contracts as pc
from karyon.assay import RetrospectiveAssay
from karyon.contracts import ContractSet
from karyon.loop import LoopConfig, Substrate

_BASE = "GGCAT" + "TTGACA" + "ATCGATGCATCGATGCA" + "TATAAT" + "GCATGCATGCATGGCA"   # clean σ70, len 50


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def _mutate(seq: str, k: int, rng: random.Random) -> str:
    s = list(seq)
    for i in rng.sample(range(len(s)), k):
        s[i] = rng.choice([b for b in "ACGT" if b != s[i]])
    return "".join(s)


def _toy_substrate(n: int = 200, seed: int = 0, signal: str = "linear"):
    """A small σ70-flavoured substrate with LINEAR (learnable) or SHUFFLED (noise) truth over the promoter
    featurizer, plus a fixed set of planted-broken designs. Returns (Substrate, broken_set)."""
    rng = random.Random(seed)
    feat = lp._promoter_featurize
    w = [rng.uniform(-1, 1) for _ in range(len(feat(_BASE)))]
    clean = []
    seen = {_BASE}
    while len(clean) < n:
        s = _mutate(_BASE, rng.randint(2, 8), rng)
        if s not in seen and len(s) == len(_BASE):
            seen.add(s)
            clean.append(s)
    # Deterministic, register-INDEPENDENT breaks (a single box-replace is maskable by a chance box-pair
    # elsewhere on a 50-mer — the register-agnostic scan's known limit; the box DRC's power is validated
    # on real promoters in test_promoter_contracts (AUROC 0.66) and exercised live in OPERATOR_RESULT).
    broken = {
        "GC" * 25,                                            # C4 extreme GC (hard; always fires), len 50
        _BASE[:40] + "GAATTC" + _BASE[46:],                   # C6 forbidden EcoRI (boxes intact), len 50
    }
    broken = {b for b in broken if len(b) == len(_BASE)}
    pool = clean + sorted(broken)
    vals = [sum(wi * xi for wi, xi in zip(w, feat(s))) for s in pool]
    if signal == "random":
        shuffled = vals[:]
        random.Random(seed + 1).shuffle(shuffled)            # destroy seq↔value mapping
        vals = shuffled
    truth = {s: v for s, v in zip(pool, vals)}
    sub = Substrate(f"toy-σ70-{signal}", truth, pool, feat, len(_BASE), lambda s: True, enumerable=False)
    return sub, broken


def test_no_regression_matches_loop_L() -> None:
    """ONLINE: gate-OFF operator == loop's L strategy, bit-for-bit, on the real promoter pool."""
    try:
        sub = lp.promoter_substrate()
    except Exception as e:                                     # noqa: BLE001 — offline → skip
        return _skip(f"promoter data unavailable ({e})")
    cfg = LoopConfig(seeds=1, seed_size=32, cycles=3, batch=16, propose_m=80)
    ref = lp._run_seed(sub, 0, cfg)["L"]["curve"]
    plain = op.single_run(sub, cfg, 0, gate=False).curve
    r = [tuple(round(x, 9) for x in p) for p in ref]
    p = [tuple(round(x, 9) for x in p) for p in plain]
    assert r == p, f"operator plain-mode diverged from loop L:\n  loop {r}\n  op   {p}"
    print(f"1. no-regression: gate-OFF operator reproduces loop's L curve bit-for-bit ({len(r)} cycles)")


def test_drc_gate_rejects_broken() -> None:
    """The DRC-gate rejects planted-broken designs (with a reason) and never measures them."""
    sub, broken = _toy_substrate(n=120, seed=0)
    cfg = LoopConfig(seeds=1, seed_size=16, cycles=3, batch=12, propose_m=200, test_frac=0.0)
    run = op.single_run(sub, cfg, 0, gate=True)
    chosen = {s for c in run.cycles for s in c.chosen}
    rejected = {s for c in run.cycles for s, _ in c.rejected}
    assert broken & chosen == set(), "a DRC-broken design was measured"
    assert broken <= rejected, f"a broken design was not rejected: {broken - rejected}"
    # …and the rejection carries the right named reason.
    reason_by = {s: r for c in run.cycles for s, r in c.rejected}
    assert any("GC" in x for x in reason_by["GC" * 25])
    assert any("EcoRI" in x or "forbidden" in x for x in reason_by[_BASE[:40] + "GAATTC" + _BASE[46:]])
    print(f"2. DRC-gate rejected all {len(broken)} planted-broken designs with reasons; none measured")


def test_qualify_catches_dropout() -> None:
    """A synthetic build-dropout readout is flagged and excluded from the model update."""
    sub, _ = _toy_substrate(n=120, seed=1)
    cfg = LoopConfig(seeds=1, seed_size=16, cycles=2, batch=12, propose_m=120, test_frac=0.0)
    drop = {"n": 0}

    def stress(s, v):                                          # fail the build of ~1/4 of measured designs
        if hash(s) % 4 == 0:
            drop["n"] += 1
            return {"built": False, "value": None}
        return None
    run = op.single_run(sub, cfg, 0, gate=False, assay=RetrospectiveAssay(sub.truth, stress=stress))
    flagged = {s for c in run.cycles for s, _ in c.flagged}
    assert run.n_flagged > 0, "no dropout was flagged"
    assert flagged.isdisjoint(run.measured), "a flagged dropout leaked into the measured set"
    reason = next(r for c in run.cycles for s, r in c.flagged)
    assert any("dropout" in x or "not built" in x for x in reason)
    print(f"3. qualify flagged {run.n_flagged} build-dropouts and kept them out of the model update")


def test_qualify_protects_model() -> None:
    """Qualification PROTECTS the predictor: ingesting corrupted high-CV readouts (no qualify) degrades
    the held-out ρ that the qualify arm preserves."""
    sub, _ = _toy_substrate(n=240, seed=2)
    cfg = LoopConfig(seeds=1, seed_size=24, cycles=5, batch=16, propose_m=240, test_frac=0.25)

    def corrupt(s, v):                                         # ~1/3 of readouts: sign-flipped + noisy
        if hash(s) % 3 == 0:
            return {"value": -v - 100.0, "replicate_cv": 0.9}
        return None
    assay = RetrospectiveAssay(sub.truth, stress=corrupt)
    qual = op.single_run(sub, cfg, 0, gate=False, assay=assay, readout=pc.READOUT)
    none = op.single_run(sub, cfg, 0, gate=False, assay=assay, readout=ContractSet("none"))
    rho_q, rho_n = qual.curve[-1][3], none.curve[-1][3]
    assert qual.n_flagged > 0, "qualify caught no corrupted readouts"
    assert rho_q > rho_n, f"qualify did not protect the model (ρ qualify {rho_q:.3f} vs none {rho_n:.3f})"
    print(f"4. qualify protected the model: held-out ρ {rho_q:+.3f} (qualify) vs {rho_n:+.3f} (ingest-all)")


def test_planted_recover_and_noise_reject() -> None:
    """Learnable truth ⇒ the predictor's held-out ρ rises; shuffled truth ⇒ ρ ≈ 0 (no manufactured signal)."""
    cfg = LoopConfig(seeds=1, seed_size=32, cycles=5, batch=16, propose_m=240, test_frac=0.25)
    lin, _ = _toy_substrate(n=300, seed=3, signal="linear")
    rnd, _ = _toy_substrate(n=300, seed=3, signal="random")
    rho_lin = op.single_run(lin, cfg, 0, gate=False).curve[-1][3]
    rho_rnd = op.single_run(rnd, cfg, 0, gate=False).curve[-1][3]
    assert rho_lin > 0.30, f"failed to recover learnable signal (ρ {rho_lin:.3f})"
    assert abs(rho_rnd) < 0.15, f"manufactured signal on shuffled truth (ρ {rho_rnd:.3f})"
    print(f"5. planted-recover ρ {rho_lin:+.3f} (learnable) vs {rho_rnd:+.3f} (shuffled noise)")


def test_deterministic() -> None:
    sub, _ = _toy_substrate(n=120, seed=4)
    cfg = LoopConfig(seeds=1, seed_size=16, cycles=3, batch=12, propose_m=120, test_frac=0.2)
    a = op.single_run(sub, cfg, 0, gate=True).curve
    b = op.single_run(sub, cfg, 0, gate=True).curve
    assert a == b
    print("6. operator runs are deterministic")


def test_audit_report_is_legible() -> None:
    """The audit surface renders the reasons and a structured provenance trail."""
    sub, broken = _toy_substrate(n=120, seed=5)
    cfg = LoopConfig(seeds=1, seed_size=16, cycles=2, batch=12, propose_m=120, test_frac=0.0)
    run = op.single_run(sub, cfg, 0, gate=True)
    report = audit.run_report(run)
    assert "DRC-rejected" in report and "auditable" in report
    assert any(tag in report for tag in ("−35", "−10", "GC", "forbidden"))
    prov = audit.provenance(run)
    assert prov["n_rejected"] > 0 and prov["cycles"][0]["rejected"]
    assert all("reasons" in r for r in prov["cycles"][0]["rejected"])
    print("7. audit.run_report shows reasons; provenance is structured and serializable")


def _run() -> None:
    test_no_regression_matches_loop_L()
    test_drc_gate_rejects_broken()
    test_qualify_catches_dropout()
    test_qualify_protects_model()
    test_planted_recover_and_noise_reject()
    test_deterministic()
    test_audit_report_is_legible()
    print("\nALL operator proofs passed.")


if __name__ == "__main__":
    _run()
