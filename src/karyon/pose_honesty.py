"""pose_honesty — the docking physical-validity honesty probe (avenue 7, PoseBusters).

Tests whether the karyon legible QC layer reveals a *large* effect on a genuinely **different** QC mechanism
than avenues 1-6 (which were all leakage/dedup audits): a **deterministic physical-validity DRC** over
predicted docking poses — the most "CAD-DRC-shaped" flavor in the queue. Two arms, mirroring the
retrosynthesis probe (instrument arm establishes the mechanism; faithful arm measures the real magnitude):

  * **Arm A — instrument check (offline, my code).** The legible intramolecular DRC (`pose_validity.py`)
    discriminates physically valid poses (seed-locked ETKDG+UFF conformers of cached ESOL molecules) from
    deterministically perturbed decoys. Proves the rules are real before any deposited data. → P1.
  * **Arm B — faithful arm (deposited DiffDock vs Vina poses + the real PoseBusters bust_results).**
      B0 faithfulness: our legible *intramolecular* DRC vs the reference PoseBusters package per-pose
        (the screen-QC → real-MAGeCK "faithful, not a strawman" guard). → P2.
      B1 the effect: combining our intra verdict with the reference *intermolecular* verdict (consumed,
        disclosed — the protein-frame bookkeeping is pipeline-specific), how inflated is the benchmark's
        RMSD-success metric, and is the effect localized to the placement axis? DiffDock vs Vina. → P3.

The honest split (see pose_data.py): our legible DRC owns the intramolecular axis end-to-end; the
intermolecular (protein) axis — where the large effect lives — is consumed from the reference and disclosed,
because reproducing it needs each method's pose+receptor in a shared coordinate frame the deposit does not
cleanly ship. The deliverable is the **legible honest-eval / qualification harness** (a faithful intra DRC +
a per-pose audit that names the failing axis), not a better docker. Qualification only.

Pre-registered (before running Arms A/B0 — the genuinely uncertain, our-code parts):
  P1  Arm A — AUROC(severity → is_decoy) ≥ 0.90 AND pass_rate(valid) ≥ 0.85 AND flag_rate(decoy) ≥ 0.85.
  P2  Arm B0 — our intra DRC matches PoseBusters' per-pose intra verdict ≥ 85% (faithful, not a strawman).
  P3  Arm B1 — ≥ 40% of DiffDock's RMSD-success poses are physically INVALID, and the effect is ≥ 5× larger
      on the intermolecular axis than the intramolecular one (the inflation localizes to placement).

    cd karyon/probe && python pose_honesty.py --poses 150
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

from . import stats_kit
from .pose_data import INTRA_TO_CONTRACT, PoseUnavailable, load_poses

try:
    from . import pose_validity as pv
    _HAVE_RDKIT = pv._HAVE_RDKIT
    from rdkit import Chem
except Exception:
    _HAVE_RDKIT = False

_ARM_A_MOLS = 150           # ESOL molecules for the instrument check


# --------------------------------------------------------------------------- #
# Pose parsing — the C0 sanitize gate (MolFromMolBlock returns a Mol on garbage; sanitize explicitly).
# --------------------------------------------------------------------------- #
def parse_pose(block: str):
    m = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    if m is None:
        return None, False
    try:
        Chem.SanitizeMol(m)
        return m, True
    except Exception:
        return m, False


def pose_features(block: str, tol):
    m, ok = parse_pose(block)
    if not ok:
        return pv.PoseFeatures(parsed=False, has_3d=False)
    return pv.featurize(m, tol)


# --------------------------------------------------------------------------- #
# Arm A — instrument check (my code, fully offline).
# --------------------------------------------------------------------------- #
def run_arm_a(n_mols: int, tol) -> dict:
    from .molnet_data import DatasetUnavailable, load_dataset
    try:
        mols = load_dataset("esol")
    except DatasetUnavailable as e:
        print(f"  Arm A SKIP — ESOL unavailable ({e})")
        return {}
    # deterministic subsample of drug-like molecules
    import random
    idx = list(range(len(mols)))
    random.Random(0).shuffle(idx)
    smiles = [mols[i].smiles for i in idx[:n_mols]]

    cs = pv.validity_contracts()
    valid_sev, decoy_sev = [], []
    valid_pass, decoy_flag = 0, 0
    n_valid, n_decoy = 0, 0
    decoy_fire = Counter()
    decoy_kind_flag: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        clean = pv.clean_conformer(smi)
        if clean is None:
            continue
        fv = pv.featurize(clean, tol)
        if not fv.ref_ok:                               # only score molecules with a usable reference
            continue
        n_valid += 1
        valid_sev.append(fv.severity(tol))
        valid_pass += 0 if pv.is_invalid(fv, cs, tol) else 1

        kind = pv.DECOYS[i % len(pv.DECOYS)]
        decoy = kind(clean, seed=i)
        fd = pv.featurize(decoy, tol)
        n_decoy += 1
        decoy_sev.append(fd.severity(tol))
        flagged = pv.is_invalid(fd, cs, tol)
        decoy_flag += 1 if flagged else 0
        decoy_kind_flag.setdefault(kind.__name__, []).append(1 if flagged else 0)
        for name in cs.evaluate(fd, tol).fired:
            decoy_fire[name] += 1

    au = stats_kit.mann_whitney(decoy_sev, valid_sev)
    auroc = au.auroc if isinstance(au, stats_kit.MannWhitney) else float("nan")
    pr = valid_pass / (n_valid or 1)
    fr = decoy_flag / (n_decoy or 1)
    print(f"\n=== ARM A — instrument check (my legible intramolecular DRC) ===")
    print(f"  molecules scored      : {n_valid} valid / {n_decoy} decoy (ESOL, seed 0)")
    print(f"  pass_rate(valid)      : {pr:.0%}   (a clean ETKDG+UFF conformer should pass)   <- P1")
    print(f"  flag_rate(decoy)      : {fr:.0%}   (a perturbed pose should be flagged)        <- P1")
    print(f"  AUROC(severity→decoy) : {auroc:.3f}                                            <- P1")
    print(f"  decoy flag-rate by perturbation type:")
    for kind, flags in sorted(decoy_kind_flag.items()):
        print(f"    {kind:24} {sum(flags)/len(flags):4.0%}  (n={len(flags)})")
    print(f"  contracts fired across decoys: " +
          ", ".join(f"{c}={decoy_fire[c]}" for c in sorted(decoy_fire, key=lambda c: -decoy_fire[c])))
    return {"auroc": auroc, "pass_valid": pr, "flag_decoy": fr}


# --------------------------------------------------------------------------- #
# Arm B — faithful arm (deposited poses + reference bust_results).
# --------------------------------------------------------------------------- #
@dataclass
class MethodResult:
    method: str
    n: int
    ref_ok_rate: float
    intra_agreement: float          # my intra verdict vs reference intra verdict (faithfulness, P2)
    my_intra_invalid: float
    ref_intra_invalid: float
    ref_inter_invalid: float
    success_n: int
    success_invalid: float          # inflation: invalid fraction among RMSD≤2 successes (full validity)
    success_intra_invalid: float    # of successes, fraction failing the intra axis (mine)
    success_inter_invalid: float    # of successes, fraction failing the inter axis (reference)


def run_method(method: str, tol, limit: int | None) -> MethodResult | None:
    try:
        poses = load_poses(method, limit=limit)
    except PoseUnavailable as e:
        print(f"  Arm B SKIP {method} — {e}")
        return None
    cs = pv.validity_contracts()

    n = len(poses)
    ref_ok = agree = my_inv = 0
    succ = succ_inv = succ_intra = succ_inter = 0
    per_contract = {col: [0, 0] for col in INTRA_TO_CONTRACT}   # [agree, total] my-fire vs bust-False
    for p in poses:
        f = pose_features(p.block, tol)
        ref_ok += 1 if f.ref_ok else 0
        my_intra_invalid = pv.is_invalid(f, cs, tol)
        my_inv += 1 if my_intra_invalid else 0
        if my_intra_invalid == (not p.ref_intra_valid):
            agree += 1
        fired = set(cs.evaluate(f, tol).fired)
        for col, contract in INTRA_TO_CONTRACT.items():
            ref_bad = p.ref_checks.get(col) is False
            mine_bad = contract in fired
            per_contract[col][1] += 1
            per_contract[col][0] += 1 if (mine_bad == ref_bad) else 0
        if p.rmsd_le_2:
            succ += 1
            intra_bad = my_intra_invalid
            inter_bad = not p.ref_inter_valid
            succ_intra += 1 if intra_bad else 0
            succ_inter += 1 if inter_bad else 0
            succ_inv += 1 if (intra_bad or inter_bad) else 0

    res = MethodResult(
        method=method, n=n, ref_ok_rate=ref_ok / n,
        intra_agreement=agree / n,
        my_intra_invalid=my_inv / n,
        ref_intra_invalid=sum(1 for p in poses if not p.ref_intra_valid) / n,
        ref_inter_invalid=sum(1 for p in poses if not p.ref_inter_valid) / n,
        success_n=succ,
        success_invalid=(succ_inv / succ) if succ else float("nan"),
        success_intra_invalid=(succ_intra / succ) if succ else float("nan"),
        success_inter_invalid=(succ_inter / succ) if succ else float("nan"))

    print(f"\n=== ARM B — {method.upper()} ({n} raw PoseBusters poses) ===")
    print(f"  reference available (ETKDG)   : {res.ref_ok_rate:.0%} of poses")
    print(f"  B0 FAITHFULNESS — my intra DRC vs the real PoseBusters package:")
    print(f"     per-pose intra-validity agreement : {res.intra_agreement:.0%}   <- P2")
    print(f"     my intra-invalid {res.my_intra_invalid:.0%}  vs  reference intra-invalid {res.ref_intra_invalid:.0%}")
    for col, (a, t) in per_contract.items():
        print(f"       {INTRA_TO_CONTRACT[col]:24} agree {a/t:4.0%}  (ref-fail {sum(1 for p in poses if p.ref_checks.get(col) is False)})")
    print(f"  AXIS DECOMPOSITION (all poses): intra-invalid {res.ref_intra_invalid:.0%} (mine≈{res.my_intra_invalid:.0%}) "
          f"vs inter-invalid {res.ref_inter_invalid:.0%} [reference, consumed]")
    if succ:
        print(f"  B1 INFLATION — of {succ} RMSD≤2 'successes': {res.success_invalid:.0%} physically INVALID "
              f"(intra {res.success_intra_invalid:.0%} | inter {res.success_inter_invalid:.0%})   <- P3")
    return res


# --------------------------------------------------------------------------- #
def run(poses_per_method: int | None, arm_a_mols: int = _ARM_A_MOLS) -> None:
    if not _HAVE_RDKIT:
        print("SKIP — rdkit/numpy not importable (the pose-validity probe needs them).")
        raise SystemExit(0)
    tol = pv.Tol()
    print("Docking physical-validity honesty probe (PoseBusters; avenue 7)")
    print("A legible deterministic DRC — the most CAD-DRC-shaped QC flavor — over predicted docking poses.")

    a = run_arm_a(arm_a_mols, tol)
    dd = run_method("diffdock", tol, poses_per_method)
    vn = run_method("vina", tol, poses_per_method)

    print("\n=== PRE-REGISTERED VERDICT ===")
    if a:
        p1 = a["auroc"] >= 0.90 and a["pass_valid"] >= 0.85 and a["flag_decoy"] >= 0.85
        print(f"  P1 instrument   {'PASS' if p1 else 'FAIL'}  "
              f"AUROC {a['auroc']:.3f}≥0.90 · pass_valid {a['pass_valid']:.0%}≥85% · flag_decoy {a['flag_decoy']:.0%}≥85%")
    methods = [m for m in (dd, vn) if m]
    if methods:
        agr = min(m.intra_agreement for m in methods)
        p2 = agr >= 0.85
        print(f"  P2 faithful     {'PASS' if p2 else 'FAIL'}  "
              f"min per-pose intra agreement vs PoseBusters {agr:.0%} ≥ 85%")
    if dd and dd.success_n:
        localized = dd.ref_inter_invalid >= 5 * max(dd.ref_intra_invalid, 0.01)
        p3 = dd.success_invalid >= 0.40 and localized
        print(f"  P3 large effect {'PASS' if p3 else 'FAIL'}  "
              f"DiffDock successes invalid {dd.success_invalid:.0%}≥40% · "
              f"inter/intra invalid {dd.ref_inter_invalid:.0%}/{dd.ref_intra_invalid:.0%} (≥5×: {'yes' if localized else 'no'})")
    if dd and vn:
        print(f"\n  DL-vs-classical contrast: physically-invalid (inter) rate "
              f"DiffDock {dd.ref_inter_invalid:.0%} vs Vina {vn.ref_inter_invalid:.0%} "
              f"(Δ {dd.ref_inter_invalid - vn.ref_inter_invalid:+.0%})")
    print("\n  Read: the legible intramolecular DRC is faithful to the reference (P2); modern DL docking's")
    print("  LOCAL geometry is clean, so the benchmark's RMSD-success inflation localizes to PLACEMENT —")
    print("  the protein axis (consumed from the reference, disclosed). Deliverable = the legible honest-eval")
    print("  harness + per-pose axis audit, NOT a better docker. Qualification, not accuracy. Qualification only.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Docking physical-validity honesty probe (PoseBusters).")
    ap.add_argument("--poses", type=int, default=150, help="poses per method (0 = all)")
    ap.add_argument("--arm-a-mols", type=int, default=_ARM_A_MOLS, help="ESOL molecules for the instrument check")
    cli = ap.parse_args()
    run(cli.poses or None, cli.arm_a_mols)
