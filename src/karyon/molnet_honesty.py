"""molnet_honesty — legible scaffold-leakage audit over MoleculeNet property benchmarks (avenue 6).

The property-prediction sibling of the retrosynthesis honesty probe (avenue 5), testing whether the
leakage-honesty thesis **generalizes** beyond route prediction. The documented ADMET-benchmark failure: models
report on **random** splits, which leak (shared Bemis-Murcko scaffolds straddle train/test), so reported
accuracy is inflated against the honest **scaffold-disjoint** split. The legible layer here:

  * a transparent **Tanimoto-weighted k-NN** baseline (reuses Morgan fingerprints — the program's
    similarity-is-the-model ethos), scored AUROC (classification) / Spearman ρ (regression);
  * the **inflation** = random-split metric − scaffold-split metric (the documented, decision-relevant gap);
  * a per-molecule **leakage DRC** (`contracts` engine reused verbatim): `NEAR_DUP_MOLECULE` (a train
    molecule at Tanimoto ≥ τ) and `SCAFFOLD_SEEN_IN_TRAIN` — every flag an auditable reason.

Pre-registered (before seeing the audit metrics):
  P1  scaffold-leakage prevalence on the random-split test ≥ 50%.
  P2  inflation (random − scaffold metric) ≥ 0.05 (AUROC for clf, ρ for reg).
  P3  the random-split metric on the leakage-free partition is ≥ 0.03 below the full random-split metric.

rdkit-gated; SKIPs cleanly if rdkit/data absent. Reuses `contracts` + `stats_kit`.

    python -m karyon.molnet_honesty            # runs bbbp + esol
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from dataclasses import dataclass

from . import contracts
from . import stats_kit
from .molnet_data import (
    DATASETS,
    DatasetUnavailable,
    Molecule,
    Split,
    load_dataset,
    random_split,
    scaffold_split,
)

try:
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False

_TAU = 0.7          # near-duplicate Morgan-Tanimoto threshold
_K = 5              # k-NN neighbours
_MORGAN_R, _MORGAN_BITS = 2, 2048


def _fps(smiles: list[str]):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=_MORGAN_R, fpSize=_MORGAN_BITS)
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(gen.GetFingerprint(m) if m is not None else None)
    return out


@dataclass(frozen=True)
class MolOutcome:
    mol: Molecule
    nn_sim: float
    scaffold_seen: bool
    pred: float                 # predicted prob (clf) or value (reg)

    @property
    def correct(self) -> bool:  # classification only (rounded); used for the sim-stratified accuracy curve
        return round(self.pred) == round(self.mol.label)


def _knn(test_fp, train_fps, labels, k: int) -> tuple[float, float]:
    """Tanimoto-weighted k-NN: returns (predicted label, nearest-neighbour similarity)."""
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(test_fp, train_fps))
    kk = min(k, len(sims))
    top = sorted(np.argpartition(-sims, kk - 1)[:kk], key=lambda i: (-sims[i], i))
    w = np.array([sims[i] for i in top])
    lab = np.array([labels[i] for i in top])
    pred = float((w * lab).sum() / w.sum()) if w.sum() > 0 else float(lab.mean())
    return pred, float(sims[top[0]])


def predict_split(split: Split) -> list[MolOutcome]:
    train_fps = _fps([m.smiles for m in split.train])
    keep = [i for i, fp in enumerate(train_fps) if fp is not None]
    tfps = [train_fps[i] for i in keep]
    labels = [split.train[i].label for i in keep]
    train_scaffolds = {m.scaffold for m in split.train if m.scaffold}
    outs = []
    for m, fp in zip(split.test, _fps([m.smiles for m in split.test])):
        if fp is None:
            continue
        pred, nn = _knn(fp, tfps, labels, _K)
        outs.append(MolOutcome(m, nn, bool(m.scaffold) and m.scaffold in train_scaffolds, pred))
    return outs


def metric(outs: list[MolOutcome], classification: bool) -> float:
    if not outs:
        return float("nan")
    if classification:
        au = stats_kit.mann_whitney([o.pred for o in outs if o.mol.label >= 0.5],
                                    [o.pred for o in outs if o.mol.label < 0.5])
        return au.auroc if isinstance(au, stats_kit.MannWhitney) else float("nan")
    r = stats_kit.spearman([o.pred for o in outs], [o.mol.label for o in outs])
    return r.rho if isinstance(r, stats_kit.Corr) else float("nan")


# --------------------------------------------------------------------------- #
# The leakage DRC (reused contracts engine).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainView:
    tau: float


def leakage_contracts() -> contracts.ContractSet:
    cs = contracts.ContractSet("molnet-leakage")
    cs.add(contracts.Contract(
        "NEAR_DUP_MOLECULE",
        lambda o, ctx: f"a train molecule is near-identical (Tanimoto {o.nn_sim:.2f} ≥ {ctx.tau})"
        if o.nn_sim >= ctx.tau else None))
    cs.add(contracts.Contract(
        "SCAFFOLD_SEEN_IN_TRAIN",
        lambda o, ctx: "this molecule's Bemis-Murcko scaffold was seen in train" if o.scaffold_seen else None))
    return cs


def _decile_curve(outs: list[MolOutcome], classification: bool, bins: int = 10):
    ordered = sorted(outs, key=lambda o: o.nn_sim)
    n = len(ordered)
    rows = []
    for b in range(bins):
        chunk = ordered[b * n // bins:(b + 1) * n // bins]
        if chunk:
            rows.append((statistics.fmean([o.nn_sim for o in chunk]), metric(chunk, classification)))
    return rows


def run_one(name: str, *, tau: float = _TAU) -> dict | None:
    spec = DATASETS[name]
    clf = spec.classification
    mlabel = "AUROC" if clf else "ρ"
    try:
        mols = load_dataset(name)
    except DatasetUnavailable as e:
        print(f"  SKIP {name} — {e}")
        return None

    rnd = predict_split(random_split(mols, seed=0))
    scf = predict_split(scaffold_split(mols, seed=0))
    m_rnd, m_scf = metric(rnd, clf), metric(scf, clf)

    cs, tv = leakage_contracts(), TrainView(tau)
    verdicts = [cs.evaluate(o, tv) for o in rnd]
    clean = [o for o, v in zip(rnd, verdicts) if v.ok]
    prevalence = sum(1 for v in verdicts if not v.ok) / (len(rnd) or 1)
    fire = Counter(name for v in verdicts for name in v.fired)
    m_clean = metric(clean, clf)

    print(f"\n=== {name.upper()} ({'classification' if clf else 'regression'}; metric={mlabel}) ===")
    print(f"  random split   {mlabel} = {m_rnd:.3f}   (train/test {len(rnd)} scored)")
    print(f"  scaffold split {mlabel} = {m_scf:.3f}   (the honest, scaffold-disjoint eval)")
    print(f"  INFLATION (random − scaffold) = {m_rnd - m_scf:+.3f}   <- P2")
    print(f"  leakage prevalence (random test): {prevalence:.1%}   <- P1")
    for c in cs.contracts:
        print(f"    {c.name:<24} {fire.get(c.name, 0) / (len(rnd) or 1):6.1%}")
    print(f"  random {mlabel}: full {m_rnd:.3f} → leakage-free partition {m_clean:.3f} "
          f"({len(clean)}/{len(rnd)} survive)   <- P3")
    print(f"  similarity-stratified {mlabel} (deciles):")
    for sim, mv in _decile_curve(rnd, clf):
        print(f"    nn_sim≈{sim:.2f}  {mlabel}={mv:.3f}")

    return {"name": name, "clf": clf, "m_rnd": m_rnd, "m_scf": m_scf, "inflation": m_rnd - m_scf,
            "prevalence": prevalence, "m_clean": m_clean}


def run(tau: float = _TAU) -> None:
    if not _HAVE_RDKIT:
        print("SKIP — rdkit/numpy not importable (the molecular-property arm needs them).")
        raise SystemExit(0)
    print("Scaffold-leakage honesty audit over MoleculeNet property benchmarks\n")
    results = [r for r in (run_one(n, tau=tau) for n in DATASETS) if r]
    if not results:
        return
    print("\n=== PRE-REGISTERED VERDICT (per dataset) ===")
    for r in results:
        unit = "AUROC" if r["clf"] else "ρ"
        p1 = r["prevalence"] >= 0.50
        p2 = r["inflation"] >= 0.05
        p3 = (r["m_rnd"] - r["m_clean"]) >= 0.03
        print(f"  {r['name']:<6} P1 leakage≥50% {'PASS' if p1 else 'FAIL'} ({r['prevalence']:.0%}) · "
              f"P2 inflation≥0.05 {'PASS' if p2 else 'FAIL'} ({r['inflation']:+.3f} {unit}) · "
              f"P3 clean-drop≥0.03 {'PASS' if p3 else 'FAIL'} ({r['m_rnd'] - r['m_clean']:+.3f})")
    print("\n  Read: does the leakage-honesty win GENERALIZE beyond retrosynthesis? The scaffold-leakage")
    print("  audit measures the random-vs-scaffold inflation per dataset with auditable per-molecule reasons.")
    print("  Deliverable = the legible honest-eval harness (qualification), not a better predictor. Qualification only")
    print("  call.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scaffold-leakage honesty audit over MoleculeNet (bbbp + esol).")
    ap.add_argument("--tau", type=float, default=_TAU)
    cli = ap.parse_args()
    run(tau=cli.tau)
