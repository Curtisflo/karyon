"""pose_validity — a legible physical-validity DRC for molecular docking poses (avenue 7, PoseBusters).

The karyon thesis: AI authors a *legible* reliability / QC layer — the DRC
design-rule-check (DRC) + `contracts` engine ported to biology — not a black-box predictor. Avenues 1-6
spanned that layer across **leakage/dedup** audits (CRISPR, retrosynthesis, ADMET). This module tests a
genuinely **different QC mechanism**: a *deterministic physical-validity DRC* over a predicted 3D ligand
pose — bond geometry, steric clashes, ring planarity, internal strain — the **most "CAD-DRC-shaped" QC
flavor** in the queue (it *is* a geometric design-rule-check, the way a CAD tool checks a printed part).

It re-derives the documented PoseBusters finding (Buttenschoen et al., Chem Sci 2024) on its own terms:
deep-learning docking (DiffDock) emits poses that score well on RMSD yet are *physically invalid*, while
classical docking (AutoDock Vina) stays valid. Here the checks are expressed as `contracts.Contract`s so
every flag carries a human-readable reason — the legible layer, validated against the real PoseBusters
package output (`pose_honesty.py`'s faithfulness cross-check, the screen-QC→real-MAGeCK precedent).

Design (mirrors `molnet_honesty.leakage_contracts`): all RDKit geometry is computed ONCE per pose in
`featurize(mol, ref, tol) -> PoseFeatures`; the contracts are thin predicates reading precomputed scalars,
so the `contracts` engine stays substrate-agnostic and the rules are trivially unit-testable on planted
features. Every threshold in `Tol` is a physical constant or reference-derived — **zero fitted parameters**.

Geometry conventions match PoseBusters where it matters:
  * bond length / angle validity = **relative to an ETKDG reference ensemble** of the same molecular graph
    (ratio in [1-tol, 1+tol]), NOT a z-score — rigid bonds have ~zero ensemble spread, so a relative bound
    is the robust form (this is *why* PoseBusters uses relative bounds; a z-score would divide by ~0).
  * aromatic-ring flatness, internal steric clash, double-bond planarity need NO reference (pure geometry),
    so they run even when ETKDG embedding fails on a hard molecule.
  * internal strain = UFF(pose) / mean UFF(relaxed reference ensemble); skipped-with-disclosure when UFF
    cannot parametrize the molecule (metal / exotic atom) — the `stats_kit.Degenerate` philosophy.

rdkit-gated; the pure contract logic over `PoseFeatures` runs without rdkit (for the falsification proofs).

    python -m karyon.pose_validity        # smoke: featurize a clean conformer, print the verdict
"""

from __future__ import annotations
from .paths import cache_dir

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from . import contracts

try:
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdMolTransforms
    from rdkit.Geometry import Point3D
    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False

_REF_SEED = 0xC0FFEE        # the ONLY stochastic element (ETKDG) — locked, so the DRC is deterministic
_REF_CONFS = 16             # reference-ensemble size (speed/stability balance)


# --------------------------------------------------------------------------- #
# Calibration — every threshold a physical constant or reference-derived (zero fitted params).
# Defaults follow the PoseBusters package so the legible reimplementation is faithful to the reference.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tol:
    bond_rel: float = 0.25       # bond length within ±25% of the ETKDG reference mean (PoseBusters default)
    angle_rel: float = 0.25      # bond angle within ±25% of reference
    ring_flat_A: float = 0.25    # Å — aromatic-ring max out-of-plane deviation (PoseBusters default)
    double_bond_deg: float = 25.0  # ° from planar (0/180) for a non-rotatable double bond
    clash_rel: float = 0.60      # non-bonded separation < 0.60·(vdwᵢ+vdwⱼ) ⇒ steric clash (the standard
    #                              ~40%-vdW-overlap criterion; this fraction calibrated to reproduce the
    #                              reference PoseBusters internal_steric_clash convention — 93%/100% agree)
    energy_ratio: float = 100.0  # UFF(pose)/mean UFF(reference) > 100× ⇒ implausible internal strain


# --------------------------------------------------------------------------- #
# The precomputed per-pose features the contracts read (the molnet_honesty.MolOutcome analogue).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PoseFeatures:
    parsed: bool = True              # sanitize succeeded (C0)
    has_3d: bool = True              # a real, non-degenerate 3D conformer (C1)
    ref_ok: bool = True              # an ETKDG reference ensemble was available (bond/angle/energy live)
    uff_ok: bool = True              # UFF parametrized the molecule (energy check live)
    worst_bond_rel_lo: float = 1.0   # min pose/reference bond-length ratio (short outliers)
    worst_bond_rel_hi: float = 1.0   # max pose/reference bond-length ratio (long outliers)
    worst_angle_rel: float = 1.0     # max |pose/reference − 1| + 1 for angles (≥1)
    max_ring_dev_A: float = 0.0      # worst aromatic-ring out-of-plane distance, Å
    max_double_bond_deg: float = 0.0  # worst non-rotatable double-bond twist, °
    min_clash_rel: float = 1.5       # min non-bonded dist / (vdwᵢ+vdwⱼ)
    energy_ratio: float = 1.0        # UFF(pose) / mean UFF(reference ensemble)
    n_bonds: int = 0
    n_clash_pairs: int = 0

    def severity(self, tol: Tol) -> float:
        """A continuous severity (Σ normalized exceedances) — the RANKING statistic for Arm A's AUROC.
        Distinct from the Verdict.score (Σ fired weights); 0.0 for a clean pose, grows with violations."""
        def relu(x: float) -> float:
            return x if x > 0 else 0.0
        s = 0.0
        if self.ref_ok:
            s += relu((1.0 - tol.bond_rel) / max(self.worst_bond_rel_lo, 1e-6) - 1.0)   # too-short
            s += relu(self.worst_bond_rel_hi / (1.0 + tol.bond_rel) - 1.0)              # too-long
            s += relu((self.worst_angle_rel - 1.0) / tol.angle_rel - 1.0)
        s += relu(self.max_ring_dev_A / tol.ring_flat_A - 1.0)
        s += relu(self.max_double_bond_deg / tol.double_bond_deg - 1.0)
        s += relu(tol.clash_rel / max(self.min_clash_rel, 1e-6) - 1.0)
        if self.ref_ok and self.uff_ok:
            s += relu(math.log10(max(self.energy_ratio, 1.0)) / math.log10(tol.energy_ratio) - 1.0)
        return s


# --------------------------------------------------------------------------- #
# The DRC — each check a legible contract reading a precomputed PoseFeatures scalar.
# --------------------------------------------------------------------------- #
def validity_contracts() -> contracts.ContractSet:
    cs = contracts.ContractSet("pose-physical-validity")

    cs.add(contracts.Contract("POSE_UNPARSEABLE",
        lambda f, t: "pose did not sanitize (valence/kekulization error)" if not f.parsed else None,
        weight=2.0))
    cs.add(contracts.Contract("NO_3D_CONFORMER",
        lambda f, t: "pose has no real 3D conformer (2D or degenerate coordinates)"
        if f.parsed and not f.has_3d else None, weight=2.0))

    cs.add(contracts.Contract("BOND_LENGTH_OUTLIER",
        lambda f, t: (
            f"a bond is {f.worst_bond_rel_lo:.0%} of its reference length (<{1 - t.bond_rel:.0%})"
            if f.worst_bond_rel_lo < 1 - t.bond_rel else
            f"a bond is {f.worst_bond_rel_hi:.0%} of its reference length (>{1 + t.bond_rel:.0%})")
        if f.ref_ok and (f.worst_bond_rel_lo < 1 - t.bond_rel or f.worst_bond_rel_hi > 1 + t.bond_rel)
        else None))
    cs.add(contracts.Contract("BOND_ANGLE_OUTLIER",
        lambda f, t: f"a bond angle is {f.worst_angle_rel:.0%} of its reference (>{1 + t.angle_rel:.0%})"
        if f.ref_ok and f.worst_angle_rel > 1 + t.angle_rel else None))

    cs.add(contracts.Contract("AROMATIC_RING_NONPLANAR",
        lambda f, t: f"an aromatic ring is non-planar (max {f.max_ring_dev_A:.2f} Å out of plane, "
        f">{t.ring_flat_A} Å)" if f.max_ring_dev_A > t.ring_flat_A else None))
    cs.add(contracts.Contract("DOUBLE_BOND_NONPLANAR",
        lambda f, t: f"a planar double bond is twisted {f.max_double_bond_deg:.0f}° (>{t.double_bond_deg:.0f}°)"
        if f.max_double_bond_deg > t.double_bond_deg else None))

    cs.add(contracts.Contract("INTERNAL_STERIC_CLASH",
        lambda f, t: f"two non-bonded atoms clash ({f.min_clash_rel:.0%} of their van-der-Waals sum, "
        f"<{t.clash_rel:.0%})" if f.min_clash_rel < t.clash_rel else None, weight=1.5))

    cs.add(contracts.Contract("INTERNAL_STRAIN_ENERGY",
        lambda f, t: f"pose internal energy is {f.energy_ratio:.0f}× the relaxed reference (>{t.energy_ratio:.0f}×)"
        if f.ref_ok and f.uff_ok and f.energy_ratio > t.energy_ratio else None))
    # Honest disclosure (weight 0.0 — informs, does not condemn): the energy oracle was unavailable.
    cs.add(contracts.Contract("ENERGY_UNCHECKABLE",
        lambda f, t: "internal-energy check skipped: UFF cannot parametrize this molecule (metal/exotic atom)"
        if f.ref_ok and not f.uff_ok else None, kind=contracts.CALIBRATED, weight=0.0))
    return cs


def is_invalid(features: PoseFeatures, cs: contracts.ContractSet, tol: Tol) -> bool:
    """A pose is physically INVALID iff a *condemning* contract fired (score > 0 — ENERGY_UNCHECKABLE,
    weight 0, discloses without condemning)."""
    return cs.evaluate(features, tol).score > 0.0


# --------------------------------------------------------------------------- #
# Geometry helpers (rdkit) — all wrapped so one bad check never crashes a pose.
# --------------------------------------------------------------------------- #
def _heavy(mol):
    """Heavy-atom view with a single 3D conformer preserved (the pose)."""
    try:
        return Chem.RemoveHs(mol)
    except Exception:
        return mol


def _aromatic_ring_dev(mol, conf) -> float:
    """Worst aromatic-ring out-of-plane deviation (Å) via an SVD best-fit plane. 0.0 if no aromatic ring."""
    worst = 0.0
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if not all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            continue
        pts = np.array([[conf.GetAtomPosition(a).x, conf.GetAtomPosition(a).y,
                         conf.GetAtomPosition(a).z] for a in ring])
        centroid = pts.mean(axis=0)
        # smallest-singular-vector = plane normal; max |projection| = out-of-plane distance
        _, _, vh = np.linalg.svd(pts - centroid)
        dev = float(np.max(np.abs((pts - centroid) @ vh[2])))
        worst = max(worst, dev)
    return worst


def _double_bond_twist(mol, conf) -> float:
    """Worst twist (°) of a non-ring double bond from planarity (nearest of 0°/180°). 0.0 if none."""
    worst = 0.0
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.IsInRing():
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        na = next((n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != b.GetIdx()), None)
        nb = next((n.GetIdx() for n in b.GetNeighbors() if n.GetIdx() != a.GetIdx()), None)
        if na is None or nb is None:
            continue                                    # a terminal/H-only double bond — dihedral undefined
        try:
            d = rdMolTransforms.GetDihedralDeg(conf, na, a.GetIdx(), b.GetIdx(), nb)
        except Exception:
            continue
        twist = min(abs(d), abs(180.0 - abs(d)))        # distance to the nearer of {0,180}
        worst = max(worst, twist)
    return worst


def _min_clash_rel(mol, conf) -> tuple[float, int]:
    """Min non-bonded (topological distance ≥3) separation as a fraction of the van-der-Waals sum, and the
    number of such pairs checked. Heavy atoms only (the robust core; H contacts are noisier)."""
    n = mol.GetNumAtoms()
    if n < 2:
        return 1.5, 0
    d3 = Chem.Get3DDistanceMatrix(mol)
    dtopo = Chem.GetDistanceMatrix(mol)
    pt = Chem.GetPeriodicTable()
    vdw = [pt.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in range(n)]
    worst, pairs = 1.5, 0
    for i in range(n):
        for j in range(i + 1, n):
            t = dtopo[i][j]
            # skip 1,2 (bonded) + 1,3 (angle) neighbours AND inter-fragment pairs (RDKit marks
            # disconnected atoms with a ~1e8 topological distance) — a multi-component ligand's
            # cross-fragment contacts are INTERmolecular, not an internal clash (the PoseBusters
            # internal_steric_clash convention; without this a 5-fragment ligand false-clashes).
            if t < 3 or t > 1e6:
                continue
            pairs += 1
            rel = d3[i][j] / (vdw[i] + vdw[j])
            if rel < worst:
                worst = rel
    return worst, pairs


# --------------------------------------------------------------------------- #
# The ETKDG reference oracle — "what should this molecular graph's geometry look like" (bond/angle/energy).
# Cached per canonical SMILES (in-memory + disk), keyed by canonical atom ranks so it is index-independent.
# --------------------------------------------------------------------------- #
@dataclass
class RefStats:
    ok: bool
    bond: dict = field(default_factory=dict)    # "rankA-rankB" -> mean reference length (Å)
    angle: dict = field(default_factory=dict)   # "rankC|rankA-rankB" -> mean reference angle (°)
    energy_mean: float = 0.0
    uff_ok: bool = False


_REF_CACHE: dict[str, RefStats] = {}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    return here.parents[2]


def _ref_disk_path() -> Path:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "pose_refstats.json"


def _load_ref_disk() -> None:
    if _REF_CACHE:
        return
    p = _ref_disk_path()
    if not p.exists():
        return
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return
    for smi, d in raw.items():
        _REF_CACHE[smi] = RefStats(d["ok"], d.get("bond", {}), d.get("angle", {}),
                                   d.get("energy_mean", 0.0), d.get("uff_ok", False))


def _save_ref_disk() -> None:
    p = _ref_disk_path()
    out = {smi: {"ok": r.ok, "bond": r.bond, "angle": r.angle,
                 "energy_mean": r.energy_mean, "uff_ok": r.uff_ok}
           for smi, r in _REF_CACHE.items()}
    try:
        p.write_text(json.dumps(out))
    except Exception:
        pass


def reference_stats(heavy_mol) -> RefStats:
    """Per-bond / per-angle reference geometry + mean UFF energy from a seed-locked ETKDG ensemble of the
    SAME molecular graph. Deterministic (fixed seed) and cached by canonical SMILES. ok=False when the
    molecule will not embed (bond/angle/energy checks then disclose-skip; geometry checks still run)."""
    try:
        smi = Chem.MolToSmiles(heavy_mol)
    except Exception:
        return RefStats(False)
    _load_ref_disk()
    if smi in _REF_CACHE:
        return _REF_CACHE[smi]

    res = _build_reference(heavy_mol)
    _REF_CACHE[smi] = res
    _save_ref_disk()
    return res


def _build_reference(heavy_mol) -> RefStats:
    ranks = list(Chem.CanonicalRankAtoms(heavy_mol))
    n_heavy = heavy_mol.GetNumAtoms()
    try:
        ref = Chem.AddHs(Chem.Mol(heavy_mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = _REF_SEED
        cids = list(AllChem.EmbedMultipleConfs(ref, numConfs=_REF_CONFS, params=params))
        if not cids:
            return RefStats(False)
        AllChem.UFFOptimizeMoleculeConfs(ref, maxIters=400)
    except Exception:
        return RefStats(False)

    def bkey(i, j):
        a, b = sorted((ranks[i], ranks[j]))
        return f"{a}-{b}"

    def akey(c, i, j):
        a, b = sorted((ranks[i], ranks[j]))
        return f"{ranks[c]}|{a}-{b}"

    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in ref.GetBonds()
             if b.GetBeginAtomIdx() < n_heavy and b.GetEndAtomIdx() < n_heavy]
    angles = []
    for c in range(n_heavy):
        nbrs = [nb.GetIdx() for nb in ref.GetAtomWithIdx(c).GetNeighbors() if nb.GetIdx() < n_heavy]
        for x in range(len(nbrs)):
            for y in range(x + 1, len(nbrs)):
                angles.append((nbrs[x], c, nbrs[y]))

    bsum: dict[str, list[float]] = {}
    asum: dict[str, list[float]] = {}
    energies: list[float] = []
    uff_ok = bool(AllChem.UFFHasAllMoleculeParams(ref))
    for cid in cids:
        conf = ref.GetConformer(cid)
        for i, j in bonds:
            bsum.setdefault(bkey(i, j), []).append(rdMolTransforms.GetBondLength(conf, i, j))
        for i, c, j in angles:
            asum.setdefault(akey(c, i, j), []).append(rdMolTransforms.GetAngleDeg(conf, i, c, j))
        if uff_ok:
            ff = AllChem.UFFGetMoleculeForceField(ref, confId=cid)
            if ff is not None:
                energies.append(ff.CalcEnergy())

    bond = {k: sum(v) / len(v) for k, v in bsum.items()}
    angle = {k: sum(v) / len(v) for k, v in asum.items()}
    energy_mean = (sum(energies) / len(energies)) if energies else 0.0
    return RefStats(True, bond, angle, energy_mean, uff_ok and bool(energies))


# --------------------------------------------------------------------------- #
# featurize — one RDKit pass per pose; the contracts read only its scalars.
# --------------------------------------------------------------------------- #
def featurize(mol, tol: Tol = Tol()) -> PoseFeatures:
    """Compute every physical-validity scalar for one predicted pose. `mol` is an already-sanitized RDKit
    Mol with a 3D conformer (the C0 sanitize gate lives in the loader; here we trust a parsed mol)."""
    if mol is None:
        return PoseFeatures(parsed=False, has_3d=False)
    heavy = _heavy(mol)
    if heavy.GetNumConformers() == 0:
        return PoseFeatures(has_3d=False)
    conf = heavy.GetConformer()
    if not conf.Is3D():
        return PoseFeatures(has_3d=False)
    # degenerate (all-zero / collapsed) coordinates → not a real pose
    pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                    for i in range(heavy.GetNumAtoms())])
    if heavy.GetNumAtoms() > 1 and float(np.ptp(pos)) < 1e-3:
        return PoseFeatures(has_3d=False)

    ring_dev = _safe(lambda: _aromatic_ring_dev(heavy, conf), 0.0)
    db_twist = _safe(lambda: _double_bond_twist(heavy, conf), 0.0)
    clash, n_pairs = _safe(lambda: _min_clash_rel(heavy, conf), (1.5, 0))

    ref = reference_stats(heavy)
    lo, hi, ang = 1.0, 1.0, 1.0
    energy_ratio, uff_ok = 1.0, False
    n_bonds = 0
    if ref.ok:
        ranks = list(Chem.CanonicalRankAtoms(heavy))
        for b in heavy.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            a, c = sorted((ranks[i], ranks[j]))
            refmean = ref.bond.get(f"{a}-{c}")
            if not refmean:
                continue
            rel = rdMolTransforms.GetBondLength(conf, i, j) / refmean
            lo, hi, n_bonds = min(lo, rel), max(hi, rel), n_bonds + 1
        for c in range(heavy.GetNumAtoms()):
            nbrs = [nb.GetIdx() for nb in heavy.GetAtomWithIdx(c).GetNeighbors()]
            for x in range(len(nbrs)):
                for y in range(x + 1, len(nbrs)):
                    i, j = nbrs[x], nbrs[y]
                    a, bb = sorted((ranks[i], ranks[j]))
                    refang = ref.angle.get(f"{ranks[c]}|{a}-{bb}")
                    if not refang:
                        continue
                    ang = max(ang, rdMolTransforms.GetAngleDeg(conf, i, c, j) / refang)
        uff_ok = ref.uff_ok
        if uff_ok and ref.energy_mean > 0:
            energy_ratio = _safe(lambda: _pose_energy(mol) / ref.energy_mean, 1.0)

    return PoseFeatures(
        parsed=True, has_3d=True, ref_ok=ref.ok, uff_ok=uff_ok,
        worst_bond_rel_lo=lo, worst_bond_rel_hi=hi, worst_angle_rel=ang,
        max_ring_dev_A=ring_dev, max_double_bond_deg=db_twist,
        min_clash_rel=clash, energy_ratio=energy_ratio, n_bonds=n_bonds, n_clash_pairs=n_pairs)


def _pose_energy(mol) -> float:
    """UFF energy of the pose as given (Hs added at idealized positions; heavy-atom geometry frozen)."""
    mh = Chem.AddHs(mol, addCoords=True)
    ff = AllChem.UFFGetMoleculeForceField(mh)
    return ff.CalcEnergy()


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# Arm-A decoy generators — deterministic perturbations that each break one contract (the instrument check).
# --------------------------------------------------------------------------- #
def clean_conformer(smiles: str, seed: int = _REF_SEED):
    """A physically valid pose by construction: a seed-locked ETKDG + UFF-minimized conformer."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    mh = Chem.AddHs(m)
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    if AllChem.EmbedMolecule(mh, p) != 0:
        return None
    AllChem.UFFOptimizeMoleculeConfs(mh, maxIters=400)
    return Chem.RemoveHs(mh)


def _clone(mol):
    return Chem.Mol(mol)


def decoy_jitter(mol, sigma: float = 0.7, seed: int = 0):
    """Cartesian noise on every atom — breaks bonds/angles/clashes globally."""
    m = _clone(mol)
    conf = m.GetConformer()
    rng = np.random.default_rng(seed)
    for i in range(m.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        d = rng.normal(0, sigma, 3)
        conf.SetAtomPosition(i, Point3D(p.x + d[0], p.y + d[1], p.z + d[2]))
    return m


def decoy_clash(mol, target_rel: float = 0.5, seed: int = 0):
    """Move one atom on top of a topologically-distant atom — a single internal steric clash."""
    m = _clone(mol)
    conf = m.GetConformer()
    n = m.GetNumAtoms()
    dtopo = Chem.GetDistanceMatrix(m)
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j and dtopo[i][j] >= 3]
    if not pairs:
        return m
    i, j = pairs[rng.integers(len(pairs))]
    pj = conf.GetAtomPosition(j)
    conf.SetAtomPosition(i, Point3D(pj.x + 0.5, pj.y, pj.z))   # ~0.5 Å away → deep clash
    return m


def decoy_stretch(mol, factor: float = 1.6, seed: int = 0):
    """Stretch one terminal bond to `factor`× its length — a bond-length outlier."""
    m = _clone(mol)
    conf = m.GetConformer()
    rng = np.random.default_rng(seed)
    terminal = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()
                if m.GetAtomWithIdx(b.GetBeginAtomIdx()).GetDegree() == 1
                or m.GetAtomWithIdx(b.GetEndAtomIdx()).GetDegree() == 1]
    if not terminal:
        terminal = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    if not terminal:
        return m
    i, j = terminal[rng.integers(len(terminal))]
    if m.GetAtomWithIdx(i).GetDegree() == 1:
        i, j = j, i                                     # j = the terminal atom to move
    pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
    v = np.array([pj.x - pi.x, pj.y - pi.y, pj.z - pi.z])
    conf.SetAtomPosition(j, Point3D(pi.x + v[0] * factor, pi.y + v[1] * factor, pi.z + v[2] * factor))
    return m


def decoy_pucker_ring(mol, amp: float = 0.6, seed: int = 0):
    """Pucker a flat aromatic ring by pushing alternating atoms ±amp along the ring normal."""
    m = _clone(mol)
    conf = m.GetConformer()
    for ring in m.GetRingInfo().AtomRings():
        if not all(m.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            continue
        pts = np.array([[conf.GetAtomPosition(a).x, conf.GetAtomPosition(a).y,
                         conf.GetAtomPosition(a).z] for a in ring])
        _, _, vh = np.linalg.svd(pts - pts.mean(axis=0))
        normal = vh[2]
        for k, a in enumerate(ring):
            p = conf.GetAtomPosition(a)
            s = amp if k % 2 == 0 else -amp
            conf.SetAtomPosition(a, Point3D(p.x + normal[0] * s, p.y + normal[1] * s, p.z + normal[2] * s))
        return m                                        # one ring is enough to make the pose invalid
    return m


def decoy_twist_double_bond(mol, deg: float = 80.0, seed: int = 0):
    """Twist a planar double bond out of plane — a double-bond-planarity violation."""
    m = _clone(mol)
    conf = m.GetConformer()
    for bond in m.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.IsInRing():
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        na = next((nb.GetIdx() for nb in a.GetNeighbors() if nb.GetIdx() != b.GetIdx()), None)
        nb = next((x.GetIdx() for x in b.GetNeighbors() if x.GetIdx() != a.GetIdx()), None)
        if na is None or nb is None:
            continue
        try:
            rdMolTransforms.SetDihedralDeg(conf, na, a.GetIdx(), b.GetIdx(), nb, deg)
            return m
        except Exception:
            continue
    return decoy_jitter(mol, sigma=0.5, seed=seed)       # fallback for molecules with no acyclic double bond


DECOYS = [decoy_jitter, decoy_clash, decoy_stretch, decoy_pucker_ring, decoy_twist_double_bond]


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not _HAVE_RDKIT:
        print("SKIP — rdkit/numpy not importable (pose_validity needs them).")
        raise SystemExit(0)
    cs, tol = validity_contracts(), Tol()
    for name, smi in [("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
                      ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C")]:
        clean = clean_conformer(smi)
        f = featurize(clean, tol)
        v = cs.evaluate(f, tol)
        print(f"\n{name}: clean conformer → {'OK' if not is_invalid(f, cs, tol) else 'INVALID'} "
              f"(severity {f.severity(tol):.2f}, ref_ok={f.ref_ok})")
        for r in v.reasons:
            print(f"    · {r.contract}: {r.message}")
        decoy = decoy_clash(clean, seed=1)
        fd = featurize(decoy, tol)
        vd = cs.evaluate(fd, tol)
        print(f"{name}: clash decoy   → {'INVALID' if is_invalid(fd, cs, tol) else 'OK'} "
              f"(severity {fd.severity(tol):.2f})")
        for r in vd.reasons:
            print(f"    · {r.contract}: {r.message}")
    print("\npose_validity smoke OK.")
