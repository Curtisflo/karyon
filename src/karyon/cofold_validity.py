"""cofold_validity — a legible *intermolecular* physical-validity DRC for co-folding poses (cofold-QC).

The sibling to `pose_validity.py`, which owns the **intra**molecular axis (bond/angle/ring/clash/strain of
the ligand) but *consumes* the **inter**molecular axis — ligand↔protein clash, volume overlap,
out-of-pocket placement — from a reference `bust_results.csv`, because a DiffDock/Vina ligand pose is not in
its receptor's coordinate frame (the honest intra/inter split). The *large* effect (71% of DiffDock
"successes" physically invalid) lives on that consumed axis.

**Co-folding models (Boltz / AF3 / Chai) predict protein + ligand in ONE frame**, so the intermolecular
checks become ownable pure geometry — the most CAD/EDA-DRC-shaped QC there is (a geometric inter-part
collision check, exactly like a CAD tool checks a printed assembly). This module owns that axis end-to-end:

  * `LIGAND_PROTEIN_CLASH`         — min ligand↔protein heavy-atom distance ÷ vdW sum < threshold.
  * `LIGAND_PROTEIN_VOLUME_OVERLAP`— fraction of (scaled) ligand vdW volume buried inside the protein.
  * `LIGAND_OUT_OF_POCKET`         — the ligand floats away (max over ligand atoms of distance-to-protein).

Each is a thin `contracts.Contract` reading a precomputed `InterFeatures` scalar — the `pose_validity`
pattern (featurize once, contracts read scalars), so the rules are unit-testable on planted features with
**no rdkit and no numpy**, and the `contracts` engine stays substrate-agnostic. Every threshold is a
physical constant or reference-calibrated (disclosed, the screen-QC→MAGeCK precedent) — zero fitted-to-
accuracy parameters. `full_verdict()` fuses this with `pose_validity.validity_contracts()` for the whole
intra+inter physical-validity call a co-folding QC skill returns.

`structure_io` (stdlib) parses the protein; rdkit parses the ligand; the geometry is plain numpy.

    python -m karyon.cofold_validity     # smoke: a clean complex passes; a buried decoy fails
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import contracts
from .structure_io import Atom

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:
    _HAVE_NUMPY = False

try:
    from . import pose_validity as pv
    from rdkit import Chem
    _HAVE_RDKIT = pv._HAVE_RDKIT
except Exception:
    _HAVE_RDKIT = False


# --------------------------------------------------------------------------- #
# van der Waals radii (Å) — Bondi 1964 + common bio elements. stdlib, so the geometry needs no rdkit.
# These are the radii PoseBusters' clash/volume checks are defined against; a fixed table keeps the DRC
# deterministic and legible.
# --------------------------------------------------------------------------- #
_VDW = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80, "S": 1.80,
    "Cl": 1.75, "Br": 1.85, "I": 1.98, "B": 1.92, "Si": 2.10, "Se": 1.90,
    "Na": 2.27, "Mg": 1.73, "K": 2.75, "Ca": 2.31, "Zn": 1.39, "Fe": 2.05,
    "Mn": 2.05, "Cu": 1.40, "Ni": 1.63, "Co": 2.0, "As": 1.85,
}
_VDW_DEFAULT = 1.70


def vdw(element: str) -> float:
    return _VDW.get(element.capitalize(), _VDW_DEFAULT)


# --------------------------------------------------------------------------- #
# Calibration — every threshold a physical constant or reference-derived (disclosed). No fit-to-accuracy.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InterTol:
    clash_rel: float = 0.75      # ligand↔protein dist < 0.75·(vdwᵢ+vdwⱼ) ⇒ steric clash (PoseBusters'
    #                              protein clash convention: ~0.75× the vdW sum; recalibrated vs the Boltz
    #                              bust_results during the faithfulness pass, the screen-QC→MAGeCK precedent)
    vol_scale: float = 1.0       # FULL (unscaled) vdW radii for volume overlap — PoseBusters' convention.
    #                              A normal binding contact sits at ~the vdW-sum distance (spheres touch ⇒
    #                              ~0 overlap), so unscaled radii measure genuine *interpenetration*; an
    #                              earlier 0.80 scaling under-measured it and missed the poses PoseBusters
    #                              flags (verified vs the Boltz bust_results — the disclosed calibration).
    vol_overlap_frac: float = 0.075  # > 7.5% of the ligand vdW volume buried ⇒ volume clash (PoseBusters default)
    not_bound_A: float = 5.0     # ligand's CLOSEST approach to the protein > 5.0 Å ⇒ it makes no contact,
    #                              i.e. floated outside any pocket (a bound ligand contacts at ~2.5-3.5 Å).
    #                              NB: this is closest-approach, NOT per-atom — a solvent-exposed ligand tail
    #                              is normal and must not flag (the calibration fix vs the crystal set).
    grid_step_A: float = 0.5         # volume-overlap grid resolution (Å)


# --------------------------------------------------------------------------- #
# The precomputed per-pose intermolecular features the contracts read.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InterFeatures:
    framed: bool = True           # a protein AND a ligand were both present in one frame (else N/A)
    min_lig_prot_rel: float = 1.5  # min ligand↔protein dist ÷ vdW sum (clash if < clash_rel)
    min_lig_prot_A: float = 3.0    # the same minimum as an absolute distance (Å) — a default bound contact,
    #                                so a bare InterFeatures() is clean (not_bound only fires when it's large)
    vol_overlap_frac: float = 0.0  # fraction of scaled ligand vdW volume inside the protein
    max_lig_min_dist_A: float = 0.0  # max over ligand atoms of distance-to-nearest-protein-atom (Å)
    n_protein_atoms: int = 0
    n_ligand_atoms: int = 0
    n_clash_pairs: int = 0

    def severity(self, tol: InterTol) -> float:
        """Continuous severity (Σ normalized exceedances) — the ranking statistic for the instrument AUROC.
        0.0 for a clean interface, grows with violations. Distinct from Verdict.score (Σ fired weights)."""
        if not self.framed:
            return 0.0

        def relu(x: float) -> float:
            return x if x > 0 else 0.0
        s = 0.0
        s += relu(tol.clash_rel / max(self.min_lig_prot_rel, 1e-6) - 1.0)         # deeper clash → bigger
        s += relu(self.vol_overlap_frac / tol.vol_overlap_frac - 1.0)
        s += relu(self.min_lig_prot_A / tol.not_bound_A - 1.0)                    # farther = more unbound
        return s


# --------------------------------------------------------------------------- #
# The DRC — each check a legible contract reading a precomputed InterFeatures scalar.
# --------------------------------------------------------------------------- #
def intermolecular_contracts() -> contracts.ContractSet:
    cs = contracts.ContractSet("cofold-intermolecular-validity")

    cs.add(contracts.Contract("LIGAND_PROTEIN_CLASH",
        lambda f, t: (f"ligand clashes into the protein: closest atoms {f.min_lig_prot_A:.2f} Å apart "
                      f"({f.min_lig_prot_rel:.0%} of their vdW sum, <{t.clash_rel:.0%}; {f.n_clash_pairs} "
                      f"clashing pair{'s' if f.n_clash_pairs != 1 else ''})")
        if f.framed and f.min_lig_prot_rel < t.clash_rel else None, weight=1.5))

    cs.add(contracts.Contract("LIGAND_PROTEIN_VOLUME_OVERLAP",
        lambda f, t: (f"ligand volume buried in the protein: {f.vol_overlap_frac:.0%} of its vdW volume "
                      f"overlaps (>{t.vol_overlap_frac:.0%})")
        if f.framed and f.vol_overlap_frac > t.vol_overlap_frac else None, weight=1.5))

    cs.add(contracts.Contract("LIGAND_OUT_OF_POCKET",
        lambda f, t: (f"ligand not bound: its closest approach to the protein is {f.min_lig_prot_A:.1f} Å "
                      f"(>{t.not_bound_A:.1f} Å) — it makes no contact, sitting outside any pocket")
        if f.framed and f.min_lig_prot_A > t.not_bound_A else None, weight=1.0))

    # Honest disclosure (weight 0.0 — informs, does not condemn): no protein+ligand pair in one frame.
    cs.add(contracts.Contract("NOT_FRAMED",
        lambda f, t: "intermolecular check skipped: a protein+ligand pair was not present in one frame"
        if not f.framed else None, weight=0.0))
    return cs


def is_inter_invalid(features: InterFeatures, cs: contracts.ContractSet, tol: InterTol) -> bool:
    """A pose is intermolecularly INVALID iff a condemning contract fired (score > 0; NOT_FRAMED is 0)."""
    return cs.evaluate(features, tol).score > 0.0


# --------------------------------------------------------------------------- #
# featurize — the geometry (numpy). Protein atoms + ligand atoms are both `structure_io.Atom` in one frame.
# --------------------------------------------------------------------------- #
def interface_features(protein: list[Atom], ligand: list[Atom], tol: InterTol = InterTol()) -> InterFeatures:
    if not protein or not ligand:
        return InterFeatures(framed=False, n_protein_atoms=len(protein), n_ligand_atoms=len(ligand))

    lig = np.array([[a.x, a.y, a.z] for a in ligand], dtype=float)
    pro = np.array([[a.x, a.y, a.z] for a in protein], dtype=float)
    rl = np.array([vdw(a.element) for a in ligand], dtype=float)
    rp = np.array([vdw(a.element) for a in protein], dtype=float)

    # full ligand×protein distance matrix (ligand is tiny; protein ~1e4 — ~1e6 floats, fine)
    diff = lig[:, None, :] - pro[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))                       # (L, P)
    rel = dist / (rl[:, None] + rp[None, :])                    # (L, P) distance as a fraction of vdW sum

    min_rel = float(rel.min())
    min_A = float(dist.min())
    n_clash = int((rel < tol.clash_rel).sum())
    max_lig_min = float(dist.min(axis=1).max())                # farthest ligand atom from the protein

    vol_frac = _volume_overlap_fraction(lig, rl, pro, rp, tol)

    return InterFeatures(
        framed=True, min_lig_prot_rel=min_rel, min_lig_prot_A=min_A,
        vol_overlap_frac=vol_frac, max_lig_min_dist_A=max_lig_min,
        n_protein_atoms=len(protein), n_ligand_atoms=len(ligand), n_clash_pairs=n_clash)


def _volume_overlap_fraction(lig, rl, pro, rp, tol: InterTol) -> float:
    """Fraction of the ligand's (scaled) vdW volume that lies inside the protein's (scaled) vdW volume,
    by a uniform grid over the ligand bounding box. Legible and deterministic; resolution = tol.grid_step_A.
    Radii are scaled by tol.vol_scale so an ordinary binding contact (touching spheres) is not 'overlap'."""
    rls = rl * tol.vol_scale
    rps = rp * tol.vol_scale
    pad = float(rls.max())
    lo = lig.min(axis=0) - pad
    hi = lig.max(axis=0) + pad
    step = tol.grid_step_A
    axes = [np.arange(lo[k], hi[k] + step, step) for k in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)   # (G, 3)
    if grid.shape[0] == 0:
        return 0.0

    # points inside any ligand sphere (the ligand vdW volume, discretized)
    inside_lig = _any_within(grid, lig, rls)
    n_lig = int(inside_lig.sum())
    if n_lig == 0:
        return 0.0

    # of those, how many also lie inside a protein sphere — prefilter protein atoms near the ligand box
    near = ((pro >= lo - rps.max()) & (pro <= hi + rps.max())).all(axis=1)
    if not near.any():
        return 0.0
    lig_pts = grid[inside_lig]
    inside_pro = _any_within(lig_pts, pro[near], rps[near])
    return float(inside_pro.sum()) / float(n_lig)


def _any_within(points, centers, radii, chunk: int = 4096) -> "np.ndarray":
    """Boolean mask: is each point within `radius_j` of ANY center j? Chunked over points to bound memory."""
    out = np.zeros(points.shape[0], dtype=bool)
    r2 = radii * radii
    for s in range(0, points.shape[0], chunk):
        pts = points[s:s + chunk]
        d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1)     # (chunk, C)
        out[s:s + chunk] = (d2 <= r2[None, :]).any(axis=1)
    return out


# --------------------------------------------------------------------------- #
# Ligand bridge — an rdkit Mol (a co-folding/SDF ligand) → heavy-atom `structure_io.Atom`s in its frame.
# --------------------------------------------------------------------------- #
def ligand_atoms_from_mol(mol) -> list[Atom]:
    """Heavy-atom (element, x, y, z) of an rdkit Mol's conformer — the ligand side of the interface."""
    if mol is None or mol.GetNumConformers() == 0:
        return []
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    out = []
    for i in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(i)
        if a.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(i)
        out.append(Atom(a.GetSymbol(), p.x, p.y, p.z))
    return out


# --------------------------------------------------------------------------- #
# full_verdict — the whole intra + inter physical-validity call a co-folding QC skill returns.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FullVerdict:
    ok: bool
    intra: contracts.Verdict
    inter: contracts.Verdict

    @property
    def reasons(self):
        return self.intra.reasons + self.inter.reasons

    @property
    def messages(self):
        return self.intra.messages + self.inter.messages


def full_verdict(protein: list[Atom], ligand_mol, tol_intra=None, tol_inter: InterTol = InterTol()
                 ) -> FullVerdict:
    """Combine the intramolecular DRC (pose_validity) over the ligand with the intermolecular DRC over the
    (protein, ligand) interface into one physical-validity verdict with per-axis legible reasons."""
    tol_intra = tol_intra or pv.Tol()
    intra_cs = pv.validity_contracts()
    inter_cs = intermolecular_contracts()

    fi = pv.featurize(ligand_mol, tol_intra)
    intra_v = intra_cs.evaluate(fi, tol_intra)

    lig_atoms = ligand_atoms_from_mol(ligand_mol)
    fx = interface_features(protein, lig_atoms, tol_inter)
    inter_v = inter_cs.evaluate(fx, tol_inter)

    ok = (intra_v.score == 0.0) and (inter_v.score == 0.0)
    return FullVerdict(ok=ok, intra=intra_v, inter=inter_v)


# --------------------------------------------------------------------------- #
# Instrument decoys — deterministic interface perturbations (translate the ligand) for the AUROC arm.
# --------------------------------------------------------------------------- #
def translate(ligand: list[Atom], vec) -> list[Atom]:
    return [Atom(a.element, a.x + vec[0], a.y + vec[1], a.z + vec[2], a.resname, a.is_hetero) for a in ligand]


def decoy_bury_into_protein(protein: list[Atom], ligand: list[Atom], depth: float = 2.0) -> list[Atom]:
    """Shove the ligand toward the protein interior along (ligand centroid → nearest protein atom), so its
    closest contact buries ~`depth` Å past the vdW contact — a guaranteed protein clash."""
    lig = np.array([[a.x, a.y, a.z] for a in ligand], dtype=float)
    pro = np.array([[a.x, a.y, a.z] for a in protein], dtype=float)
    lc = lig.mean(axis=0)
    nearest = pro[np.argmin(((pro - lc) ** 2).sum(axis=1))]
    d = nearest - lc
    n = float(np.linalg.norm(d))
    u = d / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    return translate(ligand, u * (n * 0.5 + depth))            # halfway in, plus a bury


def decoy_eject_from_pocket(protein: list[Atom], ligand: list[Atom], margin: float = 12.0) -> list[Atom]:
    """Translate the ligand clear of the protein along (protein centroid → ligand centroid), placing its
    centroid `margin` Å beyond the protein's bounding-sphere radius — guaranteed out-of-pocket, no clash
    even for a large protein (a fixed 14 Å hop often lands back in another surface region)."""
    lig = np.array([[a.x, a.y, a.z] for a in ligand], dtype=float)
    pro = np.array([[a.x, a.y, a.z] for a in protein], dtype=float)
    pc, lc = pro.mean(axis=0), lig.mean(axis=0)
    d = lc - pc
    n = float(np.linalg.norm(d))
    u = d / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    radius = float(np.sqrt(((pro - pc) ** 2).sum(axis=1)).max())
    target_centroid = pc + u * (radius + margin)
    return translate(ligand, target_centroid - lc)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not (_HAVE_NUMPY and _HAVE_RDKIT):
        print("SKIP — cofold_validity smoke needs numpy + rdkit.")
        raise SystemExit(0)
    # A synthetic one-frame complex: a benzene "ligand" seated inside a carbon "pocket" — a Fibonacci
    # shell of atoms surrounding it at vdW-contact distance, so a clean seat passes and the decoys break it.
    from rdkit.Chem import AllChem
    benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    p = AllChem.ETKDGv3(); p.randomSeed = 0xC0FFEE
    AllChem.EmbedMolecule(benzene, p)
    lig = ligand_atoms_from_mol(Chem.RemoveHs(benzene))
    _c = np.array([[a.x, a.y, a.z] for a in lig]).mean(axis=0)
    _r = 4.8                                             # vdW-contact distance for a clean seat (no overlap)
    _n = 60
    _ga = math.pi * (3.0 - math.sqrt(5.0))
    protein = []
    for k in range(_n):
        _y = 1.0 - 2.0 * (k + 0.5) / _n
        _rad = math.sqrt(max(0.0, 1.0 - _y * _y))
        protein.append(Atom("C", _c[0] + _r * math.cos(_ga * k) * _rad,
                            _c[1] + _r * _y, _c[2] + _r * math.sin(_ga * k) * _rad))
    cs, tol = intermolecular_contracts(), InterTol()

    for label, atoms in [("seated (clean)", lig),
                         ("buried decoy", decoy_bury_into_protein(protein, lig)),
                         ("ejected decoy", decoy_eject_from_pocket(protein, lig))]:
        f = interface_features(protein, atoms, tol)
        v = cs.evaluate(f, tol)
        verdict = "INVALID" if is_inter_invalid(f, cs, tol) else "OK"
        print(f"\n{label:16} → {verdict}  (min {f.min_lig_prot_A:.2f} Å / {f.min_lig_prot_rel:.0%} vdW, "
              f"overlap {f.vol_overlap_frac:.0%}, far {f.max_lig_min_dist_A:.1f} Å, sev {f.severity(tol):.2f})")
        for r in v.reasons:
            print(f"    · {r.contract}: {r.message}")
    print("\ncofold_validity smoke OK.")
