"""protein_interface_validity — a legible *intermolecular* physical-validity DRC for protein↔PROTEIN
complexes (complex-QC). The cofold-QC sibling that extends the owned interface DRC from protein↔ligand to
protein↔protein.

`cofold_validity.py` owns the ligand↔protein interface (clash / volume-overlap / out-of-pocket) over a
co-folding pose. A predicted or DESIGNED protein complex — AlphaFold-Multimer, RFdiffusion + ProteinMPNN,
Proteina-Complexa — is the same geometry with a second *protein* chain in place of the ligand: still two
heavy-atom sets in one coordinate frame, so the interface checks stay ownable pure geometry. This module owns
that axis for protein complexes:

  * `INTERFACE_CLASH`        — an inter-chain heavy-atom pair overlapping past the MolProbity convention.
  * `INTERFACE_VOLUME_OVERLAP`— a chain's interface volume buried inside the partner (gross interpenetration).
  * `CHAINS_NOT_IN_CONTACT`  — the chains' closest approach is large ⇒ no interface (failed placement, the
                               protein↔protein analog of LIGAND_OUT_OF_POCKET).

The vdW table, the grid volume-overlap, the `contracts` engine and the `InterTol` distances are reused from
`cofold_validity` **verbatim** — the geometry is symmetric two-atom-set, no ligand assumption. The one new
physical constant is the clash criterion: an atom-pair **overlap > 0.40 Å** — MolProbity's all-atom clash
convention (here on heavy↔heavy atoms), which is *the deposited wwPDB validation reference's own convention*,
so the gate is like-for-like with it (the cofold-QC faithfulness discipline). Zero parameters fitted to
accuracy.

    python -m karyon.protein_interface_validity    # smoke: a docked dimer passes; decoys fail
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from . import contracts
from . import cofold_validity as cv          # reuse: vdw(), _volume_overlap_fraction(), _any_within(), numpy gate
from .structure_io import Atom

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:
    _HAVE_NUMPY = False


# --------------------------------------------------------------------------- #
# vdW radii for the interface clash — MolProbity's OWN set (Word et al. 1999, the "explicit hydrogen" radii
# the `probe`/`reduce` clash engine uses, and therefore the radii behind every deposited wwPDB clash record).
# Using them — not Bondi — is what makes the owned heavy↔heavy clash like-for-like with the deposited
# reference (notably O = 1.40, vs Bondi 1.52, which otherwise over-flags backbone-carbonyl contacts). The
# disclosed calibration, verified against the wwPDB clash counts. Falls back to cofold's Bondi table for any
# element MolProbity doesn't special-case (metals/halogens — rare at a protein interface).
# --------------------------------------------------------------------------- #
_VDW_MP = {"C": 1.75, "N": 1.55, "O": 1.40, "S": 1.80, "P": 1.80, "H": 1.17}


def vdw_mp(element: str) -> float:
    return _VDW_MP.get(element.capitalize(), cv.vdw(element))


# --------------------------------------------------------------------------- #
# Calibration — distances reused from cofold's InterTol; the clash criterion is MolProbity's overlap
# convention (a physical constant, the reference's own). No fit-to-accuracy.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IfaceTol:
    clash_overlap_A: float = 0.40    # heavy↔heavy vdW overlap > 0.40 Å ⇒ a clash. MolProbity's all-atom
    #                                  clash convention — the SAME 0.4 Å the deposited wwPDB validation
    #                                  reference uses — applied on MolProbity's OWN vdW radii (`vdw_mp`,
    #                                  Word 1999, e.g. O=1.40 not Bondi 1.52) so the detector is like-for-like
    #                                  with the reference (the cofold-QC "match the reference's convention"
    #                                  calibration — disclosed, verified against the wwPDB clash counts, not
    #                                  fit to accuracy).
    hbond_allowance_A: float = 0.0   # OFF by design. MolProbity excludes H-bonds from clashes using added
    #                                  hydrogens + donor/acceptor geometry; a heavy-atom-only gate has neither,
    #                                  and a blanket "both-polar ⇒ allow" rule was verified to drop REAL clashes
    #                                  (recall 100%→93%) for only a marginal false-positive gain. So we keep it
    #                                  off and DISCLOSE that the gate over-flags interface H-bonds/salt-bridges
    #                                  vs MolProbity — a heavy-atom limitation, not a defect (faithfulness is
    #                                  strongest in clash-count ρ and recall, weakest in binary presence).
    disulfide_max_A: float = 2.50    # an inter-chain S···S closer than this is a DISULFIDE BOND (covalent),
    #                                  not a clash — exclude it (else a cross-chain disulfide reads as deep
    #                                  interpenetration). The reference excludes covalently-bonded atoms.
    severe_overlap_A: float = 0.90   # CONDEMNS: a heavy↔heavy overlap this deep is unambiguous interpenetration
    #                                  (backbone-through-backbone), not a refinement artifact. A deposited,
    #                                  experimentally-determined complex's clashes are SHALLOW (just past
    #                                  0.4 Å); a misplaced/interpenetrating predicted partner is DEEP. Set
    #                                  above the deposited-native distribution (disclosed calibration) so a
    #                                  valid deposited structure passes and a failed prediction is flagged.
    vol_overlap_frac: float = 0.10   # CONDEMNS: > 10% of a chain's INTERFACE vdW volume buried in the partner
    #                                  ⇒ gross interpenetration (a normal interface buries ~0; touching spheres
    #                                  ≈ no overlap).
    not_bound_A: float = 5.0         # CONDEMNS: chains' CLOSEST approach > 5 Å ⇒ no contact, a failed placement
    #                                  (the protein↔protein analog of cofold's LIGAND_OUT_OF_POCKET).
    iface_cutoff_A: float = 6.0      # an atom within this of the partner chain is an "interface" atom (a
    #                                  reported diagnostic; the burial fraction itself is over the whole chain)


# --------------------------------------------------------------------------- #
# The precomputed per-complex interface features the contracts read (the cofold InterFeatures analog).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IfaceFeatures:
    framed: bool = True              # two non-empty protein chain groups were present in one frame
    min_ab_A: float = 3.5            # closest inter-chain heavy-atom distance (Å) — a default bound contact
    min_ab_rel: float = 1.0          # that closest pair as a fraction of its vdW sum
    max_overlap_A: float = 0.0       # the DEEPEST inter-chain heavy-atom overlap (Å) — clash severity driver
    n_clash_pairs: int = 0           # inter-chain heavy-atom pairs overlapping > clash_overlap_A
    vol_overlap_frac: float = 0.0    # fraction of the smaller chain's interface vdW volume inside the partner
    n_a_atoms: int = 0
    n_b_atoms: int = 0
    n_interface_atoms: int = 0       # atoms within iface_cutoff_A of the other chain (both sides)
    clash_pairs: tuple = ()          # up to a few identified clashes (chainA,resnumA,atomA,…,overlap) — legible

    def severity(self, tol: IfaceTol) -> float:
        """Continuous severity (Σ normalized exceedances) — the ranking statistic for the instrument AUROC.
        0.0 for a clean interface, grows with violations. Mirrors cofold_validity.InterFeatures.severity."""
        if not self.framed:
            return 0.0

        def relu(x: float) -> float:
            return x if x > 0 else 0.0
        s = 0.0
        s += relu(self.max_overlap_A / tol.severe_overlap_A - 1.0)          # deep interpenetration → bigger
        s += relu(self.vol_overlap_frac / tol.vol_overlap_frac - 1.0)
        s += relu(self.min_ab_A / tol.not_bound_A - 1.0)                     # farther = more unbound
        return s


# --------------------------------------------------------------------------- #
# The DRC — each check a legible contract reading a precomputed IfaceFeatures scalar (reuses contracts).
# --------------------------------------------------------------------------- #
def protein_interface_contracts() -> contracts.ContractSet:
    cs = contracts.ContractSet("complex-interface-validity")

    # DISCLOSE (weight 0 — informs, does not condemn): the inter-chain clash count. A deposited, valid complex
    # commonly carries a few SHALLOW interface clashes (refinement artifacts the wwPDB report also lists), so
    # the presence of a clash is reported — it is the faithfulness/detection signal — but does not by itself
    # fail the structure. Condemnation is reserved for clashes deep enough to be unphysical (below).
    cs.add(contracts.Contract("INTERFACE_CLASH",
        lambda f, t: (f"interface has {f.n_clash_pairs} inter-chain heavy-atom clash"
                      f"{'es' if f.n_clash_pairs != 1 else ''} (overlap >{t.clash_overlap_A:.2f} Å; deepest "
                      f"{f.max_overlap_A:.2f} Å; closest atoms {f.min_ab_A:.2f} Å apart)")
        if f.framed and f.n_clash_pairs > 0 else None, weight=0.0))

    # CONDEMN: a clash deep enough to be backbone-through-backbone interpenetration, not a refinement artifact.
    cs.add(contracts.Contract("SEVERE_INTERFACE_CLASH",
        lambda f, t: (f"chains interpenetrate: an inter-chain pair overlaps {f.max_overlap_A:.2f} Å "
                      f"(>{t.severe_overlap_A:.2f} Å — unphysically deep, not a refinement artifact; "
                      f"{f.n_clash_pairs} clashing pair{'s' if f.n_clash_pairs != 1 else ''})")
        if f.framed and f.max_overlap_A > t.severe_overlap_A else None, weight=1.5))

    cs.add(contracts.Contract("INTERFACE_VOLUME_OVERLAP",
        lambda f, t: (f"chains interpenetrate: {f.vol_overlap_frac:.0%} of one chain's interface vdW volume "
                      f"is buried in the other (>{t.vol_overlap_frac:.0%})")
        if f.framed and f.vol_overlap_frac > t.vol_overlap_frac else None, weight=1.5))

    cs.add(contracts.Contract("CHAINS_NOT_IN_CONTACT",
        lambda f, t: (f"chains are not in contact: closest approach {f.min_ab_A:.1f} Å (>{t.not_bound_A:.1f} Å)"
                      f" — no interface, the partner makes no contact (a failed placement)")
        if f.framed and f.min_ab_A > t.not_bound_A else None, weight=1.0))

    # Honest disclosure (weight 0.0 — informs, does not condemn): no two-chain pair in one frame.
    cs.add(contracts.Contract("NOT_FRAMED",
        lambda f, t: "interface check skipped: two protein chains were not present in one frame"
        if not f.framed else None, weight=0.0))
    return cs


def is_interface_invalid(features: IfaceFeatures, cs: contracts.ContractSet, tol: IfaceTol) -> bool:
    """A complex is interface-INVALID iff a condemning contract fired (score > 0; NOT_FRAMED is weight 0)."""
    return cs.evaluate(features, tol).score > 0.0


# --------------------------------------------------------------------------- #
# featurize — the geometry (numpy). Two groups of `structure_io.Atom` (chain set A vs chain set B), one frame.
# --------------------------------------------------------------------------- #
def interface_features(group_a: list[Atom], group_b: list[Atom], tol: IfaceTol = IfaceTol()) -> IfaceFeatures:
    if not group_a or not group_b:
        return IfaceFeatures(framed=False, n_a_atoms=len(group_a), n_b_atoms=len(group_b))

    A = np.array([[a.x, a.y, a.z] for a in group_a], dtype=float)
    B = np.array([[a.x, a.y, a.z] for a in group_b], dtype=float)
    ra = np.array([vdw_mp(a.element) for a in group_a], dtype=float)
    rb = np.array([vdw_mp(a.element) for a in group_b], dtype=float)
    pa, sa = _elem_masks(group_a)
    pb, sb = _elem_masks(group_b)

    min_A, min_rel, max_overlap, n_clash, a_min, b_min, pairs = _scan_interface(
        A, ra, B, rb, tol, pa, pb, sa, sb)

    # interface atoms (within cutoff of the partner).
    a_if = a_min <= tol.iface_cutoff_A
    b_if = b_min <= tol.iface_cutoff_A
    n_iface = int(a_if.sum() + b_if.sum())

    # gross interpenetration: the fraction of the SMALLER chain's atoms whose CENTRE lies inside the partner's
    # vdW shell. ≈ 0 for a normal contact (surfaces touch, centres stay out), large when chains pass through
    # each other. An atom-centre test (O(n·m), no grid) — cheap even when interpenetration makes most atoms
    # "interface" atoms (the grid blew up there); SEVERE_INTERFACE_CLASH is the primary interpenetration gate.
    vol_frac = _burial_fraction(A, B, rb) if A.shape[0] <= B.shape[0] else _burial_fraction(B, A, ra)

    clash_ids = _identify_clashes(group_a, group_b, pairs)

    return IfaceFeatures(
        framed=True, min_ab_A=min_A, min_ab_rel=min_rel, max_overlap_A=max_overlap, n_clash_pairs=n_clash,
        vol_overlap_frac=vol_frac, n_a_atoms=len(group_a), n_b_atoms=len(group_b),
        n_interface_atoms=n_iface, clash_pairs=clash_ids)


def all_interchain_features(atoms: list[Atom], tol: IfaceTol = IfaceTol()) -> IfaceFeatures:
    """Inter-chain interface features over ALL chain pairs of a complex (not a single A-vs-B split). The
    faithfulness primitive: it measures every inter-chain heavy-atom clash in the frame, exactly the set the
    wwPDB validation report's inter-chain `<clash>` records describe — so the owned verdict is like-for-like
    with the deposited reference. (The two-group `interface_features` is the gate primitive for a known
    binder↔target split; this is the whole-complex audit.)"""
    from .structure_io import group_by_chain
    groups = group_by_chain(atoms)                      # polymer chains only
    chains = [c for c in groups if groups[c]]
    if len(chains) < 2:
        return IfaceFeatures(framed=False, n_a_atoms=len(atoms))

    min_A = math.inf
    max_overlap = 0.0
    n_clash = 0
    pairs_all: list[tuple] = []
    n_atoms = 0
    for ci in range(len(chains)):
        for cj in range(ci + 1, len(chains)):
            ga, gb = groups[chains[ci]], groups[chains[cj]]
            A = np.array([[a.x, a.y, a.z] for a in ga], dtype=float)
            B = np.array([[a.x, a.y, a.z] for a in gb], dtype=float)
            ra = np.array([vdw_mp(a.element) for a in ga], dtype=float)
            rb = np.array([vdw_mp(a.element) for a in gb], dtype=float)
            pa, sa = _elem_masks(ga)
            pb, sb = _elem_masks(gb)
            m_A, m_rel, m_ov, nc, _, _, pr = _scan_interface(A, ra, B, rb, tol, pa, pb, sa, sb)
            min_A = min(min_A, m_A)
            max_overlap = max(max_overlap, m_ov)
            n_clash += nc
            for i, j, ov in pr:                         # carry identity for the legible clash list
                pairs_all.append((ga[i], gb[j], ov))
    n_atoms = sum(len(groups[c]) for c in chains)
    clash_ids = tuple((a.chain, a.resnum, a.resname, a.atom_name, b.chain, b.resnum, b.resname,
                       b.atom_name, round(ov, 2)) for a, b, ov in sorted(pairs_all, key=lambda p: -p[2])[:8])
    return IfaceFeatures(framed=True, min_ab_A=(0.0 if min_A is math.inf else min_A),
                         max_overlap_A=max_overlap, n_clash_pairs=n_clash,
                         n_a_atoms=n_atoms, n_interface_atoms=n_atoms, clash_pairs=clash_ids)


def primary_interface_pair(atoms: list[Atom], tol: IfaceTol = IfaceTol(), max_candidates: int = 4):
    """The two chains forming the largest interface (most atoms in contact), among the `max_candidates`
    largest chains. The right native interface for the instrument arm: in a multi-copy / multi-chain deposit
    (e.g. a 3-dimer crystal, or antibody H+L+antigen) the two LARGEST chains may not touch each other — that
    would make a deposited native falsely read 'not in contact'. Returns (group_a, group_b)."""
    from .structure_io import group_by_chain
    groups = group_by_chain(atoms)
    chs = sorted((c for c in groups if groups[c]), key=lambda c: -len(groups[c]))[:max_candidates]
    if len(chs) < 2:
        return ([], [])
    best = (groups[chs[0]], groups[chs[1]])
    best_n = -1
    for i in range(len(chs)):
        for j in range(i + 1, len(chs)):
            ga, gb = groups[chs[i]], groups[chs[j]]
            A = np.array([[a.x, a.y, a.z] for a in ga], dtype=float)
            B = np.array([[a.x, a.y, a.z] for a in gb], dtype=float)
            _, _, _, _, a_min, _, _ = _scan_interface(A, np.ones(len(ga)), B, np.ones(len(gb)), tol)
            n = int((a_min <= tol.iface_cutoff_A).sum())
            if n > best_n:
                best_n, best = n, (ga, gb)
    return best


def _burial_fraction(small, big, big_radii) -> float:
    """Fraction of `small`'s atom centres lying inside ANY `big` atom's vdW sphere — a cheap, grid-free
    interpenetration proxy (reuses cofold's chunked `_any_within`). ≈0 for a normal surface contact (centres
    sit outside the partner), large when chains overlap."""
    if small.shape[0] == 0 or big.shape[0] == 0:
        return 0.0
    inside = cv._any_within(small, big, big_radii)
    return float(inside.sum()) / float(small.shape[0])


def _elem_masks(atoms: list[Atom]):
    """(polar, sulfur) boolean masks for a chain's atoms — polar = N/O (H-bond/salt-bridge donors/acceptors),
    sulfur = S (for inter-chain disulfide detection). Feed the MolProbity clash exclusions in `_scan_interface`."""
    pol = np.array([a.element in ("N", "O") for a in atoms], dtype=bool)
    sul = np.array([a.element == "S" for a in atoms], dtype=bool)
    return pol, sul


def _scan_interface(A, ra, B, rb, tol: IfaceTol, pol_a=None, pol_b=None, sul_a=None, sul_b=None,
                    chunk: int = 1024):
    """Chunked over A: the inter-chain heavy-atom geometry, applying MolProbity's clash exclusions (the
    reference's convention): a polar donor–acceptor pair (`pol_*`) gets the H-bond/salt-bridge overlap
    allowance, and a covalent inter-chain disulfide (`sul_*` within `disulfide_max_A`) is not a clash. Returns
    (min_dist_A, min_rel, max_overlap_A, n_clash_pairs, a_min_dists, b_min_dists, clash_index_pairs) where
    max_overlap and the count are over CLASH-ELIGIBLE pairs only. Bounds memory (one A-chunk at a time)."""
    na, nb = A.shape[0], B.shape[0]
    if pol_a is None:                                   # element masks are optional (geometry-only callers)
        pol_a = np.zeros(na, bool); pol_b = np.zeros(nb, bool)
        sul_a = np.zeros(na, bool); sul_b = np.zeros(nb, bool)
    min_A = math.inf
    min_rel = math.inf
    max_overlap = 0.0
    n_clash = 0
    a_min = np.full(na, math.inf)
    b_min = np.full(nb, math.inf)
    pairs: list[tuple[int, int, float]] = []           # (i, j, overlap) — capped for legibility
    for s in range(0, na, chunk):
        Ac = A[s:s + chunk]
        rac = ra[s:s + chunk]
        diff = Ac[:, None, :] - B[None, :, :]
        dist = np.sqrt((diff * diff).sum(-1))          # (chunk, nb)
        vsum = rac[:, None] + rb[None, :]
        overlap = vsum - dist                          # >0 ⇒ spheres interpenetrate
        a_min[s:s + chunk] = dist.min(axis=1)
        b_min = np.minimum(b_min, dist.min(axis=0))
        cmin = float(dist.min())
        if cmin < min_A:
            min_A = cmin
            min_rel = float((dist / vsum).min())
        # exclude the reference's non-clashes: H-bond/salt-bridge polar pairs (shallow), covalent disulfides.
        hbond = (pol_a[s:s + chunk][:, None] & pol_b[None, :]) & (overlap < tol.clash_overlap_A + tol.hbond_allowance_A)
        disulf = (sul_a[s:s + chunk][:, None] & sul_b[None, :]) & (dist < tol.disulfide_max_A)
        eligible = ~(hbond | disulf)
        clash_mask = (overlap > tol.clash_overlap_A) & eligible
        if eligible.any():
            mo = float(np.where(eligible, overlap, -np.inf).max())
            if mo > max_overlap:
                max_overlap = mo
        n_clash += int(clash_mask.sum())
        if len(pairs) < 64 and clash_mask.any():       # keep a few identified clashes (legible reasons)
            ii, jj = np.nonzero(clash_mask)
            for i, j in zip(ii[:64], jj[:64]):
                pairs.append((s + int(i), int(j), float(overlap[i, j])))
    return (float(min_A), float(min_rel), max(max_overlap, 0.0), n_clash, a_min, b_min, pairs[:64])


def _identify_clashes(group_a, group_b, pairs):
    """Map (i, j, overlap) index clashes → (chainA, resnumA/resnameA, atomA, chainB, resnumB/resnameB, atomB,
    overlap) tuples, deepest first — the legible interface-clash list."""
    out = []
    for i, j, ov in sorted(pairs, key=lambda p: -p[2])[:8]:
        a, b = group_a[i], group_b[j]
        out.append((a.chain, a.resnum, a.resname, a.atom_name,
                    b.chain, b.resnum, b.resname, b.atom_name, round(ov, 2)))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Instrument decoys — deterministic rigid-body perturbations of chain B (chain preserved), for the AUROC arm.
# --------------------------------------------------------------------------- #
def _translate(atoms: list[Atom], vec) -> list[Atom]:
    return [replace(a, x=a.x + float(vec[0]), y=a.y + float(vec[1]), z=a.z + float(vec[2])) for a in atoms]


def _rotate(atoms: list[Atom], R, center) -> list[Atom]:
    out = []
    for a in atoms:
        p = np.array([a.x, a.y, a.z]) - center
        q = R @ p + center
        out.append(replace(a, x=float(q[0]), y=float(q[1]), z=float(q[2])))
    return out


def decoy_interpenetrate(group_a: list[Atom], group_b: list[Atom], depth: float = 2.0) -> list[Atom]:
    """Drive chain B into chain A along (B centroid → nearest A atom) so the closest contact buries ~`depth`
    Å past vdW contact — a guaranteed interface clash (the protein↔protein analog of decoy_bury_into_protein)."""
    A = np.array([[a.x, a.y, a.z] for a in group_a], dtype=float)
    B = np.array([[a.x, a.y, a.z] for a in group_b], dtype=float)
    bc = B.mean(axis=0)
    nearest = A[np.argmin(((A - bc) ** 2).sum(axis=1))]
    d = nearest - bc
    n = float(np.linalg.norm(d))
    u = d / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    return _translate(group_b, u * (n * 0.5 + depth))


def decoy_separate(group_a: list[Atom], group_b: list[Atom], margin: float = 12.0) -> list[Atom]:
    """Translate chain B clear of chain A along (A centroid → B centroid), placing B's centroid beyond BOTH
    chains' bounding radii + `margin` — guaranteed not-in-contact even for a large partner (unlike a tiny
    ligand, a protein chain has tens of Å of extent, so the eject distance must include B's own radius)."""
    A = np.array([[a.x, a.y, a.z] for a in group_a], dtype=float)
    B = np.array([[a.x, a.y, a.z] for a in group_b], dtype=float)
    ac, bc = A.mean(axis=0), B.mean(axis=0)
    d = bc - ac
    n = float(np.linalg.norm(d))
    u = d / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    radius_a = float(np.sqrt(((A - ac) ** 2).sum(axis=1)).max())
    radius_b = float(np.sqrt(((B - bc) ** 2).sum(axis=1)).max())
    return _translate(group_b, (ac + u * (radius_a + radius_b + margin)) - bc)


def decoy_rotate_chain(group_b: list[Atom], axis=(0.0, 0.0, 1.0), angle_deg: float = 60.0) -> list[Atom]:
    """Rigid-body rotate chain B about its own centroid (Rodrigues) — a realistic 'wrong pose' that swings
    the partner across the interface. Less guaranteed than interpenetrate/separate; included for the realistic
    decoy bank, not the make-or-break AUROC."""
    B = np.array([[a.x, a.y, a.z] for a in group_b], dtype=float)
    c = B.mean(axis=0)
    k = np.array(axis, dtype=float)
    k = k / (np.linalg.norm(k) or 1.0)
    th = math.radians(angle_deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    return _rotate(group_b, R, c)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not _HAVE_NUMPY:
        print("SKIP — protein_interface_validity smoke needs numpy.")
        raise SystemExit(0)
    # A synthetic two-chain "dimer": chain A a Fibonacci patch, chain B a small cluster seated at vdW contact.
    def shell(center, r, n, chain):
        ga = math.pi * (3.0 - math.sqrt(5.0))
        out = []
        for k in range(n):
            y = 1.0 - 2.0 * (k + 0.5) / n
            rad = math.sqrt(max(0.0, 1.0 - y * y))
            out.append(Atom("C", center[0] + r * math.cos(ga * k) * rad, center[1] + r * y,
                            center[2] + r * math.sin(ga * k) * rad, chain=chain, atom_name="CA"))
        return out

    chain_a = shell((0.0, 0.0, 0.0), 8.0, 120, "A")
    # chain B: a little patch placed just outside A, in contact (closest ~vdW sum, no clash)
    chain_b = [Atom("C", 8.0 + 3.4 + 0.6 * i, 0.0, 0.0, chain="B", atom_name="CB") for i in range(8)]
    cs, tol = protein_interface_contracts(), IfaceTol()

    for label, b in [("docked (clean)", chain_b),
                     ("interpenetrated", decoy_interpenetrate(chain_a, chain_b)),
                     ("separated", decoy_separate(chain_a, chain_b))]:
        f = interface_features(chain_a, b, tol)
        verdict = "INVALID" if is_interface_invalid(f, cs, tol) else "OK"
        print(f"\n{label:16} → {verdict}  (min {f.min_ab_A:.2f} Å, {f.n_clash_pairs} clashes, deepest "
              f"{f.max_overlap_A:.2f} Å, burial {f.vol_overlap_frac:.0%}, sev {f.severity(tol):.2f})")
        for r in cs.evaluate(f, tol).reasons:
            print(f"    · {r.contract}: {r.message}")
    print("\nprotein_interface_validity smoke OK.")
