"""antibody_developability — a legible developability/liability DRC for antibody Fv sequences (antibody-QC).

A BioNeMo-complement QC gate on the **generative-output** axis, extended to biologics — the "unroutable net"
report for an antibody- or binder-design tool. A generative biologics model (an AlphaFold-Multimer / RFdiffusion
+ ProteinMPNN binder, an antibody language model) emits an Fv that scores well on the model's own confidence
yet can carry **developability liabilities** that make it undruggable: a free (unpaired) cysteine that scrambles
disulfides, an N-glycosylation sequon in a CDR that glycosylates the binding site, a chemically labile
deamidation / isomerization hotspot in a CDR, an extreme isoelectric point. None of these are *function*
problems the model optimizes for; they are the manufacturability/stability axis it is blind to. BioNeMo's
biologics stack proposes; this module is the deterministic gate.

It is the antibody analogue of `gen_dna_validity` (DFM for DNA): each check is a thin `contracts.Contract`
reading a precomputed `AbFeatures` scalar/list, over pure-stdlib sequence primitives (CDR anchoring, motif
scanning, Henderson–Hasselbalch charge). Faithful to the **Therapeutic Antibody Profiler** (Raybould et al.,
*PNAS* 2019) and the clinical-stage developability survey (Jain et al., *PNAS* 2017): the rare/severe liabilities
**condemn** (unpaired Cys, CDR N-glyc sequon, extreme CDR-H3 length, extreme pI), while the common chemistry
flags that even approved antibodies carry are **disclosed** (deamidation/isomerization/oxidation hotspots,
fragmentation, framework glycosylation, charge asymmetry, hydrophobicity) — reported without failing the gate,
so a real therapeutic passes. The repair loop demands the disclosed liabilities explicitly (`clear_disclosures`),
exactly as the DNA loop demands a restriction site.

Scope (honest): the gate owns the **sequence-determined** chemistry plus TAP-style *sequence* proxies. The true
spatial TAP metrics — patches of surface hydrophobicity / charge (PSH / PPC / PNC) and the structural Fv charge
symmetry parameter (SFvCSP) — need a 3D Fv model and are out of scope for v1 (the charge-asymmetry and CDR-GRAVY
disclosures are coarse sequence stand-ins). CDR boundaries are located from conserved framework anchors (no
ANARCI / HMMER); when the anchors don't resolve, the CDR-scoped checks stand down and a disclosure says so.

No numpy, no rdkit, no network: pure string geometry + arithmetic.

    python -m karyon.antibody_developability     # smoke: trastuzumab passes; a planted-liability Fv fails
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from . import contracts


# --------------------------------------------------------------------------- #
# Calibration — every threshold a developability-literature constant (TAP / Jain), disclosed, none fitted.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AntibodyTol:
    h3_min: int = 3              # CDR-H3 length window. TAP flags total-CDR-length outliers; CDR-H3 dominates
    h3_max: int = 32             #   the variance, so the reliably-anchored H3 (Cys→WGxG) carries the length gate.
    pi_min: float = 5.0          # Fv isoelectric point band — outside ~5–9.5 invites solubility/viscosity
    pi_max: float = 9.5          #   (low pI) or fast-clearance/viscosity (high pI) risk (TAP charge flags).
    charge_asym: float = 6.0     # |net charge(VH) − net charge(VL)| at pH 7.4 above this ⇒ DISCLOSE (SFvCSP
    #                              sequence proxy; the true metric is structural).
    gravy_max: float = 0.30      # mean Kyte–Doolittle hydropathy over CDR residues above this ⇒ DISCLOSE
    #                              (a hydrophobic-patch / PSH sequence proxy; the true metric is structural).
    charge_ph: float = 7.4       # physiological pH for the net-charge / asymmetry computation.
    # CDR-window offsets from the framework anchors (Kabat-ish; generous, validated on reference therapeutics).
    h1_back: int = 9             # heavy CDR-H1 ≈ the `h1_back` residues before the FR2 Trp (FR1 is long).
    h2_len: int = 18             # heavy CDR-H2 window length from FR2+`fr2_gap`.
    l2_len: int = 9              # light CDR-L2 window length from FR2+`fr2_gap`.
    fr2_gap: int = 14            # residues of FR2 between the conserved Trp and the start of CDR2.


# Liability motifs (regex over one chain). NG/NS are the fast-deamidating Asn motifs; DG/DS the labile Asp
# isomerization motifs; N[^P][ST] the N-glycosylation sequon; DP the acid-labile fragmentation bond.
_SEQUON = re.compile(r"N[^P][ST]")
_DEAMIDATION = re.compile(r"N[GS]")
_ISOMERIZATION = re.compile(r"D[GS]")
_FRAGMENTATION = re.compile(r"DP")
_J_HEAVY = re.compile(r"WG.G")        # heavy FR4 J-motif (WGxG) — closes CDR-H3
_J_LIGHT = re.compile(r"[FW]G.G")     # light FR4 J-motif (FGxG, sometimes WGxG) — closes CDR-L3

_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte–Doolittle hydropathy (PSH sequence proxy over CDR residues).
_KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
       "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
       "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}

# Side-chain pKa (EMBOSS set) for Henderson–Hasselbalch net charge / isoelectric point.
_PKA_POS = {"K": 10.8, "R": 12.5, "H": 6.5}     # protonated-when-acidic groups (+) — plus the N-terminus
_PKA_NEG = {"D": 3.9, "E": 4.1, "C": 8.5, "Y": 10.1}  # deprotonated-when-basic groups (−) — plus the C-terminus
_PKA_NTERM = 8.6
_PKA_CTERM = 3.6


@dataclass(frozen=True)
class Region:
    """One CDR span as a half-open `[start, end)` index range into its chain, with its Kabat-ish name."""

    name: str            # "H1".."H3" / "L1".."L3"
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class Liability:
    """One fired sequence-liability hit — chain, position, the matched motif, and the region it sits in."""

    chain: str           # "H" / "L"
    pos: int             # 0-based index into that chain
    motif: str
    region: str          # a CDR name ("H3") or "FR" (framework)


# --------------------------------------------------------------------------- #
# Sequence primitives (the "seq_dfm" of the antibody gate) — pure stdlib, testable, reused by the repair agent.
# --------------------------------------------------------------------------- #
def clean_chain(seq: str) -> str:
    """Uppercase and strip a chain to the 20 canonical amino acids (drop gaps/whitespace/unknowns)."""
    return "".join(c for c in seq.upper().strip() if c in _AA)


def net_charge(seq: str, pH: float) -> float:
    """Net charge of a single chain at `pH` (Henderson–Hasselbalch over the side chains + both termini)."""
    if not seq:
        return 0.0
    pos = 1.0 / (1.0 + 10 ** (pH - _PKA_NTERM))                                  # N-terminus
    for aa, pka in _PKA_POS.items():
        pos += seq.count(aa) / (1.0 + 10 ** (pH - pka))
    neg = 1.0 / (1.0 + 10 ** (_PKA_CTERM - pH))                                  # C-terminus
    for aa, pka in _PKA_NEG.items():
        neg += seq.count(aa) / (1.0 + 10 ** (pka - pH))
    return pos - neg


def isoelectric_point(seqs: list[str]) -> float:
    """The pH at which the combined chains carry zero net charge (bisection) — the Fv pI proxy."""
    chains = [s for s in seqs if s]
    if not chains:
        return 7.0

    def charge(pH: float) -> float:
        return sum(net_charge(s, pH) for s in chains)

    lo, hi = 0.0, 14.0
    for _ in range(60):                                  # ~10^-18 pH resolution — deterministic, no fitting
        mid = (lo + hi) / 2.0
        if charge(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def gravy(seq: str) -> float:
    """Mean Kyte–Doolittle hydropathy over `seq` (0.0 for an empty span)."""
    vals = [_KD[a] for a in seq if a in _KD]
    return sum(vals) / len(vals) if vals else 0.0


def find_cdrs(chain: str, heavy: bool, tol: AntibodyTol = AntibodyTol()) -> tuple[Region, ...] | None:
    """Locate the three CDRs of a variable domain from conserved framework anchors (no ANARCI):

      * the canonical disulfide cysteines (FR1 Cys≈22 and pre-CDR3 Cys≈92),
      * the FR2 tryptophan that starts FR2 (Trp≈36), and
      * the FR4 J-motif (WGxG heavy / FGxG light) that closes CDR3.

    Returns the three `Region`s (`H1/H2/H3` or `L1/L2/L3`), or `None` when the anchors don't resolve a plausible
    domain (the caller then stands the CDR-scoped checks down and discloses the uncertainty). Boundaries are
    Kabat-ish and deliberately a touch generous — they need to *contain* the loop residues for liability
    scanning, not number the domain exactly."""
    n = len(chain)
    if n < 70:                                            # too short to be a variable domain
        return None
    j_re = _J_HEAVY if heavy else _J_LIGHT
    jm = None
    for m in j_re.finditer(chain):                        # the J-motif sits in FR4, near the C-terminus
        if m.start() > n * 0.55:
            jm = m
    if jm is None:
        return None
    cys = [i for i, a in enumerate(chain) if a == "C"]
    c1 = next((i for i in cys if 15 <= i <= 30), None)    # FR1 conserved Cys
    if c1 is None:
        return None
    wf2 = chain.find("W", c1 + 4)                         # FR2 Trp (first W a few residues past the FR1 Cys)
    if wf2 < 0 or wf2 - c1 > 25:
        return None
    lbl = "H" if heavy else "L"
    stub = 3 if heavy else 1                              # the conserved C-x-x stub after the pre-CDR3 Cys

    def _plausible_c2(ci: int) -> bool:
        h3 = jm.start() - (ci + stub)                    # the resulting CDR-H3 length must be sane (the cap sits
        return wf2 < ci < jm.start() and 2 <= h3 <= 45   # above the length contract's h3_max so long H3s are
        #                                                  CONDEMNED by CDR_LENGTH_OUT_OF_RANGE, not disclosed away.

    # pre-CDR3 Cys: the conserved FR3 cysteine in the aromatic `[FY]-x-C` motif (Kabat ≈ 92). Anchor on the
    # *earliest* plausible motif so an engineered extra Cys downstream in CDR-H3 can't hijack the boundary;
    # fall back to the plain cysteine list when the motif is atypical.
    motif_c2 = sorted(m.end() - 1 for m in re.finditer(r"[FY][A-Z]C", chain) if _plausible_c2(m.end() - 1))
    c2 = motif_c2[0] if motif_c2 else min((i for i in cys if _plausible_c2(i)), default=None)
    if c2 is None or c2 <= c1:
        return None
    # CDR1: heavy's FR1 tail (Cys→Trp) is long, so take a fixed back-window before the Trp; light's is short,
    # so the whole Cys→Trp span is CDR-L1.
    c1_start = max(c1 + 1, wf2 - tol.h1_back) if heavy else c1 + 1
    cdr1 = Region(f"{lbl}1", c1_start, wf2)
    # CDR2: a fixed window opening `fr2_gap` residues past the FR2 Trp (capped before the pre-CDR3 Cys).
    c2_start = wf2 + tol.fr2_gap
    c2_end = min(c2_start + (tol.h2_len if heavy else tol.l2_len), c2)
    cdr2 = Region(f"{lbl}2", c2_start, max(c2_start, c2_end))
    # CDR3: between the pre-CDR3 Cys (+ the conserved 2–3 residue stub) and the J-motif.
    c3_start = c2 + (3 if heavy else 1)
    cdr3 = Region(f"{lbl}3", c3_start, jm.start())
    return (cdr1, cdr2, cdr3)


def _in_cdr(pos: int, cdrs: tuple[Region, ...]) -> Region | None:
    for r in cdrs:
        if r.start <= pos < r.end:
            return r
    return None


def cysteine_positions(chain: str) -> list[int]:
    return [i for i, a in enumerate(chain) if a == "C"]


# --------------------------------------------------------------------------- #
# Features — the scalars/lists the contracts read (featurize once; rules are pure predicates over `AbFeatures`).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AbFeatures:
    heavy: str = ""
    light: str | None = None
    cdr_ok: bool = False                                   # the CDRs of every present chain resolved
    odd_cys_chains: tuple[str, ...] = ()                   # chains with an unpaired (odd-count) cysteine
    cdr_sequons: tuple[Liability, ...] = ()                # N-glyc sequon inside a CDR (condemning)
    framework_sequons: tuple[Liability, ...] = ()          # N-glyc sequon in framework (disclose)
    cdr_deamidation: tuple[Liability, ...] = ()            # NG/NS in a CDR (disclose)
    cdr_isomerization: tuple[Liability, ...] = ()          # DG/DS in a CDR (disclose)
    cdr_oxidation: tuple[Liability, ...] = ()              # Met/Trp in a CDR (disclose)
    fragmentation: tuple[Liability, ...] = ()              # DP anywhere (disclose)
    nterm_pyroglu: tuple[str, ...] = ()                    # chains with an N-terminal Q/E (disclose)
    h3_len: int | None = None                              # CDR-H3 length (the length gate), None if unresolved
    total_cdr_len: int | None = None
    pI: float = 7.0
    charge_h: float = 0.0
    charge_l: float = 0.0
    cdr_gravy: float | None = None

    @property
    def net_charge(self) -> float:
        return self.charge_h + self.charge_l


def _scan_chain(lbl: str, seq: str, cdrs: tuple[Region, ...] | None,
                sequon_cdr, sequon_fr, deam, isom, oxid, frag) -> None:
    """Append this chain's liability hits into the caller's accumulators (mutating, to keep `featurize` flat)."""
    for m in _SEQUON.finditer(seq):
        r = _in_cdr(m.start(), cdrs) if cdrs else None
        (sequon_cdr if r else sequon_fr).append(
            Liability(lbl, m.start(), m.group(), r.name if r else "FR"))
    if cdrs:
        for rx, acc in ((_DEAMIDATION, deam), (_ISOMERIZATION, isom)):
            for m in rx.finditer(seq):
                r = _in_cdr(m.start(), cdrs)
                if r:
                    acc.append(Liability(lbl, m.start(), m.group(), r.name))
        for r in cdrs:
            for i in range(r.start, min(r.end, len(seq))):
                if seq[i] in "MW":
                    oxid.append(Liability(lbl, i, seq[i], r.name))
    for m in _FRAGMENTATION.finditer(seq):
        r = _in_cdr(m.start(), cdrs) if cdrs else None
        frag.append(Liability(lbl, m.start(), m.group(), r.name if r else "FR"))


def featurize(heavy: str, light: str | None = None, tol: AntibodyTol = AntibodyTol()) -> AbFeatures:
    """Precompute an Fv's developability scalars/lists from the sequence primitives (pure stdlib, testable)."""
    heavy = clean_chain(heavy)
    light = clean_chain(light) if light else None
    chains = [("H", heavy)] + ([("L", light)] if light else [])

    cdrs = {lbl: find_cdrs(seq, lbl == "H", tol) for lbl, seq in chains}
    cdr_ok = all(cdrs[lbl] is not None for lbl, _ in chains)

    sequon_cdr: list[Liability] = []
    sequon_fr: list[Liability] = []
    deam: list[Liability] = []
    isom: list[Liability] = []
    oxid: list[Liability] = []
    frag: list[Liability] = []
    odd_cys: list[str] = []
    pyroglu: list[str] = []
    cdr_residues: list[str] = []

    for lbl, seq in chains:
        _scan_chain(lbl, seq, cdrs[lbl], sequon_cdr, sequon_fr, deam, isom, oxid, frag)
        if len(cysteine_positions(seq)) % 2 == 1:                # an unpaired free thiol
            odd_cys.append(lbl)
        if seq and seq[0] in "QE":                               # N-terminal pyroglutamate precursor
            pyroglu.append(lbl)
        if cdrs[lbl]:
            for r in cdrs[lbl]:
                cdr_residues.append(seq[r.start:min(r.end, len(seq))])

    h_cdrs = cdrs.get("H")
    h3_len = next((r.length for r in h_cdrs if r.name == "H3"), None) if h_cdrs else None
    total = sum(r.length for c in cdrs.values() if c for r in c) if cdr_ok else None

    return AbFeatures(
        heavy=heavy, light=light, cdr_ok=cdr_ok,
        odd_cys_chains=tuple(odd_cys),
        cdr_sequons=tuple(sequon_cdr), framework_sequons=tuple(sequon_fr),
        cdr_deamidation=tuple(deam), cdr_isomerization=tuple(isom), cdr_oxidation=tuple(oxid),
        fragmentation=tuple(frag), nterm_pyroglu=tuple(pyroglu),
        h3_len=h3_len, total_cdr_len=total,
        pI=isoelectric_point([heavy, light or ""]),
        charge_h=net_charge(heavy, tol.charge_ph),
        charge_l=net_charge(light, tol.charge_ph) if light else 0.0,
        cdr_gravy=gravy("".join(cdr_residues)) if cdr_residues else None,
    )


# --------------------------------------------------------------------------- #
# The DRC — each check a legible contract reading a precomputed AbFeatures field. Condemning first, then
# disclosures (weight 0.0): the rare/severe liabilities fail the gate; the common chemistry flags inform.
# --------------------------------------------------------------------------- #
def _names(hits: tuple[Liability, ...]) -> str:
    return ", ".join(f"{h.chain}:{h.motif}@{h.pos}({h.region})" for h in hits)


def antibody_contracts() -> contracts.ContractSet:
    cs = contracts.ContractSet("antibody-developability")

    # — Condemning (weight > 0): the rare, genuinely disqualifying liabilities. —
    cs.add(contracts.Contract("UNPAIRED_CYSTEINE",
        lambda f, t: (f"unpaired cysteine in chain(s) {', '.join(f.odd_cys_chains)} (odd Cys count) — "
                      f"a free thiol drives disulfide scrambling and aggregation")
        if f.odd_cys_chains else None, weight=1.5))

    cs.add(contracts.Contract("N_GLYCOSYLATION_SEQUON_CDR",
        lambda f, t: (f"N-glycosylation sequon in a CDR ({_names(f.cdr_sequons)}) — variable glycosylation "
                      f"of the binding site (N-X-[S/T], X≠P)")
        if f.cdr_sequons else None, weight=1.5))

    cs.add(contracts.Contract("CDR_LENGTH_OUT_OF_RANGE",
        lambda f, t: (f"CDR-H3 length {f.h3_len} outside the typical window {t.h3_min}–{t.h3_max} — a "
                      f"developability/expression outlier")
        if f.h3_len is not None and not (t.h3_min <= f.h3_len <= t.h3_max) else None, weight=1.0))

    cs.add(contracts.Contract("EXTREME_FV_CHARGE",
        lambda f, t: (f"Fv isoelectric point {f.pI:.1f} outside the band {t.pi_min:.1f}–{t.pi_max:.1f} "
                      f"(net charge {f.net_charge:+.1f} at pH {t.charge_ph}) — solubility / viscosity / "
                      f"clearance risk")
        if not (t.pi_min <= f.pI <= t.pi_max) else None, weight=1.0))

    # — Disclose (weight 0.0): the common chemistry/biophysical flags even approved antibodies carry. —
    cs.add(contracts.Contract("DEAMIDATION_HOTSPOT_CDR",
        lambda f, t: (f"{len(f.cdr_deamidation)} Asn deamidation hotspot(s) in a CDR ({_names(f.cdr_deamidation)}) "
                      f"— charge heterogeneity / potency loss on storage")
        if f.cdr_deamidation else None, weight=0.0))

    cs.add(contracts.Contract("ISOMERIZATION_HOTSPOT_CDR",
        lambda f, t: (f"{len(f.cdr_isomerization)} Asp isomerization hotspot(s) in a CDR "
                      f"({_names(f.cdr_isomerization)}) — backbone isomerization / potency loss")
        if f.cdr_isomerization else None, weight=0.0))

    cs.add(contracts.Contract("FRAMEWORK_GLYCOSYLATION",
        lambda f, t: (f"N-glycosylation sequon in framework ({_names(f.framework_sequons)}) — usually benign "
                      f"but a heterogeneity source")
        if f.framework_sequons else None, weight=0.0))

    cs.add(contracts.Contract("OXIDATION_PRONE_CDR",
        lambda f, t: (f"{len(f.cdr_oxidation)} oxidation-prone residue(s) (Met/Trp) in a CDR "
                      f"({_names(f.cdr_oxidation)}) — methionine/tryptophan oxidation risk")
        if f.cdr_oxidation else None, weight=0.0))

    cs.add(contracts.Contract("FRAGMENTATION_DP",
        lambda f, t: (f"{len(f.fragmentation)} acid-labile Asp-Pro (DP) bond(s) ({_names(f.fragmentation)}) — "
                      f"low-pH fragmentation risk")
        if f.fragmentation else None, weight=0.0))

    cs.add(contracts.Contract("N_TERMINAL_PYROGLUTAMATE",
        lambda f, t: (f"N-terminal Gln/Glu on chain(s) {', '.join(f.nterm_pyroglu)} — pyroglutamate formation "
                      f"(charge heterogeneity; usually benign)")
        if f.nterm_pyroglu else None, weight=0.0))

    cs.add(contracts.Contract("CHARGE_ASYMMETRY",
        lambda f, t: (f"VH/VL charge asymmetry {f.charge_h - f.charge_l:+.1f} (|Δ|>{t.charge_asym}) — an SFvCSP "
                      f"sequence proxy (the true metric is structural)")
        if f.light is not None and abs(f.charge_h - f.charge_l) > t.charge_asym else None, weight=0.0))

    cs.add(contracts.Contract("HYDROPHOBICITY_HIGH",
        lambda f, t: (f"high CDR hydrophobicity (GRAVY {f.cdr_gravy:.2f} > {t.gravy_max}) — a PSH sequence proxy "
                      f"(the true metric is structural)")
        if f.cdr_gravy is not None and f.cdr_gravy > t.gravy_max else None, weight=0.0))

    cs.add(contracts.Contract("CDR_DETECTION_UNCERTAIN",
        lambda f, t: ("CDR boundaries did not resolve from the framework anchors — CDR-scoped checks stood "
                      "down (pass IMGT-numbered chains or check the input is a variable domain)")
        if f.heavy and not f.cdr_ok else None, weight=0.0))
    return cs


def validate(heavy: str, light: str | None = None, tol: AntibodyTol = AntibodyTol()) -> contracts.Verdict:
    """The Fv developability verdict: featurize then evaluate the DRC. `light=None` qualifies a single-domain
    VHH / nanobody (the VH/VL-asymmetry disclosure stands down)."""
    return antibody_contracts().evaluate(featurize(heavy, light, tol), tol)


def is_undevelopable(heavy: str, light: str | None = None, tol: AntibodyTol = AntibodyTol()) -> bool:
    """An Fv is undevelopable iff a CONDEMNING contract fired (score > 0; disclosed flags are weight 0)."""
    return validate(heavy, light, tol).score > 0.0


# --------------------------------------------------------------------------- #
# Reference Fv sequences (public, well-known approved therapeutics) — the "real drugs pass" fixtures.
# --------------------------------------------------------------------------- #
TRASTUZUMAB_VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQ"
                  "MNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS")
TRASTUZUMAB_VL = ("DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFA"
                  "TYYCQQHYTTPPTFGQGTKVEIK")
ADALIMUMAB_VH = ("EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQ"
                 "MNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS")
ADALIMUMAB_VL = ("DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVA"
                 "TYYCQRYNRAPYTFGQGTKVEIK")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tol = AntibodyTol()
    print("=== antibody-QC smoke ===")
    for name, h, l in (("trastuzumab", TRASTUZUMAB_VH, TRASTUZUMAB_VL),
                       ("adalimumab", ADALIMUMAB_VH, ADALIMUMAB_VL)):
        v = validate(h, l, tol)
        f = featurize(h, l, tol)
        print(f"\n{name:12} → {'PASS' if v.score == 0 else 'FAIL'} "
              f"(pI {f.pI:.1f}, H3 {f.h3_len}, score {v.score})")
        for r in v.reasons:
            print(f"    {'✗' if r.weight > 0 else '·'} {r.contract}: {r.message}")

    # a planted-liability Fv: an unpaired Cys + an N-glyc sequon, both in CDR-H3 → condemned.
    bad_h = TRASTUZUMAB_VH.replace("WGGDGFYAMDY", "WGGDNISYACMDY")   # NIS sequon + extra Cys in CDR-H3
    v = validate(bad_h, TRASTUZUMAB_VL, tol)
    print(f"\nplanted-liability → {'PASS' if v.score == 0 else 'FAIL'} (score {v.score})")
    for r in v.reasons:
        print(f"    {'✗' if r.weight > 0 else '·'} {r.contract}: {r.message}")
    print("\nantibody_developability smoke OK.")
