"""gen_dna_validity — a legible synthesizability/manufacturability DRC for generated DNA (gen-DNA-QC).

A BioNeMo-complement QC gate on the **generative-output** axis — the "unroutable net" report for a
generative model's output. A genomic-sequence generator (e.g. NVIDIA's **Evo2**) emits DNA that scores well
on the model's own confidence yet can be **unmanufacturable** — GC outside the synthesis envelope, a
homopolymer run that slips the synthesizer, a hairpin that won't anneal, a cloning-site collision, or
(across a batch) two sequences that hybridize to each other. BioNeMo ships *advisory* validation for this;
this module is the deterministic gate.

It is a CAD/EDA design-for-manufacture design-rule check transplanted to DNA: each check is a thin
`contracts.Contract` reading a precomputed `DNAFeatures` scalar (featurize once, contracts read scalars),
over the pure-stdlib primitives in `seq_dfm.py`. Two ownership levels:

  * **per-sequence** (a Part-level fact): GC band, homopolymer / poly-G runs, length window, hairpin stem,
    restriction sites — `dna_contracts()` over `DNAFeatures`.
  * **per-batch** (a Design-level invariant no single sequence owns): cross-hybridization across the set —
    `set_contracts()` over `SetFeatures`.

Disclose-vs-condemn tiering: the PASS/FAIL gate keys off the verdict's continuous *score* (Σ condemning
weights), so weight-0 contracts (poly-G risk, restriction-site note, a mild cross-hyb stretch) are *reported*
without failing the structure. Every threshold is a commercial-synthesis constant or calibrated to the
DnaChisel reference (disclosed) — zero fitted-to-accuracy parameters. No numpy, no rdkit: the gate is pure
string geometry.

    python -m karyon.gen_dna_validity     # smoke: a clean sequence passes; planted decoys fail
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import contracts
from . import seq_dfm


# --------------------------------------------------------------------------- #
# Calibration — every threshold a commercial-synthesis constant or DnaChisel-reference-derived (disclosed).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenDNATol:
    gc_min: float = 0.25         # GC fraction band — outside ~25–65% an oligo/gene synthesizes poorly.
    gc_max: float = 0.65         #   Calibrated to the DnaChisel EnforceGCContent reference (gen_dna_honesty).
    max_homopolymer_run: int = 8     # a run LONGER than this fails (9+ nt of one base ⇒ synthesis slippage);
    #                                  matched to a DnaChisel AvoidPattern("9x<base>") in the faithfulness arm.
    max_g_run: int = 3           # stricter cap on consecutive G specifically — GGGG+ is a G-quadruplex risk
    #                              (disclosed, not condemned: a flagged risk, not a guaranteed synth failure).
    min_length: int = 18         # below the orderable floor (short oligos don't stay annealed).
    max_length: int = 3000       # above the single-fragment gene-synthesis window (gBlocks/Twist gene).
    min_stem_len: int = 12       # a self-complementary stem >= this fails (strong hairpin); the DnaChisel
    #                              AvoidHairpins stem_size in the faithfulness arm.
    hairpin_min_loop: int = 3
    hairpin_max_loop: int = 30
    cross_hyb_kmer: int = 12         # batch pair sharing a complementary stretch >= this ⇒ DISCLOSE.
    cross_hyb_condemn_kmer: int = 20  # >= this ⇒ CONDEMN (the pair anneals to each other, not the target).


# --------------------------------------------------------------------------- #
# Per-sequence features — the scalars the contracts read (featurize once; rules are pure scalar predicates).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DNAFeatures:
    length: int = 60
    gc_frac: float = 0.5
    max_run: int = 1                 # longest homopolymer run of any base
    max_g_run: int = 1               # longest run of G specifically
    hairpin: "seq_dfm.Hairpin" = field(default_factory=lambda: seq_dfm.Hairpin(0))
    sites: tuple["seq_dfm.SiteHit", ...] = ()

    def severity(self, tol: GenDNATol) -> float:
        """Continuous severity (Σ normalized exceedances) — the ranking statistic for the instrument AUROC.
        0.0 for a clean sequence, grows with how far each condemning violation exceeds its threshold.
        Distinct from Verdict.score (Σ fired weights). Disclosed-only (poly-G/sites) do not contribute."""
        def relu(x: float) -> float:
            return x if x > 0 else 0.0
        s = 0.0
        s += relu(self.gc_frac / tol.gc_max - 1.0)                       # GC above band
        s += relu(tol.gc_min / max(self.gc_frac, 1e-6) - 1.0)           # GC below band
        s += relu(self.max_run / tol.max_homopolymer_run - 1.0)         # homopolymer over cap
        s += relu(self.hairpin.stem / tol.min_stem_len - 1.0)          # hairpin over cap
        s += relu(self.length / tol.max_length - 1.0)                   # too long
        s += relu(tol.min_length / max(self.length, 1) - 1.0)          # too short
        return s


_VALID_BASES = set("ACGT")


def featurize(seq: str, tol: GenDNATol = GenDNATol()) -> DNAFeatures:
    """Precompute a sequence's DFM scalars from the `seq_dfm` primitives (pure stdlib, testable)."""
    s = seq.upper()
    return DNAFeatures(
        length=len(s),
        gc_frac=seq_dfm.gc_fraction(s),
        max_run=seq_dfm.longest_run(s),
        max_g_run=seq_dfm.longest_run(s, "G"),
        hairpin=seq_dfm.hairpin_stem(s, tol.hairpin_min_loop, tol.hairpin_max_loop),
        sites=tuple(seq_dfm.restriction_sites(s)),
    )


# --------------------------------------------------------------------------- #
# The per-sequence DRC — each check a legible contract reading a precomputed DNAFeatures scalar.
# --------------------------------------------------------------------------- #
def dna_contracts() -> contracts.ContractSet:
    cs = contracts.ContractSet("gen-dna-validity")

    cs.add(contracts.Contract("GC_OUT_OF_BAND",
        lambda f, t: (f"GC content {f.gc_frac:.0%} outside the synthesis envelope "
                      f"{t.gc_min:.0%}–{t.gc_max:.0%}")
        if not (t.gc_min <= f.gc_frac <= t.gc_max) else None, weight=1.5))

    cs.add(contracts.Contract("HOMOPOLYMER_RUN",
        lambda f, t: (f"homopolymer run of {f.max_run} (>{t.max_homopolymer_run}) — synthesis slippage")
        if f.max_run > t.max_homopolymer_run else None, weight=1.5))

    cs.add(contracts.Contract("LENGTH_OUT_OF_RANGE",
        lambda f, t: (f"length {f.length} nt outside the orderable window "
                      f"{t.min_length}–{t.max_length} nt")
        if not (t.min_length <= f.length <= t.max_length) else None, weight=1.0))

    cs.add(contracts.Contract("STRONG_HAIRPIN",
        lambda f, t: (f"strong hairpin: a {f.hairpin.stem} bp self-complementary stem (≥{t.min_stem_len}) "
                      f"across a {f.hairpin.loop} nt loop — folds on itself, won't synthesize/anneal cleanly")
        if f.hairpin.stem >= t.min_stem_len else None, weight=1.0))

    # Disclose (weight 0.0 — informs, does not condemn): a poly-G G-quadruplex RISK.
    cs.add(contracts.Contract("POLY_G_RUN",
        lambda f, t: (f"poly-G run of {f.max_g_run} (>{t.max_g_run}) — G-quadruplex risk")
        if f.max_g_run > t.max_g_run else None, weight=0.0))

    # Disclose (weight 0.0): restriction-recognition-site collisions — a cloning hazard, not a synth failure.
    cs.add(contracts.Contract("RESTRICTION_SITE",
        lambda f, t: (f"{len(f.sites)} restriction site{'s' if len(f.sites) != 1 else ''} present "
                      f"({', '.join(sorted({h.enzyme for h in f.sites}))}) — will be cut if cloned with them")
        if f.sites else None, weight=0.0))
    return cs


def validate(seq: str, tol: GenDNATol = GenDNATol()) -> contracts.Verdict:
    """The per-sequence verdict: featurize then evaluate the DRC."""
    return dna_contracts().evaluate(featurize(seq, tol), tol)


def is_unsynthesizable(seq: str, tol: GenDNATol = GenDNATol()) -> bool:
    """A sequence is unsynthesizable iff a CONDEMNING contract fired (score > 0; disclosed flags are 0)."""
    return validate(seq, tol).score > 0.0


# --------------------------------------------------------------------------- #
# Per-batch DRC — cross-hybridization across the set (the Design-level invariant no single sequence owns).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CrossHybPair:
    """Two sequences (by index/name) that anneal to each other over ``stretch`` (a complementary run)."""

    a: str
    b: str
    stretch: str

    @property
    def length(self) -> int:
        return len(self.stretch)


@dataclass(frozen=True)
class SetFeatures:
    n: int = 0
    max_cross_hyb: int = 0                 # longest inter-sequence anneal stretch (nt)
    worst: CrossHybPair | None = None
    n_warn: int = 0                        # pairs at/above the disclose threshold
    n_condemn: int = 0                     # pairs at/above the condemn threshold


def featurize_set(named_seqs: list[tuple[str, str]], tol: GenDNATol = GenDNATol()) -> SetFeatures:
    """Pairwise cross-hyb over a batch of (name, seq). The longest stretch over which sequence a anneals to
    sequence b is ``seq_dfm.anneal_stretch(a, b)`` (the longest common substring of a and rc(b))."""
    pairs: list[CrossHybPair] = []
    for i in range(len(named_seqs)):
        ni, si = named_seqs[i]
        for j in range(i + 1, len(named_seqs)):
            nj, sj = named_seqs[j]
            stretch = seq_dfm.anneal_stretch(si.upper(), sj.upper())
            if len(stretch) >= tol.cross_hyb_kmer:
                pairs.append(CrossHybPair(ni, nj, stretch))
    worst = max(pairs, key=lambda p: p.length, default=None)
    return SetFeatures(
        n=len(named_seqs),
        max_cross_hyb=worst.length if worst else 0,
        worst=worst,
        n_warn=sum(1 for p in pairs if p.length >= tol.cross_hyb_kmer),
        n_condemn=sum(1 for p in pairs if p.length >= tol.cross_hyb_condemn_kmer),
    )


def set_contracts() -> contracts.ContractSet:
    cs = contracts.ContractSet("gen-dna-batch-validity")

    # Disclose (weight 0.0): a moderate complementary stretch between two sequences.
    cs.add(contracts.Contract("CROSS_HYBRIDIZATION",
        lambda f, t: (f"{f.n_warn} sequence pair{'s' if f.n_warn != 1 else ''} share a complementary "
                      f"stretch ≥{t.cross_hyb_kmer} nt (worst {f.worst.a}↔{f.worst.b} over {f.max_cross_hyb} nt)")
        if f.worst is not None and f.n_condemn == 0 else None, weight=0.0))

    # Condemn: a stretch long enough that the pair anneals to each other instead of their targets.
    cs.add(contracts.Contract("SEVERE_CROSS_HYBRIDIZATION",
        lambda f, t: (f"{f.n_condemn} sequence pair{'s' if f.n_condemn != 1 else ''} anneal to each other "
                      f"over ≥{t.cross_hyb_condemn_kmer} nt (worst {f.worst.a}↔{f.worst.b} over "
                      f"{f.max_cross_hyb} nt) — they will hybridize to each other, not their targets")
        if f.worst is not None and f.n_condemn > 0 else None, weight=1.5))
    return cs


@dataclass(frozen=True)
class SetVerdict:
    """A batch verdict: the per-sequence verdicts plus the set-level cross-hyb verdict. ``ok`` ⇒ no
    condemning contract fired anywhere (the gate's PASS), so disclosed notes don't fail the batch."""

    per_seq: tuple[tuple[str, contracts.Verdict], ...]
    batch: contracts.Verdict

    @property
    def ok(self) -> bool:
        return self.batch.score == 0.0 and all(v.score == 0.0 for _, v in self.per_seq)

    @property
    def reasons(self):
        out = []
        for name, v in self.per_seq:
            out.extend((name, r) for r in v.reasons)
        out.extend(("<batch>", r) for r in self.batch.reasons)
        return out


def validate_set(named_seqs: list[tuple[str, str]], tol: GenDNATol = GenDNATol()) -> SetVerdict:
    """The whole-batch gate: every sequence's per-sequence DRC + the design-level cross-hyb check."""
    cs = dna_contracts()
    per = tuple((name, cs.evaluate(featurize(seq, tol), tol)) for name, seq in named_seqs)
    batch = set_contracts().evaluate(featurize_set(named_seqs, tol), tol)
    return SetVerdict(per_seq=per, batch=batch)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tol = GenDNATol()
    clean = "ACGTACGTGGTTACCAGTCAGTACTGACTAGTCAGTGCATGCATGGTACAACGTACGTAGT"
    decoys = {
        "GC-extreme (high)": "GCGCGGCGCGGCGCGGCGCGGCGCGGCGCGGCGCGGCGCGGCGCGGCGCGGCGCGGCGCG",
        "homopolymer":       "ACGTACGTAAAAAAAAAAAAACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
        "strong hairpin":    "GGGGCCCCAATTTTGGGGCCCC".replace("AATTTT", "ATTTAG") + "GGGCCC",
        "restriction site":  "ACGTGAATTCACGTGGATCCACGTAAGCTTACGTACGTACGTACGTACGTACGTACGTAC",
    }
    print("=== gen-DNA-QC smoke (per-sequence) ===")
    v = validate(clean, tol)
    print(f"clean              → {'PASS' if v.score == 0 else 'FAIL'}  (GC {seq_dfm.gc_fraction(clean):.0%}, "
          f"run {seq_dfm.longest_run(clean)}, stem {seq_dfm.hairpin_stem(clean).stem})")
    for r in v.reasons:
        print(f"    · {r.contract}: {r.message}")
    for label, seq in decoys.items():
        v = validate(seq, tol)
        print(f"\n{label:18} → {'PASS' if v.score == 0 else 'FAIL'} (score {v.score})")
        for r in v.reasons:
            mark = "✗" if r.weight > 0 else "·"
            print(f"    {mark} {r.contract}: {r.message}")

    print("\n=== gen-DNA-QC smoke (batch cross-hyb) ===")
    a = "ACGTACGTACGTACGTTTGGCCAATTGGCCAATTGGCCAATTACGTACGTACGTACGTAC"
    b = "GCGCGCGC" + seq_dfm.reverse_complement(a[20:44]) + "GCGCGCGC"   # b carries rc of a[20:44] → anneal
    sv = validate_set([("seqA", a), ("seqB", b)], tol)
    print(f"batch              → {'PASS' if sv.ok else 'FAIL'}")
    for name, r in sv.reasons:
        mark = "✗" if r.weight > 0 else "·"
        print(f"    {mark} [{name}] {r.contract}: {r.message}")
    print("\ngen_dna_validity smoke OK.")
