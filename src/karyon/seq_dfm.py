"""seq_dfm — pure-stdlib DNA sequence primitives for the generative-output DFM gate (gen-DNA-QC).

The sequence-level half of the design-for-manufacturability question — *can this DNA actually be ordered and
cloned, and will it behave* — the DNA analogue of a CAD DFM check (printability / no-interference
contracts). This module is the stdlib-clean core: the bare-sequence DFM primitives (`gc_fraction`,
`longest_run`, `reverse_complement`, `anneal_stretch`), **plus** two new sequence-level signals a
generated-sequence gate needs that origami staples didn't:

  * `restriction_sites(seq)`  — recognition-site collisions (a cloning hazard a generator is blind to).
  * `hairpin_stem(seq)`       — the strongest self-complementary stem (secondary structure that fails
                                synthesis / annealing). ViennaRNA's ΔG is the *reference* (see
                                `gen_dna_honesty.py`); this owned signal needs no thermodynamics package.

Nothing here imports numpy or rdkit — the gate is the lightest of the karyon QC layers (pure string ops).
The `contracts`-wrapped DRC over these primitives is `gen_dna_validity.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


# --------------------------------------------------------------------------- #
# Bare-sequence primitives (the substrate-independent DFM surface).
# --------------------------------------------------------------------------- #
def gc_fraction(seq: str) -> float:
    """GC fraction of a bare sequence (0.0 for empty)."""
    return sum(c in "GC" for c in seq) / len(seq) if seq else 0.0


def longest_run(seq: str, base: str | None = None) -> int:
    """Longest run of identical bases in ``seq`` (or of ``base`` specifically)."""
    best = run = 0
    prev = ""
    for c in seq:
        if c == prev and (base is None or c == base):
            run += 1
        else:
            run = 1 if (base is None or c == base) else 0
        prev = c
        best = max(best, run)
    return best


def reverse_complement(seq: str) -> str:
    """Watson–Crick reverse complement of a bare sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


def _longest_common_substring(a: str, b: str) -> str:
    """The longest substring shared by ``a`` and ``b`` (classic DP). Used on ``a`` and
    ``reverse_complement(b)`` so the shared substring is exactly the stretch over which ``a`` anneals to
    ``b`` (see :func:`anneal_stretch`)."""
    if not a or not b:
        return ""
    prev = [0] * (len(b) + 1)
    best_len = best_end = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len, best_end = cur[j], i
        prev = cur
    return a[best_end - best_len:best_end]


def anneal_stretch(a: str, b: str) -> str:
    """The longest stretch over which sequence ``a`` anneals to sequence ``b``.

    The cross-hybridization primitive on bare sequences: the longest common substring of ``a`` and
    ``reverse_complement(b)`` (if ``a`` contains ``s`` and ``b`` contains ``rc(s)``, they pair over ``|s|``
    bases). The sequence-level core of a cross-hybridizing-pairs check."""
    return _longest_common_substring(a, reverse_complement(b))


# --------------------------------------------------------------------------- #
# Restriction sites — a curated table of common type-II recognition sites. Presence is a CLONING hazard
# (the site will be cut if the sequence is used with that enzyme), not a synthesis failure — so the DRC
# discloses it. The default table covers the BioBrick-forbidden enzymes plus common cloning workhorses.
# --------------------------------------------------------------------------- #
RESTRICTION_ENZYMES: dict[str, str] = {
    "EcoRI": "GAATTC", "BamHI": "GGATCC", "HindIII": "AAGCTT", "XhoI": "CTCGAG",
    "NotI": "GCGGCCGC", "XbaI": "TCTAGA", "SpeI": "ACTAGT", "PstI": "CTGCAG",
    "SacI": "GAGCTC", "KpnI": "GGTACC", "SalI": "GTCGAC", "NcoI": "CCATGG",
    "NdeI": "CATATG", "SmaI": "CCCGGG", "BsaI": "GGTCTC", "BbsI": "GAAGAC",
}


@dataclass(frozen=True)
class SiteHit:
    """One restriction-site occurrence: which enzyme, the recognized site, and its 0-based position."""

    enzyme: str
    site: str
    position: int


def restriction_sites(seq: str, enzymes: dict[str, str] | None = None) -> list[SiteHit]:
    """Every restriction-recognition-site occurrence in ``seq`` (both strands).

    For each enzyme, the forward strand is searched for the recognition site *and* its reverse complement
    (so non-palindromic sites — e.g. the type-IIS BsaI/BbsI — are caught on either strand); palindromic
    sites de-duplicate to one search. Hits are returned in (position, enzyme) order."""
    s = seq.upper()
    table = enzymes if enzymes is not None else RESTRICTION_ENZYMES
    hits: list[SiteHit] = []
    for enzyme, site in table.items():
        site = site.upper()
        patterns = {site, reverse_complement(site)}      # one entry when palindromic
        for pat in patterns:
            start = s.find(pat)
            while start != -1:
                hits.append(SiteHit(enzyme, pat, start))
                start = s.find(pat, start + 1)
    hits.sort(key=lambda h: (h.position, h.enzyme))
    return hits


# --------------------------------------------------------------------------- #
# Hairpins — the strongest self-complementary stem (intra-strand fold-back). A 5' arm pairs with a
# downstream 3' arm of the same strand, with an intervening loop. Strong stems impede synthesis and make
# an oligo anneal to itself instead of its target. ViennaRNA's ΔG quantifies this thermodynamically; this
# owned signal is the deterministic, package-free stem length — validated against ViennaRNA in the harness.
# --------------------------------------------------------------------------- #
_PAIR = {"A": "T", "T": "A", "G": "C", "C": "G"}


@dataclass(frozen=True)
class Hairpin:
    """A self-complementary stem: ``stem`` bp paired across a loop of ``loop`` nt, the 5' arm starting at
    ``start`` and the 3' arm ending at ``end`` (0-based, exclusive end). ``stem == 0`` ⇒ no hairpin."""

    stem: int
    loop: int = 0
    start: int = 0
    end: int = 0


def hairpin_stem(seq: str, min_loop: int = 3, max_loop: int = 30) -> Hairpin:
    """The longest perfect self-complementary stem in ``seq``, requiring a loop in ``[min_loop, max_loop]``.

    For each innermost paired position ``(l, r)`` (the base pair nearest the loop, ``r - l - 1`` the loop
    length), expand outward while the arms stay complementary; track the longest stem. Bounding the loop to
    ``max_loop`` keeps the scan O(n·max_loop·stem) — fast on gene-length input and faithful to the *local*
    hairpins that fail synthesis (long-range structure is ViennaRNA's job, not a deterministic local gate)."""
    s = seq.upper()
    n = len(s)
    best = Hairpin(0)
    for l in range(n):
        for loop in range(min_loop, max_loop + 1):
            r = l + loop + 1
            if r >= n:
                break
            if _PAIR.get(s[l]) != s[r]:
                continue
            # innermost pair (l, r) is complementary — expand outward.
            stem = 1
            li, ri = l - 1, r + 1
            while li >= 0 and ri < n and _PAIR.get(s[li]) == s[ri]:
                stem += 1
                li -= 1
                ri += 1
            if stem > best.stem:
                best = Hairpin(stem=stem, loop=loop, start=li + 1, end=ri)
    return best
