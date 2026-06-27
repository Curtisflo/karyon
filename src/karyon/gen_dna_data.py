"""gen_dna_data — corpora for the generated-DNA synthesizability gate (gen-DNA-QC).

The gate (`gen_dna_validity.py`) qualifies a *generative* model's DNA output (NVIDIA's Evo2). Producing real
Evo2 sequences needs a GPU/NIM (out of scope at QC time — generating the sequence is a separate GPU step),
so this module supplies the sequences the *honesty probe* needs to prove the gate is a real, faithful
instrument, all reproducible and offline-tolerant:

  * `load_natural_cds`   — real *E. coli* K-12 MG1655 (NC_000913.3) coding sequences from NCBI (fetch+cache,
                           offline-skip). The clean biological class (and the Markov training corpus).
  * `synthetic_clean`    — seeded random sequences guaranteed to PASS the gate (the by-construction clean set).
  * `synthetic_decoys`   — seeded sequences each carrying one CONDEMNING barrier (GC-extreme / homopolymer /
                           hairpin) — the instrument's negative class (PI-1).
  * `uniform_random`     — pure uniform ACGT: the un-gated generator baseline (some clean, some rejected).
  * `markov_generated`   — a tiny order-k Markov sampler trained on a corpus: a real (if simple) sequence
                           *generator* whose raw output the gate scores (PI-3, the "gate a generator" demo).

Everything is seeded and deterministic; only `load_natural_cds` touches the network (and offline-skips).
"""

from __future__ import annotations

import random
import socket
import urllib.error
import urllib.request

from .paths import network_allowed
from .pose_data import _cache_dir

from . import gen_dna_validity as gv

_UA = "karyon/0.1 (+https://github.com/Curtisflo/karyon)"
_TIMEOUT_S = 120
# E. coli K-12 MG1655 complete genome; rettype=fasta_cds_na returns every CDS as a nucleotide FASTA record.
_ECOLI_CDS_URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                  "?db=nuccore&id=NC_000913.3&rettype=fasta_cds_na&retmode=text")


class SeqUnavailable(RuntimeError):
    """A corpus could not be fetched and is not cached — the arm offline-skips (mirrors PoseUnavailable)."""


def _gen_cache():
    d = _cache_dir() / "gen_dna"
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """(header, sequence) records from a FASTA string. stdlib, no Biopython."""
    out: list[tuple[str, str]] = []
    header, chunks = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                out.append((header, "".join(chunks)))
            header, chunks = line[1:].strip(), []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        out.append((header, "".join(chunks)))
    return out


# --------------------------------------------------------------------------- #
# Natural CDS — real biological sequence (the clean class + Markov training corpus).
# --------------------------------------------------------------------------- #
def load_natural_cds(limit: int | None = None, *, min_len: int = 60, max_len: int = 3000) -> list[str]:
    """*E. coli* MG1655 coding sequences (uppercase ACGT-only), length-windowed. Fetched from NCBI and
    cached; offline-skip via `SeqUnavailable` if neither cache nor network is available."""
    path = _gen_cache() / "ecoli_mg1655_cds.fasta"
    if path.exists():
        text = path.read_text(errors="replace")
    else:
        if not network_allowed():
            raise SeqUnavailable("network disabled via KARYON_NO_NETWORK")
        req = urllib.request.Request(_ECOLI_CDS_URL, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
                text = r.read().decode(errors="replace")
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
            raise SeqUnavailable(f"cannot fetch E. coli MG1655 CDS from NCBI: {e}") from e
        if text.count(">") < 100:
            raise SeqUnavailable("NCBI returned too few CDS records (unexpected payload)")
        path.write_text(text)

    seqs: list[str] = []
    for _, seq in parse_fasta(text):
        s = seq.upper()
        if set(s) <= set("ACGT") and min_len <= len(s) <= max_len:
            seqs.append(s)
            if limit and len(seqs) >= limit:
                break
    if not seqs:
        raise SeqUnavailable("E. coli CDS cache present but no ACGT sequences in the length window")
    return seqs


# --------------------------------------------------------------------------- #
# Deterministic in-process corpora (seeded, offline).
# --------------------------------------------------------------------------- #
def _rand_seq(rng: random.Random, n: int, gc: float = 0.5) -> str:
    """A random ACGT sequence of length n with expected GC fraction `gc`."""
    g = c = gc / 2.0
    a = t = (1.0 - gc) / 2.0
    return "".join(rng.choices("ACGT", weights=[a, c, g, t], k=n))


def synthetic_clean(n: int = 120, seed: int = 0, length_range: tuple[int, int] = (60, 300),
                    tol: "gv.GenDNATol" = gv.GenDNATol()) -> list[str]:
    """`n` seeded random sequences guaranteed to PASS the gate (the by-construction clean class)."""
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        length = rng.randint(*length_range)
        s = _rand_seq(rng, length, gc=rng.uniform(0.42, 0.58))
        if not gv.is_unsynthesizable(s, tol):          # reject the rare random sequence that trips a rule
            out.append(s)
    return out


def synthetic_decoys(n: int = 120, seed: int = 1, length_range: tuple[int, int] = (60, 300),
                     tol: "gv.GenDNATol" = gv.GenDNATol()) -> list[str]:
    """`n` seeded sequences, each carrying one CONDEMNING barrier (cycled): a GC-extreme block, an
    over-long homopolymer, or a strong hairpin. Each is guaranteed to be flagged unsynthesizable."""
    rng = random.Random(seed)
    barriers = ("gc_high", "gc_low", "homopolymer", "hairpin")
    out: list[str] = []
    while len(out) < n:
        length = rng.randint(*length_range)
        backbone = _rand_seq(rng, length, gc=0.5)
        kind = barriers[len(out) % len(barriers)]
        if kind == "gc_high":
            s = _rand_seq(rng, length, gc=0.92)
        elif kind == "gc_low":
            s = _rand_seq(rng, length, gc=0.08)
        elif kind == "homopolymer":
            base = rng.choice("ACGT")
            run = base * (tol.max_homopolymer_run + 4)
            cut = rng.randint(0, len(backbone))
            s = backbone[:cut] + run + backbone[cut:]
        else:  # hairpin: arm + loop + reverse-complement(arm)
            from .seq_dfm import reverse_complement
            arm = _rand_seq(rng, tol.min_stem_len + 3, gc=0.5)
            loop = _rand_seq(rng, 5, gc=0.5)
            stem = arm + loop + reverse_complement(arm)
            cut = rng.randint(0, len(backbone))
            s = backbone[:cut] + stem + backbone[cut:]
        if gv.is_unsynthesizable(s, tol):              # only keep guaranteed-condemned decoys
            out.append(s)
    return out


def uniform_random(n: int = 120, seed: int = 2, length_range: tuple[int, int] = (60, 300)) -> list[str]:
    """`n` pure-uniform ACGT sequences — the un-gated generator baseline (no synthesis awareness)."""
    rng = random.Random(seed)
    return [_rand_seq(rng, rng.randint(*length_range), gc=0.5) for _ in range(n)]


def markov_generated(train: list[str], order: int = 3, n: int = 120, seed: int = 3,
                     length_range: tuple[int, int] = (60, 300)) -> list[str]:
    """`n` sequences sampled from an order-`order` Markov model trained on `train` — a real (if simple)
    sequence generator standing in for Evo2; the gate scores its RAW output. Deterministic for a given seed."""
    if not train:
        raise SeqUnavailable("markov_generated needs a non-empty training corpus")
    rng = random.Random(seed)
    # transition counts: context (order-mer) -> {next base: count}
    trans: dict[str, dict[str, int]] = {}
    starts: list[str] = []
    for s in train:
        if len(s) <= order:
            continue
        starts.append(s[:order])
        for i in range(len(s) - order):
            ctx = s[i:i + order]
            nxt = s[i + order]
            trans.setdefault(ctx, {}).setdefault(nxt, 0)
            trans[ctx][nxt] += 1
    if not starts:
        raise SeqUnavailable("training corpus too short for the chosen Markov order")

    def sample_next(ctx: str) -> str:
        opts = trans.get(ctx)
        if not opts:
            return rng.choice("ACGT")
        bases, weights = zip(*opts.items())
        return rng.choices(bases, weights=weights, k=1)[0]

    out: list[str] = []
    for _ in range(n):
        length = rng.randint(*length_range)
        s = rng.choice(starts)
        while len(s) < length:
            s += sample_next(s[-order:])
        out.append(s[:length])
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="gen-DNA-QC corpora.")
    ap.add_argument("--natural", action="store_true", help="fetch + summarize E. coli MG1655 CDS")
    ap.add_argument("--limit", type=int, default=8)
    cli = ap.parse_args()

    if cli.natural:
        try:
            seqs = load_natural_cds(limit=cli.limit)
            print(f"natural CDS: {len(seqs)} sequences (limit {cli.limit})")
            for s in seqs[:cli.limit]:
                print(f"  len {len(s):4}  GC {gv.seq_dfm.gc_fraction(s):.0%}  {'PASS' if not gv.is_unsynthesizable(s) else 'FLAG'}")
        except SeqUnavailable as e:
            print("SKIP —", e)
        raise SystemExit(0)

    clean = synthetic_clean(6)
    decoys = synthetic_decoys(6)
    rnd = uniform_random(6)
    print("synthetic_clean (all should PASS):")
    for s in clean:
        print(f"  {'PASS' if not gv.is_unsynthesizable(s) else 'FLAG':4}  len {len(s)}  GC {gv.seq_dfm.gc_fraction(s):.0%}")
    print("synthetic_decoys (all should FLAG):")
    for s in decoys:
        v = gv.validate(s)
        print(f"  {'PASS' if v.score == 0 else 'FLAG':4}  {[r.contract for r in v.reasons if r.weight > 0]}")
    print(f"uniform_random rejection: {sum(gv.is_unsynthesizable(s) for s in rnd)}/{len(rnd)}")
