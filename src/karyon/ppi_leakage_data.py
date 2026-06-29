"""ppi_leakage_data — cached loader for a sequence-based PPI benchmark (the protein/PPI leakage avenue).

The substrate for the **protein-interaction benchmark-honesty** probe — the protein-sequence sibling of the
molecular-property scaffold-leakage probe (avenue 6). Same cross-cutting diagnostic (STRATEGY §3: "data
leakage inflates ML accuracy"), here in its canonical *pair-input* form documented for PPI prediction
(Park & Marcotte 2012, Nature Methods; Bernett et al. 2024): sequence-based PPI models report on **random
pair splits**, which leak — the same proteins straddle train and test, so a model scores well by memorizing
"protein X is sticky" (node-identity reuse), not by learning interaction. The honest eval is a
**protein-disjoint split** (Park & Marcotte's C3 regime: no protein in a test pair appears in any train
pair). Reported random-split accuracy is inflated against it.

Substrate — the **Guo yeast** set (S. cerevisiae), the canonical sequence-based PPI benchmark used by
DeepPPI / PIPR (Chen et al. 2019), distributed as login-free raw-GitHub TSVs:
  * a dictionary  `protein_id <TAB> sequence`  (2,497 proteins, 20-AA alphabet);
  * an actions    `id1 <TAB> id2 <TAB> label`  (11,188 pairs, balanced 5,594 + / 5,594 −).

Each record is a `Pair` of two `Protein`s plus a 0/1 label. stdlib-only (urllib + csv); cache-first under
`.cache/bio/`, offline-skip via `DatasetUnavailable`. Mirrors `molnet_data.py`.

    python -m karyon.ppi_leakage_data     # smoke: fetch + summarize + disjoint-split disjointness
"""

from __future__ import annotations

import math
import random
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from .paths import cache_dir, network_allowed

AA = "ACDEFGHIKLMNPQRSTVWY"          # the 20-amino-acid alphabet (the validity filter for sequences)
_AA = set(AA)

_UA = "chalkeon-bio/1 (+https://chalkeon.local/karyon ppi-leakage)"
_TIMEOUT_S = 120
_BASE = "https://raw.githubusercontent.com/muhaochen/seq_ppi/master/yeast/preprocessed"
_DICT_URL = f"{_BASE}/protein.dictionary.tsv"
_ACTIONS_URL = f"{_BASE}/protein.actions.tsv"


class DatasetUnavailable(RuntimeError):
    """The PPI benchmark could not be fetched/parsed and is not cached → SKIP."""


@dataclass(frozen=True)
class Protein:
    pid: str                    # the protein id (the node-identity / leakage key)
    seq: str                    # amino-acid sequence (the similarity / near-dup key)


@dataclass(frozen=True)
class Pair:
    a: Protein
    b: Protein
    label: int                  # 1 = interacting, 0 = non-interacting


# --------------------------------------------------------------------------- #
# Cache plumbing (cache_dir(); override with $KARYON_CACHE — mirrors molnet_data.py).
# --------------------------------------------------------------------------- #
def _fetch(url: str, cache_name: str, *, refresh: bool) -> str:
    """Fetch `url` text, cache-first under the karyon cache. Raises DatasetUnavailable when neither works."""
    path = cache_dir() / cache_name
    if path.exists() and not refresh:
        return path.read_text()
    if not network_allowed():
        raise DatasetUnavailable("network disabled via KARYON_NO_NETWORK")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        text = urllib.request.urlopen(req, timeout=_TIMEOUT_S).read().decode("utf-8", "replace")
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
        raise DatasetUnavailable(f"cannot reach {url} and not cached: {e}") from e
    path.write_text(text)
    return text


def _parse_dictionary(text: str) -> dict[str, Protein]:
    out: dict[str, Protein] = {}
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        pid, seq = parts[0].strip(), parts[1].strip().upper()
        if pid and seq and set(seq) <= _AA:      # standard-AA only (the validity filter)
            out[pid] = Protein(pid, seq)
    return out


def _parse_actions(text: str, proteins: dict[str, Protein]) -> list[Pair]:
    out: list[Pair] = []
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        a, b, lab = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if a not in proteins or b not in proteins or a == b:
            continue
        try:
            label = int(float(lab))
        except ValueError:
            continue
        if label not in (0, 1):
            continue
        out.append(Pair(proteins[a], proteins[b], label))
    return out


def load_pairs(*, refresh: bool = False) -> list[Pair]:
    """The Guo-yeast PPI benchmark as a list of labelled `Pair`s. Cache-first, offline-skip."""
    proteins = _parse_dictionary(_fetch(_DICT_URL, "ppi_yeast.dictionary.tsv", refresh=refresh))
    if not proteins:
        raise DatasetUnavailable("parsed 0 proteins from the dictionary (format drift?)")
    pairs = _parse_actions(_fetch(_ACTIONS_URL, "ppi_yeast.actions.tsv", refresh=refresh), proteins)
    if not pairs:
        raise DatasetUnavailable("parsed 0 usable pairs from the actions table (format drift?)")
    return pairs


def proteins_in(pairs: list[Pair]) -> set[str]:
    return {p.a.pid for p in pairs} | {p.b.pid for p in pairs}


# --------------------------------------------------------------------------- #
# Splits: random (the leaky one everyone reports on) + protein-disjoint (the honest one).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Split:
    name: str
    train: list[Pair]
    test: list[Pair]

    @property
    def sizes(self) -> tuple[int, int]:
        return (len(self.train), len(self.test))


def random_split(pairs: list[Pair], *, seed: int = 0, test_frac: float = 0.2) -> Split:
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1.0 - test_frac))
    return Split(f"random(seed={seed})", [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]])


def protein_disjoint_split(pairs: list[Pair], *, seed: int = 0, test_frac: float = 0.2) -> Split:
    """The honest split (Park & Marcotte's C3): partition the PROTEIN set so no protein straddles. A pair
    is a test pair iff BOTH its proteins are test-proteins; a train pair iff BOTH are train-proteins; pairs
    spanning the two protein-sets (C2) are DROPPED. A pair needs both endpoints on one side, so to land
    ~test_frac of pairs in test we hold out ~sqrt(test_frac) of proteins (P(both) ≈ p²)."""
    prots = sorted(proteins_in(pairs))
    random.Random(seed).shuffle(prots)
    n_test = max(1, int(len(prots) * math.sqrt(test_frac)))
    test_prots = set(prots[:n_test])
    train, test = [], []
    for p in pairs:
        a_in, b_in = p.a.pid in test_prots, p.b.pid in test_prots
        if a_in and b_in:
            test.append(p)
        elif not a_in and not b_in:
            train.append(p)
        # else: a cross/C2 pair — dropped so no protein straddles
    return Split(f"protein_disjoint(seed={seed})", train, test)


def holdout_split(pairs: list[Pair], *, seed: int = 0, novel_frac: float = 0.3,
                  c1_test_frac: float = 0.2) -> Split:
    """The Park & Marcotte C1/C2/C3 design in one split. Hold out `novel_frac` of PROTEINS as unseen, then:
      * train  = `1 − c1_test_frac` of the pairs whose BOTH endpoints are seen (the fittable universe);
      * test   = the held-out seen-seen pairs (→ C1) + every pair touching a novel protein (→ C2/C3).
    The model's known proteins are exactly `proteins_in(train)`, so the *leakage class* of each test pair
    (both seen=C1 / one seen=C2 / neither seen=C3) is derived downstream by the contracts — the split just
    arranges that all three classes are populated. AUROC(C1) is what a random-split paper reports; AUROC(C3)
    is the honest Park & Marcotte number; the gap is the leakage inflation."""
    prots = sorted(proteins_in(pairs))
    rng = random.Random(seed)
    rng.shuffle(prots)
    novel = set(prots[:max(1, int(len(prots) * novel_frac))])
    both_seen = [p for p in pairs if p.a.pid not in novel and p.b.pid not in novel]
    touches_novel = [p for p in pairs if p.a.pid in novel or p.b.pid in novel]
    idx = list(range(len(both_seen)))
    rng.shuffle(idx)
    cut = int(len(both_seen) * (1.0 - c1_test_frac))
    train = [both_seen[i] for i in idx[:cut]]
    test = [both_seen[i] for i in idx[cut:]] + touches_novel
    return Split(f"holdout(seed={seed},novel={novel_frac})", train, test)


if __name__ == "__main__":
    print("Loading Guo-yeast sequence-based PPI benchmark\n")
    try:
        pairs = load_pairs()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    prots = proteins_in(pairs)
    pos = sum(1 for p in pairs if p.label == 1)
    lens = [len(pr) for pr in {p.a.pid: p.a.seq for p in pairs}.values()] + \
           [len(pr) for pr in {p.b.pid: p.b.seq for p in pairs}.values()]
    print(f"  pairs                   : {len(pairs)}  ({pos} positive / {len(pairs) - pos} negative)")
    print(f"  unique proteins         : {len(prots)}")

    rs = random_split(pairs)
    ds = protein_disjoint_split(pairs)
    rs_straddle = (proteins_in(rs.train) & proteins_in(rs.test))
    ds_straddle = (proteins_in(ds.train) & proteins_in(ds.test))
    print(f"\n  random split   train/test pairs: {rs.sizes}  "
          f"(proteins straddling: {len(rs_straddle)}  <- leaky by design)")
    print(f"  disjoint split train/test pairs: {ds.sizes}  "
          f"(proteins straddling: {len(ds_straddle)}  <- must be 0)")
    dpos = sum(1 for p in ds.test if p.label == 1)
    print(f"  disjoint test class balance    : {dpos}/{len(ds.test)} positive "
          f"({dpos / (len(ds.test) or 1):.0%})")
