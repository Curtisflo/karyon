"""uspto_data — cached loader for the USPTO-50k retrosynthesis benchmark.

The substrate for the **retrosynthesis benchmark-honesty** probe (karyon avenue 5, the cross-domain
one). USPTO-50k (Schneider/Liu) is the canonical public retrosynthesis benchmark: ~50k single-step
reactions extracted from US patents, each a product + its recorded reactants, labelled with one of 10
reaction superclasses. We consume **retrosim**'s `data_processed.csv` (Coley et al. 2017,
`github.com/connorcoley/retrosim`) — a login-free flat CSV, already RDKit-canonicalized and atom-mapped:

  * `prod_smiles` — the **canonical** product SMILES (clean, no maps) → exact/near-duplicate leakage is
    detectable with pure string ops, no RDKit;
  * `rxn_smiles`  — the atom-mapped `reactants>>products` reaction (the reactant side is the retriever's
    "answer"; map-stripped + `.`-sorted into an order-independent signature);
  * `class`       — the reaction superclass 1..10 (an independent, non-circular prediction target);
  * `id`          — the source US patent (provenance → a *patent-disjoint* clean split: a real, legible
    leakage source is two reactions from the same patent straddling train/test).

Why this matters: published retrosynthesis accuracy on USPTO-50k is documented to be **leakage-inflated**
(RetroXpert Top-1 70.4%→62.1% after a leak fix, RSC d4dd00007b). This loader is the substrate for a
legible reliability layer over that benchmark — a deterministic leakage audit + an honest held-out number.

Posture (the karyon charter): **stdlib-only + offline once `~/.cache/karyon/` is warm** — mirrors
`crispr_qc_data.py`/`promoter_data.py`. The 22.7 MB source CSV is parsed once into a slim cached table;
degrades to a typed `DatasetUnavailable` (the test SKIPs, never fails) when neither reachable nor cached.

    cd karyon/probe && python uspto_data.py            # smoke: fetch + summarize + split disjointness
"""

from __future__ import annotations
from .paths import cache_dir, network_allowed

import csv
import random
import re
import socket
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# retrosim's processed USPTO-50k — raw.githubusercontent is the canonical auth-free mirror.
_CSV_URL = "https://raw.githubusercontent.com/connorcoley/retrosim/master/retrosim/data/data_processed.csv"
_UA = "karyon-bio/1 (+https://github.com/connorcoley/retrosim)"
_TIMEOUT_S = 180
_ATOM_MAP = re.compile(r":\d+")           # ':12' inside '[CH2:12]' — the atom-map tag


class DatasetUnavailable(RuntimeError):
    """The USPTO-50k CSV could not be fetched/parsed and is not cached → SKIP."""


@dataclass(frozen=True)
class Reaction:
    """One single-step retrosynthesis example (product ← reactants), with its patent + class."""

    rid: str                # source US patent id (provenance; the patent-disjoint split key)
    klass: int              # USPTO-50k reaction superclass, 1..10 (the independent class target)
    product: str            # canonical product SMILES (clean) — the similarity / leakage key
    reactant_sig: str       # map-stripped, '.'-sorted reactant-set signature (the retriever's answer)
    rxn_smiles: str = ""    # the atom-mapped `reactants>>products` (needed by the RDKit template arm)


# --------------------------------------------------------------------------- #
# Cache plumbing (~/.cache/karyon/, gitignored — mirrors crispr_qc_data.py).
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    return here.parents[2]


def _cache_path() -> Path:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "uspto50k.csv"


# --------------------------------------------------------------------------- #
# Fetch + parse.
# --------------------------------------------------------------------------- #
def _fetch(url: str = _CSV_URL) -> bytes:
    if not network_allowed():
        raise DatasetUnavailable("network disabled via KARYON_NO_NETWORK")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        return urllib.request.urlopen(req, timeout=_TIMEOUT_S).read()
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        raise DatasetUnavailable(f"cannot reach the USPTO-50k CSV ({url}): {e}") from e


def reactant_signature(rxn_smiles: str) -> str:
    """Order-independent reactant-set signature from an atom-mapped `reactants>>products` SMILES.

    Takes the reactant side (before any reagent `>` or the `>>`), strips atom-map tags, and sorts the
    `.`-separated molecules. Within USPTO-50k (one canonicalizer wrote every row) this is a stable string
    key for "same recorded reactants"; it is NOT a cross-source canonical form (an honest bound: two
    chemically-identical reactions written with different atom orderings would differ → duplicate detection
    UNDER-counts, the safe direction)."""
    lhs = rxn_smiles.split(">>")[0].split(">")[0]
    mols = _ATOM_MAP.sub("", lhs).split(".")
    return ".".join(sorted(m for m in mols if m))


def _parse(raw: bytes) -> list[Reaction]:
    text = raw.decode("utf-8", errors="replace").splitlines()
    out: list[Reaction] = []
    for row in csv.DictReader(text):
        if (row.get("keep") or "True") == "False":      # retrosim's validity flag
            continue
        prod = (row.get("prod_smiles") or "").strip()
        rxn = (row.get("rxn_smiles") or "").strip()
        rid = (row.get("id") or "").strip()
        if not prod or ">>" not in rxn:
            continue
        try:
            klass = int(row["class"])
        except (KeyError, ValueError, TypeError):
            continue
        sig = reactant_signature(rxn)
        if not sig:
            continue
        out.append(Reaction(rid=rid, klass=klass, product=prod, reactant_sig=sig, rxn_smiles=rxn))
    return out


# --------------------------------------------------------------------------- #
# Cache read/write (the slim 4-column table — never the 22.7 MB source).
# --------------------------------------------------------------------------- #
_FIELDS = ["rid", "klass", "product", "reactant_sig", "rxn_smiles"]


def _write_cache(path: Path, rxns: list[Reaction]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_FIELDS)
        for r in rxns:
            w.writerow([r.rid, r.klass, r.product, r.reactant_sig, r.rxn_smiles])


def _read_cache(path: Path) -> list[Reaction]:
    out = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Reaction(row["rid"], int(row["klass"]), row["product"], row["reactant_sig"],
                                row.get("rxn_smiles", "")))
    return out


def load_reactions(*, refresh: bool = False, limit: int | None = None) -> list[Reaction]:
    """The USPTO-50k reactions (product + reactant signature + class + patent id).

    Reads `~/.cache/karyon/uspto50k.csv` if present (offline-friendly); otherwise fetches the 22.7 MB source
    CSV, parses it stdlib-only, caches the slim table, and returns. `limit` truncates (deterministic, head)
    for fast tests — leave None for the real run (subsampling changes duplication rates). Raises
    `DatasetUnavailable` when neither reachable nor cached."""
    path = _cache_path()
    if path.exists() and not refresh:
        rxns = _read_cache(path)
        print(f"  [cache] {len(rxns)} USPTO-50k reactions from {path.name}")
    else:
        rxns = _parse(_fetch())
        if not rxns:
            raise DatasetUnavailable("parsed 0 usable reactions (CSV format drift?)")
        _write_cache(path, rxns)
        print(f"  [cache] wrote {len(rxns)} USPTO-50k reactions -> {path.name}")
    return rxns[:limit] if limit else rxns


# --------------------------------------------------------------------------- #
# The two pre-registered splits.
#   random_split        — the "standard-style" split everyone reports on; it LEAKS (duplicates and
#                         same-patent reactions scatter across train/test). This is the inflated baseline.
#   patent_disjoint_split — assigns whole patents to one side, killing the same-patent leakage source.
# The third evaluation condition (the "leakage-free partition" of the random test set) is computed by the
# honesty layer, since it needs the train index.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Split:
    name: str
    train: list[Reaction]
    test: list[Reaction]

    @property
    def sizes(self) -> tuple[int, int]:
        return (len(self.train), len(self.test))


def random_split(rxns: list[Reaction], *, seed: int = 0, test_frac: float = 0.1) -> Split:
    """A seeded uniform shuffle then a test_frac tail — the standard-style (leaky) split."""
    idx = list(range(len(rxns)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1.0 - test_frac))
    train = [rxns[i] for i in idx[:cut]]
    test = [rxns[i] for i in idx[cut:]]
    return Split(f"random(seed={seed})", train, test)


def patent_disjoint_split(rxns: list[Reaction], *, seed: int = 0, test_frac: float = 0.1) -> Split:
    """Assign whole patents (by `rid`) to train/test so no patent straddles the split — removes the
    same-patent leakage source. Reactions with a blank `rid` are pooled under one sentinel patent."""
    by_patent: dict[str, list[Reaction]] = {}
    for r in rxns:
        by_patent.setdefault(r.rid or "__nopatent__", []).append(r)
    patents = list(by_patent)
    random.Random(seed).shuffle(patents)
    target_test = int(len(rxns) * test_frac)
    test: list[Reaction] = []
    train: list[Reaction] = []
    for p in patents:
        (test if len(test) < target_test else train).extend(by_patent[p])
    return Split(f"patent_disjoint(seed={seed})", train, test)


if __name__ == "__main__":
    print("Loading the USPTO-50k retrosynthesis benchmark (retrosim data_processed.csv)\n")
    try:
        rxns = load_reactions()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    patents = {r.rid for r in rxns}
    klass = Counter(r.klass for r in rxns)
    dup_products = sum(c - 1 for c in Counter(r.product for r in rxns).values() if c > 1)
    print(f"\n  reactions               : {len(rxns)}")
    print(f"  distinct patents        : {len(patents)}  (mean {len(rxns) / len(patents):.1f} rxns/patent)")
    print(f"  distinct products       : {len(set(r.product for r in rxns))}  "
          f"({dup_products} are repeats of another row's product — intrinsic duplication)")
    print(f"  reaction classes (1..10): {dict(sorted(klass.items()))}")

    rs = random_split(rxns, seed=0)
    ps = patent_disjoint_split(rxns, seed=0)
    print(f"\n  random split    train/test: {rs.sizes}")
    print(f"  patent-disjoint train/test: {ps.sizes}")
    # disjointness invariant: no patent straddles the patent-disjoint split.
    straddle = {r.rid for r in ps.train} & {r.rid for r in ps.test}
    print(f"  patents straddling the disjoint split: {len(straddle)}  (must be 0)")
