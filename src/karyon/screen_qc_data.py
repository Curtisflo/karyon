"""screen_qc_data — cached loader for a BULK pooled CRISPR screen's raw sgRNA counts + a gold standard.

The substrate for the screen-reliability QC probe (the count-level sibling of `crispr_qc_data.py`,
which loads sequence→activity). Where the activity probe asked "is this *guide* effective?", this one
asks "is this screen's *non-hit* trustworthy, or an under-powered silent failure?" — and that question
lives in the raw read **counts**, not in a derived activity/phenotype score.

  * **Screen counts** — MAGeCK's own demo `leukemia.new.csv` (T. Wang et al., *Science* 2014): a
    negative-selection (dropout) screen in two human leukemia lines. One row per sgRNA:
    `sgRNA, Gene, HL60.initial, KBM7.initial, HL60.final, KBM7.final` (raw integer counts). Essential
    genes deplete initial→final. ≈10 sgRNAs/gene → supports gene-level "is this gene powered?" QC.
  * **Gold standard** — Hart CEGv2 (core-essential) + NEGv1 (non-essential) reference gene sets
    (github.com/hart-lab/bagel). EXTERNAL truth, independent of the counts: a CEGv2 gene the screen
    fails to call is a real, gold-standard silent failure; a NEGv1 gene should *not* be flagged.

Why a bulk screen and NOT single-cell/Replogle Perturb-seq: Replogle raw counts are GB-scale `h5ad`
(needs scanpy/anndata) — off the probes' stdlib/offline posture (same call `crispr_qc_data.py` makes).
A bulk screen's counts are a small flat CSV: exactly (sgRNA, gene, counts) per guide, stdlib-parseable.
MAGeCK is the incumbent for this data; SCEPTRE/Replogle is the documented heavier fast-follow.

  * fetches the demo counts CSV (SourceForge, auth-free) + the two reference lists (GitHub raw);
  * parses stdlib-only and caches the small flat tables to `~/.cache/karyon/` (gitignored);
  * degrades to a typed `DatasetUnavailable` (the test SKIPs, never fails, offline).

    cd karyon/probe && python screen_qc_data.py     # smoke: fetch + summarize + control scan + n
"""

from __future__ import annotations
from .paths import cache_dir, network_allowed

import csv
import gzip
import math
import random
import socket
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

# MAGeCK demo counts (the GitHub demo path is dead; SourceForge is the live auth-free host).
_COUNTS_URL = "https://sourceforge.net/projects/mageck/files/example/leukemia.new.csv/download"
# Hart lab BAGEL reference gene sets (tab-separated, header GENE\tHGNC_ID\tENTREZ_ID; blank lines exist).
_CEG_URL = "https://raw.githubusercontent.com/hart-lab/bagel/master/CEGv2.txt"
_NEG_URL = "https://raw.githubusercontent.com/hart-lab/bagel/master/NEGv1.txt"
_UA = "karyon-bio/1 (+https://sourceforge.net/projects/mageck/)"
_TIMEOUT_S = 180

# A sample is an "initial" (T0 / plasmid / control) arm if its name carries one of these tokens;
# everything else is a "final" (post-selection) arm. The dropout contrast is initial -> final.
_INITIAL_TOKENS = ("initial", "plasmid", "t0", "control", "ctrl", "dropout_t0")

# Non-targeting / safe-harbor control sentinels (mirrors crispr_qc_data._sgid_is_ntc). Most Wang-era
# libraries have few/none in the demo table, so the NEGv1-as-null path is the realistic fallback.
_CONTROL_TOKENS = ("non-targeting", "nontargeting", "negative", "control", "_ntc", "safe", "olfr")


class DatasetUnavailable(RuntimeError):
    """A required file could not be fetched/parsed and is not cached → SKIP (never fail offline)."""


@dataclass(frozen=True)
class CountRow:
    """One sgRNA's raw read counts across the screen's samples."""

    sgrna: str
    gene: str
    counts: dict[str, int]      # {sample_name: raw_count}, e.g. {"HL60.initial": 312, ...}


@dataclass(frozen=True)
class ScreenCounts:
    """A bulk screen: per-sgRNA counts + the initial→final dropout contrast (which samples are which)."""

    rows: list[CountRow]
    samples: list[str]          # all count columns, in file order
    initial: list[str]          # the T0/control arm(s)
    final: list[str]            # the post-selection arm(s)

    def genes(self) -> set[str]:
        return {r.gene for r in self.rows}


class ReferenceSets(NamedTuple):
    """Hart gold-standard gene sets, already intersected with the screen's assayed genes."""

    essential: frozenset[str]       # CEGv2 ∩ screen
    nonessential: frozenset[str]    # NEGv1 ∩ screen


# --------------------------------------------------------------------------- #
# Cache plumbing (~/.cache/karyon/, gitignored — mirrors crispr_qc_data.py).
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    return here.parents[2]


def _cache_dir() -> Path:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _counts_cache() -> Path:
    return _cache_dir() / "screen_qc_counts.csv"


def _ref_cache(name: str) -> Path:
    return _cache_dir() / f"screen_qc_{name}.txt"


# --------------------------------------------------------------------------- #
# Fetch.
# --------------------------------------------------------------------------- #
def _fetch_bytes(url: str) -> bytes:
    if not network_allowed():
        raise DatasetUnavailable("network disabled via KARYON_NO_NETWORK")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        return urllib.request.urlopen(req, timeout=_TIMEOUT_S).read()
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        raise DatasetUnavailable(f"cannot reach {url}: {e}") from e


def _fetch_text(url: str) -> str:
    """Fetch text, transparently gunzipping a `.gz` URL (the Hart TKO read-count tables are gzipped)."""
    raw = _fetch_bytes(url)
    if url.endswith(".gz"):
        try:
            raw = gzip.decompress(raw)
        except OSError as e:
            raise DatasetUnavailable(f"could not gunzip {url}: {e}") from e
    return raw.decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# Counts: parse + cache.
# --------------------------------------------------------------------------- #
def _classify_samples(samples: list[str]) -> tuple[list[str], list[str]]:
    initial = [s for s in samples if any(t in s.lower() for t in _INITIAL_TOKENS)]
    final = [s for s in samples if s not in initial]
    return initial, final


def _parse_counts(text: str) -> ScreenCounts:
    delim = "\t" if ("\t" in text.splitlines()[0] and "," not in text.splitlines()[0]) else ","
    reader = csv.reader(text.splitlines(), delimiter=delim)
    header = next(reader, None)
    if not header or len(header) < 3:
        raise DatasetUnavailable("counts file has no usable header (got a redirect/HTML page?)")
    # Column 0 = sgRNA id, column 1 = gene, the rest = per-sample raw counts.
    samples = [h.strip() for h in header[2:]]
    rows: list[CountRow] = []
    for rec in reader:
        if len(rec) < 3 or not rec[0].strip():
            continue
        try:
            counts = {s: int(float(v)) for s, v in zip(samples, rec[2:])}
        except (ValueError, TypeError):
            continue
        # Uppercase gene symbols so they match the (uppercase) Hart reference sets — otherwise a
        # mixed-case symbol is silently dropped from the CEGv2/NEGv1 intersection downstream.
        rows.append(CountRow(rec[0].strip(), rec[1].strip().upper(), counts))
    if not rows:
        raise DatasetUnavailable("parsed 0 sgRNA count rows (format drift / not a counts CSV?)")
    initial, final = _classify_samples(samples)
    if not initial or not final:
        raise DatasetUnavailable(
            f"could not split samples into initial/final (samples={samples})")
    return ScreenCounts(rows, samples, initial, final)


def _write_counts_cache(path: Path, sc: ScreenCounts) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sgRNA", "Gene", *sc.samples])
        for r in sc.rows:
            w.writerow([r.sgrna, r.gene, *(r.counts[s] for s in sc.samples)])


def _read_counts_cache(path: Path) -> ScreenCounts:
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        samples = header[2:]
        rows = [CountRow(rec[0], rec[1].upper(), {s: int(v) for s, v in zip(samples, rec[2:])})
                for rec in reader if len(rec) >= 3]
    initial, final = _classify_samples(samples)
    return ScreenCounts(rows, samples, initial, final)


def load_counts(*, refresh: bool = False) -> ScreenCounts:
    """The bulk screen's raw sgRNA counts. Reads `~/.cache/karyon/screen_qc_counts.csv` if present;
    otherwise fetches the demo CSV, parses stdlib-only, caches the flat table, and returns. Raises
    `DatasetUnavailable` when neither reachable nor cached."""
    path = _counts_cache()
    if path.exists() and not refresh:
        sc = _read_counts_cache(path)
        print(f"  [cache] {len(sc.rows)} sgRNAs / {len(sc.genes())} genes from "
              f"{path.name}")
        return sc
    sc = _parse_counts(_fetch_text(_COUNTS_URL))
    _write_counts_cache(path, sc)
    print(f"  [cache] wrote {len(sc.rows)} sgRNAs / {len(sc.genes())} genes -> "
          f"{path.name}")
    return sc


def _restrict_finals(sc: ScreenCounts, token: str) -> ScreenCounts:
    """Keep only final samples whose name contains `token` (case-insensitive) — e.g. one timepoint
    ('T6') of a multi-timepoint screen — plus all initial samples. Lets a probe select an early/weak
    dropout window (the naturally-low-power arm) from a screen with several timepoints."""
    tok = token.lower()
    finals = [s for s in sc.final if tok in s.lower()]
    if not finals:
        raise DatasetUnavailable(f"no final sample contains '{token}' (finals={sc.final})")
    keep = list(sc.initial) + finals
    rows = [CountRow(r.sgrna, r.gene, {s: r.counts[s] for s in keep}) for r in sc.rows]
    return ScreenCounts(rows, keep, list(sc.initial), finals)


def load_named_screen(url: str, cache_name: str, *, finals_contains: str | None = None,
                      refresh: bool = False) -> ScreenCounts:
    """Fetch + cache ANY bulk screen's raw counts (gz auto-detected by `.gz`) in the same (sgRNA, gene,
    counts) shape as the demo, optionally restricting the final arm to one timepoint token. Lets the
    avenue probes point the SAME baseline + QC machinery at a different screen (e.g. a naturally
    low-power one). The full table is cached; the timepoint restriction is applied on every load."""
    path = _cache_dir() / f"screen_{cache_name}.csv"
    if path.exists() and not refresh:
        sc = _read_counts_cache(path)
    else:
        sc = _parse_counts(_fetch_text(url))
        _write_counts_cache(path, sc)
        print(f"  [cache] wrote {len(sc.rows)} sgRNAs / {len(sc.genes())} genes -> "
              f"{path.name}")
    return _restrict_finals(sc, finals_contains) if finals_contains else sc


# --------------------------------------------------------------------------- #
# Reference gene sets: parse + cache.
# --------------------------------------------------------------------------- #
def _parse_ref(text: str) -> set[str]:
    """Hart reference list: tab-separated, header row, gene symbol in column 0; skip blank lines."""
    genes: set[str] = set()
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or i == 0:                  # skip blank lines and the GENE\tHGNC_ID header
            continue
        sym = line.split("\t")[0].strip().upper()
        if sym:
            genes.add(sym)
    return genes


def _load_one_ref(name: str, url: str, *, refresh: bool) -> set[str]:
    path = _ref_cache(name)
    if path.exists() and not refresh:
        return {ln.strip().upper() for ln in path.read_text().splitlines() if ln.strip()}
    genes = _parse_ref(_fetch_text(url))
    if not genes:
        raise DatasetUnavailable(f"parsed 0 genes from the {name} reference list")
    path.write_text("\n".join(sorted(genes)) + "\n")
    return genes


def load_references(screen_genes: set[str], *, refresh: bool = False) -> ReferenceSets:
    """Hart CEGv2 (essential) + NEGv1 (non-essential), each intersected with the screen's assayed
    genes (a gene not in the library can't be flagged — avoids universe leakage in the metrics)."""
    ceg = _load_one_ref("ceg", _CEG_URL, refresh=refresh)
    neg = _load_one_ref("neg", _NEG_URL, refresh=refresh)
    su = {g.upper() for g in screen_genes}
    return ReferenceSets(frozenset(ceg & su), frozenset(neg & su))


def detect_controls(rows: list[CountRow]) -> list[CountRow]:
    """sgRNAs that look like non-targeting / safe-harbor controls (define a clean null band if present;
    most bulk demo libraries have few/none, in which case the QC layer falls back to a NEGv1 null)."""
    out = []
    for r in rows:
        tag = f"{r.sgrna} {r.gene}".lower()
        if any(t in tag for t in _CONTROL_TOKENS):
            out.append(r)
    return out


def downsample_counts(sc: ScreenCounts, target_genes: set[str], target_initial: float,
                      seed: int = 0) -> ScreenCounts:
    """Return a copy of the screen with `target_genes`' guides read-depth-THINNED toward
    `target_initial` initial reads, PRESERVING each guide's fold-change (so the genes stay truly
    essential — just under-sequenced). Per guide p = target_initial / its mean initial count (≤1),
    applied to all of its samples via a normal-approximation binomial thinning. Used to stress-test the
    count-floor contracts on a library that is otherwise well-sequenced (so they normally lie dormant)."""
    rng = random.Random(seed)

    def thin(c: int, p: float) -> int:
        if p >= 1.0 or c <= 0:
            return c
        mean, var = c * p, c * p * (1.0 - p)
        return max(0, round(rng.gauss(mean, math.sqrt(var)) if var > 0 else mean))

    rows = []
    for r in sc.rows:
        if r.gene in target_genes:
            init_mean = statistics.fmean([r.counts[s] for s in sc.initial]) or 1.0
            p = min(1.0, target_initial / init_mean)
            rows.append(CountRow(r.sgrna, r.gene, {s: thin(r.counts[s], p) for s in sc.samples}))
        else:
            rows.append(r)
    return ScreenCounts(rows, sc.samples, sc.initial, sc.final)


def subsample_guides(sc: ScreenCounts, target_genes: set[str], k: int,
                     seed: int = 0) -> ScreenCounts:
    """Return a copy of the screen with `target_genes`' guides RANDOMLY SUBSAMPLED to at most `k`
    sgRNAs per gene — counts kept at FULL DEPTH, only the NUMBER of independent guides drops.

    This is the count-level sibling of `downsample_counts`, built for a different job. Read-depth
    thinning preserves each guide's fold-change, so the baseline's count-moderation absorbs it and the
    gene stays just as callable — useless for creating baseline head-room. Dropping *guides* instead
    attacks the axis the gene-level call actually uses: fewer sgRNAs ⇒ a weaker per-gene rank-sum
    (`screen_baseline.call_genes`) ⇒ marginal genes genuinely fall below the FDR cut into the non-hit
    pile. That is the honest lower-power regime the Q4 lift test needs.

    Genes outside `target_genes`, and target genes with ≤ k guides, pass through unchanged. Gene
    iteration is sorted so the draw is deterministic in `seed` regardless of row order."""
    rng = random.Random(seed)
    idx_by_gene: dict[str, list[int]] = {}
    for i, r in enumerate(sc.rows):
        idx_by_gene.setdefault(r.gene, []).append(i)
    keep: set[int] = set()
    for gene in sorted(idx_by_gene):
        idxs = idx_by_gene[gene]
        if gene in target_genes and len(idxs) > k:
            keep.update(rng.sample(idxs, k))
        else:
            keep.update(idxs)
    rows = [r for i, r in enumerate(sc.rows) if i in keep]
    return ScreenCounts(rows, sc.samples, sc.initial, sc.final)


if __name__ == "__main__":
    print("Loading the bulk CRISPR screen counts (MAGeCK demo: T. Wang 2014 leukemia dropout)\n")
    try:
        sc = load_counts()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    gene_counts: dict[str, int] = {}
    for r in sc.rows:
        gene_counts[r.gene] = gene_counts.get(r.gene, 0) + 1
    guides_per_gene = statistics.fmean(gene_counts.values())
    print(f"  sgRNAs                 : {len(sc.rows)}")
    print(f"  genes                  : {len(sc.genes())}  (mean {guides_per_gene:.1f} sgRNAs/gene)")
    print(f"  samples                : {sc.samples}")
    print(f"  initial / final        : {sc.initial}  ->  {sc.final}")
    for s in sc.samples:
        col = [r.counts[s] for r in sc.rows]
        zeros = sum(1 for c in col if c == 0)
        print(f"    {s:<16}: min {min(col)} / med {int(statistics.median(col))} / max {max(col)} "
              f"  ({zeros} zeros, {zeros / len(col):.1%})")

    controls = detect_controls(sc.rows)
    print(f"  control-like sgRNAs    : {len(controls)}  "
          f"({'NEGv1-as-null fallback' if not controls else 'usable NTC null band'})")

    print("\nIntersecting Hart CEGv2 / NEGv1 reference sets with the screen's gene universe\n")
    try:
        refs = load_references(sc.genes())
    except DatasetUnavailable as e:
        print(f"SKIP (references) — {e}")
        raise SystemExit(0)
    print(f"  CEGv2 ∩ screen (essential)    : {len(refs.essential)}")
    print(f"  NEGv1 ∩ screen (non-essential): {len(refs.nonessential)}")
    overlap = refs.essential & refs.nonessential
    print(f"  CEGv2 ∩ NEGv1 (should be ~0)   : {len(overlap)}")
