"""utr5_data — a cached loader for the Optimus 5-Prime 5'UTR MPRA (mean ribosome load) dataset.

A THIRD substrate for the probes — eukaryotic translation efficiency, the highest-interest thread
("mRNA drugs"). Sample et al. (*Nat. Biotechnol.* 2019, "Optimus 5-Prime") measured a massively
parallel polysome-profiling assay over a random-5'UTR library: ~280k random 50-mer 5'UTRs upstream
of eGFP, each with a **mean ribosome load (MRL)** — the polysome-weighted mean number of ribosomes
on the transcript, i.e. a direct readout of translation efficiency. This loader gets that flat
sequence→MRL table onto the desk, cheaply and reproducibly:

  * The canonical deposit is GEO **GSE114002**; the main training library is the sample
    `GSM3130435_egfp_unmod_1.csv.gz` — random 5'UTRs, eGFP, unmodified, replicate 1 (the set Optimus
    trains on). NCBI serves it over HTTPS with a stable per-sample suppl path.
  * The file is a 63 MB gzip and the rows are **pre-sorted by `total_reads` (read depth) descending**
    — read depth IS label reliability (the paper keeps the top ~280k by depth and discards the noisy
    tail). So a *streamed prefix* of the first `n` rows is precisely the `n` HIGHEST-confidence
    measurements — an honest, principled $0 subsample, not an arbitrary one. We stream-decompress and
    stop after `n` usable rows; we never hold or write the whole file.
  * Columns are resolved BY HEADER NAME (`utr` → sequence, `rl` → MRL, `total_reads` → depth), so a
    re-export with shuffled columns still loads. The kept (sequence, MRL, depth) triples are cached to
    `~/.cache/karyon/` (gitignored); a second run — and every offline run — reads the cache with no network.

Network failure raises `DatasetUnavailable` so callers (the test) can SKIP rather than fail.

    python -m karyon.utr5_data        # smoke: fetch a subsample, print summary
"""

from __future__ import annotations
from .paths import cache_dir, network_allowed

import csv
import gzip
import io
import json
import socket
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# GEO GSE114002 / GSM3130435 — random 5'UTR library, eGFP unmodified rep 1 (the Optimus training set).
DATASET_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130435/suppl/"
    "GSM3130435_egfp_unmod_1.csv.gz"
)
_UA = "karyon-bio-benchmark/1 (+https://github.com/pjsample/human_5utr_modeling)"
_TIMEOUT_S = 120
UTR_LEN = 50                          # the random library is fixed-length 50-mers
_DEFAULT_N = 20_000                   # the paper's held-out eval size; the top-depth (cleanest) slice
_COL = {"utr": "utr", "rl": "rl", "total_reads": "total_reads"}   # field -> header name


class DatasetUnavailable(RuntimeError):
    """The 5'UTR dataset could not be fetched (offline / network error) and is not cached → SKIP."""


@dataclass(frozen=True)
class Record:
    """One measured 5'UTR — sequence, mean ribosome load, and read depth (a confidence weight)."""

    utr: str                 # the 50-nt 5'UTR sequence
    mrl: float               # measured mean ribosome load (translation efficiency); ~2..9
    total_reads: float       # read depth — higher = more reliable (the paper's QC axis)


# --------------------------------------------------------------------------- #
# Cache plumbing (~/.cache/karyon/, gitignored — mirrors toehold_data.py / emopec_data.py).
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    return here.parents[2]


def _cache_path(n: int) -> Path:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"utr5_egfp_unmod1_n{n}.csv"


# --------------------------------------------------------------------------- #
# Fetch (stream-decompress a prefix; offline degrades to DatasetUnavailable).
# --------------------------------------------------------------------------- #
def _usable(seq: str, mrl) -> bool:
    return (isinstance(seq, str) and len(seq) == UTR_LEN and set(seq) <= set("ACGT")
            and isinstance(mrl, (int, float)))


def _fetch_prefix(n: int) -> list[Record]:
    """Stream-decompress the gzip and keep the first `n` usable rows (the highest read depth).

    The file is sorted by `total_reads` descending, so the prefix is the cleanest subsample; we stop
    as soon as `n` usable rows are in hand and never decompress the whole 63 MB."""
    if not network_allowed():
        raise DatasetUnavailable("network disabled via KARYON_NO_NETWORK")
    req = urllib.request.Request(DATASET_URL, headers={"User-Agent": _UA})
    try:
        resp = urllib.request.urlopen(req, timeout=_TIMEOUT_S)
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        raise DatasetUnavailable(f"cannot reach the 5'UTR dataset ({DATASET_URL}): {e}") from e
    try:
        gz = gzip.GzipFile(fileobj=resp)
        reader = csv.reader(io.TextIOWrapper(gz, encoding="utf-8", errors="replace"))
        header = next(reader, None)
        if not header:
            raise DatasetUnavailable("5'UTR dataset is empty (no header)")
        pos = {name: i for i, name in enumerate(header)}
        missing = [h for h in _COL.values() if h not in pos]
        if missing:
            raise DatasetUnavailable(f"5'UTR header missing expected columns: {missing}")
        i_utr, i_rl, i_tr = pos[_COL["utr"]], pos[_COL["rl"]], pos[_COL["total_reads"]]
        recs: list[Record] = []
        for fields in reader:
            if len(fields) <= max(i_utr, i_rl, i_tr):
                continue
            seq = fields[i_utr].strip().upper()
            try:
                mrl = float(fields[i_rl])
                depth = float(fields[i_tr])
            except ValueError:
                continue
            if not _usable(seq, mrl):
                continue
            recs.append(Record(seq, mrl, depth))
            if len(recs) >= n:
                break
    except (urllib.error.URLError, socket.timeout, ConnectionError, EOFError, OSError) as e:
        raise DatasetUnavailable(f"5'UTR stream failed mid-read: {e}") from e
    if not recs:
        raise DatasetUnavailable("fetched 0 usable 5'UTR rows (parse/format drift?)")
    return recs


# --------------------------------------------------------------------------- #
# Cache read/write (the small kept subset only — never the raw 63 MB).
# --------------------------------------------------------------------------- #
def _write_cache(path: Path, recs: list[Record]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["utr", "mrl", "total_reads"])
        for r in recs:
            w.writerow([r.utr, r.mrl, r.total_reads])


def _read_cache(path: Path) -> list[Record]:
    out: list[Record] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Record(row["utr"], float(row["mrl"]), float(row["total_reads"])))
    return out


def load_records(*, n: int = _DEFAULT_N, refresh: bool = False) -> list[Record]:
    """The top-`n` (highest read depth = cleanest) measured 5'UTR→MRL records.

    Reads `~/.cache/karyon/utr5_egfp_unmod1_n{n}.csv` if present (offline-friendly); otherwise stream-fetches
    the prefix, caches, and returns. Raises `DatasetUnavailable` when neither reachable nor cached."""
    path = _cache_path(n)
    if path.exists() and not refresh:
        recs = _read_cache(path)
        print(f"  [cache] {len(recs)} 5'UTR records from {path.name}")
        return recs
    recs = _fetch_prefix(n)
    _write_cache(path, recs)
    print(f"  [cache] wrote {len(recs)} records -> {path.name} "
          f"(top-{n} by read depth from GSM3130435)")
    return recs


if __name__ == "__main__":
    print(f"Loading Optimus 5-Prime 5'UTR MRL dataset (GEO GSE114002 / GSM3130435)\n"
          f"  from {DATASET_URL}\n  (63 MB gzip, sorted by read depth; streaming a top-N prefix, "
          f"nothing committed)\n")
    try:
        rows = load_records()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    mrl = [r.mrl for r in rows]
    print(f"\n  records              : {len(rows)}")
    print(f"  UTR length (all ==50): {set(len(r.utr) for r in rows)}")
    print(f"  MRL min/med/max      : {min(mrl):.3f} / {statistics.median(mrl):.3f} / {max(mrl):.3f}")
    depth = [r.total_reads for r in rows]
    print(f"  read depth min/med/max: {min(depth):.0f} / {statistics.median(depth):.0f} / {max(depth):.0f}")
    top = sorted(rows, key=lambda r: r.mrl, reverse=True)[:5]
    print(f"  top-5 MRL UTRs       : {[(r.utr[:12] + '...', round(r.mrl, 2)) for r in top]}")
