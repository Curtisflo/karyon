"""stability_data — a cached loader for a $0 slice of the Tsuboyama/Rocklin mega-scale protein
stability dataset (a PROTEIN substrate for the discovery-lever probes).

The point of this loader is the data-scout finding made operational. The full deposit is large
(Zenodo record 7992926: `Processed_K50_dG_datasets.zip` is **1.0 GB**; the HuggingFace mirror
`RosettaCommons/MegaScale` is multi-GB parquet behind the `datasets` library). Neither is a $0
stdlib fetch. The cheap path is the HuggingFace **dataset-viewer** REST API
(`datasets-server.huggingface.co/rows`), which serves the parquet rows as **paginated JSON over
HTTP** — no `datasets` dependency, no multi-GB download, stdlib `urllib` only.

The representative SUBSET we take is the protein analogue of EMOPEC's bounded 6-nt SD space: the
**complete single-mutant deep-mutational-scan (DMS) of ONE protein domain** — all single-substitution
variants of a fixed wild-type sequence, with their measured folding ΔG (kcal/mol; higher = more
stable). The `dataset3_single` config is sorted by `WT_name`, so one domain's variants are a
contiguous block; we page from offset 0 of a split and collect the first domain encountered until it
ends (then stop). This gives a clean, single-length, bounded mutational landscape — the right shape
for the discovery-lever test, and small enough to cache (~a few hundred KB).

  * fetches `/rows` pages with generous 429 backoff (the viewer is rate-limited; over-fetch → HTTP 429);
  * DEDUPES by mutation (the raw deposit repeats each measurement many times, one row per read/barcode);
  * keeps single substitutions + the wild-type row, joins the measured ΔG with the dataset's own
    `ddG_ML` ML prediction (ThermoMPNN-class, for a context head-to-head — NOT a fair held-out baseline);
  * caches the small per-domain table to `~/.cache/karyon/` (gitignored), offline-skip via DatasetUnavailable.

    python -m karyon.stability_data        # smoke: fetch + summarize one domain's DMS
"""

from __future__ import annotations
from .paths import cache_dir, network_allowed

import csv
import json
import os
import re
import socket
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_HOST = "https://datasets-server.huggingface.co"
_DATASET = "RosettaCommons/MegaScale"
_CONFIG = "dataset3_single"          # sorted by WT_name; ΔG ceiling-trimmed (WT dG < 4.75 kcal/mol)
_SPLIT = "test"                      # any split works; we just take the first contiguous domain block
_UA = "karyon-bio-benchmark/1 (+https://huggingface.co/datasets/RosettaCommons/MegaScale)"
_TIMEOUT_S = 60
_PAGE = 100                          # rows per /rows request (the viewer's max page is 100)
_MAX_PAGES = 120                     # safety cap (one small domain's block is well under this)
_POLITE_S = 2.0                      # gap between page fetches (the viewer rate-limits hard)
AA = "ACDEFGHIKLMNPQRSTVWY"          # the 20-amino-acid alphabet (NOT nucleotides)
_SINGLE_MUT = re.compile(r"[A-Z]\d+[A-Z]")   # e.g. "D1Q" — one substitution


class DatasetUnavailable(RuntimeError):
    """MegaScale rows could not be fetched (offline / rate-limited / network error) and are not
    cached → SKIP (never a test failure)."""


@dataclass(frozen=True)
class Record:
    """One protein-domain variant with measured folding stability."""

    wt_name: str                  # the wild-type domain this variant belongs to
    mut_type: str                 # "wt" or a single substitution like "D1Q"
    aa_seq: str                   # the full variant amino-acid sequence (fixed length within a domain)
    deltaG: float                 # measured folding ΔG, kcal/mol — HIGHER = MORE STABLE
    ddg_ml: float | None          # the deposit's own ML ΔΔG prediction (context only; not held-out)


# --------------------------------------------------------------------------- #
# Cache plumbing (~/.cache/karyon/, gitignored — mirrors emopec_data.py).
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
    return d / "stability.csv"


# --------------------------------------------------------------------------- #
# Fetch (HF dataset-viewer /rows, with 429 backoff).
# --------------------------------------------------------------------------- #
def _get(url: str) -> dict:
    """One `/rows` GET. Retries ONLY on HTTP 429 (the viewer rate-limits hard); a connection failure
    means offline → raise at once so the offline-skip is fast (mirrors the sibling loaders)."""
    for attempt in range(4):
        if not network_allowed():
            raise DatasetUnavailable("network disabled via KARYON_NO_NETWORK")
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=_TIMEOUT_S).read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:           # rate-limited → back off and retry
                time.sleep(15 * (attempt + 1))
                continue
            raise DatasetUnavailable(f"HTTP {e.code} fetching MegaScale rows ({url}): {e}") from e
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            raise DatasetUnavailable(f"cannot reach the HF dataset-viewer ({url}): {e}") from e
    raise DatasetUnavailable(f"rate-limited (HTTP 429) by the HF dataset-viewer after retries: {url}")


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fetch_first_domain() -> list[Record]:
    """Page `/rows` from offset 0 and collect the FIRST domain's complete single-mutant DMS.

    The config is sorted by `WT_name`, so the first domain's variants form a contiguous prefix; we
    stop once a different domain has clearly begun. Dedupe by mutation (the deposit repeats rows)."""
    base = f"{_HOST}/rows?dataset={_DATASET}&config={_CONFIG}&split={_SPLIT}"
    first: str | None = None
    by_mut: dict[str, Record] = {}
    seen_other = 0
    for page in range(_MAX_PAGES):
        d = _get(f"{base}&offset={page * _PAGE}&length={_PAGE}")
        rows = d.get("rows", [])
        if not rows:
            break
        for entry in rows:
            row = entry.get("row", {})
            wt = row.get("WT_name")
            mut = row.get("mut_type")
            seq = row.get("aa_seq")
            dg = _to_float(row.get("deltaG"))
            if first is None:
                first = wt
            if wt != first:
                seen_other += 1
                continue
            if not (isinstance(seq, str) and set(seq) <= set(AA) and dg is not None):
                continue
            if mut != "wt" and not _SINGLE_MUT.fullmatch(str(mut)):
                continue                                # keep wild-type + single substitutions only
            by_mut[str(mut)] = Record(wt, str(mut), seq, dg, _to_float(row.get("ddG_ML")))
        if seen_other > _PAGE:                          # well past the first domain's block → done
            break
        time.sleep(_POLITE_S)
    if not by_mut:
        raise DatasetUnavailable("fetched 0 usable variants (format drift?)")
    return list(by_mut.values())


# --------------------------------------------------------------------------- #
# Cache read/write (the small per-domain table only).
# --------------------------------------------------------------------------- #
def _write_cache(path: Path, recs: list[Record]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wt_name", "mut_type", "aa_seq", "deltaG", "ddg_ml"])
        for r in recs:
            w.writerow([r.wt_name, r.mut_type, r.aa_seq, r.deltaG,
                        "" if r.ddg_ml is None else r.ddg_ml])


def _read_cache(path: Path) -> list[Record]:
    out = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            ml = row["ddg_ml"]
            out.append(Record(row["wt_name"], row["mut_type"], row["aa_seq"],
                              float(row["deltaG"]), float(ml) if ml not in ("", None) else None))
    return out


def load_records(*, refresh: bool = False) -> list[Record]:
    """One protein domain's complete single-mutant DMS (variant AA seq + measured folding ΔG).

    Reads `~/.cache/karyon/stability.csv` if present (offline-friendly); otherwise pages the HF
    dataset-viewer for the first domain's block, caches, and returns. Raises `DatasetUnavailable`
    when neither reachable nor cached."""
    path = _cache_path()
    if path.exists() and not refresh:
        recs = _read_cache(path)
        print(f"  [cache] {len(recs)} stability variants from {path.name}")
        return recs
    recs = _fetch_first_domain()
    _write_cache(path, recs)
    n_ml = sum(r.ddg_ml is not None for r in recs)
    print(f"  [cache] wrote {len(recs)} variants of {recs[0].wt_name} "
          f"({n_ml} with a ΔΔG_ML value) -> {path.name}")
    return recs


if __name__ == "__main__":
    print(f"Loading a protein-stability DMS slice from the HF dataset-viewer ({_DATASET})\n")
    try:
        rows = load_records()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    dg = [r.deltaG for r in rows]
    lens = sorted({len(r.aa_seq) for r in rows})
    print(f"\n  domain (WT_name)     : {rows[0].wt_name}")
    print(f"  variants             : {len(rows)} (single-mutants + wild-type, deduped)")
    print(f"  aa_seq length(s)     : {lens}  (alphabet ⊆ 20 AA: {set(''.join(r.aa_seq for r in rows[:20])) <= set(AA)})")
    print(f"  ΔG (kcal/mol) min/med/max: {min(dg):.2f} / {statistics.median(dg):.2f} / {max(dg):.2f}")
    wt = [r for r in rows if r.mut_type == "wt"]
    if wt:
        print(f"  wild-type ΔG         : {wt[0].deltaG:.2f}  (stabilizing muts = those above this)")
    with_ml = [r for r in rows if r.ddg_ml is not None]
    print(f"  with ΔΔG_ML pred     : {len(with_ml)}/{len(rows)}")
    top = sorted(rows, key=lambda r: r.deltaG, reverse=True)[:5]
    print(f"  top-5 most stable    : {[(r.mut_type, round(r.deltaG, 2)) for r in top]}")
