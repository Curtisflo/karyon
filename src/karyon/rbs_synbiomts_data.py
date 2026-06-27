"""rbs_synbiomts_data — a cached loader for E. coli designed-RBS data from SynBioMTS.

Probe #2's REAL predictor margin needs a diverse, full-RBS dataset + a fair runnable baseline. The
Salis lab's SynBioMTS (Reis & Salis, ACS Synth. Biol. 2020) curates measured translation across many
studies — the corpus the RBS Calculator was validated on. This loader ports the parse recipes from
`SalisLabCode/ModelTestSystem/examples/RBS/initdb.py` for the **E. coli designed-RBS** studies (so
the E. coli anti-Shine-Dalgarno baseline is valid — Bacteroides/B. subtilis/diverse-organism sets are
excluded), fetches each `.xls` from GitHub raw, and builds records:

    SEQUENCE = 5'UTR + CDS,  STARTPOS = len(5'UTR),  PROT_MEAN = measured expression,  DATASET = study

NOTE ON POSTURE: this file (and the OSTIR baseline) admit non-stdlib deps — `xlrd` for ingest — which
probe #2 explicitly authorizes for a fair biophysical baseline. The learned core ([linmodel.py]) stays
stdlib. Network/parse failure degrades to a typed `DatasetUnavailable` (the test SKIPs). Cached to
`~/.cache/karyon/` so every later run (and the offline learned-core LOSO) reads the cache.

    python -m karyon.rbs_synbiomts_data        # fetch + summarize
"""

from __future__ import annotations
from .paths import cache_dir, network_allowed

import csv
import os
import socket
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    import xlrd
except ImportError:                                  # ingest dep; degrade cleanly if absent
    xlrd = None

_RAW = ("https://raw.githubusercontent.com/hsalis/SalisLabCode/master/"
        "ModelTestSystem/examples/RBS/datasets/{}.xls")
_UA = "karyon-bio-benchmark/1 (+https://github.com/hsalis/SalisLabCode)"
_TIMEOUT_S = 60
_STARTS = ("ATG", "GTG", "CTG", "TTG")

# Simple recipes: (utr_col, cds_col, prot_col, row_start, row_end), 0-based, end-exclusive.
# Ported verbatim from initdb.py. _extended is merged into its parent study.
_RECIPES: dict[str, list[tuple[int, int, int, int, int]]] = {
    "EspahBorujeni_NAR_2013": [(5, 6, 8, 5, 141), None],   # None -> the _extended sheet (special-cased)
    "EspahBorujeni_JACS_2016": [(3, 4, 9, 6, 42)],
    "EspahBorujeni_Footprint": [(3, 4, 10, 5, 32)],
    "Tian_NAR_2015": [(4, 5, 12, 2, 26)],
}


class DatasetUnavailable(RuntimeError):
    """SynBioMTS RBS data could not be fetched/parsed and is not cached → SKIP."""


@dataclass(frozen=True)
class Record:
    dataset: str
    sequence: str          # 5'UTR + CDS, uppercase DNA
    startpos: int          # index of the start codon's first base (== len(5'UTR))
    prot_mean: float       # measured expression (study-specific units; compare by rank within study)


# --------------------------------------------------------------------------- #
# Cache + fetch.
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
    return d / "rbs_synbiomts.csv"


def _fetch(name: str) -> bytes:
    if not network_allowed():
        raise DatasetUnavailable("network disabled via KARYON_NO_NETWORK")
    url = _RAW.format(name)
    try:
        return urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": _UA}), timeout=_TIMEOUT_S).read()
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        raise DatasetUnavailable(f"cannot reach {url}: {e}") from e


def _clean(seq: str) -> str:
    return str(seq).strip().upper().replace("U", "T")


def _valid(seq: str, startpos: int) -> bool:
    return (bool(seq) and set(seq) <= set("ACGT") and 0 < startpos < len(seq) - 6
            and seq[startpos:startpos + 3] in _STARTS)


def _records_from_simple(name: str, recipe) -> list[Record]:
    """Studies whose sheet gives 5'UTR / CDS / PROT columns directly."""
    sheet = xlrd.open_workbook(file_contents=_fetch(name)).sheet_by_index(0)
    out: list[Record] = []
    for spec in recipe:
        utr_c, cds_c, prot_c, r0, r1 = spec
        utrs = sheet.col_values(utr_c, r0, r1)
        cdss = sheet.col_values(cds_c, r0, r1)
        prots = sheet.col_values(prot_c, r0, r1)
        for utr, cds, prot in zip(utrs, cdss, prots):
            rec = _make(name, _clean(utr), _clean(cds), prot)
            if rec:
                out.append(rec)
    return out


def _records_extended_nar2013() -> list[Record]:
    """The EspahBorujeni_NAR_2013_extended sheet (cols 3/5/8, rows 3..42) — merged into NAR_2013."""
    sheet = xlrd.open_workbook(file_contents=_fetch("EspahBorujeni_NAR_2013_extended")).sheet_by_index(0)
    utrs, cdss, prots = (sheet.col_values(c, 3, 42) for c in (3, 5, 8))
    out = [_make("EspahBorujeni_NAR_2013", _clean(u), _clean(c), p) for u, c, p in zip(utrs, cdss, prots)]
    return [r for r in out if r]


def _records_salis2009() -> list[Record]:
    """Salis 2009 (the original RBS Calculator paper): start found via the SacI site, then backed up
    to the nearest start codon — ported from initdb.py."""
    sheet = xlrd.open_workbook(file_contents=_fetch("Salis_Nat_Biotech_2009")).sheet_by_index(0)
    rfp_cds = _clean(sheet.cell_value(1, 3))
    seqs = [_clean(s) for s in sheet.col_values(3, 3, 135)]
    prots = sheet.col_values(4, 3, 135)
    out: list[Record] = []
    for seq, prot in zip(seqs, prots):
        saci = seq.find("GAGCTC")
        if saci < 5:
            continue
        sp = saci - 5
        while sp >= 0 and seq[sp:sp + 3] not in _STARTS:
            sp -= 3
        if sp < 0:
            continue
        full = seq[:saci] + rfp_cds            # 5'UTR + (early CDS up to SacI) + RFP1 body
        rec = _make("Salis_Nat_Biotech_2009", full[:sp], full[sp:], prot)
        if rec:
            out.append(rec)
    return out


def _make(dataset: str, utr: str, cds: str, prot) -> Record | None:
    try:
        p = float(prot)
    except (ValueError, TypeError):
        return None
    seq, startpos = utr + cds, len(utr)
    if p > 0.0 and _valid(seq, startpos):
        return Record(dataset, seq, startpos, p)
    return None


# --------------------------------------------------------------------------- #
# Cache read/write + public loader.
# --------------------------------------------------------------------------- #
def _write_cache(path: Path, recs: list[Record]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "sequence", "startpos", "prot_mean"])
        for r in recs:
            w.writerow([r.dataset, r.sequence, r.startpos, r.prot_mean])


def _read_cache(path: Path) -> list[Record]:
    with path.open(newline="") as fh:
        return [Record(row["dataset"], row["sequence"], int(row["startpos"]), float(row["prot_mean"]))
                for row in csv.DictReader(fh)]


def load_records(*, refresh: bool = False) -> list[Record]:
    """All E. coli SynBioMTS RBS records. Reads `~/.cache/karyon/rbs_synbiomts.csv` if present; otherwise
    fetches + parses every study, caches, returns. Raises `DatasetUnavailable` if neither works."""
    path = _cache_path()
    if path.exists() and not refresh:
        recs = _read_cache(path)
        print(f"  [cache] {len(recs)} RBS records from {path.name}")
        return recs
    if xlrd is None:
        raise DatasetUnavailable("xlrd not installed (pip install xlrd) and no cache present")
    recs: list[Record] = []
    for name, recipe in _RECIPES.items():
        simple = [s for s in recipe if s is not None]
        recs += _records_from_simple(name, simple)
        if None in recipe:                              # the NAR_2013 _extended sheet
            recs += _records_extended_nar2013()
    recs += _records_salis2009()
    if not recs:
        raise DatasetUnavailable("parsed 0 usable RBS records (recipe/format drift?)")
    _write_cache(path, recs)
    by = {}
    for r in recs:
        by[r.dataset] = by.get(r.dataset, 0) + 1
    print(f"  [cache] wrote {len(recs)} records -> {path.name}; by study: {by}")
    return recs


if __name__ == "__main__":
    print("Loading E. coli RBS data from SynBioMTS (hsalis/SalisLabCode)\n")
    try:
        rows = load_records()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r.dataset, []).append(r.prot_mean)
    print(f"\n  total records: {len(rows)}")
    for ds, ps in sorted(by.items()):
        print(f"    {ds:<32} n={len(ps):>3}  expr range {min(ps):.3g}..{max(ps):.3g}")
    ex = rows[0]
    print(f"\n  example: {ex.dataset}  startpos={ex.startpos}  codon={ex.sequence[ex.startpos:ex.startpos+3]}"
          f"  len={len(ex.sequence)}  prot={ex.prot_mean:.3g}")
