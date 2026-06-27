"""promoter_data — a cached loader for the La Fleur/Salis 2022 σ70 promoter-strength dataset.

A THIRD substrate for the probes — bacterial transcription initiation (σ70 promoter strength) — the
mid-learnability point between toehold (model ρ≈0.20) and RBS/EMOPEC (ρ≈0.79). La Fleur, Hossain &
Salis (*Nat. Commun.* 2022, the "Promoter Calculator") characterized large massively-parallel promoter
libraries. Their Supplementary Data 1 is a multi-sheet `.xlsx`; the cleanest flat sequence→strength
table in it is the **Urtecho et al. set** (sheet "Urtecho et al (Fig 3c, S7b)"): one fixed-length
150-nt promoter sequence per row + one measured transcription rate (`Observed TX [au]`), 10,898 unique
promoters, plus the Promoter Calculator's own prediction column for free context. This loader:

  * fetches the one ~21 MB `.xlsx` from Springer static-content (the canonical host; the NCBI/PMC
    mirror is CAPTCHA-gated) — one HTTP GET, no auth;
  * parses it **stdlib-only** (`zipfile` + `xml.etree`, streaming the one sheet we need) — `.xlsx` is
    a zip of XML, so no `openpyxl`/`pandas`; shared strings are resolved by index;
  * keeps only usable rows (a clean 150-nt ACGT sequence + a positive numeric strength) and caches the
    SMALL joined flat table to `~/.cache/karyon/promoter.csv` (gitignored), not the 21 MB source;
  * degrades to a typed `DatasetUnavailable` (the test SKIPs, never fails, offline).

Strength note: `Observed TX [au]` spans ~0.06..35 (heavy-tailed), so probes model log-strength; the
loader stores the raw `tx` and lets the featurizer take the log. The Promoter Calculator column is
"Predicted log(TX/Txref)" — a log-RATIO whose sign is INVERTED vs raw TX (more negative = stronger),
stored verbatim as `calc_pred` for an |ρ| context check, NOT as a fair held-out baseline.

    python -m karyon.promoter_data        # smoke: fetch + summarize
"""

from __future__ import annotations
from .paths import cache_dir, network_allowed

import csv
import io
import math
import os
import re
import socket
import statistics
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

# Springer static-content is the canonical, auth-free host for the supplementary xlsx (the PMC/NCBI
# mirror returns a CAPTCHA interstitial). MOESM5 == "Supplementary Data 1".
_XLSX_URL = ("https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-32829-5/"
             "MediaObjects/41467_2022_32829_MOESM5_ESM.xlsx")
_UA = "karyon-bio-benchmark/1 (+https://www.nature.com/articles/s41467-022-32829-5)"
_TIMEOUT_S = 120
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# The target sheet's display name + its flat columns (Urtecho et al; one sequence col, one strength col).
_SHEET_NAME = "Urtecho et al (Fig 3c, S7b)"
_COL_SEQ = "B"        # Promoter Sequence (fixed 150 nt)
_COL_TX = "C"         # Observed TX [au]  (raw transcription rate; heavy-tailed)
_COL_CALC = "E"       # Predicted log(TX/Txref) — the Promoter Calculator's own number (context only)
PROMOTER_LEN = 150    # the Urtecho fixed promoter length


class DatasetUnavailable(RuntimeError):
    """The promoter xlsx could not be fetched/parsed and is not cached → SKIP."""


@dataclass(frozen=True)
class Record:
    """One measured σ70 promoter."""

    seq: str                    # the 150-nt promoter sequence (uppercase ACGT)
    tx: float                   # measured transcription rate, Observed TX [au] (>0; model on log)
    calc_pred: float | None     # Promoter Calculator's Predicted log(TX/Txref) (context; sign-inverted)

    @property
    def strength(self) -> float:
        """log-transcription-rate — the analysis target (raw TX is heavy-tailed ~0.06..35)."""
        return math.log(self.tx)


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
    return d / "promoter.csv"


# --------------------------------------------------------------------------- #
# Fetch + parse the xlsx, stdlib only (zipfile + xml.etree; no openpyxl/pandas).
# --------------------------------------------------------------------------- #
def _fetch_xlsx() -> zipfile.ZipFile:
    """The supplementary `.xlsx` as an in-memory zip (it is a zip of XML)."""
    if not network_allowed():
        raise DatasetUnavailable("network disabled via KARYON_NO_NETWORK")
    req = urllib.request.Request(_XLSX_URL, headers={"User-Agent": _UA})
    try:
        raw = urllib.request.urlopen(req, timeout=_TIMEOUT_S).read()
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        raise DatasetUnavailable(f"cannot reach the promoter xlsx ({_XLSX_URL}): {e}") from e
    try:
        return zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as e:
        raise DatasetUnavailable(f"fetched bytes are not a valid xlsx zip: {e}") from e


def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    """The workbook's shared-string table; cell `t='s'` values index into this list."""
    try:
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(t.text or "" for t in si.iter(_NS + "t")) for si in root.iter(_NS + "si")]


def _sheet_path(z: zipfile.ZipFile, name: str) -> str:
    """The `xl/worksheets/sheetN.xml` member backing the sheet whose display name is `name`."""
    wb = z.read("xl/workbook.xml").decode("utf-8", "replace")
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8", "replace")
    rid = dict(re.findall(r'<sheet[^>]*name="([^"]*)"[^>]*r:id="([^"]*)"', wb)).get(name)
    target = dict(re.findall(r'<Relationship[^>]*Id="([^"]*)"[^>]*Target="([^"]*)"', rels)).get(rid)
    if not target:
        raise DatasetUnavailable(f"sheet {name!r} not found in the promoter xlsx (format drift?)")
    return "xl/" + target.lstrip("/")


def _col(ref: str) -> str:
    """The column letters of a cell reference ('B12' -> 'B')."""
    return re.match(r"[A-Z]+", ref).group()


def _stream_rows(z: zipfile.ZipFile, sheet_path: str, ss: list[str]):
    """Yield each worksheet row as {col_letter: text}, streaming (the file is big)."""
    with z.open(sheet_path) as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag != _NS + "row":
                continue
            row: dict[str, str] = {}
            for c in el.findall(_NS + "c"):
                v = c.find(_NS + "v")
                if v is None or v.text is None:
                    continue
                row[_col(c.get("r"))] = ss[int(v.text)] if c.get("t") == "s" else v.text
            el.clear()
            yield row


def _usable(seq: str, tx) -> bool:
    return (isinstance(seq, str) and len(seq) == PROMOTER_LEN and set(seq) <= set("ACGT")
            and isinstance(tx, float) and tx > 0.0)


def _parse(z: zipfile.ZipFile) -> list[Record]:
    ss = _shared_strings(z)
    sheet = _sheet_path(z, _SHEET_NAME)
    out: list[Record] = []
    for i, row in enumerate(_stream_rows(z, sheet, ss)):
        if i == 0:                                   # header
            continue
        seq = (row.get(_COL_SEQ) or "").strip().upper().replace("U", "T")
        try:
            tx = float(row[_COL_TX])
        except (KeyError, ValueError, TypeError):
            continue
        if not _usable(seq, tx):
            continue
        calc = row.get(_COL_CALC)
        try:
            cp = float(calc) if calc not in (None, "") else None
        except (ValueError, TypeError):
            cp = None
        out.append(Record(seq, tx, cp))
    return out


# --------------------------------------------------------------------------- #
# Cache read/write (the small flat table only).
# --------------------------------------------------------------------------- #
def _write_cache(path: Path, recs: list[Record]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seq", "tx", "calc_pred"])
        for r in recs:
            w.writerow([r.seq, r.tx, "" if r.calc_pred is None else r.calc_pred])


def _read_cache(path: Path) -> list[Record]:
    out = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            cp = row["calc_pred"]
            out.append(Record(row["seq"], float(row["tx"]),
                              float(cp) if cp not in ("", None) else None))
    return out


def load_records(*, refresh: bool = False) -> list[Record]:
    """The measured σ70 promoters (150-nt sequence + Observed TX + Promoter-Calc prediction).

    Reads `~/.cache/karyon/promoter.csv` if present (offline-friendly); otherwise fetches the one ~21 MB
    supplementary xlsx, parses the Urtecho sheet stdlib-only, caches the small flat table, and returns.
    Raises `DatasetUnavailable` when neither reachable nor cached."""
    path = _cache_path()
    if path.exists() and not refresh:
        recs = _read_cache(path)
        print(f"  [cache] {len(recs)} promoter records from {path.name}")
        return recs
    recs = _parse(_fetch_xlsx())
    if not recs:
        raise DatasetUnavailable("parsed 0 usable promoter records (sheet/format drift?)")
    _write_cache(path, recs)
    n_calc = sum(r.calc_pred is not None for r in recs)
    print(f"  [cache] wrote {len(recs)} records ({n_calc} with Promoter-Calc predictions) "
          f"-> {path.name}")
    return recs


if __name__ == "__main__":
    print("Loading La Fleur/Salis 2022 σ70 promoter dataset (Urtecho sheet, Supplementary Data 1)\n")
    try:
        rows = load_records()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    tx = [r.tx for r in rows]
    print(f"\n  records                 : {len(rows)}")
    print(f"  promoter length (all==150): {set(len(r.seq) for r in rows)}")
    print(f"  Observed TX min/med/max : {min(tx):.3f} / {statistics.median(tx):.3f} / {max(tx):.3f}")
    print(f"  log(TX) min/med/max     : {math.log(min(tx)):+.2f} / "
          f"{math.log(statistics.median(tx)):+.2f} / {math.log(max(tx)):+.2f}")
    with_calc = [r for r in rows if r.calc_pred is not None]
    print(f"  with Promoter-Calc pred : {len(with_calc)}/{len(rows)}")
    top = sorted(rows, key=lambda r: r.tx, reverse=True)[:5]
    print(f"  top-5 measured TX       : {[(round(r.tx, 2)) for r in top]}")
