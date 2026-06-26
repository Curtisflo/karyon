"""hossain_data — cached loader for the La Fleur/Salis 2022 Hossain IN-VIVO σ70 promoter set.

The in-vivo companion to [promoter_data.py]'s Urtecho (in-vitro) set — the SAME Supplementary Data 1
xlsx, the `Hossain et al (Fig 3d, S7d)` sheet. Used by [hossain_predictor.py] as a predictor-margin test
with a **cleaner baseline** than Urtecho: this sheet carries the Promoter Calculator's per-row residual
columns (Abs(residual)/MSE/MAE), marking it the calc's BENCHMARK set — i.e. likely *held-out* for the
calc — so beating its prediction here removes the in-sample caveat the Urtecho probe carried.

Sheet layout (verified by inspection): A=ID, B=Upstream DNA (constant flank), C=Promoter Sequence (the
variable element), D=Downstream DNA (constant flank), E=Observed TX [au] (measured strength), ...,
G=Predicted log(TX/Txref) (the Promoter Calculator's own number; a sign-INVERTED log-ratio vs strength).

  * reuses [promoter_data.py]'s stdlib xlsx machinery (zipfile + xml.etree) by import — no duplication;
  * keeps the VARIABLE promoter (col C) as the sequence (B/D are constant flanks → constant features);
  * caches the small flat table to `~/.cache/karyon/hossain.csv`; offline-skip via the shared DatasetUnavailable.

    cd bio/probe && python hossain_data.py
"""

from __future__ import annotations
from .paths import cache_dir

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from . import promoter_data as pdat  # reuse the xlsx machinery + cache root + DatasetUnavailable

DatasetUnavailable = pdat.DatasetUnavailable

_SHEET_NAME = "Hossain et al (Fig 3d, S7d)"
_COL_SEQ = "C"        # Promoter Sequence (the variable element we model)
_COL_TX = "E"         # Observed TX [au]  (measured in-vivo transcription rate; heavy-tailed)
_COL_CALC = "G"       # Predicted log(TX/Txref) — Promoter Calculator's own number (sign-inverted)
_COL_UP = "B"         # Upstream DNA   (expected constant flank — verified at parse time)
_COL_DN = "D"         # Downstream DNA (expected constant flank — verified at parse time)


@dataclass(frozen=True)
class Record:
    """One measured in-vivo σ70 promoter (the variable element only)."""

    seq: str                    # the variable promoter sequence (uppercase ACGT)
    tx: float                   # measured in-vivo transcription rate, Observed TX [au] (>0; model on log)
    calc_pred: float | None     # Promoter Calculator's Predicted log(TX/Txref) (sign-inverted)

    @property
    def strength(self) -> float:
        """log-transcription-rate — the analysis target (raw TX is heavy-tailed)."""
        return math.log(self.tx)


def _cache_path() -> Path:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "hossain.csv"


def _usable(seq, tx) -> bool:
    return (isinstance(seq, str) and len(seq) > 0 and set(seq) <= set("ACGT")
            and isinstance(tx, float) and tx > 0.0)


def _row_to_record(row: dict) -> Record | None:
    """Map one worksheet row {col_letter: text} to a Record, or None if unusable. Pure (no xlsx state)
    so the column mapping + filtering is unit-testable without constructing a workbook."""
    seq = (row.get(_COL_SEQ) or "").strip().upper().replace("U", "T")
    try:
        tx = float(row[_COL_TX])
    except (KeyError, ValueError, TypeError):
        return None
    if not _usable(seq, tx):
        return None
    calc = row.get(_COL_CALC)
    try:
        cp = float(calc) if calc not in (None, "") else None
    except (ValueError, TypeError):
        cp = None
    return Record(seq, tx, cp)


def _parse(z) -> tuple[list[Record], dict]:
    """Stream the Hossain sheet → Records (col C / E / G), collecting flank/length sanity meta."""
    ss = pdat._shared_strings(z)
    sheet = pdat._sheet_path(z, _SHEET_NAME)
    out: list[Record] = []
    ups: set[str] = set()
    dns: set[str] = set()
    lens: set[int] = set()
    for i, row in enumerate(pdat._stream_rows(z, sheet, ss)):
        if i == 0:                                   # header
            continue
        rec = _row_to_record(row)
        if rec is None:
            continue
        out.append(rec)
        if row.get(_COL_UP):
            ups.add(row[_COL_UP].strip().upper())
        if row.get(_COL_DN):
            dns.add(row[_COL_DN].strip().upper())
        lens.add(len(rec.seq))
    return out, {"n_up_flanks": len(ups), "n_dn_flanks": len(dns), "seq_lens": sorted(lens)}


# --------------------------------------------------------------------------- #
# Cache read/write (the small flat table only) — same 3-column shape as promoter_data.
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
    """The measured in-vivo σ70 promoters (variable element + Observed TX + Promoter-Calc prediction).

    Reads `~/.cache/karyon/hossain.csv` if present (offline-friendly); otherwise fetches the one ~21 MB
    supplementary xlsx (via promoter_data), parses the Hossain sheet stdlib-only, caches the small flat
    table, and returns. Raises `DatasetUnavailable` when neither reachable nor cached."""
    path = _cache_path()
    if path.exists() and not refresh:
        recs = _read_cache(path)
        print(f"  [cache] {len(recs)} Hossain records from {path.name}")
        return recs
    recs, meta = _parse(pdat._fetch_xlsx())
    if not recs:
        raise DatasetUnavailable("parsed 0 usable Hossain records (sheet/format drift?)")
    _write_cache(path, recs)
    n_calc = sum(r.calc_pred is not None for r in recs)
    print(f"  [cache] wrote {len(recs)} records ({n_calc} with Promoter-Calc predictions); "
          f"upstream flanks={meta['n_up_flanks']}, downstream flanks={meta['n_dn_flanks']}, "
          f"promoter lengths={meta['seq_lens']} -> {path.name}")
    return recs


if __name__ == "__main__":
    import statistics
    print("Loading La Fleur/Salis 2022 Hossain IN-VIVO σ70 promoter set (Fig 3d sheet, Suppl. Data 1)\n")
    try:
        rows = load_records()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    tx = [r.tx for r in rows]
    with_calc = [r for r in rows if r.calc_pred is not None]
    print(f"\n  records                 : {len(rows)}")
    print(f"  promoter length(s)      : {sorted(set(len(r.seq) for r in rows))}")
    print(f"  Observed TX min/med/max : {min(tx):.3f} / {statistics.median(tx):.3f} / {max(tx):.3f}")
    print(f"  with Promoter-Calc pred : {len(with_calc)}/{len(rows)}")
