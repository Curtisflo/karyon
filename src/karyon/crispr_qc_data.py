"""crispr_qc_data — cached loader for Horlbeck et al. 2016 CRISPRi sgRNA activity scores.

The ground-truth substrate for the screen-QC probe. Horlbeck, Gilbert, Villalta … Weissman
(*eLife* 2016, "Compact and highly active next-generation libraries for CRISPR-mediated gene
repression and activation"; eLife 19760) deposited per-sgRNA empirical **CRISPRi activity scores**
(Supplementary Data 1) — a relative on-target activity learned from tiling screens, the closest
public thing to a *measured* knockdown-efficacy label with the protospacer sequence attached. The
CRISPRi sheet is one row per sgRNA:

  * gene symbol (≈12 sgRNAs per gene — supports gene-level "is this gene powered?" QC),
  * sgRNA protospacer sequence (18–25 nt; a leading transcription-G inflates some lengths), and
  * CRISPRi activity score (higher = stronger knockdown; the weak tail activity<0.20 is ≈44% of
    guides, matching the field's "~40–50% of guides are ineffective").

Why this and not Replogle Perturb-seq directly: Replogle 2022 raw per-guide data is GB-scale `h5ad`
(needs scanpy/anndata) — not the probes' stdlib/offline posture. Horlbeck SD1 is a 1.4 MB xlsx with
exactly (sequence, gene, measured efficacy) per guide. The real-screen growth phenotypes (SD7) are a
documented fast-follow that joins on (chrom, coord, strand); this loader keeps those join keys.

  * fetches the 1.4 MB Supplementary Data 1 xlsx from the eLife CDN (auth-free) — one HTTP GET;
  * parses the CRISPRi sheet stdlib-only via `xlsx_kit` (`.xlsx` is a zip of XML);
  * keeps usable rows (clean ACGT protospacer + numeric activity) and caches the small flat table to
    `~/.cache/karyon/crispr_qc.csv` (gitignored), not the source xlsx;
  * degrades to a typed `DatasetUnavailable` (the test SKIPs, never fails, offline).

    cd karyon/probe && python crispr_qc_data.py     # smoke: fetch + summarize
"""

from __future__ import annotations
from .paths import cache_dir

import csv
import socket
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import xlsx_kit

# eLife CDN is the canonical auth-free host for the supplementary xlsx. supp1 == "Supplementary
# Data 1: CRISPRi and CRISPRa activity score datasets." (discovered via the eLife article API).
_XLSX_URL = "https://cdn.elifesciences.org/articles/19760/elife-19760-supp1-v2.xlsx"
_UA = "karyon-bio/1 (+https://elifesciences.org/articles/19760)"
_TIMEOUT_S = 180
_SHEET = "CRISPRi"

# SD1 CRISPRi column layout (header row 0):
#   A gene symbol | B chromosome | C PAM genomic coordinate [hg19] | D strand targeted
#   E sgRNA length (including PAM) | F sgRNA sequence | G CRISPRi activity score
_COL_GENE, _COL_CHROM, _COL_COORD, _COL_STRAND, _COL_SEQ, _COL_ACT = "A", "B", "C", "D", "F", "G"

# SD7 = the real K562 growth-phenotype screen for the hCRISPRi-v2 library. Joined to SD1 on
# (gene, strand, coord) it yields, per guide, (sequence, activity score, measured phenotype gamma) — the
# real-screen confirmation: a weak guide on an essential gene shows ~0 gamma (a silent failure). sgId is
# "{gene}_{strand}_{coord}.{len}-P{n}"; non-targeting controls carry no gene and define the null band.
_SD7_URL = "https://cdn.elifesciences.org/articles/19760/elife-19760-supp7-v2.xlsx"
_SD7_SHEET = "Sheet1"
_SD7_COL_SGID, _SD7_COL_GAMMA = "A", "H"          # sgId | gamma, ave_Rep1_Rep2
_NTC_GENE = "__NTC__"                              # sentinel gene for NT controls in the cached screen table


class DatasetUnavailable(RuntimeError):
    """The CRISPRi activity xlsx could not be fetched/parsed and is not cached → SKIP."""


@dataclass(frozen=True)
class Record:
    """One measured CRISPRi sgRNA (activity score = relative on-target knockdown activity)."""

    gene: str                   # target gene symbol (many sgRNAs share a gene)
    seq: str                    # protospacer sequence, uppercase ACGT (variable length 18–25)
    activity: float             # CRISPRi activity score; higher = stronger knockdown
    chrom: str = ""             # join keys for the SD7 real-screen growth-phenotype join
    coord: str = ""
    strand: str = ""


@dataclass(frozen=True)
class ScreenRecord:
    """A guide present in BOTH the activity set (SD1) and the real K562 growth screen (SD7)."""

    gene: str
    seq: str
    activity: float             # CRISPRi activity score (SD1)
    gamma: float                # growth phenotype, SD7 ave of 2 reps; essential-gene knockdown → negative


# --------------------------------------------------------------------------- #
# Cache plumbing (~/.cache/karyon/, gitignored — mirrors promoter_data.py).
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
    return d / "crispr_qc.csv"


# --------------------------------------------------------------------------- #
# Fetch + parse.
# --------------------------------------------------------------------------- #
def _fetch(url: str = _XLSX_URL) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        return urllib.request.urlopen(req, timeout=_TIMEOUT_S).read()
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        raise DatasetUnavailable(f"cannot reach the CRISPRi xlsx ({url}): {e}") from e


def _parse(raw: bytes) -> list[Record]:
    try:
        z = xlsx_kit.workbook(raw)
    except Exception as e:  # BadZipFile etc. — a CAPTCHA/HTML body, not an xlsx
        raise DatasetUnavailable(f"fetched bytes are not a valid xlsx: {e}") from e
    out: list[Record] = []
    for i, row in enumerate(xlsx_kit.rows(z, _SHEET)):
        if i == 0:                                   # header
            continue
        seq = (row.get(_COL_SEQ) or "").strip().upper().replace("U", "T")
        gene = (row.get(_COL_GENE) or "").strip()
        try:
            activity = float(row[_COL_ACT])
        except (KeyError, ValueError, TypeError):
            continue
        if not gene or not seq or set(seq) - set("ACGT"):
            continue                                 # drop blank/non-ACGT (mixed-case mismatch rows)
        out.append(Record(gene, seq, activity,
                          chrom=(row.get(_COL_CHROM) or "").strip(),
                          coord=(row.get(_COL_COORD) or "").strip(),
                          strand=(row.get(_COL_STRAND) or "").strip()))
    return out


# --------------------------------------------------------------------------- #
# Cache read/write (the small flat table only).
# --------------------------------------------------------------------------- #
_FIELDS = ["gene", "seq", "activity", "chrom", "coord", "strand"]


def _write_cache(path: Path, recs: list[Record]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_FIELDS)
        for r in recs:
            w.writerow([r.gene, r.seq, r.activity, r.chrom, r.coord, r.strand])


def _read_cache(path: Path) -> list[Record]:
    out = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Record(row["gene"], row["seq"], float(row["activity"]),
                              chrom=row.get("chrom", ""), coord=row.get("coord", ""),
                              strand=row.get("strand", "")))
    return out


def load_records(*, refresh: bool = False) -> list[Record]:
    """The measured CRISPRi sgRNAs (gene + protospacer + activity score).

    Reads `~/.cache/karyon/crispr_qc.csv` if present (offline-friendly); otherwise fetches the 1.4 MB
    Supplementary Data 1 xlsx, parses the CRISPRi sheet stdlib-only, caches the small flat table, and
    returns. Raises `DatasetUnavailable` when neither reachable nor cached."""
    path = _cache_path()
    if path.exists() and not refresh:
        recs = _read_cache(path)
        print(f"  [cache] {len(recs)} CRISPRi sgRNAs from {path.name}")
        return recs
    recs = _parse(_fetch())
    if not recs:
        raise DatasetUnavailable("parsed 0 usable CRISPRi sgRNAs (sheet/format drift?)")
    _write_cache(path, recs)
    print(f"  [cache] wrote {len(recs)} CRISPRi sgRNAs -> {path.name}")
    return recs


# --------------------------------------------------------------------------- #
# The real-screen join (SD7) — optional; the activity loader above stands alone.
# --------------------------------------------------------------------------- #
def _screen_cache_path() -> Path:
    return _cache_path().with_name("crispr_qc_screen.csv")


def _sgid_is_ntc(sgid: str) -> bool:
    """A non-targeting / negative-control sgRNA (defines the null band; carries no target gene)."""
    low = sgid.lower()
    return any(t in low for t in ("non-targeting", "negative", "control", "_ntc"))


def _sgid_key(sgid: str) -> tuple[str, str, str] | None:
    """Parse a targeting sgId ('AARS_+_70323441.23-P1') into the SD1 join key (gene, strand, coord)."""
    parts = sgid.split("_")
    if len(parts) < 3:
        return None
    return (parts[0], parts[1], parts[2].split(".")[0].split("-")[0])


def _parse_screen(raw: bytes, sd1: dict[tuple[str, str, str], tuple[str, float]]
                  ) -> tuple[list[ScreenRecord], list[float]]:
    """Join SD7 growth phenotypes onto SD1 (sequence, activity) by (gene, strand, coord); split off the
    non-targeting controls (no gene) whose gamma defines the screen's null band."""
    try:
        z = xlsx_kit.workbook(raw)
    except Exception as e:
        raise DatasetUnavailable(f"SD7 bytes are not a valid xlsx: {e}") from e
    recs: list[ScreenRecord] = []
    ntc: list[float] = []
    for i, row in enumerate(xlsx_kit.rows(z, _SD7_SHEET)):
        if i < 2:                                    # row0 group header, row1 column header
            continue
        sgid = row.get(_SD7_COL_SGID)
        if not sgid:
            continue
        try:
            gamma = float(row[_SD7_COL_GAMMA])
        except (KeyError, ValueError, TypeError):
            continue
        if gamma != gamma:                           # nan (a replicate dropped out)
            continue
        if _sgid_is_ntc(sgid):
            ntc.append(gamma)
            continue
        key = _sgid_key(sgid)
        if key is not None and key in sd1:
            recs.append(ScreenRecord(key[0], sd1[key][0], sd1[key][1], gamma))
    return recs, ntc


def _write_screen_cache(path: Path, recs: list[ScreenRecord], ntc: list[float]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gene", "seq", "activity", "gamma"])
        for r in recs:
            w.writerow([r.gene, r.seq, r.activity, r.gamma])
        for g in ntc:
            w.writerow([_NTC_GENE, "", "", g])


def _read_screen_cache(path: Path) -> tuple[list[ScreenRecord], list[float]]:
    recs, ntc = [], []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["gene"] == _NTC_GENE:
                ntc.append(float(row["gamma"]))
            else:
                recs.append(ScreenRecord(row["gene"], row["seq"], float(row["activity"]),
                                         float(row["gamma"])))
    return recs, ntc


def load_screen(*, refresh: bool = False) -> tuple[list[ScreenRecord], list[float]]:
    """The SD1×SD7 join: guides carrying (sequence, activity, real-screen gamma), plus the non-targeting
    controls' gamma (the null band). Reads the cached joined table if present; otherwise fetches the 17 MB
    SD7 xlsx, joins it onto the (cached) activity set, and caches the small result. Raises
    `DatasetUnavailable` when neither reachable nor cached."""
    path = _screen_cache_path()
    if path.exists() and not refresh:
        recs, ntc = _read_screen_cache(path)
        print(f"  [cache] {len(recs)} screen-matched guides + {len(ntc)} NT controls "
              f"from {path.name}")
        return recs, ntc
    sd1 = {(r.gene, r.strand, r.coord): (r.seq, r.activity) for r in load_records()}
    recs, ntc = _parse_screen(_fetch(_SD7_URL), sd1)
    if not recs:
        raise DatasetUnavailable("parsed 0 screen-matched guides (SD7 join produced nothing)")
    _write_screen_cache(path, recs, ntc)
    print(f"  [cache] wrote {len(recs)} screen-matched guides + {len(ntc)} NT controls "
          f"-> {path.name}")
    return recs, ntc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Horlbeck CRISPRi loader (activity; --screen adds the SD7 join).")
    ap.add_argument("--screen", action="store_true", help="also fetch + join the SD7 K562 growth screen")
    cli = ap.parse_args()
    print("Loading Horlbeck et al. 2016 CRISPRi sgRNA activity scores (eLife 19760, Suppl. Data 1)\n")
    try:
        rows = load_records()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    act = [r.activity for r in rows]
    genes = {r.gene for r in rows}
    lens = [len(r.seq) for r in rows]
    weak = sum(a < 0.20 for a in act) / len(act)
    print(f"\n  sgRNAs                  : {len(rows)}")
    print(f"  genes                   : {len(genes)}  (mean {len(rows) / len(genes):.1f} sgRNAs/gene)")
    print(f"  activity min/med/max    : {min(act):+.3f} / {statistics.median(act):+.3f} / {max(act):+.3f}")
    print(f"  protospacer length range: {min(lens)}–{max(lens)} nt")
    print(f"  weak tail activity<0.20 : {weak:.1%}  (the field's ~40–50% ineffective-guide rate)")
    top = sorted(rows, key=lambda r: r.activity, reverse=True)[:3]
    print(f"  strongest 3             : {[(r.gene, round(r.activity, 2)) for r in top]}")

    if cli.screen:
        print("\nJoining the real K562 growth screen (eLife 19760, Suppl. Data 7)\n")
        try:
            srecs, ntc = load_screen()
        except DatasetUnavailable as e:
            print(f"SKIP — {e}")
            raise SystemExit(0)
        gam = [r.gamma for r in srecs]
        print(f"\n  screen-matched guides   : {len(srecs)}  ({len(srecs) / len(rows):.0%} of the activity set)")
        print(f"  NT controls (null band) : {len(ntc)}  med {statistics.median(ntc):+.3f} "
              f"sd {statistics.pstdev(ntc):.3f}")
        print(f"  gamma min/med/max       : {min(gam):+.3f} / {statistics.median(gam):+.3f} / {max(gam):+.3f}")
