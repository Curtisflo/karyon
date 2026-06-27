"""rbs_hollerer_data — a cached loader for the Höllerer 2020 300k RBS sequence-function set.

The [RBS_PREDICTOR_RESULT.md] LOSO probe over 394 SynBioMTS constructs left one question open: does a
*strong* learned model on a BIG dataset beat the biophysical RBS Calculator de-novo? (The literature —
Höllerer et al. 2020 — says yes; we had only tested cheap cores on small data.) This loader fetches the
heavy-but-not-multi-GB exception that admits the already-installed deps: the **processed** RBS
sequence→function table from Höllerer et al., "Large-scale DNA-based phenotypic recording and deep
learning enable highly accurate sequence-function mapping," *Nat. Commun.* 11:3551 (2020), hosted at
`github.com/JeschekLab/uASPIre` (`RBS_data/uASPIre_RBS_300k_r*.txt.gz`, ~4–10 MB gzipped each — NOT the
multi-GB raw NGS on SRA, which is infeasible and unneeded; the processed table is all we need).

Each row is one library variant:

    RBS         a 17-nt variable region (the N17 directly 5' of the bxb1-sfGFP ATG; the start codon and
                all flanks are CONSTANT across the library — only this 17-mer varies).
    0..720      fraction-flipped at nine time points (min); the raw uASPIre phenotype recording.
    IFP480      integrated fraction of protein over 480 min ∈ [0,1] — THE function/activity value.
    total_reads read coverage (the authors pre-filtered; min ≈ 200, so no extra depth filter is needed).

We keep (sequence=17-mer, activity=IFP480). The learned core ([linmodel.py]) consumes the bare 17-mer
exactly as the paper's SAPIENs net does (`code/training/utils.py:seq2onehot` one-hot-encodes the bare
`<U17`; the published test set `data/sequences_test.npy` is 27,654 length-17 ACGT strings — confirmed).
The OSTIR baseline ([rbs_hollerer_predictor.py]) reconstructs the fixed flanks around the 17-mer so the
biophysical tool can fold; see that file for the (documented, bounded) construct reconstruction.

POSTURE (mirrors [rbs_synbiomts_data.py]): stdlib-only ingest (gzip + csv), network/parse failure →
typed `DatasetUnavailable` (the test SKIPs), cached to `~/.cache/karyon/` so every later run reads the cache
offline. The replicate is selectable; default r3 (the smallest 300k replicate, ~154k variants).

    cd bio/probe && python rbs_hollerer_data.py            # fetch + summarize (default replicate r3)
    cd bio/probe && python rbs_hollerer_data.py --rep r2   # the largest replicate (~300k variants)
"""

from __future__ import annotations
from .paths import cache_dir, network_allowed

import csv
import gzip
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_RAW = ("https://raw.githubusercontent.com/JeschekLab/uASPIre/master/"
        "RBS_data/uASPIre_RBS_300k_{rep}.txt.gz")
_UA = "karyon-bio-benchmark/1 (+https://github.com/JeschekLab/uASPIre)"
_TIMEOUT_S = 120
_REPS = ("r1", "r2", "r3")
_RBS_LEN = 17                      # the N17 variable region (paper + SAPIENs data confirm length 17)


class DatasetUnavailable(RuntimeError):
    """Höllerer uASPIre RBS data could not be fetched/parsed and is not cached → SKIP."""


@dataclass(frozen=True)
class Record:
    sequence: str          # the 17-nt variable RBS region, uppercase DNA
    ifp: float             # IFP480 — integrated fraction of protein over 480 min ∈ [0,1] (the function)
    reads: int             # total_reads coverage


# --------------------------------------------------------------------------- #
# Cache + fetch.
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    return here.parents[2]


def _cache_path(rep: str) -> Path:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"rbs_hollerer_300k_{rep}.csv"


def _fetch(rep: str) -> bytes:
    if not network_allowed():
        raise DatasetUnavailable("network disabled via KARYON_NO_NETWORK")
    url = _RAW.format(rep=rep)
    try:
        return urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": _UA}), timeout=_TIMEOUT_S).read()
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        raise DatasetUnavailable(f"cannot reach {url}: {e}") from e


def _parse_gz(raw: bytes) -> list[Record]:
    """Parse the tab-delimited gz: header then rows; keep (17-mer, IFP480, total_reads)."""
    text = gzip.decompress(raw).decode("utf-8", "replace")
    lines = text.splitlines()
    if not lines:
        raise DatasetUnavailable("empty uASPIre file")
    header = lines[0].split("\t")
    try:
        i_seq = header.index("RBS")
        i_ifp = header.index("IFP480")
        i_reads = header.index("total_reads")
    except ValueError as e:
        raise DatasetUnavailable(f"unexpected uASPIre header {header}: {e}") from e
    out: list[Record] = []
    for line in lines[1:]:
        p = line.split("\t")
        if len(p) <= max(i_seq, i_ifp, i_reads):
            continue
        seq = p[i_seq].strip().upper().replace("U", "T")
        if len(seq) != _RBS_LEN or set(seq) - set("ACGT"):
            continue
        try:
            ifp = float(p[i_ifp])
            reads = int(float(p[i_reads]))
        except (ValueError, TypeError):
            continue
        out.append(Record(seq, ifp, reads))
    if not out:
        raise DatasetUnavailable("parsed 0 usable uASPIre records (format drift?)")
    return out


# --------------------------------------------------------------------------- #
# Cache read/write + public loader.
# --------------------------------------------------------------------------- #
def _write_cache(path: Path, recs: list[Record]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence", "ifp", "reads"])
        for r in recs:
            w.writerow([r.sequence, r.ifp, r.reads])


def _read_cache(path: Path) -> list[Record]:
    with path.open(newline="") as fh:
        return [Record(row["sequence"], float(row["ifp"]), int(row["reads"]))
                for row in csv.DictReader(fh)]


def load_records(*, rep: str = "r3", refresh: bool = False) -> list[Record]:
    """All Höllerer 300k RBS records for a replicate. Reads `~/.cache/karyon/rbs_hollerer_300k_<rep>.csv`
    if present; otherwise fetches + parses the gz, caches, returns. Raises `DatasetUnavailable` if
    neither works (the test SKIPs offline)."""
    if rep not in _REPS:
        raise ValueError(f"rep must be one of {_REPS}, got {rep!r}")
    path = _cache_path(rep)
    if path.exists() and not refresh:
        recs = _read_cache(path)
        print(f"  [cache] {len(recs)} Höllerer RBS records ({rep}) from {path.name}")
        return recs
    recs = _parse_gz(_fetch(rep))
    _write_cache(path, recs)
    print(f"  [cache] wrote {len(recs)} records ({rep}) -> {path.name}")
    return recs


if __name__ == "__main__":
    import argparse
    import statistics as st
    ap = argparse.ArgumentParser(description="Fetch + summarize the Höllerer 2020 300k RBS set")
    ap.add_argument("--rep", default="r3", choices=_REPS, help="biological replicate (default r3)")
    ap.add_argument("--refresh", action="store_true", help="re-fetch even if cached")
    args = ap.parse_args()
    print(f"Loading Höllerer 2020 300k RBS data ({args.rep}) from JeschekLab/uASPIre\n")
    try:
        rows = load_records(rep=args.rep, refresh=args.refresh)
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
    ifp = [r.ifp for r in rows]
    reads = [r.reads for r in rows]
    print(f"\n  total records: {len(rows)} (all 17-nt, ACGT)")
    print(f"  IFP480 (activity): min {min(ifp):.3f}  mean {st.mean(ifp):.3f}  "
          f"median {st.median(ifp):.3f}  max {max(ifp):.3f}")
    print(f"  total_reads:       min {min(reads)}  median {int(st.median(reads))}  max {max(reads)}")
    ex = rows[0]
    print(f"\n  example: seq={ex.sequence} (len {len(ex.sequence)})  IFP480={ex.ifp:.3f}  reads={ex.reads}")
