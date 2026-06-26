"""emopec_data — a cached loader for the EMOPEC measured RBS (Shine-Dalgarno) expression dataset.

A SECOND substrate for the probes — bacterial translation initiation, structurally unlike toehold's
RNA switches. Bonde et al. (Nat. Methods 2015) systematically characterized the Shine-Dalgarno
sequence: **3,070 SD hexamers with measured relative expression** (normalized 0..1; AGGAGA = the
reference maximum). The data ships embedded as Python dicts in the EMOPEC package; this loader:

  * fetches the two ~110 KB source files via the GitHub contents API (default branch, base64 — no
    branch-guessing, no `git clone`);
  * parses the dict literals with `ast` (`literal_eval`, never `exec`) — so a malicious source file
    cannot run code here;
  * joins the MEASURED expression (`_sequences.py`) with the EMOPEC MODEL's own predictions
    (`_predicted_sequences.py`) on the hexamer, so the probe can score our learned core head-to-head
    against the published model;
  * caches the small joined table to `~/.cache/karyon/` (gitignored) and degrades to a typed
    `DatasetUnavailable` (the test SKIPs, never fails, offline).

    cd bio/probe && python emopec_data.py        # smoke: fetch + summarize
"""

from __future__ import annotations
from .paths import cache_dir

import ast
import base64
import csv
import json
import os
import socket
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_REPO = "smsaladi/EMOPEC"                          # a clean fork of the Bonde 2015 original
_MEASURED = "emopec/_sequences.py"
_PREDICTED = "emopec/_predicted_sequences.py"
_UA = "karyon-bio-benchmark/1 (+https://github.com/smsaladi/EMOPEC)"
_TIMEOUT_S = 45
SD_LEN = 6                                         # the Shine-Dalgarno hexamer length


class DatasetUnavailable(RuntimeError):
    """EMOPEC source could not be fetched (offline / network error) and is not cached → SKIP."""


@dataclass(frozen=True)
class Record:
    """One measured Shine-Dalgarno hexamer."""

    sd: str                       # the 6-nt SD sequence
    expression: float             # measured relative expression, 0..1 (AGGAGA = 1.0 reference)
    emopec_pred: float | None     # the EMOPEC model's predicted expression (None if not present)


# --------------------------------------------------------------------------- #
# Cache plumbing (~/.cache/karyon/, gitignored — mirrors toehold_data.py).
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
    return d / "emopec.csv"


# --------------------------------------------------------------------------- #
# Fetch + parse (ast.literal_eval, never exec).
# --------------------------------------------------------------------------- #
def _fetch_source(path_in_repo: str) -> str:
    """The decoded text of a file in the EMOPEC repo, via the GitHub contents API (default branch)."""
    url = f"https://api.github.com/repos/{_REPO}/contents/{path_in_repo}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/vnd.github+json"})
    try:
        raw = urllib.request.urlopen(req, timeout=_TIMEOUT_S).read()
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        raise DatasetUnavailable(f"cannot reach EMOPEC source ({url}): {e}") from e
    try:
        return base64.b64decode(json.loads(raw)["content"]).decode("utf-8", "replace")
    except (ValueError, KeyError) as e:
        raise DatasetUnavailable(f"unexpected GitHub API response for {path_in_repo}: {e}") from e


def _extract_mapping(source: str) -> dict[str, float]:
    """The largest sequence→value mapping in an EMOPEC source file, parsed with ast (no exec).

    Handles both shapes EMOPEC uses: `OrderedDict([(k, v), ...])` (a Call wrapping a list of pairs)
    and a bare `{k: v, ...}` dict literal."""
    best: dict = {}
    for node in ast.walk(ast.parse(source)):
        m = None
        if isinstance(node, ast.Dict):
            m = _safe_literal(node)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
            pairs = _safe_literal(node.args[0])
            m = dict(pairs) if isinstance(pairs, (list, tuple)) else None
        if isinstance(m, dict) and len(m) > len(best):
            best = m
    if not best:
        raise DatasetUnavailable("no sequence→value mapping found in EMOPEC source")
    return best


def _safe_literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def _usable(sd: str, val) -> bool:
    return (isinstance(sd, str) and len(sd) == SD_LEN and set(sd) <= set("ACGT")
            and isinstance(val, (int, float)))


# --------------------------------------------------------------------------- #
# Cache read/write (the small joined table only).
# --------------------------------------------------------------------------- #
def _write_cache(path: Path, recs: list[Record]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sd", "expression", "emopec_pred"])
        for r in recs:
            w.writerow([r.sd, r.expression, "" if r.emopec_pred is None else r.emopec_pred])


def _read_cache(path: Path) -> list[Record]:
    out = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            pred = row["emopec_pred"]
            out.append(Record(row["sd"], float(row["expression"]),
                              float(pred) if pred not in ("", None) else None))
    return out


def load_records(*, refresh: bool = False) -> list[Record]:
    """The 3,070 measured SD hexamers (measured expression + EMOPEC-model prediction).

    Reads `~/.cache/karyon/emopec.csv` if present (offline-friendly); otherwise fetches the two source
    files, joins, caches, and returns. Raises `DatasetUnavailable` when neither reachable nor cached."""
    path = _cache_path()
    if path.exists() and not refresh:
        recs = _read_cache(path)
        print(f"  [cache] {len(recs)} EMOPEC records from {path.name}")
        return recs
    measured = _extract_mapping(_fetch_source(_MEASURED))
    predicted = _extract_mapping(_fetch_source(_PREDICTED))
    recs = [Record(sd, float(expr), predicted.get(sd))
            for sd, expr in measured.items() if _usable(sd, expr)]
    if not recs:
        raise DatasetUnavailable("fetched 0 usable EMOPEC records (parse/format drift?)")
    _write_cache(path, recs)
    n_pred = sum(r.emopec_pred is not None for r in recs)
    print(f"  [cache] wrote {len(recs)} records ({n_pred} with EMOPEC predictions) "
          f"-> {path.name}")
    return recs


if __name__ == "__main__":
    print(f"Loading EMOPEC SD-expression dataset from github.com/{_REPO}\n")
    try:
        rows = load_records()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    expr = [r.expression for r in rows]
    print(f"\n  records              : {len(rows)}")
    print(f"  SD length (all == 6) : {set(len(r.sd) for r in rows)}")
    print(f"  expression min/med/max: {min(expr):.3f} / {statistics.median(expr):.3f} / {max(expr):.3f}")
    with_pred = [r for r in rows if r.emopec_pred is not None]
    print(f"  with EMOPEC pred     : {len(with_pred)}/{len(rows)}")
    top = sorted(rows, key=lambda r: r.expression, reverse=True)[:5]
    print(f"  top-5 measured       : {[(r.sd, round(r.expression, 3)) for r in top]}")
