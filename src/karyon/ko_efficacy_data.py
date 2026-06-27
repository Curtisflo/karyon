"""ko_efficacy_data — cached loader for CRISPR **knockout** on-target *efficacy* datasets.

The substrate for the guide-efficacy *ownership* probe (avenue 3): "how much of CRISPRko guide
efficacy is captured by the published Rule-Set/Azimuth models (which read a 30-nt genomic context)
vs. a cheap, legible sequence-only model?" The earlier ownership benchmark
(`CRISPR_QC_BENCHMARK_RESULT.md` Q2) refused to score Azimuth/Rule-Set on the Horlbeck table because
that table is protospacer-only CRISPR**i** *knockdown* — cross-mechanism for tools built on CRISPR**ko**
*cutting*. This loader supplies the valid substrate: real CRISPRko datasets that ship, per guide, the
30-nt context AND the published tools' deposited scores AND a measured cleavage label.

Source: the CRISPOR benchmark aggregate (Haeussler 2016), `github.com/maximilianh/crisporPaper`,
`effData/<name>.scores.tab` (GitHub-raw, login-free, small flat TSV). One row per guide; the columns
this loader uses:

  * `seq`          — 23-mer (20-nt protospacer + 3-nt NGG PAM); protospacer = `seq[:20]`.
  * `modFreq`      — the **measured** on-target efficacy label (indel/depletion; scale varies by dataset,
                     but every downstream metric is Spearman ρ, which is invariant to the monotone scale).
  * `longSeq100Bp` — a ~100-nt strand-oriented window the guide reads forward in; the Azimuth 30-mer
                     (4 up + 20 protospacer + 3 PAM + 3 down) is sliced from it — **no genome download**.
  * deposited published scores: `fusi` = Rule Set 2 / Azimuth (the owned incumbent of record),
    `doench` = Rule Set 1, plus `ssc` / `crisprScan` / `chariRaw` / `wang` / `wuCrispr` (the wider
    tool panel). These are predictions; `modFreq` is the truth they are scored against.

Only **RS2-independent** datasets are offered (Xu/Wang HL60, Chari 293T, Moreno-Mateos zebrafish,
Doench 2014) — Doench **2016** is deliberately absent because it is Rule Set 2's *training* set
(scoring `fusi` on it would be circular).

  * fetches the chosen dataset's `.scores.tab`, parses stdlib-only, derives the 30-mer, caches the small
    flat table to `~/.cache/karyon/` (gitignored); degrades to a typed `DatasetUnavailable` (test SKIPs, never
    fails, offline).

    cd karyon/probe && python ko_efficacy_data.py     # smoke: fetch + summarize the source ladder
"""

from __future__ import annotations
from .paths import cache_dir, network_allowed

import csv
import socket
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

_BASE = "https://raw.githubusercontent.com/maximilianh/crisporPaper/master/effData"
_UA = "karyon-bio/1 (+https://github.com/maximilianh/crisporPaper)"
_TIMEOUT_S = 180

# The RS2-independent CRISPRko efficacy datasets, in ladder order (biggest/most-relevant first). Each
# entry: (cache key, human label). load_efficacy(None) returns the first reachable; the probe panel
# iterates them. Doench 2016 is intentionally excluded (RS2 training set → circular).
_DATASETS: list[tuple[str, str]] = [
    ("xu2015TrainHl60", "Xu/Wang 2015 — HL60 (human, 2076 guides, depletion-derived efficacy; RS2-independent)"),
    ("chari2015Train293T", "Chari 2015 — HEK293T (human, 1234 guides, indel; RS2-independent)"),
    ("morenoMateos2015", "Moreno-Mateos 2015 — zebrafish (1020 guides, indel; RS2-independent)"),
    ("doench2014-Hs", "Doench 2014 — human (881 guides, indel; RS1+RS2 TRAINING LINEAGE → in-sample control)"),
]
# The clean ownership verdict rests on the first three (RS2-independent). doench2014 is kept as an honest
# in-sample POSITIVE CONTROL: RS2/Azimuth (Fusi-Doench 2016) shares training lineage with Doench 2014, so
# it should — and does — win there, confirming the probe detects real ownership rather than being rigged.
_RS2_INDEPENDENT = {"xu2015TrainHl60", "chari2015Train293T", "morenoMateos2015"}

# Deposited published tool scores kept per guide. `fusi` (Rule Set 2 / Azimuth) is the incumbent of
# record; `doench` (Rule Set 1) is the second incumbent; the rest round out the owned-tool panel.
_DEPOSITED_KEYS = ["fusi", "doench", "ssc", "crisprScan", "chariRaw", "wang", "wuCrispr"]


class DatasetUnavailable(RuntimeError):
    """A required `.scores.tab` could not be fetched/parsed and is not cached → SKIP (never fail offline)."""


@dataclass(frozen=True)
class KoRecord:
    """One measured CRISPRko guide: sequence at three resolutions + label + deposited tool scores."""

    guide: str                  # the dataset's guide id (unique within a dataset)
    dataset: str                # which source table this came from
    proto20: str                # 20-nt protospacer (the sequence-only view)
    seq23: str                  # 23-mer: protospacer + 3-nt PAM
    context30: str              # Azimuth 30-mer: 4 up + 20 protospacer + 3 PAM + 3 down
    activity: float             # ORIENTED efficacy: higher = more effective guide (see _orient_sign)
    activity_raw: float = 0.0   # the dataset's native label (modFreq) before orientation
    deposited: dict[str, float] = field(default_factory=dict)   # {tool: published score}, numeric only


def available_datasets() -> list[tuple[str, str]]:
    """The (key, label) ladder, in priority order."""
    return list(_DATASETS)


def rs2_independent(dataset: str) -> bool:
    """True if RS2/Azimuth did NOT train on this dataset's lineage (so its ρ here is a fair generalization,
    not in-sample). The clean ownership verdict uses only these; doench2014 is the in-sample control."""
    return dataset in _RS2_INDEPENDENT


# --------------------------------------------------------------------------- #
# Cache plumbing (~/.cache/karyon/, gitignored — mirrors screen_qc_data.py / crispr_qc_data.py).
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


def _cache_path(dataset: str) -> Path:
    return _cache_dir() / f"ko_efficacy_{dataset}.csv"


# --------------------------------------------------------------------------- #
# Fetch.
# --------------------------------------------------------------------------- #
def _fetch_text(url: str) -> str:
    if not network_allowed():
        raise DatasetUnavailable("network disabled via KARYON_NO_NETWORK")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        return urllib.request.urlopen(req, timeout=_TIMEOUT_S).read().decode("utf-8", "replace")
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        raise DatasetUnavailable(f"cannot reach {url}: {e}") from e


# --------------------------------------------------------------------------- #
# Parse.
# --------------------------------------------------------------------------- #
def _derive_context30(seq23: str, long_seq: str) -> str | None:
    """The Azimuth 30-mer (4 up + 23-mer + 3 down) sliced from the strand-oriented window the guide
    reads forward in. Returns None when the guide is absent or sits too close to an edge to flank."""
    i = long_seq.find(seq23)
    if i < 4 or i + len(seq23) + 3 > len(long_seq):     # need 4 nt upstream + 3 nt downstream
        return None
    ctx = long_seq[i - 4: i + len(seq23) + 3]
    return ctx if len(ctx) == 30 and not (set(ctx) - set("ACGT")) else None


def _orient_sign(raws: list[float], deposited: list[dict[str, float]]) -> int:
    """+1 if the native label already reads "higher = more effective", −1 if it is reversed (e.g. a
    *depletion* fold-change, where an effective guide drops the count → a more-negative value).

    Self-calibrating, no hardcoded per-dataset flips: the published tools are all oriented higher=better,
    so the label's true direction is the MAJORITY sign of cov(tool, label) across the deposited panel —
    a vote of (up to) 7 independent tools, so no single tool (e.g. the RS2 we benchmark) decides it."""
    n = len(raws)
    if n < 10:
        return 1
    mean_raw = sum(raws) / n
    votes = 0
    for key in _DEPOSITED_KEYS:
        xs = [(d[key], r) for d, r in zip(deposited, raws) if key in d]
        if len(xs) < 10:
            continue
        ms = sum(s for s, _ in xs) / len(xs)
        cov = sum((s - ms) * (r - mean_raw) for s, r in xs)
        votes += 1 if cov > 0 else -1
    return 1 if votes >= 0 else -1


def _parse_scores(text: str, dataset: str) -> list[KoRecord]:
    reader = csv.reader(text.splitlines(), delimiter="\t")
    header = next(reader, None)
    if not header or "seq" not in header or "modFreq" not in header:
        raise DatasetUnavailable(f"{dataset}: not a scores.tab (got a redirect/HTML body?)")
    idx = {c: i for i, c in enumerate(header)}
    i_seq, i_mf = idx["seq"], idx["modFreq"]
    i_long = idx.get("longSeq100Bp")
    i_guide = idx.get("guide")
    if i_long is None:
        raise DatasetUnavailable(f"{dataset}: no longSeq100Bp column (cannot derive the 30-mer)")

    staged: list[tuple] = []
    for rec in reader:
        if len(rec) <= max(i_seq, i_mf, i_long):
            continue
        seq23 = rec[i_seq].strip().upper().replace("U", "T")
        if len(seq23) != 23 or set(seq23) - set("ACGT"):
            continue
        try:
            activity = float(rec[i_mf])
        except (ValueError, TypeError):
            continue
        ctx30 = _derive_context30(seq23, rec[i_long].strip().upper())
        if ctx30 is None:
            continue
        deposited: dict[str, float] = {}
        for k in _DEPOSITED_KEYS:
            j = idx.get(k)
            if j is not None and j < len(rec):
                try:
                    deposited[k] = float(rec[j])
                except (ValueError, TypeError):
                    pass
        guide = rec[i_guide].strip() if i_guide is not None and i_guide < len(rec) else f"g{len(staged)}"
        staged.append((guide, seq23, ctx30, activity, deposited))
    if not staged:
        raise DatasetUnavailable(f"{dataset}: parsed 0 usable guides (format drift?)")

    # Orient the label so higher = more effective (depletion screens like Xu store it reversed), by the
    # deposited-tool consensus — so the learned and owned arms are sign-comparable downstream.
    sign = _orient_sign([s[3] for s in staged], [s[4] for s in staged])
    return [KoRecord(g, dataset, seq23[:20], seq23, ctx30, sign * raw, raw, dep)
            for g, seq23, ctx30, raw, dep in staged]


# --------------------------------------------------------------------------- #
# Cache read/write (the small flat table only).
# --------------------------------------------------------------------------- #
_FIELDS = ["guide", "dataset", "proto20", "seq23", "context30", "activity", "activity_raw",
           *_DEPOSITED_KEYS]


def _write_cache(path: Path, recs: list[KoRecord]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_FIELDS)
        for r in recs:
            w.writerow([r.guide, r.dataset, r.proto20, r.seq23, r.context30, r.activity, r.activity_raw,
                        *(r.deposited.get(k, "") for k in _DEPOSITED_KEYS)])


def _read_cache(path: Path) -> list[KoRecord]:
    out: list[KoRecord] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            deposited = {}
            for k in _DEPOSITED_KEYS:
                v = row.get(k, "")
                if v != "":
                    try:
                        deposited[k] = float(v)
                    except ValueError:
                        pass
            out.append(KoRecord(row["guide"], row["dataset"], row["proto20"], row["seq23"],
                                row["context30"], float(row["activity"]),
                                float(row.get("activity_raw", row["activity"])), deposited))
    return out


def load_efficacy(dataset: str | None = None, *, refresh: bool = False) -> list[KoRecord]:
    """Load one CRISPRko efficacy dataset's guides (sequence at 20/23/30-nt + measured label + deposited
    tool scores). With `dataset=None`, walks the ladder and returns the first reachable/cached one. Raises
    `DatasetUnavailable` only when *nothing* in the requested scope is reachable nor cached (→ SKIP)."""
    targets = [dataset] if dataset is not None else [d for d, _ in _DATASETS]
    last_err: DatasetUnavailable | None = None
    for name in targets:
        path = _cache_path(name)
        if path.exists() and not refresh:
            recs = _read_cache(path)
            if recs:
                return recs
        try:
            recs = _parse_scores(_fetch_text(f"{_BASE}/{name}.scores.tab"), name)
        except DatasetUnavailable as e:
            last_err = e
            continue
        _write_cache(path, recs)
        return recs
    raise last_err or DatasetUnavailable(f"no dataset reachable in {targets}")


if __name__ == "__main__":
    print("Loading CRISPRko on-target efficacy datasets (CRISPOR/Haeussler aggregate)\n")
    any_ok = False
    for name, label in _DATASETS:
        try:
            recs = load_efficacy(name)
        except DatasetUnavailable as e:
            print(f"  SKIP {name:<20} — {e}")
            continue
        any_ok = True
        acts = [r.activity for r in recs]
        with_rs2 = sum(1 for r in recs if "fusi" in r.deposited)
        ctx_ok = all(len(r.context30) == 30 for r in recs)
        flipped = sum(1 for r in recs if r.activity != r.activity_raw) > 0
        print(f"  {name:<20} {len(recs):>4} guides | activity(oriented) "
              f"[{min(acts):+.3g},{max(acts):+.3g}] med {statistics.median(acts):+.3g} | "
              f"RS2(fusi) {with_rs2}/{len(recs)} | 30-mer ok={ctx_ok} | label-reversed={flipped}")
        print(f"        {label}")
    if not any_ok:
        print("\n  (all datasets unreachable — offline; the probe SKIPs)")
        raise SystemExit(0)
