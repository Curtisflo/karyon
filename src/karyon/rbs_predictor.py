"""rbs_predictor — probe #2's REAL predictor margin: our learned core vs the RBS Calculator, de-novo.

The honest test the the design notes scoreboard claims for RBS: does a learned model
beat the biophysical RBS Calculator on *de-novo* designs? Here both are evaluated on a
**leave-one-study-out (LOSO)** split over SynBioMTS's E. coli designed-RBS studies
([rbs_synbiomts_data.py]) — for each held-out study, neither model has seen it (our core was trained
on the *other* studies; the RBS Calculator is biophysical, untrained). Scored by **within-study
Spearman ρ** (rank — scale-free across the studies' different reporters/units), exactly how the RBS
Calculator is benchmarked.

  * baseline = the RBS Calculator via **OSTIR** (open-source; ViennaRNA) — the actual biophysical tool,
    run on each construct at its known start codon. (This file admits OSTIR + ViennaRNA, which probe
    #2 authorizes for a fair baseline; the learned core [linmodel.py] stays stdlib. OSTIR predictions
    are cached; if OSTIR is absent the learned-core LOSO still runs from cache, baseline marked n/a.)
  * learned core = `linmodel` k-mer ridge over the RBS window (5'UTR footprint + early CDS), target =
    per-study-standardized log10(expression) so the pooled cross-study regression is scale-consistent.

PRE-REGISTERED expectation: with only ~300 cross-study training sequences this is a HARD, honest test —
the RBS Calculator is on home turf (these are its validation sets). If the cheap core matches/beats it
de-novo, that is a real desk edge; if not, the finding is that the RBS predictor margin is *data-hungry*
(needs the big sets), i.e. not a cheap-desk win — which is itself decision-relevant for the scoreboard.

    python -m karyon.rbs_predictor
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict

from . import linmodel as lm
from . import rbs_synbiomts_data as rd
from . import stats_kit as sk

try:
    import ostir
except ImportError:                                  # baseline dep; learned core still runs from cache
    ostir = None

KS = (1, 2, 3)                # k-mer spectra over the RBS window
WIN_UP, WIN_DN = 35, 15       # window = [startpos-35, startpos+15): SD/standby footprint + early CDS
LAM = 1.0


def _window(rec: rd.Record) -> str:
    return rec.sequence[max(0, rec.startpos - WIN_UP): rec.startpos + WIN_DN]


def _features(records: list[rd.Record]) -> list[list[float]]:
    return [lm.featurize(_window(r), KS) for r in records]


def _zscore_by_study(records: list[rd.Record], targets: list[float]) -> list[float]:
    """Standardize the target WITHIN each study, so pooling studies of different scales is coherent."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[r.dataset].append(i)
    z = [0.0] * len(records)
    for ids in groups.values():
        vals = [targets[i] for i in ids]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
        for i in ids:
            z[i] = (targets[i] - m) / sd
    return z


# --------------------------------------------------------------------------- #
# OSTIR baseline (cached).
# --------------------------------------------------------------------------- #
def _ostir_cache_path():
    return rd._cache_path().parent / "rbs_ostir.csv"


def _ostir_one(rec: rd.Record) -> float | None:
    try:
        res = ostir.run_ostir(rec.sequence, start=rec.startpos + 1, threads=1, verbosity=0)
    except Exception:
        return None
    match = [d for d in res if d.get("start_position") == rec.startpos + 1]
    chosen = match[0] if match else (res[0] if res else None)
    return None if chosen is None else chosen.get("expression")


def ostir_predict(records: list[rd.Record], refresh: bool = False) -> dict[str, float | None]:
    """sequence -> OSTIR predicted expression (None if it failed), cached to `~/.cache/karyon/`."""
    path = _ostir_cache_path()
    cache: dict[str, float | None] = {}
    if path.exists() and not refresh:
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                cache[row["sequence"]] = float(row["expr"]) if row["expr"] else None
    todo = [r for r in records if r.sequence not in cache]
    if todo and ostir is None:
        print(f"  [ostir] not installed — baseline n/a ({len(cache)} cached)")
        return cache
    if todo:
        print(f"  [ostir] scoring {len(todo)} constructs ({len(cache)} cached)…")
        for r in todo:
            cache[r.sequence] = _ostir_one(r)
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sequence", "expr"])
            for seq, e in cache.items():
                w.writerow([seq, "" if e is None else e])
    return cache


# --------------------------------------------------------------------------- #
# Leave-one-study-out predictor margin.
# --------------------------------------------------------------------------- #
def _spearman(xs, ys) -> float | None:
    r = sk.spearman(xs, ys)
    return r.rho if isinstance(r, sk.Corr) else None


def run(refresh: bool = False, refresh_ostir: bool = False) -> dict:
    records = rd.load_records(refresh=refresh)
    studies = sorted({r.dataset for r in records})
    X = _features(records)
    logp = [math.log10(r.prot_mean) for r in records]
    ostir_pred = ostir_predict(records, refresh=refresh_ostir)
    have_ostir = any(v is not None for v in ostir_pred.values())

    print("\n=== rbs_predictor: learned core vs the RBS Calculator (OSTIR), leave-one-study-out ===")
    print(f"  {len(records)} E. coli RBS constructs, {len(studies)} studies; feature dim={len(X[0])}; "
          f"within-study Spearman ρ (scale-free); de-novo = held-out study unseen by either model.")
    print(f"\n      {'held-out study':<32}{'n':>5}{'ours ρ':>10}{'RBS-Calc ρ':>13}")
    our_rhos, ost_rhos = [], []
    for s in studies:
        tr = [i for i, r in enumerate(records) if r.dataset != s]
        te = [i for i, r in enumerate(records) if r.dataset == s]
        z = _zscore_by_study([records[i] for i in tr], [logp[i] for i in tr])
        model = lm.BayesRidge(len(X[0]), lam=LAM)
        model.observe_all([X[i] for i in tr], z)
        truth = [logp[i] for i in te]
        our = _spearman([model.predict(X[i]) for i in te], truth)
        ost = _spearman([ostir_pred.get(records[i].sequence) for i in te], truth) if have_ostir else None
        our_rhos.append(our)
        ost_rhos.append(ost)
        print(f"      {s:<32}{len(te):>5}{_fmt(our):>10}{_fmt(ost):>13}")
    our_mean = _mean([r for r in our_rhos if r is not None])
    ost_mean = _mean([r for r in ost_rhos if r is not None]) if have_ostir else None
    print(f"      {'— mean —':<32}{'':>5}{_fmt(our_mean):>10}{_fmt(ost_mean):>13}")

    print()
    if not have_ostir:
        print("  RBS Calculator baseline unavailable (OSTIR/ViennaRNA not installed) — only the learned")
        print("  core's de-novo ρ is reported. The published RBS Calculator de-novo figure is ≈<0.2 (weak).")
    elif our_mean is not None and ost_mean is not None:
        if our_mean >= ost_mean:
            print(f"  WIN: the cheap learned core MATCHES/BEATS the RBS Calculator de-novo "
                  f"(ours {our_mean:+.3f} ≥ RBS-Calc {ost_mean:+.3f}) — a real predictor edge with no biophysics.")
        else:
            print(f"  RBS Calculator wins de-novo (ours {our_mean:+.3f} < RBS-Calc {ost_mean:+.3f}). The "
                  f"predictor margin is DATA-HUNGRY — ~{len(records)} cross-study seqs isn't enough for a")
            print(f"  cheap learned core to beat the biophysical tool on its home turf; it needs the big sets.")
    return {"studies": studies, "our": our_rhos, "ostir": ost_rhos,
            "our_mean": our_mean, "ostir_mean": ost_mean}


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RBS predictor margin: learned core vs RBS Calculator (LOSO)")
    ap.add_argument("--refresh", action="store_true", help="re-fetch the dataset")
    ap.add_argument("--refresh-ostir", action="store_true", help="recompute OSTIR predictions")
    args = ap.parse_args()
    try:
        run(refresh=args.refresh, refresh_ostir=args.refresh_ostir)
    except rd.DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
