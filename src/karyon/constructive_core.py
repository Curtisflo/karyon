"""constructive_core — does declare→derive GENERATION beat naive generation on TRUE function?

The three named cores of the design notes are **predict** (a learned predictor),
**choose** (active learning), and **construct** (declare a functional spec → DERIVE candidate
sequences under sound contracts — a DRC doctrine ported to sequence). RANK (AL) and
PREDICT (predictor margin) have desk evidence on real data. The **constructive core is the one still
unproven on measured truth**: the rungs (earlier constructive-design probes) proved declare→derive is
*ergonomic* (fewer dumb moves, feasibility-by-construction) — they NEVER checked that its *output
quality* beats naive generation against a gel.

This probe checks exactly that, on the one substrate with **lookup-able truth over (almost) the whole
space**: EMOPEC's Shine-Dalgarno hexamers ([emopec_data.py]). The SD space is 4^6 = 4096 hexamers and
~3,070 are *measured* — so we can GENERATE a hexamer and look up its true expression. The sharp,
load-bearing question:

  **Does optimizing a cheap learned model's predicted expression under sound constraints yield
  sequences with high TRUE measured expression — or does it just exploit the model's blind spots
  (the classic "adversarial to your own predictor" failure)?**

That distinction decides the thesis: if the generator finds model artifacts, the constructive loop
*needs the wet dataloop to correct it* (which is the edge story); if it finds genuinely good
sequences, the constructive core can stand at the desk.

Three methods, all generating top-N hexamers, all feasibility-checked the same way:
  * **naive** — sample random *valid* (constraint-passing) hexamers. The competent-scripter generator.
  * **constructive (exhaustive)** — declare `maximize predicted expression s.t. GC band + no long
    homopolymer run`; DERIVE by scoring the LEARNED MODEL over the feasible space and taking the top
    N. Feasibility-by-construction; this is the *exact* optimum of the declared objective.
  * **constructive (local search)** — the same spec reached by hill-climbing single-base edits from
    random feasible starts (the realistic derive when the space is too big to enumerate). Sanity that
    the win is the *objective*, not the enumeration.

Two reads:
  (i) HEADLINE — among each method's top-N, the mean/median TRUE measured expression (lookup). Does
      constructive beat naive on TRUE function?
  (ii) ARTIFACT CHECK (the decisive one) — train the model on a SUBSET, generate restricted to
      candidates whose truth lies in the HELD-OUT split (the model never saw them), then score by that
      held-out truth. If constructive's true-function advantage SURVIVES → it finds genuinely good
      sequences. If it COLLAPSES (esp. below naive) → it found model artifacts.

HONEST BOUND (stated up front): the optimizer's objective is OUR OWN learned model — this is the desk
stand-in for a wet design-build-test loop, not a wet result. The artifact check is precisely the test
of whether that stand-in is trustworthy without the wet correction. EMOPEC is also an easy, near-
complete, low-noise 6-nt space; absolute margins reflect that ease (the *direction* is the finding).

    python -m karyon.constructive_core --seeds 5
"""

from __future__ import annotations

import argparse
import itertools
import random
import statistics
from dataclasses import dataclass

from . import emopec_data as ed  # the loader (3,070 measured SD hexamers), reused VERBATIM
from . import linmodel as lm  # the learned core (BayesRidge + featurizers), reused VERBATIM
from . import stats_kit as sk  # spearman/pearson, reused VERBATIM

SD_LEN = ed.SD_LEN
_BASES = "ACGT"

# Sound synthesizability constraints (the declarative spec's "subject to ..."). These are real
# oligo-synthesis failure modes — wide GC bands and long homopolymer runs (poly-G quadruplexes,
# poly-A slippage) hurt synthesis — and they are the SAME class `dfm.py` owns per-oligo. On a 6-nt
# hexamer they are deliberately permissive (a hexamer can't carry much), so they constrain the space
# without trivially selecting for expression: GC↔expression is only ρ≈0.27 on EMOPEC (measured below),
# so a GC band is NOT a backdoor expression filter.
GC_BAND = (1 / 3, 2 / 3)         # 2..4 of 6 bases are G/C
MAX_RUN = 3                      # no run of 4+ identical bases (AAAA, GGGG, ...)


# --------------------------------------------------------------------------- #
# The sound constraints (a feasibility predicate over a bare hexamer).
# --------------------------------------------------------------------------- #
def gc_fraction(seq: str) -> float:
    return sum(c in "GC" for c in seq) / len(seq) if seq else 0.0


def longest_run(seq: str) -> int:
    best = run = 0
    prev = ""
    for c in seq:
        run = run + 1 if c == prev else 1
        prev = c
        best = max(best, run)
    return best


def is_feasible(seq: str) -> bool:
    """The declared spec's hard constraints — sound, deterministic, sequence-only."""
    return (set(seq) <= set(_BASES)
            and GC_BAND[0] - 1e-9 <= gc_fraction(seq) <= GC_BAND[1] + 1e-9
            and longest_run(seq) <= MAX_RUN)


def all_hexamers() -> list[str]:
    """The full 4^6 = 4096 SD space."""
    return ["".join(p) for p in itertools.product(_BASES, repeat=SD_LEN)]


# --------------------------------------------------------------------------- #
# Features — the SAME featurization rbs_emopec.py uses (so this is the cheap learned core that
# already ranks SD expression at ρ≈0.79), exposed for a bare sequence so the generator can score
# candidates it has never seen a label for.
# --------------------------------------------------------------------------- #
def _featurize(sd: str) -> list[float]:
    return [1.0] + lm.position_onehot(sd, SD_LEN) + lm.featurize(sd, (2,))[1:]


def _fit_model(train_seqs: list[str], train_y: list[float], lam: float) -> lm.BayesRidge:
    p = len(_featurize("A" * SD_LEN))
    model = lm.BayesRidge(p, lam=lam)
    model.observe_all([_featurize(s) for s in train_seqs], train_y)
    return model


# --------------------------------------------------------------------------- #
# The generators. All return a ranked list of distinct hexamers (best-first), drawn from `pool`.
# --------------------------------------------------------------------------- #
def gen_naive(pool: list[str], n: int, rng: random.Random) -> list[str]:
    """Competent scripter: random VALID hexamers (constraint-passing). No model used."""
    feasible = [s for s in pool if is_feasible(s)]
    rng.shuffle(feasible)
    return feasible[:n]


def gen_constructive_exhaustive(pool: list[str], model: lm.BayesRidge, n: int,
                                featurize=_featurize, is_feasible=is_feasible) -> list[str]:
    """Declare→derive, exact: score the LEARNED MODEL over the feasible pool, take the top N.

    Feasibility-by-construction — infeasible candidates are never candidates, so the spec's
    constraints cannot be violated by the output (no post-hoc filtering, no re-loop). `featurize` and
    `is_feasible` default to the EMOPEC hexamer surface; another substrate (e.g. the loop's
    pool-restricted promoter design space) passes its own — the derive logic is substrate-agnostic."""
    feasible = [s for s in pool if is_feasible(s)]
    return sorted(feasible, key=lambda s: model.predict(featurize(s)), reverse=True)[:n]


def gen_constructive_local(pool: list[str], model: lm.BayesRidge, n: int,
                           rng: random.Random, restarts: int = 64) -> list[str]:
    """Declare→derive via hill-climbing single-base edits from random feasible starts.

    The realistic derive when the space is too big to enumerate: each restart climbs the model's
    predicted expression, stepping only to FEASIBLE neighbours (feasibility-by-construction holds
    along the whole search path). Returns the best N distinct optima found."""
    feasible_set = {s for s in pool if is_feasible(s)}
    if not feasible_set:
        return []
    pool_set = set(pool)
    found: dict[str, float] = {}
    starts = list(feasible_set)
    rng.shuffle(starts)
    for start in starts[:restarts]:
        cur = start
        cur_score = model.predict(_featurize(cur))
        while True:
            best_neighbour, best_score = cur, cur_score
            for i in range(SD_LEN):
                for b in _BASES:
                    if b == cur[i]:
                        continue
                    cand = cur[:i] + b + cur[i + 1:]
                    # stay on the measured pool AND feasible — feasibility holds every step
                    if cand not in pool_set or not is_feasible(cand):
                        continue
                    sc_ = model.predict(_featurize(cand))
                    if sc_ > best_score:
                        best_neighbour, best_score = cand, sc_
            if best_neighbour == cur:
                break
            cur, cur_score = best_neighbour, best_score
        found[cur] = cur_score
    return [s for s, _ in sorted(found.items(), key=lambda kv: kv[1], reverse=True)][:n]


# --------------------------------------------------------------------------- #
# Truth lookup + scoring.
# --------------------------------------------------------------------------- #
def _truth_table(records: list[ed.Record]) -> dict[str, float]:
    return {r.sd: r.expression for r in records}


@dataclass(frozen=True)
class MethodScore:
    name: str
    n: int
    true_mean: float
    true_median: float
    true_best: float
    measurable: int          # how many of the N picks had a measured truth value
    requested: int           # N requested (so coverage = measurable / requested for full-space gen)


def _score(name: str, picks: list[str], truth: dict[str, float], requested: int) -> MethodScore:
    vals = [truth[s] for s in picks if s in truth]
    if not vals:
        return MethodScore(name, 0, float("nan"), float("nan"), float("nan"), 0, requested)
    return MethodScore(name, len(vals), statistics.mean(vals), statistics.median(vals),
                       max(vals), len(vals), requested)


# --------------------------------------------------------------------------- #
# Read (i): HEADLINE — generate from the measured pool, score by true expression.
# --------------------------------------------------------------------------- #
def headline(records: list[ed.Record], seeds: int, n: int, lam: float) -> dict:
    """Model trained on the WHOLE measured set; generate from the measured pool; score by truth.

    This is the in-sample-favourable read (the model has seen every candidate's label) — it asks the
    weakest version of the question: even WITH a model fit on everything, does optimizing it pick
    high-true-expression hexamers, and does it beat naive? (The artifact check removes the in-sample
    advantage.) Naive is averaged over seeds; the two constructive methods are deterministic given the
    model, so only local-search (random restarts) varies with the seed."""
    measured_pool = [r.sd for r in records]
    truth = _truth_table(records)
    seqs = [r.sd for r in records]
    ys = [r.expression for r in records]
    model = _fit_model(seqs, ys, lam)

    naive_means, naive_meds = [], []
    local_means, local_meds = [], []
    for s in range(seeds):
        rng = random.Random(4000 + s)
        nv = _score("naive", gen_naive(measured_pool, n, rng), truth, n)
        lc = _score("constructive·local", gen_constructive_local(measured_pool, model, n, rng), truth, n)
        naive_means.append(nv.true_mean); naive_meds.append(nv.true_median)
        local_means.append(lc.true_mean); local_meds.append(lc.true_median)
    ex = _score("constructive·exhaustive", gen_constructive_exhaustive(measured_pool, model, n), truth, n)

    # The pool baseline: the true top-N (an oracle ceiling) and the pool median (a floor).
    pool_sorted = sorted(ys, reverse=True)
    oracle_topN_mean = statistics.mean(pool_sorted[:n])
    pool_median = statistics.median(ys)
    return {
        "n": n, "pool": len(records),
        "naive_mean": statistics.mean(naive_means), "naive_median": statistics.mean(naive_meds),
        "exhaustive_mean": ex.true_mean, "exhaustive_median": ex.true_median, "exhaustive_best": ex.true_best,
        "local_mean": statistics.mean(local_means), "local_median": statistics.mean(local_meds),
        "oracle_topN_mean": oracle_topN_mean, "pool_median": pool_median,
        "model_rho": _full_rho(records, lam),
    }


def _full_rho(records: list[ed.Record], lam: float) -> float:
    """Held-out test-ρ of the cheap core on this substrate (context — the ranking signal it carries)."""
    rng = random.Random(0)
    idx = list(range(len(records)))
    test = set(rng.sample(idx, len(idx) // 4))
    train = [i for i in idx if i not in test]
    model = _fit_model([records[i].sd for i in train], [records[i].expression for i in train], lam)
    test_sorted = sorted(test)
    r = sk.spearman([model.predict(_featurize(records[i].sd)) for i in test_sorted],
                    [records[i].expression for i in test_sorted])
    return r.rho if isinstance(r, sk.Corr) else 0.0


# --------------------------------------------------------------------------- #
# Read (ii): the ARTIFACT CHECK — the decisive experiment.
# --------------------------------------------------------------------------- #
def artifact_check(records: list[ed.Record], seeds: int, n: int, lam: float, test_frac: float) -> dict:
    """Train on a SUBSET; generate restricted to HELD-OUT candidates; score by held-out truth.

    For each seed: split the measured set into train / held-out test. Fit the model on TRAIN ONLY.
    The candidate pool for generation is the HELD-OUT test hexamers — the model has NEVER seen their
    labels. Both naive and constructive draw from this same held-out pool, so the only difference is
    the selection rule. If constructive's true-expression advantage SURVIVES on held-out truth, the
    generator is finding genuinely good sequences (the model's ranking generalizes). If it COLLAPSES
    — especially toward or below naive — the generator was exploiting in-sample model artifacts.

    Reported per method: mean/median TRUE held-out expression of the top-N picks, averaged over seeds;
    and the gap to the held-out ORACLE (true top-N within the held-out pool) — how much of the
    achievable true expression the generator captured without ever seeing these labels."""
    rows = []
    nv_means, nv_meds = [], []
    ex_means, ex_meds = [], []
    lc_means, lc_meds = [], []
    oracle_means, rhos = [], []
    for s in range(seeds):
        rng = random.Random(5000 + s)
        idx = list(range(len(records)))
        test_idx = set(rng.sample(idx, int(len(idx) * test_frac)))
        train_idx = [i for i in idx if i not in test_idx]
        model = _fit_model([records[i].sd for i in train_idx],
                           [records[i].expression for i in train_idx], lam)

        held_pool = [records[i].sd for i in sorted(test_idx)]
        truth = {records[i].sd: records[i].expression for i in test_idx}

        # held-out model ranking quality (the lever the generator rides)
        ts = sorted(test_idx)
        r = sk.spearman([model.predict(_featurize(records[i].sd)) for i in ts],
                        [records[i].expression for i in ts])
        rho = r.rho if isinstance(r, sk.Corr) else 0.0

        nv = _score("naive", gen_naive(held_pool, n, rng), truth, n)
        ex = _score("exhaustive", gen_constructive_exhaustive(held_pool, model, n), truth, n)
        lc = _score("local", gen_constructive_local(held_pool, model, n, rng), truth, n)
        oracle_mean = statistics.mean(sorted(truth.values(), reverse=True)[:n])

        nv_means.append(nv.true_mean); nv_meds.append(nv.true_median)
        ex_means.append(ex.true_mean); ex_meds.append(ex.true_median)
        lc_means.append(lc.true_mean); lc_meds.append(lc.true_median)
        oracle_means.append(oracle_mean); rhos.append(rho)
        rows.append(f"seed{s}: held-ρ={rho:+.3f}  naive μ={nv.true_mean:.3f}  "
                    f"constructive(exh) μ={ex.true_mean:.3f}  (local) μ={lc.true_mean:.3f}  "
                    f"oracle μ={oracle_mean:.3f}")

    def _m(xs): return statistics.mean(xs)
    nv_m, ex_m, lc_m, orc = _m(nv_means), _m(ex_means), _m(lc_means), _m(oracle_means)
    return {
        "n": n, "test_frac": test_frac, "seeds": seeds,
        "naive_mean": nv_m, "naive_median": _m(nv_meds),
        "exhaustive_mean": ex_m, "exhaustive_median": _m(ex_meds),
        "local_mean": lc_m, "local_median": _m(lc_meds),
        "oracle_mean": orc, "held_rho": _m(rhos),
        "lift_exhaustive": (ex_m - nv_m) / nv_m if nv_m else 0.0,
        "lift_local": (lc_m - nv_m) / nv_m if nv_m else 0.0,
        "capture_exhaustive": ex_m / orc if orc else 0.0,
        "capture_naive": nv_m / orc if orc else 0.0,
        # sign-consistency across seeds (a win must hold every seed, per the AL-probe discipline)
        "exhaustive_beats_naive_all": all(e > v for e, v in zip(ex_means, nv_means)),
        "local_beats_naive_all": all(l > v for l, v in zip(lc_means, nv_means)),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Read (iii): the React-test framing — dumb moves & feasibility-by-construction.
# --------------------------------------------------------------------------- #
def feasibility_audit(records: list[ed.Record], seeds: int, n: int, lam: float) -> dict:
    """Constructive output is feasible BY CONSTRUCTION; quantify how much hand-work naive needs.

    The naive generator must hand-filter for feasibility (GC band, homopolymer runs) and re-loop when
    the surviving set is short; the constructive derive never emits an infeasible candidate. We report
    (a) the full-space feasible fraction (the re-loop tax a naive *unconstrained* sampler pays), and
    (b) that 100% of constructive picks are feasible vs the rejection rate naive would hit sampling the
    raw 4^6 space."""
    full = all_hexamers()
    feasible_full = [s for s in full if is_feasible(s)]
    truth = _truth_table(records)
    measured_pool = [r.sd for r in records]
    model = _fit_model([r.sd for r in records], [r.expression for r in records], lam)

    # full-space constructive generation → measurable coverage (the honest "free generation" footnote)
    ex_full = gen_constructive_exhaustive(full, model, n)
    coverage = sum(1 for s in ex_full if s in truth) / len(ex_full) if ex_full else 0.0
    ex_full_scored = _score("exhaustive·full-space", ex_full, truth, n)
    return {
        "full_space": len(full), "feasible_full": len(feasible_full),
        "feasible_frac": len(feasible_full) / len(full),
        "naive_unconstrained_reject_rate": 1 - len(feasible_full) / len(full),
        "constructive_feasible_rate": 1.0,   # by construction
        "fullspace_coverage": coverage,
        "fullspace_measurable_mean": ex_full_scored.true_mean,
        "fullspace_measurable_n": ex_full_scored.measurable,
    }


# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
def run(seeds: int = 5, n: int = 50, lam: float = 1.0, test_frac: float = 0.5,
        refresh: bool = False) -> dict:
    records = ed.load_records(refresh=refresh)

    # context: GC is NOT a backdoor expression filter on this substrate
    gc_rho = sk.spearman([gc_fraction(r.sd) for r in records], [r.expression for r in records])
    gc_rho_v = gc_rho.rho if isinstance(gc_rho, sk.Corr) else 0.0

    print(f"  loaded {len(records)} EMOPEC SD hexamers (4^6 = {4 ** SD_LEN}; coverage "
          f"{len(records) / 4 ** SD_LEN:.0%}); top-N={n}, seeds={seeds}, λ={lam}")
    print(f"  declared spec: maximize predicted expression  s.t.  GC∈[{GC_BAND[0]:.2f},{GC_BAND[1]:.2f}], "
          f"no run>{MAX_RUN}.  (GC↔expression ρ={gc_rho_v:+.3f} — the GC band is not a backdoor "
          f"expression filter)")

    h = headline(records, seeds, n, lam)
    print(f"\n  --- Read (i) HEADLINE: generate from the measured pool, score by TRUE expression ---")
    print(f"      model held-out ρ (ranking signal) ............ {h['model_rho']:+.3f}")
    print(f"      pool median true expression (floor) .......... {h['pool_median']:.3f}")
    print(f"      naive (random valid)            top-{n} true μ {h['naive_mean']:.3f}  med {h['naive_median']:.3f}")
    print(f"      constructive·local  (hill-climb) top-{n} true μ {h['local_mean']:.3f}  med {h['local_median']:.3f}")
    print(f"      constructive·exhaustive (argmax) top-{n} true μ {h['exhaustive_mean']:.3f}  med {h['exhaustive_median']:.3f}  best {h['exhaustive_best']:.3f}")
    print(f"      ORACLE true top-{n} (ceiling) ................. μ {h['oracle_topN_mean']:.3f}")
    h_lift = (h['exhaustive_mean'] - h['naive_mean']) / h['naive_mean'] if h['naive_mean'] else 0.0
    print(f"      → constructive(exh) beats naive on TRUE μ by {h_lift:+.0%} (in-sample-favourable read)")

    a = artifact_check(records, seeds, n, lam, test_frac)
    print(f"\n  --- Read (ii) ARTIFACT CHECK: train on {1 - test_frac:.0%}, generate from HELD-OUT, "
          f"score by HELD-OUT truth ---")
    print(f"      (the decisive read: does the true-function advantage SURVIVE off the model's "
          f"training labels?)")
    for r in a["rows"]:
        print(f"      {r}")
    print(f"      mean over {seeds} seeds  (held-out ρ ≈ {a['held_rho']:+.3f}):")
    print(f"        naive .................... true μ {a['naive_mean']:.3f}  med {a['naive_median']:.3f}  "
          f"(captures {a['capture_naive']:.0%} of held-out oracle)")
    print(f"        constructive·local ....... true μ {a['local_mean']:.3f}  med {a['local_median']:.3f}  "
          f"({a['lift_local']:+.0%} vs naive; all seeds: {a['local_beats_naive_all']})")
    print(f"        constructive·exhaustive .. true μ {a['exhaustive_mean']:.3f}  med {a['exhaustive_median']:.3f}  "
          f"({a['lift_exhaustive']:+.0%} vs naive; all seeds: {a['exhaustive_beats_naive_all']})")
    print(f"        held-out ORACLE .......... true μ {a['oracle_mean']:.3f}  "
          f"(constructive captures {a['capture_exhaustive']:.0%})")
    survived = a["lift_exhaustive"] >= 0.20 and a["exhaustive_beats_naive_all"]
    print(f"      → VERDICT: the true-function advantage {'SURVIVES' if survived else 'COLLAPSES'} the "
          f"held-out artifact check "
          f"({'genuine good sequences' if survived else 'model exploitation — needs the wet dataloop'})")

    f = feasibility_audit(records, seeds, n, lam)
    print(f"\n  --- Read (iii) React-test: feasibility-by-construction & dumb-moves ---")
    print(f"      feasible fraction of the raw 4^6 space ........ {f['feasible_frac']:.0%} "
          f"(a naive UNCONSTRAINED sampler rejects {f['naive_unconstrained_reject_rate']:.0%})")
    print(f"      constructive feasible rate .................... {f['constructive_feasible_rate']:.0%} "
          f"(by construction — no hand-filter, no re-loop)")
    print(f"      full-space constructive top-{n}: {f['fullspace_coverage']:.0%} fall in the measured "
          f"set; those have true μ {f['fullspace_measurable_mean']:.3f}")

    return {"headline": h, "artifact": a, "feasibility": f, "survived": survived,
            "gc_rho": gc_rho_v}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="constructive-core probe: declare→derive generation vs naive, on TRUE function")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=50, help="top-N generated candidates per method")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.5, help="held-out fraction for the artifact check")
    ap.add_argument("--refresh", action="store_true", help="re-fetch even if cached")
    args = ap.parse_args()
    try:
        run(seeds=args.seeds, n=args.n, lam=args.lam, test_frac=args.test_frac, refresh=args.refresh)
    except ed.DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
