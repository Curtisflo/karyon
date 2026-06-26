"""loop — the closed AI-operated DBTL loop: wire predict + choose + construct into ONE cycle.

the design notes / the roadmap flag the **AI-operated autonomous
loop** as the one edge pillar still "unproven, integration-level": every probe so far froze one core
and held the rest fixed; *nothing wires all three into a closed cycle.* This module does. Each cycle:

    declare a spec → CONSTRUCT candidates (the construct core) → CHOOSE the cheapest, most-informative
    to measure (the choose core) → INGEST the readout (oracle) → UPDATE the predictor → recurse.

It is **batch Bayesian optimization with a constructive proposal stage**, run RETROSPECTIVELY against a
cached real dataset (the oracle is a lookup; the loop only chooses what to build & reveal). We claim
**integration** — the loop closes, runs end-to-end, and is label-efficient — **not a novel algorithm**
(the cores are commoditized: FLEXS/ALDE, Apache-2.0). The edge is the proprietary same-assay dataloop +
autonomous operation on the open expression-element substrate, which this makes real at the desk.

Substrate: **EMOPEC** Shine-Dalgarno hexamers — the only substrate with near-complete-space truth
(3,070 of 4⁶=4096 measured), so CONSTRUCT can generate **off the measured pool** over the full feasible
4⁶ space and the loop can still "measure" ~88% of what it builds. That off-pool generation is what makes
construct and choose do *different* work (else both reduce to greedy on the same finite pool); the
harness PRINTS a selection-overlap diagnostic to show whether they actually separate on this substrate
or fuse (a clean finding either way).

The three contrasts, all at equal total budget, 3 seeds, the program's ≥20% sign-consistent bar:
  * **L**  — full loop: greedy CONSTRUCT (off-pool) + UCB CHOOSE + recurse.
  * **B1** — one-shot: fit on the seed, design the whole budget at once, measure once. *Ablates
             recursion* → the HEADLINE contrast (does the dataloop buy anything over design-once?).
  * **B0** — naive loop: random construct + random choose, recursing. The floor + the random control
             for the per-cycle compounding metric Δ_c.

HONEST, PRE-STATED: EMOPEC is the loop's BEST case (6-nt, near-complete, low-noise, ρ≈0.79). One-shot
already captures ~85% of the oracle (the constructive-core results), so the dataloop's headroom over
design-once is *structurally small here*. We therefore expect to prove **integration + non-inferiority +
a lower bound on compounding**, and to likely see **front-loading** (Δ_c positive early, decaying), not
durable compounding — whose real test is a mid-learnability, non-enumerable substrate (promoter, Phase
2). A null (L ≈ B1) is a reported finding, not a failure.

    cd bio/probe && python loop.py --seeds 3
"""

from __future__ import annotations

import argparse
import functools
import random
import statistics
import time
from dataclasses import dataclass, replace
from typing import Callable

from . import acquisition as acq  # the CHOOSE core (shared primitive)
from . import constructive_core as cc  # the CONSTRUCT core (declare→derive) + EMOPEC featurizer, reused verbatim
from . import emopec_data as ed  # the oracle / substrate loader (EMOPEC)
from . import linmodel as lm  # the PREDICT core (BayesRidge: predict + variance + incremental update)
from . import promoter_data as pmd  # the 2nd substrate loader (σ70 promoter; stdlib-only, no t0 dep)
from . import stats_kit as sk  # spearman

MARGIN = 0.20                     # the program's pre-registered "meaningful win" bar (mirrors active_learning)
SEED_BASE = 7000


@dataclass(frozen=True)
class LoopConfig:
    seeds: int = 3
    seed_size: int = 64           # initial measured set (counts toward the budget)
    cycles: int = 10              # design-measure-learn rounds for the recursive strategies
    batch: int = 48               # measurements committed per cycle (k)
    propose_m: int = 200          # candidates CONSTRUCT proposes per cycle (M ≫ k → room for CHOOSE)
    beta: float = 1.0             # UCB exploration weight in CHOOSE
    top_q: float = 0.05           # "winners" = the true top 5% of the acquirable pool
    test_frac: float = 0.20       # held-out, never-measurable set for the predictor-ρ context curve
    lam: float = 1.0
    best_n: int = 20              # design-quality read = mean true value of the best-N measured

    @property
    def budget(self) -> int:
        return self.seed_size + self.cycles * self.batch


@dataclass(frozen=True)
class Verdict:
    name: str
    ok: bool
    metric: str
    detail: str


# --------------------------------------------------------------------------- #
# Substrate — a thin adapter (loader + featurizer + feasibility + truth oracle). Two substrates: EMOPEC
# (near-complete 4⁶ → construct generates OFF-POOL) and the σ70 promoter (10.9k deposited of a 4¹⁵⁰
# space → construct is POOL-RESTRICTED). A new substrate is a loader + a featurizer, not a rewrite.
# Featurizers are memoized — features are a pure function of sequence, so caching turns the per-cycle
# full-pool re-ranking into O(1) lookups (the model, not the featurizer, is what changes each cycle).
# The cached list is only ever READ (predict/observe), so sharing one object per sequence is safe.
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _emopec_featurize(seq: str) -> list[float]:
    return cc._featurize(seq)                    # positional one-hot + dinuc (the constructive probe's surface)


@functools.lru_cache(maxsize=None)
def _promoter_featurize(seq: str) -> list[float]:
    gc = (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0
    return lm.featurize(seq, (1, 2, 3)) + [gc]   # k-mer spectra + GC (promoter_la_fleur's surface, ρ≈0.51)


@dataclass
class Substrate:
    name: str
    truth: dict[str, float]              # sequence -> true measured function (the oracle)
    feasible_pool: list[str]             # the design space CONSTRUCT generates/selects over
    featurize: Callable[[str], list[float]]
    seq_len: int
    is_feasible: Callable[[str], bool]   # the spec's hard constraints (trivially True for a vetted pool)
    enumerable: bool                     # True ⟹ off-pool generation possible; False ⟹ pool-restricted


def emopec_substrate(refresh: bool = False) -> Substrate:
    """EMOPEC: truth over the measured SD hexamers; design space = the full feasible 4⁶ space.

    Reuses `constructive_core` — the SAME feasibility predicate + featurizer the constructive probe
    validated (held-out ρ≈0.79). Near-complete-space truth ⟹ construct generates OFF the measured pool."""
    records = ed.load_records(refresh=refresh)
    truth = {r.sd: r.expression for r in records}
    feasible = [s for s in cc.all_hexamers() if cc.is_feasible(s)]
    return Substrate("EMOPEC SD-hexamer", truth, feasible, _emopec_featurize, ed.SD_LEN,
                     cc.is_feasible, enumerable=True)


def promoter_substrate(refresh: bool = False) -> Substrate:
    """σ70 promoter (La Fleur/Salis Urtecho): 10,898 deposited 150-nt promoters of a 4¹⁵⁰ space.

    The MID-learnability test (k-mer ridge ρ≈0.51) — the model starts genuinely wrong, so recursion has
    room to compound (the durable-compounding test EMOPEC's saturation cannot run). The space is NOT
    enumerable, so construct is POOL-RESTRICTED: it ranks the deposited pool minus what's been measured.
    Every deposited promoter is vetted & measurable, so `is_feasible` is trivially True and there are no
    coverage skips. Featurizer = promoter_la_fleur's (k-mer spectra + GC); target = log(TX)."""
    records = pmd.load_records(refresh=refresh)
    truth = {r.seq: r.strength for r in records}        # log(TX) — the optimisation target
    pool = list(truth)                                   # the deposited design space (pool-restricted)
    return Substrate("La Fleur σ70 promoter", truth, pool, _promoter_featurize, pmd.PROMOTER_LEN,
                     lambda s: True, enumerable=False)


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #
def _topset(measure_truth: dict[str, float], q: float) -> set[str]:
    """The true top-q fraction of the acquirable pool — the 'winners' discovery aims to recover."""
    items = sorted(measure_truth, key=lambda s: measure_truth[s], reverse=True)
    return set(items[: max(1, int(len(items) * q))])


def _recall(measured: set[str], topset: set[str]) -> float:
    return len(measured & topset) / len(topset) if topset else 0.0


def _best_n_mean(measured: set[str], truth: dict[str, float], n: int) -> float:
    vals = sorted((truth[s] for s in measured if s in truth), reverse=True)[:n]
    return statistics.mean(vals) if vals else 0.0


def _rho_on(model: lm.BayesRidge, feat, test_seqs: list[str], truth: dict[str, float]) -> float:
    """Held-out ranking ρ of the current predictor — the dataloop's effect ON the model (context)."""
    if len(test_seqs) < 3:
        return 0.0
    r = sk.spearman([model.predict(feat(s)) for s in test_seqs], [truth[s] for s in test_seqs])
    return r.rho if isinstance(r, sk.Corr) else 0.0


def _labels_to_reach(curve: list[tuple], target: float) -> int | None:
    """First label count at which recall ≥ target (mirrors active_learning._labels_to_reach)."""
    for n_lab, recall, *_ in curve:
        if recall >= target:
            return n_lab
    return None


# --------------------------------------------------------------------------- #
# One strategy's trajectory — the closed cycle (or one-shot when cycles=1).
# --------------------------------------------------------------------------- #
def _run_strategy(sub: Substrate, design_space: list[str], measure_truth: dict[str, float],
                  test_seqs: list[str], topset: set[str], seed_set: list[str],
                  construct_policy: str, choose_policy: str, cfg: LoopConfig, rng: random.Random,
                  *, cycles: int, batch: int, m: int, record_overlap: bool = False) -> dict:
    """Run declare→construct→choose→ingest→update for `cycles` rounds; record a metric per round.

    CONSTRUCT proposes `m` feasible candidates from the design space minus what's been measured
    (greedy = the construct core's model-argmax; random = the naive baseline). CHOOSE ranks them and
    we measure the first `batch` that have lookup-able truth (skip-and-refill the unmeasurable ~12% —
    never silently substitute the pool oracle; the skips are counted). INGEST is the oracle lookup;
    UPDATE folds the new labels into the predictor incrementally."""
    feat, truth = sub.featurize, sub.truth
    model = lm.BayesRidge(len(feat(seed_set[0])), lam=cfg.lam)
    model.observe_all([feat(s) for s in seed_set], [measure_truth[s] for s in seed_set])
    measured = set(seed_set)
    curve = [(len(measured), _recall(measured, topset),
              _best_n_mean(measured, truth, cfg.best_n), _rho_on(model, feat, test_seqs, truth))]
    overlaps: list[float] = []
    skips: list[int] = []

    for _ in range(cycles):
        pool = [s for s in design_space if s not in measured]
        if not pool:
            break
        # CONSTRUCT — rank the whole unmeasured feasible design space (the construct core's full
        # model-argmax order); CHOOSE operates on the top-M window of it.
        if construct_policy == "random":
            rng.shuffle(pool)
            ranked_seqs = pool
        else:
            ranked_seqs = cc.gen_constructive_exhaustive(pool, model, len(pool),
                                                         featurize=feat, is_feasible=sub.is_feasible)
        proposals = ranked_seqs[:m]
        Xp = [feat(s) for s in proposals]
        cand = list(range(len(proposals)))

        # Diagnostic: do CHOOSE (UCB) and pure exploit (greedy) pick DIFFERENT batches, or fuse?
        if record_overlap and len(proposals) > batch:
            ucb_sel = set(acq.acquire(model, Xp, cand, "ucb", batch, beta=cfg.beta))
            grd_sel = set(acq.acquire(model, Xp, cand, "greedy", batch))
            union = ucb_sel | grd_sel
            overlaps.append(len(ucb_sel & grd_sel) / len(union) if union else 1.0)

        # CHOOSE — UCB order over the window; measure the first `batch` with lookup-able truth.
        # Skip unmeasurable designs (coverage loss — counted, never faked); if the window is
        # measurable-poor, backstop down the construct ranking so each cycle commits a full batch
        # (keeps the budget exact across strategies).
        order = acq.acquire(model, Xp, cand, choose_policy, len(proposals), rng=rng, beta=cfg.beta)
        chosen, seen, n_skip = [], set(), 0
        for li in order:
            s = proposals[li]
            if s in measure_truth:
                chosen.append(s)
                seen.add(s)
                if len(chosen) >= batch:
                    break
            else:
                n_skip += 1
        if len(chosen) < batch:
            for s in ranked_seqs[m:]:
                if s in measure_truth and s not in seen:
                    chosen.append(s)
                    seen.add(s)
                    if len(chosen) >= batch:
                        break
        if not chosen:
            break

        # INGEST (oracle lookup) + UPDATE (incremental).
        model.observe_all([feat(s) for s in chosen], [measure_truth[s] for s in chosen])
        measured.update(chosen)
        skips.append(n_skip)
        curve.append((len(measured), _recall(measured, topset),
                      _best_n_mean(measured, truth, cfg.best_n), _rho_on(model, feat, test_seqs, truth)))

    return {"curve": curve, "overlaps": overlaps, "skips": skips, "measured": measured}


# --------------------------------------------------------------------------- #
# One RNG seed — L, B1, B0 (+ the β=0 ablation) off a SHARED seed set (paired).
# --------------------------------------------------------------------------- #
def _run_seed(sub: Substrate, seed: int, cfg: LoopConfig) -> dict:
    rng = random.Random(SEED_BASE + seed)
    acquirable_all = [s for s in sub.feasible_pool if s in sub.truth]
    rng.shuffle(acquirable_all)
    n_test = int(len(acquirable_all) * cfg.test_frac)
    test_seqs = acquirable_all[:n_test]
    test_set = set(test_seqs)

    design_space = [s for s in sub.feasible_pool if s not in test_set]
    measure_truth = {s: sub.truth[s] for s in design_space if s in sub.truth}
    topset = _topset(measure_truth, cfg.top_q)

    seed_pool = list(measure_truth.keys())
    rng.shuffle(seed_pool)
    seed_set = seed_pool[: cfg.seed_size]
    oneshot_batch = cfg.cycles * cfg.batch       # B1 measures the whole post-seed budget in one round

    def strat(construct, choose, cyc, bat, m, beta, off):
        c = replace(cfg, beta=beta)
        return _run_strategy(sub, design_space, measure_truth, test_seqs, topset, seed_set,
                             construct, choose, c, random.Random(SEED_BASE + seed + off),
                             cycles=cyc, batch=bat, m=m, record_overlap=(off == 1))

    L = strat("greedy", "ucb", cfg.cycles, cfg.batch, cfg.propose_m, cfg.beta, off=1)
    B0 = strat("random", "random", cfg.cycles, cfg.batch, cfg.propose_m, cfg.beta, off=2)
    B1 = strat("greedy", "greedy", 1, oneshot_batch, oneshot_batch * 2, cfg.beta, off=3)
    Lb0 = strat("greedy", "ucb", cfg.cycles, cfg.batch, cfg.propose_m, 0.0, off=1)   # choose-signal ablation

    return {"L": L, "B0": B0, "B1": B1, "Lb0": Lb0, "test_seqs": test_seqs,
            "n_acquirable": len(measure_truth), "n_winners": len(topset)}


# --------------------------------------------------------------------------- #
# Aggregation → verdicts.
# --------------------------------------------------------------------------- #
def _final(curve: list[tuple], idx: int = 1) -> float:
    return curve[-1][idx]


def _delta_curve(seed_results: list[dict]) -> list[tuple[int, float]]:
    """Mean per-cycle compounding advantage Δ_c = recall_L(@n) − recall_B0(@n), aligned by cycle."""
    n_pts = min(len(s["L"]["curve"]) for s in seed_results)
    out = []
    for i in range(n_pts):
        n = seed_results[0]["L"]["curve"][i][0]
        d = statistics.mean(s["L"]["curve"][i][1] - s["B0"]["curve"][i][1] for s in seed_results)
        out.append((n, d))
    return out


def _verdicts(seed_results: list[dict], cfg: LoopConfig) -> list[Verdict]:
    out: list[Verdict] = []

    # Context: the predictor's held-out ρ at the end of the loop (the model the cores ride).
    rhos = [_final(s["L"]["curve"], idx=3) for s in seed_results]
    out.append(Verdict("predictor held-out ρ at budget (context, not a claim)",
                       ok=statistics.mean(rhos) >= 0.10, metric=f"ρ={statistics.mean(rhos):+.3f}",
                       detail=f"per seed {[round(r, 3) for r in rhos]}"))

    # HEADLINE — recursion: L vs B1 (one-shot) final recall at equal budget.
    rec, rows = [], []
    for s in seed_results:
        lf, bf = _final(s["L"]["curve"]), _final(s["B1"]["curve"])
        p = (lf - bf) / bf if bf else 0.0
        rec.append(p)
        rows.append(f"L={lf:.1%} one-shot={bf:.1%} ({p:+.0%})")
    m = statistics.mean(rec)
    out.append(Verdict("dataloop: L (recurse) vs B1 (design-once) — top-5% recall at budget",
                       ok=(m >= MARGIN and all(p > 0 for p in rec)),
                       metric=f"{m:+.0%} recall", detail=" | ".join(rows)))

    # Label efficiency vs the naive loop: L final recall vs B0, + labels for L to match B0's final.
    rec2, rows2 = [], []
    for s in seed_results:
        lf, bf = _final(s["L"]["curve"]), _final(s["B0"]["curve"])
        p = (lf - bf) / bf if bf else 0.0
        match = _labels_to_reach(s["L"]["curve"], bf) or cfg.budget
        rec2.append(p)
        rows2.append(f"L={lf:.1%} naive={bf:.1%} ({p:+.0%}); L matches naive's final by {match} labels "
                     f"(−{(cfg.budget - match) / cfg.budget:.0%})")
    m2 = statistics.mean(rec2)
    out.append(Verdict("label efficiency: L vs B0 (naive loop) — top-5% recall at budget",
                       ok=(m2 >= MARGIN and all(p > 0 for p in rec2)),
                       metric=f"{m2:+.0%} recall", detail=" | ".join(rows2)))

    # Durable compounding: does the L−B0 gap HOLD or WIDEN across cycles (vs decay = front-loading)?
    durable, rows3 = [], []
    for s in seed_results:
        dc = [(s["L"]["curve"][i][1] - s["B0"]["curve"][i][1])
              for i in range(min(len(s["L"]["curve"]), len(s["B0"]["curve"])))]
        first, last = dc[1] if len(dc) > 1 else 0.0, dc[-1]
        durable.append(last >= first * 0.9 and last > 0)
        rows3.append(f"Δ first={first:+.1%} last={last:+.1%}")
    out.append(Verdict("sustained edge vs naive loop: Δ_c (L−B0 gap) non-decaying through budget",
                       ok=all(durable),
                       metric=("sustained" if all(durable) else "front-loaded"),
                       detail=" | ".join(rows3)))
    return out


# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
def _curve_at(curve: list[tuple], n: int, idx: int = 1) -> float:
    return min(curve, key=lambda pt: abs(pt[0] - n))[idx]


def report(seed_results: list[dict], verdicts: list[Verdict], cfg: LoopConfig, substrate: str) -> str:
    cps = [int(cfg.seed_size + f * (cfg.budget - cfg.seed_size)) for f in (0.25, 0.5, 0.75, 1.0)]
    head = "".join(f"{('@' + str(c)):>10}" for c in cps)
    L = [f"\n=== loop: closed DBTL (construct+choose+update) vs one-shot & naive on {substrate} "
         f"(seeds={cfg.seeds}) ===",
         f"  budget={cfg.budget} (seed {cfg.seed_size} + {cfg.cycles}×{cfg.batch}); "
         f"acquirable winners (true top-{cfg.top_q:.0%}) = {seed_results[0]['n_winners']} "
         f"of {seed_results[0]['n_acquirable']}; propose M={cfg.propose_m}.",
         "  Pre-registered: a win clears ≥20% AND is sign-consistent across seeds; a null is reported."]

    # Discovery curves (top-q recall ↑) for the three strategies.
    L.append(f"\n  top-{cfg.top_q:.0%} recall at n labels  (the loop curve should dominate; B1 is one-shot)")
    L.append(f"      {'(recall)':<26}{head}")
    for name, key in (("B0 naive loop", "B0"), ("B1 one-shot", "B1"), ("L full loop (ours)", "L")):
        curve_mean = [(c, statistics.mean(_curve_at(s[key]["curve"], c) for s in seed_results)) for c in cps]
        cells = "".join(f"{v * 100:>9.1f}%" for _, v in curve_mean)
        L.append(f"      {name:<26}{cells}")

    # Design quality (best-N true value found) at budget.
    L.append(f"\n  best-{cfg.best_n} true value found at budget (design quality)")
    for name, key in (("B0 naive loop", "B0"), ("B1 one-shot", "B1"), ("L full loop (ours)", "L")):
        bestn = statistics.mean(_final(s[key]["curve"], idx=2) for s in seed_results)
        L.append(f"      {name:<26}{bestn:>9.3f}")

    # Δ_c compounding table (mean over seeds).
    L.append("\n  Δ_c per cycle = recall_L − recall_B0 at equal labels (positive+non-decaying = compounding)")
    dc = _delta_curve(seed_results)
    L.append("      " + "".join(f"{('@' + str(n)):>9}" for n, _ in dc))
    L.append("      " + "".join(f"{d * 100:>+8.1f}%" for _, d in dc))

    # Separability diagnostic (the anti-theater instrument).
    ov = [o for s in seed_results for o in s["L"]["overlaps"]]
    lb0 = statistics.mean(_final(s["Lb0"]["curve"]) for s in seed_results)
    lf = statistics.mean(_final(s["L"]["curve"]) for s in seed_results)
    skips = [k for s in seed_results for k in s["L"]["skips"]]
    mean_skip = statistics.mean(skips) if skips else 0.0
    cov = (f"off-pool coverage — {mean_skip:.0f} of M={cfg.propose_m} proposals/cycle unmeasurable "
           f"(construct generates beyond the measured pool)" if mean_skip >= 1 else
           f"pool-restricted — construct selects from the deposited design space (no coverage skips)")
    L.append("\n  diagnostics:")
    L.append(f"      CHOOSE separability — Jaccard(UCB batch, greedy batch) mean = {statistics.mean(ov):.2f} "
             f"({'≈1 ⟹ construct/choose FUSE here' if statistics.mean(ov) > 0.9 else 'separable'})")
    L.append(f"      β-ablation — L(β={cfg.beta}) recall {lf:.1%} vs L(β=0, =greedy choose) {lb0:.1%} "
             f"(Δ {lf - lb0:+.1%} ⟹ the explore signal {'adds little here' if abs(lf - lb0) < 0.02 else 'matters'})")
    L.append(f"      {cov}")

    # Verdicts.
    L.append("\n  verdicts:")
    for v in verdicts:
        L.append(f"    [{'WIN' if v.ok else ' · '}] {v.name:<52}{v.metric:>14}")
        L.append(f"          {v.detail}")
    wins = sum(v.ok for v in verdicts[1:])
    L.append(f"\n  {wins}/3 loop claims clear the ≥{MARGIN:.0%} bar, sign-consistent across seeds.")
    return "\n".join(L)


def run(cfg: LoopConfig, substrate: Substrate | None = None, refresh: bool = False) -> dict:
    sub = substrate or emopec_substrate(refresh=refresh)
    mode = "off-pool over the full feasible space" if sub.enumerable else "pool-restricted (the deposited set)"
    print(f"  substrate: {sub.name} — {len(sub.truth)} measured, {len(sub.feasible_pool)} in the design "
          f"space; construct = {mode}; λ={cfg.lam}, β={cfg.beta}")
    t0 = time.time()
    seed_results = [_run_seed(sub, s, cfg) for s in range(cfg.seeds)]
    verdicts = _verdicts(seed_results, cfg)
    print(report(seed_results, verdicts, cfg, sub.name))
    print(f"\n  ({cfg.seeds} seeds × (L + B1 + B0 + β0) in {time.time() - t0:.1f}s)")
    return {"seed_results": seed_results, "verdicts": verdicts, "cfg": cfg, "substrate": sub.name}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="closed DBTL loop: construct+choose+update vs one-shot & naive")
    ap.add_argument("--substrate", choices=("emopec", "promoter"), default="emopec",
                    help="emopec = near-complete 4⁶ (off-pool construct); promoter = mid-ρ, pool-restricted")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed-size", type=int, default=64)
    ap.add_argument("--propose-m", type=int, default=200)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--refresh", action="store_true", help="re-fetch even if cached")
    args = ap.parse_args()
    cfg = LoopConfig(seeds=args.seeds, cycles=args.cycles, batch=args.batch, seed_size=args.seed_size,
                     propose_m=args.propose_m, beta=args.beta)
    try:
        sub = (promoter_substrate(refresh=args.refresh) if args.substrate == "promoter"
               else emopec_substrate(refresh=args.refresh))
        run(cfg, substrate=sub)
    except (ed.DatasetUnavailable, pmd.DatasetUnavailable) as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
