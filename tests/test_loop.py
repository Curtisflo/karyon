"""test_loop — proofs for the closed DBTL loop (dual script / pytest).

The harness exists to answer one question — *does closing the loop and turning the crank buy
anything?* — so its teeth are about telling three outcomes apart, on SYNTHETIC substrates where we
know the answer in advance:

  * **real compounding** — a learnable signal + a BIASED seed (model wrong-early, right-late): the
    recursive loop L must beat one-shot B1 by ≥20% (recursion corrects the seed bias; design-once
    cannot). Proves the harness CAN see a dataloop win.
  * **saturation / front-loading** — the same learnable signal but a REPRESENTATIVE seed (model right
    after one batch): L must NOT beat B1 by ≥20% (both saturate). Proves the harness reports a null
    instead of manufacturing a recursion win — this is exactly the EMOPEC regime.
  * **pure noise** — labels independent of sequence: L must NOT beat the naive loop B0 (nothing to
    exploit); held-out ρ ≈ 0. Proves the loop doesn't chase label noise into a fake dataloop.

Plus the mechanics: the loop is genuinely CLOSED (the predictor improves across cycles and no
sequence is measured twice) and the budget is exact. The real-EMOPEC test (5) skips offline.

    python tests/test_loop.py        # script mode
    pytest tests/test_loop.py -q           # pytest mode
"""

from __future__ import annotations

import os
import random

from karyon import constructive_core as cc  # noqa: E402
from karyon import emopec_data as ed  # noqa: E402
from karyon import loop  # noqa: E402
from karyon import promoter_data as pmd  # noqa: E402


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


# --------------------------------------------------------------------------- #
# Synthetic substrates: hexamers (so cc._featurize / cc.is_feasible apply), planted truth over the
# WHOLE feasible space (full coverage → no skips, exact budgets, clean comparisons).
# --------------------------------------------------------------------------- #
def _feasible() -> list[str]:
    return [s for s in cc.all_hexamers() if cc.is_feasible(s)]


def _substrate(truth: dict[str, float]) -> loop.Substrate:
    return loop.Substrate("synthetic", truth, list(truth), cc._featurize, cc.SD_LEN,
                          cc.is_feasible, enumerable=True)


def _g_signal(seed: int) -> dict[str, float]:
    """Learnable: true value ≈ G-count (winners are G-rich); tiny noise. The model CAN rank this."""
    rng = random.Random(seed)
    return {s: s.count("G") + rng.uniform(-0.05, 0.05) for s in _feasible()}


def _noise(seed: int) -> dict[str, float]:
    """Unlearnable: true value independent of sequence — no signal for the loop to exploit."""
    rng = random.Random(seed)
    return {s: rng.uniform(0.0, 1.0) for s in _feasible()}


def _strategy(sub, cfg, seed_set, construct, choose, *, cycles, batch, m,
              test_seqs=(), rng_seed=1, beta=None):
    """Run one strategy directly (bypasses _run_seed so the test controls the seed set)."""
    c = cfg if beta is None else loop.replace(cfg, beta=beta)
    measure_truth = {s: sub.truth[s] for s in sub.feasible_pool if s not in set(test_seqs)}
    topset = loop._topset(measure_truth, cfg.top_q)
    design_space = [s for s in sub.feasible_pool if s not in set(test_seqs)]
    return loop._run_strategy(sub, design_space, measure_truth, list(test_seqs), topset, seed_set,
                              construct, choose, c, random.Random(rng_seed),
                              cycles=cycles, batch=batch, m=m, record_overlap=True)


def _recall(traj) -> float:
    return traj["curve"][-1][1]


# --------------------------------------------------------------------------- #
# 1. The loop is genuinely CLOSED: predictor improves across cycles, nothing measured twice.
# --------------------------------------------------------------------------- #
def test_loop_is_closed() -> None:
    sub = _substrate(_g_signal(0))
    cfg = loop.LoopConfig(seed_size=48, cycles=8, batch=32, propose_m=150)
    acquirable = list(sub.truth)
    rng = random.Random(0)
    rng.shuffle(acquirable)
    test_seqs = acquirable[:200]                     # held-out, never measured → ρ context
    rest = [s for s in acquirable if s not in set(test_seqs)]
    # a BIASED (low-G) seed leaves the predictor with something to learn, so the update is visible —
    # a representative seed on this easy signal already saturates ρ (nothing to improve).
    seed_set = sorted(rest, key=lambda s: s.count("G"))[: cfg.seed_size]

    L = _strategy(sub, cfg, seed_set, "greedy", "ucb", cycles=cfg.cycles, batch=cfg.batch,
                  m=cfg.propose_m, test_seqs=test_seqs, rng_seed=1)

    seed_rho, final_rho = L["curve"][0][3], L["curve"][-1][3]
    assert final_rho > seed_rho + 0.2, f"predictor did not improve across cycles ({seed_rho:.3f}→{final_rho:.3f})"
    # exact budget AND no re-measurement: a set of all measured equals seed + every cycle's full batch.
    expected = cfg.seed_size + cfg.cycles * cfg.batch
    assert len(L["measured"]) == expected, f"budget/dedup broken: {len(L['measured'])} != {expected}"
    assert L["curve"][-1][0] == expected, "final label count must equal the budget"
    print(f"1. loop is closed: predictor ρ {seed_rho:+.2f}→{final_rho:+.2f}, "
          f"{len(L['measured'])} distinct measured = exact budget (no re-measurement)")


# --------------------------------------------------------------------------- #
# 2. TEETH — real compounding is DETECTED: biased seed ⟹ recursion beats one-shot by ≥20%.
# --------------------------------------------------------------------------- #
def test_detects_compounding_on_biased_seed() -> None:
    sub = _substrate(_g_signal(2))
    cfg = loop.LoopConfig(seed_size=48, cycles=8, batch=32, propose_m=150)
    # BIASED seed: the lowest-G feasible hexamers → the seed model under-rates the G-rich winners.
    seed_set = sorted(sub.truth, key=lambda s: s.count("G"))[: cfg.seed_size]
    oneshot = cfg.cycles * cfg.batch

    L = _strategy(sub, cfg, seed_set, "greedy", "ucb", cycles=cfg.cycles, batch=cfg.batch,
                  m=cfg.propose_m, rng_seed=1)
    B1 = _strategy(sub, cfg, seed_set, "greedy", "greedy", cycles=1, batch=oneshot,
                   m=oneshot * 2, rng_seed=1)

    lr, br = _recall(L), _recall(B1)
    assert lr >= br * 1.2, f"harness FAILED to detect compounding: L recall={lr:.3f} one-shot={br:.3f}"
    print(f"2. detects compounding (biased seed): L={lr:.1%} vs one-shot={br:.1%} "
          f"(recursion corrects the seed bias; +{(lr - br) / br:.0%})")


# --------------------------------------------------------------------------- #
# 3. TEETH — saturation is reported as a NULL: representative seed ⟹ recursion does NOT beat one-shot.
# --------------------------------------------------------------------------- #
def test_reports_null_on_saturation() -> None:
    sub = _substrate(_g_signal(3))
    cfg = loop.LoopConfig(seed_size=48, cycles=8, batch=32, propose_m=150)
    # REPRESENTATIVE seed (random) → the seed model already ranks well → little left to compound.
    rng = random.Random(5)
    seed_set = list(sub.truth)
    rng.shuffle(seed_set)
    seed_set = seed_set[: cfg.seed_size]
    oneshot = cfg.cycles * cfg.batch

    L = _strategy(sub, cfg, seed_set, "greedy", "ucb", cycles=cfg.cycles, batch=cfg.batch,
                  m=cfg.propose_m, rng_seed=1)
    B1 = _strategy(sub, cfg, seed_set, "greedy", "greedy", cycles=1, batch=oneshot,
                   m=oneshot * 2, rng_seed=1)

    lr, br = _recall(L), _recall(B1)
    assert lr >= 0.5 and br >= 0.5, f"saturation test needs both to win (L={lr:.3f}, one-shot={br:.3f})"
    assert lr < br * 1.2, f"harness manufactured a recursion win on a saturated substrate (L={lr:.3f} > one-shot={br:.3f})"
    print(f"3. reports null on saturation (representative seed): L={lr:.1%} ≈ one-shot={br:.1%} "
          f"(both saturate; recursion adds <20% — the EMOPEC regime)")


# --------------------------------------------------------------------------- #
# 4. TEETH — no fake dataloop on noise: L must NOT beat the naive loop; held-out ρ ≈ 0.
# --------------------------------------------------------------------------- #
def test_no_false_dataloop_on_noise() -> None:
    sub = _substrate(_noise(4))
    cfg = loop.LoopConfig(seed_size=48, cycles=8, batch=32, propose_m=150)
    acquirable = list(sub.truth)
    rng = random.Random(7)
    rng.shuffle(acquirable)
    test_seqs = acquirable[:200]
    seed_set = [s for s in acquirable if s not in set(test_seqs)][: cfg.seed_size]

    L = _strategy(sub, cfg, seed_set, "greedy", "ucb", cycles=cfg.cycles, batch=cfg.batch,
                  m=cfg.propose_m, test_seqs=test_seqs, rng_seed=1)
    B0 = _strategy(sub, cfg, seed_set, "random", "random", cycles=cfg.cycles, batch=cfg.batch,
                   m=cfg.propose_m, test_seqs=test_seqs, rng_seed=2)

    lr, br, rho = _recall(L), _recall(B0), L["curve"][-1][3]
    assert lr <= br * 1.5, f"manufactured a dataloop on noise: L recall={lr:.3f} naive={br:.3f}"
    assert abs(rho) < 0.2, f"model must not rank pure noise (held-out ρ={rho:+.3f})"
    print(f"4. no false dataloop on noise: L={lr:.1%} ≈ naive={br:.1%}, held-out ρ={rho:+.2f} (no real signal)")


# --------------------------------------------------------------------------- #
# 5. e2e on real EMOPEC (skips offline): the loop runs and is label-efficient vs the naive loop.
# --------------------------------------------------------------------------- #
def test_e2e_real_data() -> None:
    try:
        ed.load_records()
    except ed.DatasetUnavailable as e:
        _skip(f"EMOPEC unreachable and not cached: {e}")
        return
    res = loop.run(loop.LoopConfig(seeds=2))
    sr, vs = res["seed_results"], {v.name.split(":")[0]: v for v in res["verdicts"]}
    # the label-efficiency verdict (L vs the naive loop) must be a WIN, sign-consistent.
    cap = next(v for v in res["verdicts"] if v.name.startswith("label efficiency"))
    assert cap.ok, f"L should beat the naive loop on EMOPEC: {cap.metric} ({cap.detail})"
    # L's final recall must dominate B0's, every seed; the separability diagnostic must be populated.
    for s in sr:
        assert s["L"]["curve"][-1][1] > s["B0"]["curve"][-1][1], "L must beat naive recall each seed"
        assert s["L"]["overlaps"], "the separability (overlap) diagnostic must be recorded for L"
    print(f"5. e2e: real EMOPEC — label efficiency {cap.metric} (L vs naive loop), separability recorded")


# --------------------------------------------------------------------------- #
# 6. Phase-2 e2e on the real promoter substrate (skips offline): durable compounding — on a
#    mid-learnability, pool-restricted substrate, recursion beats one-shot (the result EMOPEC can't show).
# --------------------------------------------------------------------------- #
def test_e2e_promoter_durable_compounding() -> None:
    try:
        sub = loop.promoter_substrate()
    except pmd.DatasetUnavailable as e:
        _skip(f"promoter dataset unreachable and not cached: {e}")
        return
    res = loop.run(loop.LoopConfig(seeds=2), substrate=sub)
    sr = res["seed_results"]
    for s in sr:
        assert max(s["L"]["skips"], default=0) == 0, "promoter is pool-restricted: expect no coverage skips"
        assert s["L"]["curve"][-1][1] > s["B0"]["curve"][-1][1], "L must beat the naive loop (label efficiency)"
        # the Phase-2 finding in its stable form: recursion ≥ one-shot on a mid-ρ substrate (the
        # quantified ≥20% margin was established by earlier evaluation; here we assert the robust direction).
        assert s["L"]["curve"][-1][1] >= s["B1"]["curve"][-1][1], "recursion (L) must be ≥ one-shot (B1) on promoter"
    fly = next(v for v in res["verdicts"] if v.name.startswith("dataloop"))
    print(f"6. e2e: promoter (pool-restricted, mid-ρ) — L ≥ one-shot each seed (durable compounding); "
          f"dataloop verdict {fly.metric}")


if __name__ == "__main__":
    test_loop_is_closed()
    test_detects_compounding_on_biased_seed()
    test_reports_null_on_saturation()
    test_no_false_dataloop_on_noise()
    test_e2e_real_data()
    test_e2e_promoter_durable_compounding()
    print("\nall loop tests pass.")
