"""ppi_leakage — legible node-/sequence-identity-leakage audit over a sequence-based PPI benchmark.

The pair-input sibling of the MoleculeNet scaffold-leakage probe (avenue 6), testing whether the
leakage-honesty thesis spans to PPI prediction. The documented PPI-benchmark failure (Park & Marcotte 2012,
Nature Methods; Bernett et al. 2024): sequence-based PPI models report on **random pair splits**, which leak —
the same proteins straddle train and test, so the model scores by memorizing *which proteins are sticky*
(node-identity / degree reuse), not by learning interaction. Park & Marcotte's instrument is the **C1/C2/C3**
stratification of a held-out test set: C1 = both proteins seen in training, C2 = exactly one seen, C3 =
neither seen (the honest, protein-disjoint regime). Reported (≈C1) accuracy is inflated against honest (C3).

The legible layer assigns every test pair its leakage class with auditable reasons and scores two
*transparent* baselines per class:
  * **node-propensity** (the primary, the documented mechanism) — a pair's score is the product of its two
    members' positive-degree in the training graph; novel proteins have degree 0. No sequences, no fitting:
    it is pure node-identity memorization, so its C1 vs C3 gap *is* the leakage.
  * **sequence k-mer** (the second channel) — a similarity-weighted k-NN over PAIRS, pair similarity =
    product of the two members' **amino-acid k-mer-set Jaccard** (the protein analogue of avenue 6's
    Morgan/Tanimoto; reuses the AA-featurizer idea). It generalizes across proteins by homology, so any C3
    lift it shows is *sequence-identity* leakage that the protein-id-disjoint split fails to remove (the
    Bernett 2024 point: de-duplicate by sequence identity, not just by id).

The DRC (`contracts` engine, reused verbatim): `BOTH_PARTNERS_SEEN` (C1) / `PARTNER_SEEN_IN_TRAIN` (C2) /
`NEAR_DUP_PROTEIN` (a train protein at k-mer Jaccard ≥ τ) — every flag a human-readable reason.

Pre-registered (mechanism-based, before the audit metrics):
  P1  node-leakage prevalence on the held-out test (≥1 partner seen) ≥ 50%.
  P2  node-identity inflation = AUROC_node(C1) − AUROC_node(C3) ≥ 0.05  (the headline).
  P3  sequence-identity is a SECOND channel surviving the id-disjoint split:
      AUROC_seq(C3) ≥ AUROC_node(C3) + 0.03.

stdlib-only; reuses `contracts` + `stats_kit`. SKIPs cleanly if the benchmark is unreachable + uncached.

    python -m karyon.ppi_leakage
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

from . import contracts
from . import stats_kit
from .ppi_leakage_data import (
    DatasetUnavailable,
    Pair,
    Split,
    holdout_split,
    load_pairs,
    proteins_in,
)

_K = 3              # amino-acid k-mer length (the per-protein fingerprint)
_M = 25             # neighbours kept per protein (the block size for the pair k-NN)
_KNN = 5            # train pairs voting on a test pair
_TAU = 0.7          # near-duplicate k-mer-Jaccard threshold (sequence-identity proxy)


# --------------------------------------------------------------------------- #
# Fingerprints + similarity (the AA analogue of Morgan + Tanimoto).
# --------------------------------------------------------------------------- #
def kmer_set(seq: str, k: int = _K) -> frozenset[str]:
    return frozenset(seq[i:i + k] for i in range(len(seq) - k + 1)) if len(seq) >= k else frozenset((seq,))


# --------------------------------------------------------------------------- #
# The two transparent baselines. Protein similarity = k-mer-set Jaccard (the Tanimoto analogue), inlined in
# `_top_neighbours` below — it is the inner loop's hot path (~5M sims/seed).
# --------------------------------------------------------------------------- #
def node_propensity(train: list[Pair]) -> Counter:
    """positive-degree per protein in the training graph — the 'stickiness' a model memorizes."""
    pos_deg: Counter = Counter()
    for p in train:
        if p.label == 1:
            pos_deg[p.a.pid] += 1
            pos_deg[p.b.pid] += 1
    return pos_deg


def _top_neighbours(test_pids: list[str], train_pids: list[str],
                    fp: dict[str, frozenset[str]], m: int) -> dict[str, list[tuple[float, str]]]:
    """For each test protein, its top-`m` most similar train proteins as (sim, train_pid), descending."""
    train_fps = [(t, fp[t]) for t in train_pids]
    out: dict[str, list[tuple[float, str]]] = {}
    for tp in test_pids:
        f = fp[tp]
        nf = len(f)
        sims = []
        for t, tf in train_fps:
            inter = len(f & tf)
            if inter:
                sims.append((inter / (nf + len(tf) - inter), t))
        sims.sort(reverse=True)
        out[tp] = sims[:m]
    return out


@dataclass(frozen=True)
class Outcome:
    pair: Pair
    a_seen: bool
    b_seen: bool
    near_dup: bool              # either protein has a ≥τ train homolog (sequence-identity proxy)
    max_homolog_sim: float
    node_pred: float           # node-propensity score (degree memorization)
    seq_pred: float            # sequence k-mer pair-kNN score

    @property
    def leak_class(self) -> str:
        if self.a_seen and self.b_seen:
            return "C1"
        if self.a_seen or self.b_seen:
            return "C2"
        return "C3"


def predict(train: list[Pair], test: list[Pair], fp: dict[str, frozenset[str]],
            *, tau: float = _TAU) -> list[Outcome]:
    train_prots = proteins_in(train)
    pos_deg = node_propensity(train)
    train_pairs_by_key: dict[frozenset[str], int] = {
        frozenset((p.a.pid, p.b.pid)): p.label for p in train}
    seq_prior = (sum(p.label for p in train) / len(train)) if train else 0.5

    topsim = _top_neighbours(sorted(proteins_in(test)), sorted(train_prots), fp, _M)

    outs: list[Outcome] = []
    for p in test:
        # node-propensity: SUM of the two members' positive-degree (novel proteins → 0). Sum (not product)
        # so a C2 pair keeps its one seen partner's stickiness → the canonical monotone C1>C2>C3 gradient;
        # a C3 pair (both novel) scores 0 for every pair → AUROC exactly 0.5, the honest memorization floor.
        node = pos_deg.get(p.a.pid, 0) + pos_deg.get(p.b.pid, 0)

        # sequence k-mer pair-kNN: most similar train pairs, similarity-weighted vote
        na, nb = topsim.get(p.a.pid, []), topsim.get(p.b.pid, [])
        cands: list[tuple[float, int]] = []
        for sa, a2 in na:
            for sb, b2 in nb:
                if a2 == b2:
                    continue
                lab = train_pairs_by_key.get(frozenset((a2, b2)))
                if lab is not None:
                    cands.append((sa * sb, lab))
        cands.sort(reverse=True)
        top = cands[:_KNN]
        if top:
            wsum = sum(s for s, _ in top)
            seq = (sum(s * lab for s, lab in top) / wsum) if wsum > 0 else seq_prior
        else:
            seq = seq_prior

        homolog = max(na[0][0] if na else 0.0, nb[0][0] if nb else 0.0)
        outs.append(Outcome(
            pair=p, a_seen=p.a.pid in train_prots, b_seen=p.b.pid in train_prots,
            near_dup=homolog >= tau, max_homolog_sim=homolog,
            node_pred=float(node), seq_pred=seq))
    return outs


def auroc(outs: list[Outcome], attr: str) -> float:
    """AUROC of `attr` separating interacting (label 1) from non-interacting (label 0) pairs."""
    pos = [getattr(o, attr) for o in outs if o.pair.label == 1]
    neg = [getattr(o, attr) for o in outs if o.pair.label == 0]
    mw = stats_kit.mann_whitney(pos, neg)
    return mw.auroc if isinstance(mw, stats_kit.MannWhitney) else float("nan")


# --------------------------------------------------------------------------- #
# The leakage DRC (reused contracts engine).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainView:
    tau: float


def leakage_contracts() -> contracts.ContractSet:
    cs = contracts.ContractSet("ppi-leakage")
    cs.add(contracts.Contract(
        "BOTH_PARTNERS_SEEN",
        lambda o, ctx: "both proteins appeared in training pairs (C1 node-identity leakage)"
        if o.a_seen and o.b_seen else None,
        weight=2.0))
    cs.add(contracts.Contract(
        "PARTNER_SEEN_IN_TRAIN",
        lambda o, ctx: "a protein in this pair appeared in a training pair (C2 node leakage)"
        if o.a_seen or o.b_seen else None))
    cs.add(contracts.Contract(
        "NEAR_DUP_PROTEIN",
        lambda o, ctx: f"a near-identical protein is in train (k-mer Jaccard {o.max_homolog_sim:.2f} ≥ {ctx.tau})"
        if o.near_dup else None))
    return cs


def _stratum(outs: list[Outcome], cls: str) -> list[Outcome]:
    return [o for o in outs if o.leak_class == cls]


def _au(outs: list[Outcome], attr: str) -> str:
    a = auroc(outs, attr)
    return f"{a:.3f}" if a == a else "  n/a"  # noqa: PLR0124 (a!=a ⇒ nan)


def run_one(*, seed: int = 0, tau: float = _TAU, pairs: list[Pair] | None = None,
            split: Split | None = None) -> dict | None:
    if pairs is None:
        try:
            pairs = load_pairs()
        except DatasetUnavailable as e:
            print(f"  SKIP — {e}")
            return None
    if split is None:
        split = holdout_split(pairs, seed=seed)

    fp = {p.pid: kmer_set(p.seq) for pr in pairs for p in (pr.a, pr.b)}
    outs = predict(split.train, split.test, fp, tau=tau)

    cs, tv = leakage_contracts(), TrainView(tau)
    verdicts = [cs.evaluate(o, tv) for o in outs]
    prevalence = sum(1 for v in verdicts if not v.ok) / (len(outs) or 1)
    fire = Counter(n for v in verdicts for n in v.fired)

    c1, c2, c3 = _stratum(outs, "C1"), _stratum(outs, "C2"), _stratum(outs, "C3")
    au_full = auroc(outs, "node_pred")
    au_c1n, au_c3n = auroc(c1, "node_pred"), auroc(c3, "node_pred")
    au_c3s = auroc(c3, "seq_pred")
    c3_dup, c3_clean = [o for o in c3 if o.near_dup], [o for o in c3 if not o.near_dup]

    print(f"\n=== GUO YEAST PPI (sequence-based; metric = AUROC; seed {seed}) ===")
    print(f"  held-out test pairs: {len(outs)}  (train pairs {len(split.train)})")
    print(f"  node-leakage prevalence (≥1 partner seen): {prevalence:.1%}   <- P1")
    for c in cs.contracts:
        print(f"    {c.name:<24} {fire.get(c.name, 0) / (len(outs) or 1):6.1%}")
    print(f"\n  leakage class    n      pos%   AUROC(node)   AUROC(seq k-mer)")
    for name, st in (("C1 both seen ", c1), ("C2 one seen  ", c2), ("C3 neither   ", c3)):
        pos = sum(1 for o in st if o.pair.label == 1)
        pct = f"{pos / (len(st) or 1):.0%}"
        print(f"    {name}  {len(st):5d}   {pct:>4}     {_au(st, 'node_pred'):>7}        {_au(st, 'seq_pred'):>7}")
    print(f"\n  NODE-IDENTITY INFLATION  AUROC_node(C1) − AUROC_node(C3) = {au_c1n - au_c3n:+.3f}   <- P2")
    print(f"    (reported-eval ≈ C1 {au_c1n:.3f}  vs  honest Park-Marcotte C3 {au_c3n:.3f})")
    print(f"  SEQUENCE CHANNEL on the honest C3 split: AUROC_seq {au_c3s:.3f} vs AUROC_node {au_c3n:.3f}"
          f"  (Δ {au_c3s - au_c3n:+.3f})   <- P3")
    print(f"    C3 near-dup pairs {len(c3_dup)} (AUROC_seq {_au(c3_dup, 'seq_pred')}) "
          f"vs C3 clean {len(c3_clean)} (AUROC_seq {_au(c3_clean, 'seq_pred')})")

    return {"seed": seed, "prevalence": prevalence, "au_full_node": au_full,
            "au_c1_node": au_c1n, "au_c3_node": au_c3n, "au_c3_seq": au_c3s,
            "inflation": au_c1n - au_c3n, "n_test": len(outs),
            "n_c1": len(c1), "n_c2": len(c2), "n_c3": len(c3)}


def run(seeds: int = 3, tau: float = _TAU) -> None:
    print("Node-/sequence-identity-leakage honesty audit over a sequence-based PPI benchmark (Guo yeast)\n")
    results = [r for r in (run_one(seed=s, tau=tau) for s in range(seeds)) if r]
    if not results:
        return
    print("\n=== PRE-REGISTERED VERDICT (per seed) ===")
    for r in results:
        p1 = r["prevalence"] >= 0.50
        p2 = r["inflation"] >= 0.05
        p3 = (r["au_c3_seq"] - r["au_c3_node"]) >= 0.03
        print(f"  seed {r['seed']}  P1 leakage≥50% {'PASS' if p1 else 'FAIL'} ({r['prevalence']:.0%}) · "
              f"P2 node-inflation≥0.05 {'PASS' if p2 else 'FAIL'} ({r['inflation']:+.3f}) · "
              f"P3 seq-channel≥0.03 {'PASS' if p3 else 'FAIL'} ({r['au_c3_seq'] - r['au_c3_node']:+.3f})")
    mean = lambda k: sum(r[k] for r in results) / len(results)  # noqa: E731
    print(f"\n  mean node inflation {mean('inflation'):+.3f}  "
          f"(reported≈C1 {mean('au_c1_node'):.3f} → honest C3 {mean('au_c3_node'):.3f}); "
          f"mean C3 seq-channel {mean('au_c3_seq'):.3f}")
    print("  Read: does the leakage-honesty layer span to PAIR-INPUT (PPI) prediction? The audit assigns each")
    print("  test pair its C1/C2/C3 leakage class and reports AUROC per class — node-identity memorization")
    print("  inflates the reported number; the honest C3 split is the qualification. Deliverable = the legible")
    print("  honest-eval harness, not a better predictor (qualification, not accuracy).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Node-/sequence-leakage audit over the Guo-yeast PPI benchmark.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tau", type=float, default=_TAU)
    cli = ap.parse_args()
    run(seeds=cli.seeds, tau=cli.tau)
