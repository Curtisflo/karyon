"""retro_baseline — a legible similarity-based retrosynthesis baseline over USPTO-50k (stdlib only).

The incumbent for the benchmark-honesty probe. The point is NOT a strong retro model — it is a
*transparent* one whose apparent accuracy the honesty layer can then explain. The design mirrors
**retrosim** (Coley et al. 2017, a published, interpretable, competitive USPTO-50k baseline): predict a
product's precursors by **nearest-neighbour over products** — find the most similar train products and
copy their recorded reactants. Real retrosim transfers the neighbour's *reaction template* (extract +
apply, RDKit); this stdlib version copies the neighbour's reactant set verbatim — the **retrieval lower
bound** on retrosim. That is deliberate: it isolates exactly the duplication the audit measures (the
faithful template arm is the RDKit hardening follow-on; see RETRO_HONESTY_RESULT.md).

Two prediction targets, both from the same product-NN machinery:
  * **reactant recovery** (the retrosynthesis-shaped task) — is the true reactant signature among the
    top-k neighbours' reactant sets? Its success is, by measurement, largely product duplication — that is
    the finding, stated honestly, not hidden.
  * **reaction class 1..10** (an INDEPENDENT, non-circular corroboration) — sim-weighted neighbour vote.
    A product can have many same-class non-duplicate neighbours, so class accuracy is NOT defined by
    duplication → "leakage explains correctness" is a genuine empirical claim here, not a tautology.

Similarity = Jaccard over **character k-mers** of the canonical product SMILES — the program's k-mer ethos
(`linmodel.py`), with the named bound that string-k-mer is a *proxy* for chemical (fingerprint) similarity,
the same proxy-vs-real disclosure the program makes for stdlib-thermo-vs-ViennaRNA. An inverted index with
a document-frequency cap keeps 5k×45k retrieval tractable in pure Python.

    cd karyon/probe && python retro_baseline.py        # top-k tables on both splits + the sim distribution
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .uspto_data import (
    DatasetUnavailable,
    Reaction,
    Split,
    load_reactions,
    patent_disjoint_split,
    random_split,
)

_K = 5                      # character k-mer length (specific enough to keep inverted lists short)
_DF_CAP_FRAC = 0.05         # skip k-mers occurring in >5% of train products (uninformative "stop-grams")
_TOP_N = 10                 # neighbours retrieved per query (top-1..10 metrics)


def kmers(s: str, k: int = _K) -> set[str]:
    """The set of character k-mers of `s` (the whole string if shorter than k)."""
    return {s[i:i + k] for i in range(len(s) - k + 1)} or {s}


class ProductIndex:
    """Inverted character-k-mer index over the train products → fast approximate Jaccard NN.

    `query` gathers candidates sharing ≥1 (non-capped) k-mer, prefilters by shared-k-mer count, then
    computes exact Jaccard on the short-list. `exact_count` backs the EXACT_PRODUCT_IN_TRAIN contract."""

    def __init__(self, train_products: list[str], k: int = _K, df_cap_frac: float = _DF_CAP_FRAC):
        self.k = k
        self.ksets: list[set[str]] = [kmers(p, k) for p in train_products]
        self.exact: dict[str, int] = Counter(train_products)
        inv: dict[str, list[int]] = defaultdict(list)
        for i, ks in enumerate(self.ksets):
            for km in ks:
                inv[km].append(i)
        cap = max(1, int(len(train_products) * df_cap_frac))
        # Drop ultra-common k-mers: they cost the most and discriminate the least (rarer shared k-mers
        # carry the near-duplicate signal). Keeps per-query work bounded.
        self.inv = {km: idxs for km, idxs in inv.items() if len(idxs) <= cap}

    def exact_count(self, product: str) -> int:
        return self.exact.get(product, 0)

    def query(self, product: str, top_n: int = _TOP_N) -> list[tuple[int, float]]:
        """Top-n (train_idx, jaccard) for `product`, best first. Empty if no shared informative k-mer."""
        qk = kmers(product, self.k)
        shared: Counter[int] = Counter()
        for km in qk:
            for i in self.inv.get(km, ()):  # capped lists only
                shared[i] += 1
        if not shared:
            return []
        # Deterministic regardless of PYTHONHASHSEED (set iteration order must not move the numbers):
        # rank candidates by (shared-k-mer count desc, train index), then by (Jaccard desc, train index).
        pre = sorted(shared, key=lambda i: (-shared[i], i))[:max(top_n * 5, 50)]
        sims = [(i, len(qk & self.ksets[i]) / len(qk | self.ksets[i])) for i in pre]
        sims.sort(key=lambda t: (-t[1], t[0]))
        return sims[:top_n]


@dataclass(frozen=True)
class Outcome:
    """Per-test-reaction baseline result; the honesty layer joins leakage reasons onto these."""

    rxn: Reaction
    nn_sim: float                 # max Jaccard product-similarity to any train product (0 if none found)
    exact_in_train: bool          # the canonical product string appears verbatim in train
    reactant_rank: int | None     # 1-based rank at which the TRUE reactant set is recovered (exact), else None
    reactant_overlap: float       # Jaccard of the top-1 neighbour's reactant molecules vs the true set
    class_correct: bool           # sim-weighted neighbour class vote == the true class


def _predict(test_rxn: Reaction, train: list[Reaction], index: ProductIndex,
             top_n: int) -> Outcome:
    nbrs = index.query(test_rxn.product, top_n)
    nn_sim = nbrs[0][1] if nbrs else 0.0

    # reactant recovery: first rank whose neighbour's reactant signature equals the truth (exact match —
    # a near-duplicate-reaction detector without template transfer), plus a softer top-1 reactant-molecule
    # overlap (a similar product often shares a major building block even when the full set differs).
    reactant_rank: int | None = None
    for rank, (i, _) in enumerate(nbrs, start=1):
        if train[i].reactant_sig == test_rxn.reactant_sig:
            reactant_rank = rank
            break
    true_mols = set(test_rxn.reactant_sig.split("."))
    if nbrs:
        pred_mols = set(train[nbrs[0][0]].reactant_sig.split("."))
        union = true_mols | pred_mols
        reactant_overlap = len(true_mols & pred_mols) / len(union) if union else 0.0
    else:
        reactant_overlap = 0.0

    # class: similarity-weighted vote over the neighbours' classes (independent of duplication).
    votes: dict[int, float] = defaultdict(float)
    for i, sim in nbrs:
        votes[train[i].klass] += sim
    # highest summed similarity wins; ties broken by lowest class id (deterministic).
    pred_class = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0] if votes else 0
    return Outcome(test_rxn, nn_sim, index.exact_count(test_rxn.product) > 0,
                   reactant_rank, reactant_overlap, pred_class == test_rxn.klass)


def run_baseline(split: Split, *, top_n: int = _TOP_N, k: int = _K) -> tuple[ProductIndex, list[Outcome]]:
    """Build the train index for `split` and score every test reaction. Returns (index, outcomes)."""
    index = ProductIndex([r.product for r in split.train], k=k)
    outcomes = [_predict(r, split.train, index, top_n) for r in split.test]
    return index, outcomes


# --------------------------------------------------------------------------- #
# Aggregate metrics.
# --------------------------------------------------------------------------- #
def topk_reactant_accuracy(outcomes: list[Outcome], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[int, float]:
    n = len(outcomes) or 1
    return {k: sum(1 for o in outcomes if o.reactant_rank is not None and o.reactant_rank <= k) / n
            for k in ks}


def class_accuracy(outcomes: list[Outcome]) -> float:
    n = len(outcomes) or 1
    return sum(1 for o in outcomes if o.class_correct) / n


def _fmt_topk(d: dict[int, float]) -> str:
    return "  ".join(f"top{k}={v:.1%}" for k, v in d.items())


def _report_split(split: Split, top_n: int = _TOP_N) -> None:
    index, outs = run_baseline(split, top_n=top_n)
    react = topk_reactant_accuracy(outs)
    cls = class_accuracy(outs)
    sims = sorted(o.nn_sim for o in outs)
    n = len(sims)
    q = lambda f: sims[min(n - 1, int(f * n))]
    exact = sum(o.exact_in_train for o in outs)
    print(f"\n[{split.name}]  train/test = {split.sizes}")
    print(f"  reactant recovery : {_fmt_topk(react)}")
    print(f"  reaction class    : top1={cls:.1%}  (majority-class floor = predict the commonest)")
    print(f"  nn product sim    : median={q(0.5):.3f}  p25={q(0.25):.3f}  p75={q(0.75):.3f}  max={sims[-1]:.3f}")
    print(f"  exact product in train : {exact} / {n}  ({exact / n:.1%})")


if __name__ == "__main__":
    print("Similarity-based retrosynthesis baseline on USPTO-50k (retrosim-lite, stdlib)\n")
    try:
        rxns = load_reactions()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
    _report_split(random_split(rxns, seed=0))
    _report_split(patent_disjoint_split(rxns, seed=0))
    print("\n(The standard random split is the leaky one; patent-disjoint removes the same-patent leakage "
          "source. The honesty layer adds the similarity-stratified inflation table + the leakage audit.)")
