"""acquisition — the shared CHOOSE primitive: rank candidates by an acquisition policy.

the design notes's three cores are **predict** ([linmodel.py]), **choose**
(active learning), and **construct** ([constructive_core.py]). The *choose* core is one tiny,
stateless operation — "given a learned model and a set of candidates, pick the k most worth
measuring" — that two callers now need:

  * [active_learning.py] (probe #1): which already-deposited switches to reveal next;
  * [loop.py] (the closed DBTL loop): which of the constructed proposals to actually "measure".

Rather than copy the policy sort into both (the very dumb-move the program scores itself on
removing), it lives here once. The policies, all rank-by-a-scalar except `random`:

  * `random`      — sample k uniformly (the control; needs `rng`);
  * `greedy`      — argmax predicted value (exploit the model's ranking — the discovery lever);
  * `uncertainty` — argmax posterior variance xᵀA⁻¹x (explore where the model is least sure);
  * `ucb`         — predict + β·sd (exploit + explore, the blended objective).

The model is a duck: anything with `.predict(x)` and (for uncertainty/ucb) `.variance(x)` works —
[linmodel.BayesRidge] gives both from one fit, which is why the choose core needs no ensemble.

    python -m karyon.acquisition        # self-tests (policy ranking + edge cases)
"""

from __future__ import annotations

import math
import random

POLICIES = ("random", "greedy", "uncertainty", "ucb")


def score(model, x: list[float], policy: str, beta: float = 1.0) -> float:
    """The acquisition value of one candidate feature vector under `policy` (higher = acquire first).

    `random` has no score (it is sampled, not ranked) and raises if asked — callers branch on it."""
    if policy == "greedy":
        return model.predict(x)
    if policy == "uncertainty":
        return model.variance(x)
    if policy == "ucb":
        return model.predict(x) + beta * math.sqrt(max(0.0, model.variance(x)))
    if policy == "random":
        raise ValueError("`random` is sampled, not scored — call acquire(..., 'random', rng=...)")
    raise ValueError(f"unknown acquisition policy {policy!r} (expected one of {POLICIES})")


def acquire(model, X: list[list[float]], candidates: list[int], policy: str, k: int,
            *, rng: random.Random | None = None, beta: float = 1.0) -> list[int]:
    """Pick up to k of `candidates` (indices into X) to measure next, by `policy`.

    Deterministic given `rng`: `random` consumes `rng`; the ranked policies are a pure sort (ties
    break by the order of `candidates`, matching `sorted(..., reverse=True)`). Returns ≤ k indices —
    fewer only when `len(candidates) < k`."""
    k = min(k, len(candidates))
    if policy == "random":
        if rng is None:
            raise ValueError("the 'random' policy needs an rng")
        return rng.sample(candidates, k)
    return sorted(candidates, key=lambda i: score(model, X[i], policy, beta), reverse=True)[:k]


# --------------------------------------------------------------------------- #
# Self-tests.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    class _Stub:
        """A model whose predict/variance are lookups — isolates the acquisition logic from any fit."""

        def __init__(self, pred: dict[int, float], var: dict[int, float]):
            self._p, self._v = pred, var

        def predict(self, x):                       # x is the index itself in these tests
            return self._p[x]

        def variance(self, x):
            return self._v[x]

    # X is the identity here: feature "vector" i is just i, so the stub keys on i directly.
    X = list(range(5))
    pred = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.3, 4: 0.7}
    var = {0: 9.0, 1: 0.1, 2: 1.0, 3: 4.0, 4: 0.2}
    m = _Stub(pred, var)
    cands = [0, 1, 2, 3, 4]

    # greedy: highest predict first.
    assert acquire(m, X, cands, "greedy", 2) == [1, 4], acquire(m, X, cands, "greedy", 2)
    # uncertainty: highest variance first.
    assert acquire(m, X, cands, "uncertainty", 2) == [0, 3], acquire(m, X, cands, "uncertainty", 2)
    # ucb with beta=0 collapses to greedy; with large beta it chases variance.
    assert acquire(m, X, cands, "ucb", 2, beta=0.0) == acquire(m, X, cands, "greedy", 2)
    assert acquire(m, X, cands, "ucb", 1, beta=100.0) == [0]   # var[0]=9 dominates
    print("1. greedy / uncertainty / ucb rank by the right scalar; ucb(β=0)=greedy")

    # random: deterministic under a seeded rng; distinct; drawn from candidates.
    r1 = acquire(m, X, cands, "random", 3, rng=random.Random(0))
    r2 = acquire(m, X, cands, "random", 3, rng=random.Random(0))
    assert r1 == r2 and len(set(r1)) == 3 and set(r1) <= set(cands)
    print(f"2. random is rng-deterministic and draws distinct candidates ({r1})")

    # k larger than the candidate set returns the whole set (ranked / sampled), never errors.
    assert sorted(acquire(m, X, cands, "greedy", 99)) == cands
    assert sorted(acquire(m, X, cands, "random", 99, rng=random.Random(1))) == cands
    print("3. k > len(candidates) returns all candidates, no overrun")

    # candidate subset: ranking is restricted to the given indices, not all of X.
    assert acquire(m, X, [0, 3], "greedy", 1) == [3]            # pred[3]=0.3 > pred[0]=0.1
    # unknown policy / missing rng raise loudly.
    for bad in ("nope", ""):
        try:
            acquire(m, X, cands, bad, 1)
            raise AssertionError(f"policy {bad!r} should have raised")
        except ValueError:
            pass
    try:
        acquire(m, X, cands, "random", 1)                       # no rng
        raise AssertionError("random without rng should have raised")
    except ValueError:
        pass
    print("4. candidate subsetting works; unknown policy and rng-less random raise")

    print("\nacquisition self-tests pass.")
