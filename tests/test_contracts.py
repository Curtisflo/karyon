"""test_contracts — proofs for the substrate-agnostic contract engine (dual script / pytest).

Fully offline. The load-bearing proof is FAITHFULNESS: the engine re-expresses the existing
`crispr_qc.hard_contracts` instance byte-identically over a battery of sequences — so `contracts.py`
GENERALIZES the two single-substrate instances, it does not silently change them. Plus: planted
contracts fire/pass correctly; the loose check-return normalization (None/False/True/str/Reason); the
calibration `ctx` actually reaches calibrated contracts and changes the verdict; score = Σ weights;
`hard_only()` drops calibrated rules; evaluation is deterministic.

    python tests/test_contracts.py
"""

from __future__ import annotations

import random

from karyon import contracts
from karyon import crispr_qc as qc
from karyon.contracts import CALIBRATED, Contract, ContractSet, Reason


def _rand_seq(rng: random.Random, n: int = 20) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


# --------------------------------------------------------------------------- #
# The mirror: crispr_qc.hard_contracts re-expressed as a ContractSet (same messages, same order).
# --------------------------------------------------------------------------- #
def _crispr_hard_set() -> ContractSet:
    cs = ContractSet("crispr-hard")
    cs.add(Contract("TTTT",
                    lambda s, ctx: "TTTT: Pol-III terminator truncates the sgRNA" if "TTTT" in s else None))

    def _gc(s, ctx):
        g = qc.gc(s)
        if g < 0.20:
            return f"GC {g:.0%} <20%: poor RISC loading"
        if g > 0.80:
            return f"GC {g:.0%} >80%: over-stable / poor specificity"
        return None
    cs.add(Contract("GC", _gc))
    cs.add(Contract("homopolymer",
                    lambda s, ctx: f"homopolymer run {qc.max_run(s)}: synthesis/folding risk"
                    if qc.max_run(s) >= 5 else None))
    return cs


def test_faithful_to_crispr_hard_contracts() -> None:
    """The engine reproduces the legacy instance EXACTLY — messages and order — over planted failures,
    a clean guide, and 500 random guides. This is the generalization-is-faithful proof."""
    planted = [
        "GGAACTTTTGGCCAAGGTTCA",   # TTTT
        "GGGGGAACCTTGGAACCTTGG",   # homopolymer run≥5
        "GCGCGCGCGCGCGCGCGCGC",    # extreme GC (high)
        "ATATATATATATATATATAT",    # extreme GC (low, 0%)
        "GCAACTTGGACCTTGAACCT",    # clean
        "TTTTGGGGGAT",             # TTTT + homopolymer (two reasons, order matters)
    ]
    cs = _crispr_hard_set()
    rng = random.Random(0)
    seqs = planted + [_rand_seq(rng, rng.randint(18, 25)) for _ in range(500)]
    mism = [s for s in seqs if cs.evaluate(s).messages != qc.hard_contracts(s)]
    assert not mism, f"engine diverged from crispr_qc.hard_contracts on {len(mism)} seqs, e.g. {mism[:2]}"
    # And the multi-reason case really did carry both, in order.
    assert cs.evaluate("TTTTGGGGGAT").messages == qc.hard_contracts("TTTTGGGGGAT")
    assert len(cs.evaluate("TTTTGGGGGAT").reasons) == 2
    print(f"1. engine faithful to crispr_qc.hard_contracts on {len(seqs)} sequences (messages + order)")


def test_planted_fire_and_clean_pass() -> None:
    """A registry fires the contracts whose predicate is true and passes a clean design."""
    cs = ContractSet("toy")
    cs.add(Contract("has_X", lambda d, ctx: "contains X" if "X" in d else None))
    cs.add(Contract("too_long", lambda d, ctx: f"len {len(d)}>5" if len(d) > 5 else None))
    v = cs.evaluate("aXbcdef")
    assert not v.ok and v.fired == ["has_X", "too_long"] and v.score == 2.0
    clean = cs.evaluate("abc")
    assert clean.ok and clean.reasons == () and clean.score == 0.0
    print("2. planted contracts fire (with reasons) and a clean design passes")


def test_check_return_normalization() -> None:
    """A check may return None/False (clean), True (fired→name), a str (fired→message), or a Reason."""
    cs = ContractSet("norm")
    cs.add(Contract("none_clean", lambda d, ctx: None))
    cs.add(Contract("false_clean", lambda d, ctx: False))
    cs.add(Contract("true_fires", lambda d, ctx: True))
    cs.add(Contract("str_fires", lambda d, ctx: "because reasons"))
    cs.add(Contract("reason_fires", lambda d, ctx: Reason("reason_fires", "custom", 3.0)))
    v = cs.evaluate("x")
    assert v.fired == ["true_fires", "str_fires", "reason_fires"]
    assert v.messages == ["true_fires", "because reasons", "custom"]
    assert v.score == 1.0 + 1.0 + 3.0
    print("3. check-return normalization (None/False/True/str/Reason) handled")


def test_calibration_ctx_reaches_contracts() -> None:
    """A calibrated contract reads the ctx — changing the calibration flips the verdict (the plumbing the
    operator's qualify step depends on)."""
    cs = ContractSet("cal")
    cs.add(Contract("floor", lambda d, ctx: f"{d} below floor {ctx}" if d < ctx else None,
                    kind=CALIBRATED, weight=2.0))
    fired = cs.evaluate(5, ctx=10)
    assert not fired.ok and fired.score == 2.0 and "below floor 10" in fired.messages[0]
    assert cs.evaluate(5, ctx=3).ok, "lowering the calibrated floor should clear the design"
    print("4. calibration ctx reaches calibrated contracts and changes the verdict")


def test_hard_only_drops_calibrated() -> None:
    """The DRC spine = the hard subset, runnable with no calibration (design-time gate)."""
    cs = ContractSet("mix")
    cs.add(Contract("rule", lambda d, ctx: None))
    cs.add(Contract("needs_cal", lambda d, ctx: None, kind=CALIBRATED))
    spine = cs.hard_only()
    assert spine.names() == ["rule"] and len(cs.contracts) == 2
    print("5. hard_only() yields the calibration-free DRC spine")


def test_deterministic() -> None:
    """Same (design, ctx) ⇒ identical verdict (no hidden state / ordering nondeterminism)."""
    cs = _crispr_hard_set()
    rng = random.Random(7)
    for _ in range(200):
        s = _rand_seq(rng, rng.randint(18, 25))
        assert cs.evaluate(s) == cs.evaluate(s)
    print("6. evaluation is deterministic")


def test_verdict_to_dict_schema() -> None:
    """Verdict/Reason serialize to the stable, JSON-safe wire schema the spine emits."""
    import json

    cs = _crispr_hard_set()
    v = cs.evaluate("TTTTGGGGGAT")            # fires TTTT + homopolymer (two reasons, order matters)
    d = v.to_dict()
    assert set(d) == {"ok", "score", "reasons"}
    assert d["ok"] is False and d["score"] == 2.0 and len(d["reasons"]) == 2
    assert set(d["reasons"][0]) == {"contract", "message", "weight"}
    assert [r["contract"] for r in d["reasons"]] == v.fired          # order preserved
    assert [r["message"] for r in d["reasons"]] == v.messages
    # JSON round-trip is lossless (no non-serializable types leak in).
    assert json.loads(json.dumps(d)) == d
    # A clean verdict serializes too.
    clean = cs.evaluate("GCAACTTGGACCTTGAACCT").to_dict()
    assert clean == {"ok": True, "score": 0.0, "reasons": []}
    print("7. Verdict/Reason to_dict() emit the stable JSON schema (keys + order + round-trip)")


def test_disclosure_only_passes_gate_but_is_not_clean() -> None:
    """A weight-0 disclosure fires but does NOT fail the gate. `ok` is PASSED-THE-GATE (`score == 0`);
    `clean` is the strict 'nothing fired'. They diverge exactly here — and `to_dict()['ok']` follows `ok`,
    so a verdict serialized directly carries the SAME notion of `ok` the qualify/CLI spine emits (the
    disclosure-only case is where the old `ok = not reasons` field used to disagree with `score == 0`)."""
    cs = ContractSet("disc")
    cs.add(Contract("CONDEMNING", lambda d, ctx: "bad" if "x" in d else None, weight=1.0))
    cs.add(Contract("DISCLOSE", lambda d, ctx: "note" if "y" in d else None, weight=0.0))

    disc = cs.evaluate("y")                          # only the weight-0 disclosure fires
    assert disc.score == 0.0 and disc.fired == ["DISCLOSE"]
    assert disc.ok is True                           # passed the gate — disclosures inform, they don't fail
    assert disc.clean is False                       # …but it is not strictly silent
    assert disc.to_dict()["ok"] is True              # the wire schema reports passed-the-gate, not "silent"

    cond = cs.evaluate("x")                          # a condemning contract fires
    assert cond.score == 1.0 and cond.ok is False and cond.clean is False

    both = cs.evaluate("xy")
    assert both.ok is False and both.fired == ["CONDEMNING", "DISCLOSE"]

    silent = cs.evaluate("z")                        # nothing fires
    assert silent.reasons == () and silent.ok is True and silent.clean is True
    print("8. disclosure-only verdict: ok (passed gate) True, clean False; to_dict ok follows ok")


def _run() -> None:
    test_faithful_to_crispr_hard_contracts()
    test_planted_fire_and_clean_pass()
    test_check_return_normalization()
    test_calibration_ctx_reaches_contracts()
    test_hard_only_drops_calibrated()
    test_deterministic()
    test_verdict_to_dict_schema()
    test_disclosure_only_passes_gate_but_is_not_clean()
    print("\nALL contracts-engine proofs passed.")


if __name__ == "__main__":
    _run()
