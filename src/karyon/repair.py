"""repair — the agent self-repair loop: generate → qualify → fix-from-reasons → re-qualify → converge.

karyon's gates don't just say PASS/FAIL — every rejection **names its reason** (`HOMOPOLYMER_RUN`,
`EXTREME_PROPERTY`, `RESTRICTION_SITE present (EcoRI)` …). A named reason is a **repair instruction**: an
agent can read it and make the corresponding edit, then re-qualify. A black-box pass/fail can't drive that
loop; a legible one can. This module is the small harness that closes it.

    artifact = agent.propose(spec)            # make something
    loop:
        verdict = qualify(artifact, modality) # check it (the karyon gate)
        if clean: stop (converged)
        artifact, action = agent.revise(artifact, verdict, spec)   # fix from the NAMED reasons

The agent is **pluggable** (the `Agent` protocol). In real use the agent is your *harness* — e.g. Claude
Code in your terminal, no API key: it writes a candidate, runs `karyon qualify`, reads the reasons, edits,
re-runs (see `examples/agent_loop/`). The two reference agents here (`DnaRepairAgent`, `MolRepairAgent`)
make the loop **runnable and testable with zero LLM** — they map each named contract to a deterministic
edit, so the loop converges on its own and CI can prove it.

Qualification, not accuracy: the loop produces an artifact that *passes the manufacturability/validity gate*,
not a guaranteed-functional one.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from . import seq_dfm
from .contracts import Reason, Verdict
from .spine import qualify


# --------------------------------------------------------------------------- #
# The trajectory — a legible record of make → check → fix → … (the audit trail).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RepairStep:
    """One round: the artifact as qualified, its verdict, and the action the agent took in response.

    `action` is "" on the converged (final) step and the budget-exhausted step."""

    round: int
    artifact: str
    ok: bool                         # the gate PASSED this round (no condemning contract)
    score: float
    reasons: tuple[Reason, ...]
    action: str

    def to_dict(self) -> dict:
        return {"round": self.round, "artifact": self.artifact, "ok": self.ok, "score": self.score,
                "reasons": [r.to_dict() for r in self.reasons], "action": self.action}


@dataclass(frozen=True)
class RepairTrajectory:
    """The whole loop: every step, the final artifact, and whether the goal was met within budget."""

    modality: str
    steps: tuple[RepairStep, ...]
    final: str
    converged: bool

    @property
    def rounds(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {"modality": self.modality, "converged": self.converged, "rounds": self.rounds,
                "final": self.final, "steps": [s.to_dict() for s in self.steps]}


# --------------------------------------------------------------------------- #
# The agent role — pluggable. The reference agents below implement it; so does your harness (Claude Code).
# --------------------------------------------------------------------------- #
@runtime_checkable
class Agent(Protocol):
    def propose(self, spec: Any) -> str:
        """Make a first-draft artifact for `spec` (a SMILES, a DNA sequence, …)."""

    def revise(self, artifact: str, verdict: Verdict, spec: Any) -> tuple[str, str]:
        """Read the verdict's NAMED reasons and return (fixed_artifact, action_description)."""


def _needs_work(verdict: Verdict, clear_disclosures: tuple[str, ...]) -> bool:
    """True iff the artifact still has work: a condemning contract fired, OR a disclosure the spec
    explicitly asked to clear (e.g. `RESTRICTION_SITE` for a Golden-Gate-clean insert)."""
    if verdict.score > 0.0:
        return True
    return any(c in verdict.fired for c in clear_disclosures)


def repair_loop(spec: Any, agent: Agent, modality: str, *, max_rounds: int = 8,
                clear_disclosures: tuple[str, ...] = ()) -> RepairTrajectory:
    """Drive an agent through generate → qualify → fix → re-qualify until the gate is satisfied (no
    condemning contract, and none of `clear_disclosures` still firing) or `max_rounds` is spent."""
    artifact = agent.propose(spec)
    steps: list[RepairStep] = []
    converged = False
    for r in range(max_rounds + 1):
        verdict = qualify(artifact, modality=modality).items[0][1]
        if not _needs_work(verdict, clear_disclosures):
            steps.append(RepairStep(r, artifact, True, verdict.score, verdict.reasons, ""))
            converged = True
            break
        if r == max_rounds:                                  # budget spent — record the unrepaired state
            steps.append(RepairStep(r, artifact, verdict.score == 0.0, verdict.score,
                                    verdict.reasons, "(budget exhausted)"))
            break
        new_artifact, action = agent.revise(artifact, verdict, spec)
        steps.append(RepairStep(r, artifact, verdict.score == 0.0, verdict.score, verdict.reasons, action))
        artifact = new_artifact
    return RepairTrajectory(modality, tuple(steps), artifact, converged)


def format_trajectory(traj: RepairTrajectory) -> str:
    """A human-readable rendering of a trajectory (the demo print + the CLI human output)."""
    lines = [f"repair loop · {traj.modality} · "
             f"{'CONVERGED' if traj.converged else 'NOT converged'} in {traj.rounds - 1} edit(s)"]
    for s in traj.steps:
        head = "PASS" if s.ok else "FAIL"
        flags = ", ".join(f"{r.contract}" for r in s.reasons) or "clean"
        lines.append(f"  round {s.round}: {head}  [{flags}]")
        if s.action:
            lines.append(f"           ↳ {s.action}")
    lines.append(f"  final: {traj.final[:64]}{'…' if len(traj.final) > 64 else ''}")
    return "\n".join(lines)


# =========================================================================== #
# Reference agent #1 — DNA (pure stdlib): SURGICAL edits keyed on the named contract.
# Each fired contract maps to a targeted seq_dfm-driven edit; one defect fixed per round so the trajectory
# reads cleanly. This is what makes the loop runnable + CI-tested with no LLM.
# =========================================================================== #
@dataclass(frozen=True)
class DnaSpec:
    """A DNA design goal: a synthesizable insert of ~`length` nt at ~`gc_target` GC, free of the named
    restriction enzymes and strong hairpins. `seed` makes `propose` deterministic."""

    length: int = 240
    gc_target: float = 0.50
    avoid_enzymes: tuple[str, ...] = ("EcoRI", "BsaI")
    seed: int = 0


_BASES = "ACGT"


def _rng_for(seq: str) -> random.Random:
    """A deterministic RNG seeded by the sequence — so `revise` is reproducible for tests."""
    return random.Random(zlib.crc32(seq.encode()))


def _longest_run_span(seq: str, base: str | None = None) -> tuple[int, int]:
    """(start, length) of the longest run of identical bases (or of `base` specifically)."""
    best_start = best_len = 0
    i = 0
    n = len(seq)
    while i < n:
        if base is not None and seq[i] != base:
            i += 1
            continue
        j = i
        while j < n and seq[j] == seq[i]:
            j += 1
        if j - i > best_len:
            best_start, best_len = i, j - i
        i = j
    return best_start, best_len


def _mutate(seq: str, pos: int, rng: random.Random, *, avoid: str = "") -> str:
    """Replace base at `pos` with a different base (avoiding any in `avoid` when possible)."""
    choices = [b for b in _BASES if b != seq[pos] and b not in avoid] or [b for b in _BASES if b != seq[pos]]
    return seq[:pos] + rng.choice(choices) + seq[pos + 1:]


@dataclass
class DnaRepairAgent:
    """Surgical reference agent for the `dna` gate. `propose` emits a rough first draft (out-of-band GC + an
    injected homopolymer + an injected EcoRI site) so the loop has visible work; `revise` fixes ONE named
    contract per round using the gate's own `seq_dfm` primitives — no message parsing."""

    # priority: structural defects before composition before disclosures (one fix per round, fixed order).
    _PRIORITY = ("LENGTH_OUT_OF_RANGE", "HOMOPOLYMER_RUN", "STRONG_HAIRPIN",
                 "GC_OUT_OF_BAND", "RESTRICTION_SITE", "POLY_G_RUN")

    def propose(self, spec: DnaSpec) -> str:
        rng = random.Random(spec.seed)
        seq = "".join(rng.choices("AT" * 4 + "GC", k=spec.length))     # deliberately AT-rich ⇒ GC below band
        # inject two realistic, named defects so the trajectory shows targeted repair:
        seq = seq[:40] + "AAAAAAAAAAAA" + seq[52:]                     # a 12-mer homopolymer (>8)
        seq = seq[:80] + "GAATTC" + seq[86:]                          # an EcoRI site
        return seq

    def revise(self, seq: str, verdict: Verdict, spec: DnaSpec) -> tuple[str, str]:
        rng = _rng_for(seq)
        fired = verdict.fired
        for contract in self._PRIORITY:
            if contract in fired:
                return getattr(self, f"_fix_{contract.lower()}")(seq, spec, rng)
        return seq, "no actionable contract"

    # — one targeted fix per contract, each re-deriving the defect from seq_dfm —
    def _fix_gc_out_of_band(self, seq, spec, rng) -> tuple[str, str]:
        s = list(seq)
        before = seq_dfm.gc_fraction(seq)
        lo, hi = 0.32, 0.58                                            # comfortably inside the 0.25–0.65 band
        guard = 0
        while seq_dfm.gc_fraction("".join(s)) < lo and guard < len(s) * 3:
            i = rng.randrange(len(s))
            if s[i] in "AT":
                s[i] = rng.choice("GC")
            guard += 1
        while seq_dfm.gc_fraction("".join(s)) > hi and guard < len(s) * 6:
            i = rng.randrange(len(s))
            if s[i] in "GC":
                s[i] = rng.choice("AT")
            guard += 1
        out = "".join(s)
        return out, f"rebalanced GC {before:.0%}→{seq_dfm.gc_fraction(out):.0%} into the synthesis band"

    def _fix_homopolymer_run(self, seq, spec, rng) -> tuple[str, str]:
        start, length = _longest_run_span(seq)
        pos = start + length // 2
        nbrs = seq[max(0, pos - 1)] + seq[min(len(seq) - 1, pos + 1)]
        return _mutate(seq, pos, rng, avoid=nbrs), f"broke a {length}-base homopolymer run at position {pos}"

    def _fix_poly_g_run(self, seq, spec, rng) -> tuple[str, str]:
        start, length = _longest_run_span(seq, "G")
        pos = start + length // 2
        return _mutate(seq, pos, rng, avoid="G"), f"broke a {length}-base poly-G run at position {pos}"

    def _fix_strong_hairpin(self, seq, spec, rng) -> tuple[str, str]:
        hp = seq_dfm.hairpin_stem(seq)
        # mutate the MIDDLE of the 5' stem arm — splits the stem into two short halves in one edit.
        pos = (hp.start + hp.stem // 2) if hp.stem else rng.randrange(len(seq))
        nbrs = seq[max(0, pos - 1)] + seq[min(len(seq) - 1, pos + 1)]
        return _mutate(seq, pos, rng, avoid=nbrs), \
            f"disrupted a {hp.stem} bp hairpin stem at position {pos}"

    def _fix_length_out_of_range(self, seq, spec, rng) -> tuple[str, str]:
        if len(seq) < 18:
            pad = "".join(rng.choices(_BASES, k=18 - len(seq)))
            return seq + pad, f"padded to the orderable floor ({len(seq)}→{len(seq) + len(pad)} nt)"
        return seq[:3000], f"trimmed to the single-fragment window ({len(seq)}→3000 nt)"

    def _fix_restriction_site(self, seq, spec, rng) -> tuple[str, str]:
        sites = seq_dfm.restriction_sites(seq)
        target = next((h for h in sites if h.enzyme in spec.avoid_enzymes), sites[0])
        pos = target.position + len(target.site) // 2
        return _mutate(seq, pos, rng), f"removed the {target.enzyme} site at position {target.position}"


# =========================================================================== #
# Reference agent #2 — molecules (rdkit): REASON-GUIDED variant search.
# Structural surgery on a SMILES needs atom/bond editing (the harness's job); the deterministic reference
# instead reads the named flaw and returns a molecule that CLEARS it from a generated pool, reason-directed.
# rdkit is lazy-imported so this module stays stdlib for the DNA path.
# =========================================================================== #
@dataclass(frozen=True)
class MolSpec:
    """A molecule design goal: a valid, synthesizable, in-range small molecule. `seed` makes it deterministic."""

    seed: int = 0


@dataclass
class MolRepairAgent:
    """Reference agent for the `mol` gate. `propose` emits a flawed molecule (extreme / unsynthesizable /
    invalid); `revise` reads the named condemning contract and returns the most drug-like molecule from a
    generated pool that PASSES the gate. This is variant *search*, not structural surgery — the honest
    deterministic reference; a real harness (Claude Code) edits the structure directly."""

    _pool_cache: list[str] = field(default_factory=list, repr=False)

    def propose(self, spec: MolSpec) -> str:
        from .mol_qc_data import planted_decoys
        return planted_decoys(n=1, seed=spec.seed)[0]                  # an extreme-MW / high-SA decoy

    def revise(self, smiles: str, verdict: Verdict, spec: MolSpec) -> tuple[str, str]:
        from . import mol_qc as mq
        why = verdict.fired[0] if verdict.fired else "INVALID_MOLECULE"
        best = None
        best_qed = -1.0
        for cand in self._pool(spec):
            v = mq.validate(cand)
            if v.score == 0.0:                                        # passes the gate
                f = mq.featurize(cand)
                if f.qed > best_qed:                                  # reason-directed: the most drug-like pass
                    best, best_qed = cand, f.qed
        if best is None:
            return smiles, "no passing analogue in the pool"
        f = mq.featurize(best)
        return best, (f"swapped to a passing analogue (cleared {why}; "
                      f"MW {f.mw:.0f}, SA {f.sa:.1f}, QED {f.qed:.2f})")

    def _pool(self, spec: MolSpec) -> list[str]:
        if not self._pool_cache:
            from .mol_qc_data import brics_generated, reference_drugs
            self._pool_cache = list(reference_drugs()) + list(brics_generated(n=60, seed=spec.seed))
        return self._pool_cache
