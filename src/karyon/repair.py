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

from . import gen_dna_validity as gv
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
    """The whole loop: every step, the final artifact, and WHY it stopped.

    `stop_reason` is the honest classification of the ending state:
      * "converged" — the gate is satisfied (no condemning contract, no demanded disclosure firing).
      * "stalled"   — the potential made no new best for `stall_window` rounds (a fix kept reintroducing
                      what another cleared); the final step names the unresolved contracts.
      * "cycled"    — an artifact state was revisited exactly (the strongest non-progress witness).
      * "budget"    — `max_rounds` spent while still making (or able to make) progress.
    Only "converged" sets `converged` True — the three others are honest non-convergence, not a timeout."""

    modality: str
    steps: tuple[RepairStep, ...]
    final: str
    stop_reason: str

    @property
    def converged(self) -> bool:
        return self.stop_reason == "converged"

    @property
    def rounds(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {"modality": self.modality, "converged": self.converged, "stop_reason": self.stop_reason,
                "rounds": self.rounds, "final": self.final, "steps": [s.to_dict() for s in self.steps]}


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
    if not verdict.ok:                                   # a condemning contract fired (didn't pass the gate)
        return True
    return any(c in verdict.fired for c in clear_disclosures)


def _potential(verdict: Verdict, clear_disclosures: tuple[str, ...]) -> tuple[float, int]:
    """The loop's descent measure: `(Σ condemning weights, # demanded disclosures still firing)`,
    compared lexicographically. It is ≥ (0, 0) and equals (0, 0) **iff** `not _needs_work(...)`, so
    "converged" and "potential zero" are one definition. A revision makes real progress iff it strictly
    lowers this — clearing a condemning contract drops the first coordinate, clearing a demanded
    disclosure drops the second. Tracked best-so-far (not per-round), so a legitimate transient worsening
    — a GC rebalance that momentarily mints a higher-priority hairpin, cleared the next round — is allowed,
    while a fix that keeps trading one defect for another (no new best) is caught as a stall."""
    demanded = sum(1 for c in clear_disclosures if c in verdict.fired)
    return (verdict.score, demanded)


def _stop_action(kind: str, verdict: Verdict, stuck_fired: list[tuple[str, ...]], no_improve: int) -> str:
    """A named reason for an honest non-convergence stop — the legible replacement for a blind timeout.
    Lists the contracts left unresolved across the no-progress streak (the ones a fix kept reintroducing)."""
    unresolved = sorted({c for fs in stuck_fired for c in fs} | set(verdict.fired)) or ["none"]
    flags = ", ".join(unresolved)
    if kind == "cycled":
        return f"(cycle: artifact state revisited; unresolved [{flags}])"
    return f"(stalled: no progress for {no_improve} rounds; unresolved [{flags}])"


def repair_loop(spec: Any, agent: Agent, modality: str, *, max_rounds: int = 8,
                clear_disclosures: tuple[str, ...] = (), stall_window: int = 3) -> RepairTrajectory:
    """Drive an agent through generate → qualify → fix → re-qualify until the gate is satisfied (no
    condemning contract, and none of `clear_disclosures` still firing) or the loop stops honestly.

    Termination is guaranteed and *classified* (`RepairTrajectory.stop_reason`), not left to a blind
    `max_rounds` timeout: the loop tracks the best `_potential` seen and stops as **stalled** when no new
    best appears for `stall_window` rounds (a fix keeps trading one defect for another), as **cycled** when
    an artifact state is revisited exactly, as **converged** when the gate is clean, or as **budget** when
    `max_rounds` is reached while still improving. The best-so-far rule tolerates a legitimate transient
    worsening (see `_potential`) but catches true thrash even when the artifact never exactly repeats.

    The defaults (`max_rounds=8`, `stall_window=3`) suit the bundled reference agents, which descend
    monotonically and converge in a few rounds — they never stall. `stall_window` is "rounds without a NEW
    best before declaring a stall", so it assumes a *working* agent makes progress at least that often: a
    more exploratory / LLM agent that legitimately needs several lateral moves before a breakthrough may be
    cut off early as `stalled`, and a harder design may need more `max_rounds` (the result is then the honest
    `budget`, never a false `converged`). For such agents raise both; keep `stall_window <= max_rounds` so
    stall detection stays reachable (otherwise the loop always reports `budget`)."""
    if max_rounds < 0:
        raise ValueError(f"max_rounds must be >= 0 (0 = qualify-only, no edits); got {max_rounds}")
    if stall_window < 1:
        raise ValueError(f"stall_window must be >= 1 (rounds without a new best before stalling); "
                         f"got {stall_window}")
    artifact = agent.propose(spec)
    steps: list[RepairStep] = []
    seen: set[str] = set()
    best_phi: tuple[float, int] | None = None
    no_improve = 0
    stuck_fired: list[tuple[str, ...]] = []                  # fired-sets over the current no-progress streak
    stop_reason = "budget"

    for r in range(max_rounds + 1):
        verdict = qualify(artifact, modality=modality).items[0][1]

        if not _needs_work(verdict, clear_disclosures):
            steps.append(RepairStep(r, artifact, True, verdict.score, verdict.reasons, ""))
            stop_reason = "converged"
            break

        if artifact in seen:                                 # exact-state repeat — a witnessed cycle
            steps.append(RepairStep(r, artifact, verdict.ok, verdict.score, verdict.reasons,
                                    _stop_action("cycled", verdict, stuck_fired, no_improve)))
            stop_reason = "cycled"
            break

        phi = _potential(verdict, clear_disclosures)
        if best_phi is None or phi < best_phi:               # progress == a NEW best (tolerates transient worsening)
            best_phi, no_improve, stuck_fired = phi, 0, []
        else:
            no_improve += 1
            stuck_fired.append(tuple(verdict.fired))

        if no_improve >= stall_window:                       # no new best for `stall_window` rounds — real thrash
            steps.append(RepairStep(r, artifact, verdict.ok, verdict.score, verdict.reasons,
                                    _stop_action("stalled", verdict, stuck_fired, no_improve)))
            stop_reason = "stalled"
            break

        if r == max_rounds:                                  # budget spent while still improving
            steps.append(RepairStep(r, artifact, verdict.ok, verdict.score,
                                    verdict.reasons, "(budget exhausted)"))
            stop_reason = "budget"
            break

        new_artifact, action = agent.revise(artifact, verdict, spec)
        steps.append(RepairStep(r, artifact, verdict.ok, verdict.score, verdict.reasons, action))
        seen.add(artifact)
        artifact = new_artifact
    return RepairTrajectory(modality, tuple(steps), artifact, stop_reason)


_STOP_LABEL = {"converged": "CONVERGED", "stalled": "STALLED", "cycled": "CYCLED",
               "budget": "BUDGET EXHAUSTED"}


def format_trajectory(traj: RepairTrajectory) -> str:
    """A human-readable rendering of a trajectory (the demo print + the CLI human output)."""
    label = _STOP_LABEL.get(traj.stop_reason, traj.stop_reason.upper())
    lines = [f"repair loop · {traj.modality} · {label} in {traj.rounds - 1} edit(s)"]
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


# The condemning (weight>0) contracts of the `dna` gate — what a fix must never newly introduce.
_CONDEMNING = ("LENGTH_OUT_OF_RANGE", "HOMOPOLYMER_RUN", "STRONG_HAIRPIN", "GC_OUT_OF_BAND")


def _condemning_fired(seq: str) -> frozenset[str]:
    """The condemning contracts firing on `seq`, read from the SAME gate the loop qualifies against — so a
    fix and the verdict can never disagree about what's wrong (no second, drifting copy of the rules)."""
    return frozenset(c for c in gv.validate(seq).fired if c in _CONDEMNING)


def _creates_long_run(s: list[str], pos: int, base: str, cap: int = 3) -> bool:
    """Would setting `s[pos] = base` create a run of identical bases longer than `cap` around `pos`?
    A cheap O(run) local check — keeps the bulk GC rebalance from ever minting a HOMOPOLYMER (>8) or a
    POLY_G (>3), so the only condemning defect a rebalance can introduce is a (rare) hairpin."""
    left = pos - 1
    while left >= 0 and s[left] == base:
        left -= 1
    right = pos + 1
    while right < len(s) and s[right] == base:
        right += 1
    return (right - left - 1) > cap


def _safe_mutate(seq: str, pos: int, rng: random.Random, *, avoid: str = "") -> str:
    """Mutate base at `pos` to one that introduces NO new condemning contract (its condemning set stays a
    subset of what already fires). This is what makes a fix monotone: it can only clear defects, never add
    one. Falls back to a plain `_mutate` when no fully-safe base exists — the loop's stall guard then
    catches the absence of progress, instead of the fix silently making things worse."""
    allowed = _condemning_fired(seq)
    choices = [b for b in _BASES if b != seq[pos] and b not in avoid] or [b for b in _BASES if b != seq[pos]]
    rng.shuffle(choices)
    for b in choices:
        cand = seq[:pos] + b + seq[pos + 1:]
        if _condemning_fired(cand) <= allowed:
            return cand
    return _mutate(seq, pos, rng, avoid=avoid)


def _correct_new_condemning(seq: str, allowed: frozenset[str], rng: random.Random,
                            max_fixes: int = 8) -> str:
    """Clear any condemning contract NOT in `allowed` that a bulk edit (the GC rebalance) introduced, via
    bounded safe single-base fixes — so the bulk fix returns a sequence whose condemning set ⊆ `allowed`.
    By construction the rebalance can only add a STRONG_HAIRPIN (runs are guarded); this removes it."""
    for _ in range(max_fixes):
        extra = _condemning_fired(seq) - allowed
        if "STRONG_HAIRPIN" in extra:
            hp = seq_dfm.hairpin_stem(seq)
            pos = hp.start + hp.stem // 2 if hp.stem else rng.randrange(len(seq))
        elif "HOMOPOLYMER_RUN" in extra:
            start, length = _longest_run_span(seq)
            pos = start + length // 2
        else:
            break                                            # nothing this corrector can address
        nbrs = seq[max(0, pos - 1)] + seq[min(len(seq) - 1, pos + 1)]
        seq = _safe_mutate(seq, pos, rng, avoid=nbrs)
    return seq


@dataclass
class DnaRepairAgent:
    """Surgical reference agent for the `dna` gate. `propose` emits a rough first draft (out-of-band GC + an
    injected homopolymer + an injected EcoRI site) so the loop has visible work; `revise` fixes ONE named
    contract per round using the gate's own `seq_dfm` primitives — no message parsing.

    Every fix is **defect-safe**: it clears its target and introduces no new condemning contract (checked
    against the same gate, via `_safe_mutate` / `_correct_new_condemning`). So each round strictly lowers
    `_potential` — score drops by the cleared contract's weight, or a demanded disclosure leaves — which
    makes `repair_loop` provably converge whenever a defect-safe edit exists (and stop with a named stall
    when one doesn't), rather than relying on the magnitude accident the un-guarded loop did."""

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
        before = seq_dfm.gc_fraction(seq)
        allowed = _condemning_fired(seq) - {"GC_OUT_OF_BAND"}           # clear GC; keep everything else ⊆ this
        s = list(seq)
        lo, hi = 0.32, 0.58                                            # band centre, ~7pt margin inside 0.25–0.65
        guard = 0
        while seq_dfm.gc_fraction("".join(s)) < lo and guard < len(s) * 4:
            i = rng.randrange(len(s))
            if s[i] in "AT":
                for b in ("G", "C"):                                   # flip toward band, never extending a run
                    if not _creates_long_run(s, i, b):
                        s[i] = b
                        break
            guard += 1
        while seq_dfm.gc_fraction("".join(s)) > hi and guard < len(s) * 8:
            i = rng.randrange(len(s))
            if s[i] in "GC":
                for b in ("A", "T"):
                    if not _creates_long_run(s, i, b):
                        s[i] = b
                        break
            guard += 1
        out = _correct_new_condemning("".join(s), allowed, rng)        # clear any hairpin the rebalance minted
        return out, f"rebalanced GC {before:.0%}→{seq_dfm.gc_fraction(out):.0%} into the synthesis band"

    # Each fix FULLY clears its targeted contract in one round (it may take several safe edits — a long run
    # splits in halves, a set may carry several sites) so the contract leaves the verdict and `_potential`
    # strictly drops. "One defect per round" means one named CONTRACT per round, not one base edit.
    def _fix_homopolymer_run(self, seq, spec, rng) -> tuple[str, str]:
        cap = gv.GenDNATol().max_homopolymer_run
        longest, n, guard = seq_dfm.longest_run(seq), 0, 0
        while seq_dfm.longest_run(seq) > cap and guard < len(seq):
            start, length = _longest_run_span(seq)
            pos = start + length // 2
            nbrs = seq[max(0, pos - 1)] + seq[min(len(seq) - 1, pos + 1)]
            seq = _safe_mutate(seq, pos, rng, avoid=nbrs)
            n, guard = n + 1, guard + 1
        return seq, f"broke {n} homopolymer run(s) below the {cap}-base cap (longest was {longest})"

    def _fix_poly_g_run(self, seq, spec, rng) -> tuple[str, str]:
        cap = gv.GenDNATol().max_g_run
        longest, n, guard = seq_dfm.longest_run(seq, "G"), 0, 0
        while seq_dfm.longest_run(seq, "G") > cap and guard < len(seq):
            start, length = _longest_run_span(seq, "G")
            pos = start + length // 2
            seq = _safe_mutate(seq, pos, rng, avoid="G")
            n, guard = n + 1, guard + 1
        return seq, f"broke {n} poly-G run(s) below the {cap}-base cap (longest was {longest})"

    def _fix_strong_hairpin(self, seq, spec, rng) -> tuple[str, str]:
        min_stem = gv.GenDNATol().min_stem_len
        strongest, n, guard = seq_dfm.hairpin_stem(seq).stem, 0, 0
        while seq_dfm.hairpin_stem(seq).stem >= min_stem and guard < len(seq):
            hp = seq_dfm.hairpin_stem(seq)
            pos = hp.start + hp.stem // 2                       # split the stem at its middle — halves it per edit
            nbrs = seq[max(0, pos - 1)] + seq[min(len(seq) - 1, pos + 1)]
            seq = _safe_mutate(seq, pos, rng, avoid=nbrs)
            n, guard = n + 1, guard + 1
        return seq, f"disrupted {n} hairpin stem(s) below the {min_stem} bp cap (strongest was {strongest})"

    def _fix_length_out_of_range(self, seq, spec, rng) -> tuple[str, str]:
        if len(seq) < 18:
            pad = "".join(rng.choices(_BASES, k=18 - len(seq)))
            return seq + pad, f"padded to the orderable floor ({len(seq)}→{len(seq) + len(pad)} nt)"
        return seq[:3000], f"trimmed to the single-fragment window ({len(seq)}→3000 nt)"

    def _fix_restriction_site(self, seq, spec, rng) -> tuple[str, str]:
        # the contract fires while ANY tabled site remains, so clear them all (spec.avoid_enzymes first).
        enzymes, n, guard = set(), 0, 0
        while seq_dfm.restriction_sites(seq) and guard < len(seq):
            sites = seq_dfm.restriction_sites(seq)
            target = next((h for h in sites if h.enzyme in spec.avoid_enzymes), sites[0])
            pos = target.position + len(target.site) // 2
            seq = _safe_mutate(seq, pos, rng)
            enzymes.add(target.enzyme)
            n, guard = n + 1, guard + 1
        return seq, f"removed {n} restriction site(s) ({', '.join(sorted(enzymes))})"


# =========================================================================== #
# Reference agent #2 — molecules (rdkit): GATE-FILTERED variant search.
# Structural surgery on a SMILES needs atom/bond editing (the harness's job); the deterministic reference
# instead returns the most drug-like molecule that CLEARS the gate from a generated pool. It does NOT branch
# on the named flaw (every passing candidate clears all of them) — the reason is surfaced in the action, not
# used to steer; that reason-directed structural edit is what a real harness does.
# rdkit is lazy-imported so this module stays stdlib for the DNA path.
# =========================================================================== #
@dataclass(frozen=True)
class MolSpec:
    """A molecule design goal: a valid, synthesizable, in-range small molecule. `seed` makes it deterministic."""

    seed: int = 0


@dataclass
class MolRepairAgent:
    """Reference agent for the `mol` gate. `propose` emits a flawed molecule (extreme / unsynthesizable /
    invalid); `revise` runs a GATE-FILTERED variant search — it returns the most drug-like molecule from a
    generated pool that PASSES the gate. Unlike the surgical `DnaRepairAgent` (which dispatches on the named
    contract), the choice here does NOT branch on which reason fired: every passing candidate clears all of
    them, so the search is gate-directed and the named reason is *surfaced* in the action for legibility, not
    used to steer. This is variant *search*, not structural surgery — the honest deterministic reference; a
    real harness (Claude Code) reads the named flaw and edits the structure directly."""

    _pool_cache: list[str] = field(default_factory=list, repr=False)

    def propose(self, spec: MolSpec) -> str:
        from .mol_qc_data import planted_decoys
        return planted_decoys(n=1, seed=spec.seed)[0]                  # an extreme-MW / high-SA decoy

    def revise(self, smiles: str, verdict: Verdict, spec: MolSpec) -> tuple[str, str]:
        from . import mol_qc as mq
        why = verdict.fired[0] if verdict.fired else "INVALID_MOLECULE"   # surfaced in the action only
        best = None
        best_qed = -1.0
        for cand in self._pool(spec):
            v = mq.validate(cand)
            if v.ok:                                                  # passes the gate (clears every contract)
                f = mq.featurize(cand)
                if f.qed > best_qed:                                  # gate-filtered: the most drug-like pass
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
