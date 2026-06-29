# `karyon.repair` — the agent self-repair loop

karyon's gates name their reasons, and a named reason is a repair instruction. `karyon.repair` is the small
harness that closes **generate → qualify → fix-from-reasons → re-qualify → converge** over `karyon.qualify`.

## The loop

```python
from karyon import repair_loop, DnaRepairAgent, DnaSpec, format_trajectory

traj = repair_loop(
    DnaSpec(length=240, avoid_enzymes=("EcoRI", "BsaI")),
    DnaRepairAgent(),                       # the agent (pluggable; see below)
    "dna",                                  # modality (any karyon.qualify modality)
    max_rounds=8,
    clear_disclosures=("RESTRICTION_SITE",),  # also clear these weight-0 (disclose) contracts
)
print(format_trajectory(traj))
```

`repair_loop(spec, agent, modality, *, max_rounds=8, clear_disclosures=(), stall_window=3) -> RepairTrajectory`

Each round runs `qualify(artifact, modality=modality)` on the current artifact (an inline DNA/SMILES
string) and stops when the artifact is **converged**: no condemning contract fired (`verdict.score == 0`)
**and** none of `clear_disclosures` still firing. Otherwise it calls `agent.revise(...)` and repeats.

## Termination is classified, not a blind timeout

The loop tracks a lexicographic **potential** `(Σ condemning weights, # demanded disclosures still
firing)` — minimized at `(0,0)`, which is exactly the converged state. It stops with a named
`stop_reason`:

- **`converged`** — the gate is satisfied.
- **`stalled`** — the potential reached no new best for `stall_window` rounds (a fix kept reintroducing
  what another cleared); the final step's `action` *names the unresolved contracts*.
- **`cycled`** — an artifact state was revisited exactly (the strongest non-progress witness).
- **`budget`** — `max_rounds` spent while still improving.

Tracking the *best potential so far* (rather than demanding a per-round drop) deliberately tolerates a
legitimate transient worsening — e.g. a GC rebalance that momentarily mints a higher-priority hairpin,
cleared the next round — while still catching true thrash even when the artifact never exactly repeats.
Only `converged` sets `RepairTrajectory.converged` True; the other three are honest non-convergence.

**Defaults and pluggable agents.** `max_rounds=8` / `stall_window=3` suit the bundled reference agents,
which descend monotonically and converge in a few rounds (they never stall). `stall_window` is "rounds
without a *new best* before declaring a stall", so it assumes a working agent makes progress at least that
often. A more exploratory / LLM agent that legitimately needs several lateral moves before a breakthrough
may be cut off early as `stalled`, and a harder design may need more rounds (you then get the honest
`budget`, never a false `converged`) — for such agents raise both, and keep `stall_window ≤ max_rounds` so
stall detection stays reachable. `stall_window` must be ≥ 1 and `max_rounds` ≥ 0 (both validated).

**Reference-agent guarantee.** `DnaRepairAgent`'s fixes are *defect-safe* — each round fully clears one
named contract and introduces no new condemning contract (checked against the same gate) — so the
potential strictly decreases every round and the loop **provably converges** in ≤ `⌈initial score⌉ +
demanded-disclosures` rounds whenever a defect-safe edit exists, and otherwise stops with a named stall.
(The loop itself can't guarantee convergence for an arbitrary pluggable agent — e.g. an LLM harness — so
for those the guarantee is honest classification, not convergence.)

## The agent role

```python
class Agent(Protocol):
    def propose(self, spec) -> str: ...                                    # make a first draft
    def revise(self, artifact: str, verdict: Verdict, spec) -> tuple[str, str]: ...   # (fixed, action)
```

`revise` receives the full `Verdict` — read `verdict.fired` (the contract names) and `verdict.reasons`
(each `Reason(contract, message, weight)`) and return the edited artifact plus a one-line description of
what you did. **Your harness is an agent**: in Claude Code, `propose`/`revise` are just your turns, with
`karyon qualify` as the check tool — no API key (see `examples/agent_loop/README.md`).

Built-in reference agents (so the loop runs and is CI-tested with zero LLM):

- `DnaRepairAgent` / `DnaSpec` — surgical, pure stdlib. **Dispatches on the named contract**: each fired
  DNA contract maps to a targeted `seq_dfm`-driven edit (rebalance GC, break a homopolymer, split a hairpin,
  remove a named restriction site). One named contract fully cleared per round, defect-safe (see the
  guarantee above).
- `MolRepairAgent` / `MolSpec` — **gate-filtered** variant search (needs `karyon[chem]`). Returns the most
  drug-like molecule from a generated pool that passes the gate. Unlike the DNA agent it does **not** branch
  on which contract fired — every passing candidate clears all of them, so the search is gate-directed and
  the cleared reason is only *surfaced* in the action (for legibility), not used to steer. Reason-directed
  structural editing of a SMILES is what a real harness (Claude Code) does.

## Trajectory JSON schema

`RepairTrajectory.to_dict()` (stable, JSON-safe):

```json
{
  "modality": "dna",
  "converged": true,
  "stop_reason": "converged",
  "rounds": 4,
  "final": "GTAATATCATCTATAACCGCG…",
  "steps": [
    {
      "round": 0,
      "artifact": "…the artifact qualified this round…",
      "ok": false,
      "score": 3.0,
      "reasons": [
        {"contract": "HOMOPOLYMER_RUN", "message": "homopolymer run of 14 (>8) — synthesis slippage", "weight": 1.5}
      ],
      "action": "broke a 14-base homopolymer run at position 46"
    }
  ]
}
```

- `ok` on a step = the gate passed that round (`score == 0`); the trajectory's `converged` additionally
  accounts for `clear_disclosures` (and is True only when `stop_reason == "converged"`).
- `action` is `""` on the converged final step, a named reason on a `stalled`/`cycled` final step
  (e.g. `"(stalled: no progress for 3 rounds; unresolved [GC_OUT_OF_BAND, HOMOPOLYMER_RUN])"`), and
  `"(budget exhausted)"` when `max_rounds` was spent while still improving.

## CLI

```bash
karyon repair -m dna --clear RESTRICTION_SITE     # reference agent proposes a flawed draft + repairs it
karyon repair draft.fasta -m dna --json           # repair your draft; emit the trajectory JSON
karyon repair "CCCC…(SMILES)…" -m mol             # molecule repair (needs karyon[chem])
```

Exit code: `0` converged, `1` not converged (`stalled` / `cycled` / `budget` — see `stop_reason` in the
output), `2` on a usage error.
