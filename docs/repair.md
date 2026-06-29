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

`repair_loop(spec, agent, modality, *, max_rounds=8, clear_disclosures=()) -> RepairTrajectory`

Each round runs `qualify(artifact, modality=modality)` on the current artifact (an inline DNA/SMILES
string) and stops when the artifact is **converged**: no condemning contract fired (`verdict.score == 0`)
**and** none of `clear_disclosures` still firing. Otherwise it calls `agent.revise(...)` and repeats, up to
`max_rounds`.

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

- `DnaRepairAgent` / `DnaSpec` — surgical, pure stdlib. Maps each fired DNA contract to a targeted
  `seq_dfm`-driven edit (rebalance GC, break a homopolymer, split a hairpin, remove a named restriction
  site). One defect per round.
- `MolRepairAgent` / `MolSpec` — reason-guided variant search (needs `karyon[chem]`). Reads the named
  condemning contract and returns the most drug-like molecule from a generated pool that passes the gate.

## Trajectory JSON schema

`RepairTrajectory.to_dict()` (stable, JSON-safe):

```json
{
  "modality": "dna",
  "converged": true,
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
  accounts for `clear_disclosures`.
- `action` is `""` on the converged final step, and `"(budget exhausted)"` if `max_rounds` was spent
  without converging.

## CLI

```bash
karyon repair -m dna --clear RESTRICTION_SITE     # reference agent proposes a flawed draft + repairs it
karyon repair draft.fasta -m dna --json           # repair your draft; emit the trajectory JSON
karyon repair "CCCC…(SMILES)…" -m mol             # molecule repair (needs karyon[chem])
```

Exit code: `0` converged, `1` not converged within budget, `2` on a usage error.
