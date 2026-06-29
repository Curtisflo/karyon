# Agent self-repair loop

**karyon's gates don't just say PASS/FAIL — every rejection names its reason.** A named reason is a
*repair instruction*: an agent can read it and make the corresponding edit, then re-check. A black-box
pass/fail can't drive that loop; a legible one can. This directory is that loop, closed:

```
    artifact = agent.propose(spec)             # make something
    repeat:
        verdict = qualify(artifact, modality)  # check it  (the karyon gate)
        if clean: done
        artifact, action = agent.revise(        # fix from the NAMED reasons
            artifact, verdict, spec)
```

## Run the reference loop (no API, no rdkit for DNA)

```bash
python examples/agent_loop/repair_dna.py        # pure stdlib
python examples/agent_loop/repair_mol.py        # needs: pip install "karyon[chem]"
```

DNA output reads like an agent talking itself into a synthesizable insert:

```
repair loop · dna · CONVERGED in 3 edit(s)
  round 0: FAIL  [GC_OUT_OF_BAND, HOMOPOLYMER_RUN, RESTRICTION_SITE]
           ↳ broke a 14-base homopolymer run at position 46
  round 1: FAIL  [GC_OUT_OF_BAND, RESTRICTION_SITE]
           ↳ rebalanced GC 22%→32% into the synthesis band
  round 2: PASS  [RESTRICTION_SITE]
           ↳ removed the EcoRI site at position 80
  round 3: PASS  [clean]
```

Same thing from the CLI:

```bash
karyon repair -m dna --clear RESTRICTION_SITE          # the reference agent proposes + repairs
karyon repair my_draft.fasta -m dna --json             # repair YOUR draft, emit the trajectory as JSON
```

## The real agent is **you, in Claude Code** (no API key)

The deterministic agents here exist so the loop is runnable and CI-tested. In normal use **you don't need
any of them** — your harness is the agent. Sitting in Claude Code in your terminal, the loop is just your
turn-by-turn cycle, with `karyon qualify` as the check tool:

> 1. Ask Claude Code to design the artifact (write `candidate.fasta` / a SMILES).
> 2. It runs `karyon qualify candidate.fasta -m dna --json` (exit 1 + named reasons on failure).
> 3. It reads the reasons — `homopolymer run of 12 (>8)`, `EcoRI site present` — and edits the file.
> 4. It re-runs `karyon qualify` until exit 0.

karyon's only job in that loop is to make the check **deterministic** and the reasons **specific enough to
act on** (the enzyme by name, the run length, the GC value). That is the whole thesis: *legible QC is what
makes agentic self-repair possible.*

## How the reference agents map reasons → fixes

Both key on the **contract name** (`verdict.fired`) and re-derive the defect from the gate's own primitives
(`seq_dfm`) — no message-string parsing.

| modality | reason | reference fix |
|---|---|---|
| dna | `GC_OUT_OF_BAND` | swap A/T↔G/C toward the synthesis band |
| dna | `HOMOPOLYMER_RUN` | locate the run, mutate its middle base |
| dna | `STRONG_HAIRPIN` | `hairpin_stem()` → split the stem at its middle |
| dna | `RESTRICTION_SITE` | `restriction_sites()` → mutate a base in the named site |
| dna | `LENGTH_OUT_OF_RANGE` | pad / trim to the orderable window |
| mol | `EXTREME_PROPERTY` / `UNSYNTHESIZABLE` / `INVALID_MOLECULE` | return the most drug-like passing analogue from a generated pool |

The DNA agent is **surgical** (one targeted edit per round). The molecule agent is **reason-guided variant
search** — structural surgery on a SMILES is the harness's job, not a 200-line reference's. Either way the
loop is driven entirely by the gate's named reasons.

Library surface (`from karyon import …`): `repair_loop`, `RepairTrajectory`, `Agent`, `DnaRepairAgent`,
`DnaSpec`, `MolRepairAgent`, `MolSpec`, `format_trajectory`. See [`docs/repair.md`](../../docs/repair.md)
for the loop contract and the trajectory JSON schema.
