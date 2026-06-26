# Contributing to karyon

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test,chem,seqdesign,data]"
pytest                       # data-backed tests skip cleanly when offline
```

Set `KARYON_CACHE` to a directory with cached datasets to avoid re-fetching during test runs.

## The contract pattern

Every QC check is a **named, legible contract** registered on a `contracts.ContractSet`. A check returns
`None` (clean) or a string/`Reason` (fired, with a human-readable message). `ContractSet.evaluate(...)`
returns a `Verdict` whose `.messages` and `.fired` explain exactly what failed and why. This legibility is
the point — prefer a deterministic contract that names its reason over an opaque score.

```python
from karyon import contracts
cs = contracts.ContractSet("my-substrate")

@cs.rule("GC_BAND")
def _(seq, ctx):
    g = sum(c in "GC" for c in seq) / len(seq)
    return f"GC {g:.0%} outside 20–80%" if not 0.2 <= g <= 0.8 else None

verdict = cs.evaluate("GACC...")          # -> Verdict(ok=..., reasons=..., score=...)
```

## Adding a QC module or skill

1. Add `src/karyon/<name>.py` exposing a `ContractSet` (and a loader in `<name>_data.py` if it needs data —
   fetch-and-cache, never redistribute; document it in `DATASETS.md`).
2. Add `tests/test_<name>.py`. Tests must skip (not fail) when a dataset is unavailable.
3. To expose it to agents, add `skills/<name>/SKILL.md` (YAML frontmatter + instructions) and register it
   in `.claude-plugin/marketplace.json`.

## Scope

karyon qualifies the outputs of bio-AI tools; it does not reimplement predictors or solvers. Keep
contributions on the reliability/QC/qualification axis.
