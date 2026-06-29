"""cli — the `karyon` command-line entry point: one tool over the whole QC spine.

Two verbs mirror the library's two families:

  * ``karyon qualify <artifact> [--modality M] [--json]`` — a per-artifact gate (pose / cofold / complex /
    mol / dna / promoter) → a `qualify.QualifyResult`. Exit code is 0 on PASS, 1 on FAIL (a condemning
    contract fired), 2 on a usage/setup error, so it gates a pipeline directly.
  * ``karyon audit <leakage|screen> [--json]`` — a dataset-level audit (benchmark leakage / CRISPR-screen
    under-power) → a serialized report. Exit 0 on a produced report, 2 if the public dataset is unavailable.

  * ``karyon list`` — enumerate the modalities and audits.

Thin by design: `qualify` delegates to `karyon.qualify`; `audit` wraps the existing audit modules' report
surfaces. No QC logic lives here — only argument parsing, dispatch, and presentation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys

from . import spine as _q


# --------------------------------------------------------------------------- #
# qualify
# --------------------------------------------------------------------------- #
def _print_human(result: _q.QualifyResult) -> None:
    for name, v in result.items:
        head = "PASS" if v.ok else "FAIL"
        print(f"{head}  {name}")
        for r in v.reasons:
            print(f"   {'✗' if r.weight > 0 else '·'} {r.contract}: {r.message}")
    if result.batch is not None:
        b = result.batch
        print(f"{'PASS' if b.ok else 'FAIL'}  [batch / set-level]")
        for r in b.reasons:
            print(f"   {'✗' if r.weight > 0 else '·'} {r.contract}: {r.message}")
    n_pass = sum(1 for _, v in result.items if v.ok)
    print(f"\n{n_pass}/{len(result.items)} pass · "
          f"{'PASS' if result.ok else 'FAIL'} overall ({result.modality})")


def _cmd_qualify(args) -> int:
    try:
        result = _q.qualify(
            args.artifact, modality=args.modality,
            ligand=args.ligand, ligand_resname=args.ligand_resname,
            chain_a=args.chain_a, chain_b=args.chain_b,
        )
    except _q.QualifyError as e:
        print(f"ERROR — {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_human(result)
    return 0 if result.ok else 1


# --------------------------------------------------------------------------- #
# repair — the agent self-repair loop, driven by the built-in reference agent.
# --------------------------------------------------------------------------- #
class _StartFrom:
    """Wraps a reference agent so the loop starts from a GIVEN artifact (the user's draft) rather than the
    agent's own first draft; `revise` still delegates to the reference agent's reason→edit logic."""

    def __init__(self, start: str, base) -> None:
        self._start, self._base = start, base

    def propose(self, spec):
        return self._start

    def revise(self, artifact, verdict, spec):
        return self._base.revise(artifact, verdict, spec)


def _read_artifact(source: str, modality: str) -> str:
    """The raw sequence / SMILES to start from — an inline string or the first record of a file."""
    if modality == "dna":
        return _q._dna_records(source)[0][1]
    return _q._mol_load(source, {})[0][1]


def _cmd_repair(args) -> int:
    from .repair import DnaRepairAgent, DnaSpec, MolRepairAgent, MolSpec, format_trajectory, repair_loop
    if args.modality == "dna":
        agent, spec = DnaRepairAgent(), DnaSpec(length=args.length)
        clear = tuple(args.clear) if args.clear else ()
    else:
        agent, spec = MolRepairAgent(), MolSpec()
        clear = tuple(args.clear) if args.clear else ()
    try:
        if args.artifact:
            agent = _StartFrom(_read_artifact(args.artifact, args.modality), agent)
        traj = repair_loop(spec, agent, args.modality, max_rounds=args.rounds, clear_disclosures=clear)
    except _q.QualifyError as e:
        print(f"ERROR — {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(traj.to_dict(), indent=2))
    else:
        print(format_trajectory(traj))
    return 0 if traj.converged else 1


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
_LEAKAGE_RETRO = ("uspto50k", "retro", "retrosynthesis")
_LEAKAGE_PPI = ("ppi", "guo", "ppi-leakage")


def _print_screen_report(rep: dict, source: str) -> None:
    """Human summary of a single-cell screen audit — the calibration, the Q1–Q4 line, the partition, and
    the decision-relevant payload: the no-phenotype calls a user should not trust, each with its reasons."""
    cal = rep["calibration"]
    print(f"single-cell screen QC · {source}")
    print(f"  {rep['n_targeting']} targeting / {rep['n_controls']} control perturbations")
    print(f"  incumbent calibration: {cal['targeting_hit_rate']:.0%} targeting vs {cal['control_hit_rate']:.0%} "
          f"control hit-rate  ({'credible' if cal['credible'] else 'weak — check the control set'})")
    print(f"  no-phenotype calls (silent-failure denominator): {rep['n_nophenotype']}")
    enr, rho = rep["weak_kd_enrichment"], rep["rho_knockdown_vs_significance"]
    enr_s = "n/a" if enr is None else f"{enr:.1f}×"
    rho_s = "n/a" if rho is None else f"{rho:.3f}"
    print(f"  flagged untrustworthy: {rep['flagged_rate']:.1%}  ·  weak-KD enrichment "
          f"{enr_s}  ·  |ρ| vs significance {rho_s}")
    part = rep["partition"]
    print(f"  partition: {part['untrustworthy']} untrustworthy · {part['trustworthy_negative']} trustworthy "
          f"negatives · {part['unmeasurable']} unmeasurable")
    flagged = rep["flagged"]
    if flagged:
        shown = min(len(flagged), 12)
        print(f"  no-phenotype calls you should NOT trust (top {shown} of {rep['n_flagged']}):")
        for f in flagged[:shown]:
            kd = "KD unmeasured" if f["knockdown_residual"] is None else f"{f['knockdown_residual']:.0%} residual"
            cells = "" if f["n_cells"] is None else f", {f['n_cells']} cells"
            names = ", ".join(r["contract"] for r in f["reasons"])
            print(f"     {f['target']:<12} {kd:<14} energy-p {f['energy_p']:.2g}{cells}  [{names}]")


def _run_single_cell_screen(input_path: str | None) -> dict:
    """The single-cell Perturb-seq screen audit: qualify a user's own screen (`--input`, core install only)
    or the bundled Replogle reference (needs `karyon[singlecell]`)."""
    from . import perturbseq_qc as pq
    if input_path:
        perts = pq.load_user_screen(input_path)                 # QualifyError on a bad/missing table
        source = input_path
    else:
        from .perturbseq_data import DatasetUnavailable, load_perturbations
        try:
            perts = load_perturbations()
        except DatasetUnavailable as e:
            raise _q.QualifyError(
                f"single-cell reference screen unavailable ({e}). Install the reader with "
                f'`pip install "karyon[singlecell]"`, or pass --input YOUR_SCREEN.csv to qualify your own '
                f"screen (CSV/TSV, no h5py needed).")
        source = "Replogle K562-essential (bundled reference)"
    rep = pq.audit_report(perts=perts)
    _print_screen_report(rep, source)
    return {"audit": "screen", "source": source, **rep}


def _run_audit(kind: str, benchmark: str, seeds: int | None,
               *, single_cell: bool = False, input_path: str | None = None) -> dict:
    """Produce a JSON-safe audit report by delegating to the audit modules. Their human prints are the
    caller's to route (captured for --json)."""
    if kind == "screen":
        if single_cell:
            return _run_single_cell_screen(input_path)
        from . import screen_qc
        rep = screen_qc.run() if seeds is None else screen_qc.run(seeds=seeds)
        return {"audit": "screen", **rep}
    if single_cell or input_path:
        raise _q.QualifyError("--single-cell / --input apply only to `karyon audit screen`")
    # kind == "leakage"
    if benchmark in _LEAKAGE_RETRO:
        from . import retro_honesty as rh
        rep = rh.report() if seeds is None else rh.report(seeds=seeds)
        return {"audit": "leakage", **rep}
    if benchmark in _LEAKAGE_PPI:
        from . import ppi_leakage as pl
        rep = pl.report() if seeds is None else pl.report(seeds=seeds)
        if rep is None:
            raise _q.QualifyError("dataset 'ppi' (Guo yeast PPI) unavailable (offline?)")
        return {"audit": "leakage", **rep}
    from . import molnet_honesty as mh
    from .molnet_data import DATASETS
    if benchmark not in DATASETS:
        raise _q.QualifyError(
            f"unknown leakage benchmark {benchmark!r}; choose one of "
            f"{list(_LEAKAGE_RETRO[:1]) + ['ppi'] + list(DATASETS)}")
    rep = mh.run_one(benchmark)
    if rep is None:
        raise _q.QualifyError(f"dataset {benchmark!r} unavailable (offline?)")
    return {"audit": "leakage", **rep}


def _cmd_audit(args) -> int:
    sink = io.StringIO() if args.json else sys.stdout
    try:
        with contextlib.redirect_stdout(sink):              # keep stdout clean for --json
            rep = _run_audit(args.kind, args.benchmark, args.seeds,
                             single_cell=args.single_cell, input_path=args.input)
    except _q.QualifyError as e:
        print(f"ERROR — {e}", file=sys.stderr)
        return 2
    except SystemExit:                                       # an audit module's "SKIP — dataset unavailable"
        print("ERROR — dataset unavailable (network required, or set up the cache)", file=sys.stderr)
        return 2
    except Exception as e:                                   # noqa: BLE001 — surface dataset/setup failures
        print(f"ERROR — audit failed: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rep, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
def _cmd_list(args) -> int:
    print("qualify modalities (karyon qualify <artifact> --modality M):")
    for m in _q.modalities():
        g = _q.GATES[m]
        extras = f"  [needs {', '.join(g.extras)}]" if g.extras else ""
        print(f"  {m:<9} {', '.join(g.extensions) or '(inline only)'}{extras}")
    print("\naudits (karyon audit KIND):")
    print("  leakage    --benchmark uspto50k | bbbp | esol | ppi")
    print("  screen     bulk dropout (Wang-2014 reference)")
    print("  screen --single-cell             Perturb-seq silent-failure QC (Replogle reference)")
    print("  screen --single-cell --input F   qualify your own single-cell screen (CSV/TSV, core install)")
    print("\nagent self-repair loop (karyon repair [artifact] -m MODALITY):")
    print("  dna        surgical reason→edit fixes (GC / homopolymer / hairpin / restriction site)")
    print("  mol        reason-guided variant search (invalid / extreme / unsynthesizable)")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="karyon",
        description="A legible reliability/QC gate over commodity bio-AI tools — the model proposes, "
                    "karyon qualifies.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    qp = sub.add_parser("qualify", help="qualify an artifact through a karyon QC gate")
    qp.add_argument("artifact", help="a file, a directory/glob of files, or (mol/dna/promoter) an inline string")
    qp.add_argument("-m", "--modality", choices=_q.modalities(),
                    help="required for structure (cofold/complex) and DNA (dna/promoter) inputs; "
                         "inferred for .sdf (pose) / .smi (mol)")
    qp.add_argument("--json", action="store_true", help="emit the stable JSON verdict schema")
    qp.add_argument("--ligand", help="cofold: a ligand SDF (bond orders) → adds the intramolecular DRC")
    qp.add_argument("--ligand-resname", help="cofold: explicit ligand residue id")
    qp.add_argument("--chain-a", help="complex: chain id(s) for partner A (comma-separated)")
    qp.add_argument("--chain-b", help="complex: chain id(s) for partner B (comma-separated)")
    qp.set_defaults(fn=_cmd_qualify)

    rp = sub.add_parser("repair", help="run the agent self-repair loop (generate → qualify → fix → converge)")
    rp.add_argument("artifact", nargs="?",
                    help="a draft to repair (inline DNA/SMILES or a file); omit to let the reference agent "
                         "propose a flawed draft (a self-contained demo)")
    rp.add_argument("-m", "--modality", choices=["dna", "mol"], required=True,
                    help="which reference agent to drive (dna = surgical edits; mol = variant search)")
    rp.add_argument("--rounds", type=int, default=8, help="max repair rounds (default 8)")
    rp.add_argument("--length", type=int, default=240, help="dna: target insert length when proposing")
    rp.add_argument("--clear", action="append", metavar="CONTRACT",
                    help="also clear a DISCLOSE-only contract (e.g. --clear RESTRICTION_SITE); repeatable")
    rp.add_argument("--json", action="store_true", help="emit the repair trajectory as JSON")
    rp.set_defaults(fn=_cmd_repair)

    apr = sub.add_parser("audit", help="audit a dataset (benchmark leakage / CRISPR-screen under-power)")
    apr.add_argument("kind", choices=["leakage", "screen"])
    apr.add_argument("--benchmark", default="uspto50k",
                     help="leakage: uspto50k (retrosynthesis), bbbp / esol (MoleculeNet), ppi (Guo-yeast PPI)")
    apr.add_argument("--single-cell", action="store_true",
                     help="screen: single-cell Perturb-seq silent-failure QC (knockdown/cell) instead of "
                          "the bulk dropout screen")
    apr.add_argument("--input", metavar="FILE",
                     help="screen --single-cell: qualify YOUR screen summary (CSV/TSV, one row per "
                          "perturbation) instead of the bundled reference — core install, no h5py")
    apr.add_argument("--seeds", type=int, help="override the audit's seed count")
    apr.add_argument("--json", action="store_true", help="emit the report as JSON")
    apr.set_defaults(fn=_cmd_audit)

    lp = sub.add_parser("list", help="list the QC modalities and audits")
    lp.set_defaults(fn=_cmd_list)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
