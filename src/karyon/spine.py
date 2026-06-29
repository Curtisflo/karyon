"""spine — the unified per-artifact QC surface: one `qualify(artifact, modality) -> QualifyResult` over
every karyon gate, emitting one stable JSON schema. (Named `spine` so the headline function
`karyon.qualify` does not collide with its own module.)

Each gate module owns its DRC and a uniform `validate(artifact) -> contracts.Verdict` (see contracts.py).
This module is the *spine*: a registry that maps a modality to (how to LOAD the artifact, how to CHECK it,
and any set-level BATCH check), so a caller — or the `karyon qualify` CLI, or an agent skill — hits ONE
surface regardless of whether the artifact is a docking pose, a co-folding structure, a protein complex, a
SMILES, or a DNA sequence. "The model proposes, karyon qualifies."

Two design rules:
  * **Lazy.** Only `contracts` + stdlib are imported at module load; every gate module (numpy/rdkit) is
    imported inside its adapter, so `import karyon` (and `from karyon import qualify`) stays light.
  * **One pass semantic.** A per-artifact item PASSES iff its verdict `score == 0.0` — i.e. no *condemning*
    contract fired. Weight-0 reasons are disclosures: they inform (and ride in `reasons`) but do not fail
    the gate. This matches every karyon gate's own `is_invalid` / `score > 0` convention.

The loaders here consolidate the file parsing that used to be copy-pasted across the per-skill scripts
(SDF supplier, FASTA, .smi, PDB/mmCIF + chain/ligand splitting).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import contracts


class QualifyError(ValueError):
    """A user-facing problem with the request (unknown/ambiguous modality, unreadable input, missing dep).
    Carries an actionable message; the CLI prints it and exits non-zero."""


# --------------------------------------------------------------------------- #
# The registry entry.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Gate:
    """One modality's spine wiring. `load` turns a source (path / dir / glob / inline string) into named
    artifacts; `check` qualifies one; `batch` (optional) runs a set-level check over all of them."""

    modality: str
    extensions: tuple[str, ...]                       # file types this gate can load (docs + load filter)
    extras: tuple[str, ...]                           # optional deps required (actionable error if absent)
    load: Callable[[Any, dict], list[tuple[str, Any]]]
    check: Callable[[Any], contracts.Verdict]
    batch: Callable[[list[tuple[str, Any]]], "contracts.Verdict | None"] | None = None


# --------------------------------------------------------------------------- #
# The uniform result envelope (1..N artifacts + an optional set-level verdict) and its stable JSON schema.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QualifyResult:
    """The verdict(s) for one `qualify` call. `items` is (name, Verdict) per artifact; `batch` is the
    set-level verdict (e.g. DNA cross-hybridization) or None."""

    modality: str
    items: tuple[tuple[str, contracts.Verdict], ...]
    batch: contracts.Verdict | None = None

    @property
    def ok(self) -> bool:
        """PASS iff no condemning contract fired anywhere (every item and the batch passed the gate)."""
        items_ok = all(v.ok for _, v in self.items)
        return items_ok and (self.batch is None or self.batch.ok)

    def to_dict(self) -> dict:
        """The stable, JSON-safe wire schema the CLI emits and consumers depend on. Each `ok` is
        passed-the-gate (`Verdict.ok`) — the same notion as a directly-serialized `Verdict.to_dict()`."""
        return {
            "modality": self.modality,
            "ok": self.ok,
            "items": [
                {"name": name, "ok": v.ok, "score": v.score,
                 "reasons": [r.to_dict() for r in v.reasons]}
                for name, v in self.items
            ],
            "batch": None if self.batch is None else {
                "ok": self.batch.ok, "score": self.batch.score,
                "reasons": [r.to_dict() for r in self.batch.reasons],
            },
        }


# --------------------------------------------------------------------------- #
# Shared loader helpers (consolidated from the per-skill scripts).
# --------------------------------------------------------------------------- #
def _fmt_of(path) -> str:
    ext = Path(str(path)).suffix.lower().lstrip(".")
    return {"cif": "cif", "mmcif": "cif", "pdb": "pdb", "pdbqt": "pdbqt", "ent": "pdb"}.get(ext, "pdb")


def _csv(arg) -> list[str] | None:
    return [c.strip() for c in str(arg).split(",") if c.strip()] if arg else None


def _is_existing_file(source) -> bool:
    return isinstance(source, (str, os.PathLike)) and Path(source).is_file()


def _collect(source, exts: tuple[str, ...]) -> list[str]:
    """Expand a directory / glob / file into a sorted list of paths with one of `exts`."""
    s = str(source)
    if os.path.isdir(s):
        out: list[str] = []
        for e in exts:
            out += glob.glob(os.path.join(s, f"*{e}"))
        return sorted(out)
    return sorted(glob.glob(s)) or ([s] if os.path.exists(s) else [])


def _parse_fasta(text: str) -> list[tuple[str, str]]:
    """(name, sequence) records from FASTA text (stdlib; no Biopython)."""
    out: list[tuple[str, str]] = []
    name, chunks = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(chunks)))
            header = line[1:].strip()
            name = header.split()[0] if header else f"seq{len(out)}"
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if name is not None:
        out.append((name, "".join(chunks)))
    return out


def _read_smi(text: str) -> list[tuple[str, str]]:
    """(name, smiles) from a .smi file: 'SMILES [name]' per line; blank / '#' lines skipped."""
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        out.append((parts[1] if len(parts) > 1 else f"mol{len(out)}", parts[0]))
    return out


def _dna_records(source) -> list[tuple[str, str]]:
    """Normalized (name, ACGT-sequence) records from a FASTA file or an inline sequence. Shared by the
    `dna` and `promoter` gates (both consume ACGT). Raises on empty / non-ACGT input."""
    if _is_existing_file(source):
        records = _parse_fasta(Path(source).read_text())
    else:
        records = [("input", str(source))]
    out: list[tuple[str, str]] = []
    for name, seq in records:
        s = seq.upper().replace(" ", "")
        bad = set(s) - set("ACGT")
        if not s or bad:
            raise QualifyError(f"{name}: sequence empty or has non-ACGT characters {sorted(bad)}")
        out.append((name, s))
    return out


# --------------------------------------------------------------------------- #
# Per-gate adapters (each lazy-imports its module).
# --------------------------------------------------------------------------- #
def _pose_load(source, opts) -> list[tuple[str, Any]]:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")                      # rdkit parse chatter is not our verdict
    files = _collect(source, (".sdf",))
    if not files:
        raise QualifyError(f"no .sdf poses found at {source!r}")
    out: list[tuple[str, Any]] = []
    for path in files:
        mols = list(Chem.SDMolSupplier(path, sanitize=True)) or [None]   # garbage file -> 1 unparseable
        for k, mol in enumerate(mols):
            out.append((path if len(mols) == 1 else f"{path}[{k}]", mol))
    return out


def _pose_check(mol) -> contracts.Verdict:
    from . import pose_validity as pv
    return pv.validate(mol, pv.Tol())


def _mol_load(source, opts) -> list[tuple[str, Any]]:
    if _is_existing_file(source):
        return _read_smi(Path(source).read_text())
    return [("input", str(source).strip())]


def _mol_check(smiles) -> contracts.Verdict:
    from . import mol_qc as mq
    return mq.validate(smiles)


def _dna_load(source, opts) -> list[tuple[str, Any]]:
    return _dna_records(source)


def _dna_check(seq) -> contracts.Verdict:
    from . import gen_dna_validity as gv
    return gv.validate(seq)


def _dna_batch(items) -> contracts.Verdict | None:
    if len(items) <= 1:
        return None
    from . import gen_dna_validity as gv
    tol = gv.GenDNATol()
    return gv.set_contracts().evaluate(gv.featurize_set(list(items), tol), tol)


def _promoter_check(seq) -> contracts.Verdict:
    from . import promoter_contracts as pc
    return pc.validate(seq)


def _cofold_load(source, opts) -> list[tuple[str, Any]]:
    if not _is_existing_file(source):
        raise QualifyError(f"cofold needs a structure file (PDB/mmCIF); got {source!r}")
    from . import structure_io as sio
    atoms = sio.read_atoms(Path(source).read_text(), fmt=_fmt_of(source))
    protein, lig_atoms = sio.split_protein_ligand(atoms, ligand_resname=opts.get("ligand_resname"))
    ligand: Any = lig_atoms
    if opts.get("ligand"):                              # optional SDF → enables the intramolecular DRC
        from .cofold_data import ligand_mol
        ligand = ligand_mol(Path(opts["ligand"]).read_text())
    if not protein or (isinstance(ligand, list) and not ligand):
        raise QualifyError(f"could not resolve a protein+ligand pair in {source!r} "
                           f"({len(protein)} protein / {len(lig_atoms)} ligand atoms)")
    return [(Path(source).name, (protein, ligand))]


def _cofold_check(parsed) -> contracts.Verdict:
    from . import cofold_validity as cv
    protein, ligand = parsed
    return cv.validate(protein, ligand)


def _complex_load(source, opts) -> list[tuple[str, Any]]:
    if not _is_existing_file(source):
        raise QualifyError(f"complex needs a structure file (PDB/mmCIF); got {source!r}")
    from . import structure_io as sio
    atoms = sio.read_atoms(Path(source).read_text(), fmt=_fmt_of(source))
    ca, cb = _csv(opts.get("chain_a")), _csv(opts.get("chain_b"))
    if ca and cb:
        ga, gb = sio.split_by_chain(atoms, ca, cb)
        a_label, b_label = ",".join(ca), ",".join(cb)
    else:
        groups = sio.group_by_chain(atoms)
        if len(groups) < 2:
            raise QualifyError(f"need ≥2 chains for an interface; found {sorted(groups)}")
        ranked = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
        (a_label, ga), (b_label, gb) = ranked[0], ranked[1]
    if not ga or not gb:
        raise QualifyError(f"could not resolve two chain groups "
                           f"(A={a_label!r}:{len(ga)} / B={b_label!r}:{len(gb)})")
    return [(f"{Path(source).name}:{a_label}|{b_label}", (ga, gb))]


def _complex_check(parsed) -> contracts.Verdict:
    from . import protein_interface_validity as piv
    ga, gb = parsed
    return piv.validate(ga, gb)


# --------------------------------------------------------------------------- #
# The registry.
# --------------------------------------------------------------------------- #
_STRUCT_EXTS = (".pdb", ".cif", ".mmcif", ".ent", ".pdbqt")
_DNA_EXTS = (".fasta", ".fa", ".fna")

GATES: dict[str, Gate] = {
    "pose": Gate("pose", (".sdf",), ("rdkit",), _pose_load, _pose_check),
    "mol": Gate("mol", (".smi",), ("rdkit",), _mol_load, _mol_check),
    "dna": Gate("dna", _DNA_EXTS, (), _dna_load, _dna_check, batch=_dna_batch),
    "promoter": Gate("promoter", _DNA_EXTS, (), _dna_load, _promoter_check),
    "cofold": Gate("cofold", _STRUCT_EXTS, ("numpy",), _cofold_load, _cofold_check),
    "complex": Gate("complex", _STRUCT_EXTS, ("numpy",), _complex_load, _complex_check),
}

# Inference: ONLY extensions that map to exactly one modality auto-resolve. Structure files
# (pose↔cofold↔complex) and DNA files (dna↔promoter) are intentionally ambiguous → `modality` required.
_INFER = {".sdf": "pose", ".smi": "mol"}
_AMBIGUOUS = {e: ("cofold", "complex") for e in _STRUCT_EXTS}
_AMBIGUOUS.update({e: ("dna", "promoter") for e in _DNA_EXTS})

_EXTRA_PIP = {
    "rdkit": 'pip install "karyon[chem]"',
    "rdchiral": 'pip install "karyon[chem]"',
    "numpy": "pip install karyon  (numpy is a core dependency)",
}


def modalities() -> list[str]:
    """The registered modalities, in registry order."""
    return list(GATES)


def _resolve_gate(artifact, modality: str | None) -> Gate:
    if modality is not None:
        try:
            return GATES[modality]
        except KeyError:
            raise QualifyError(f"unknown modality {modality!r}; choose one of {modalities()}")
    # infer from a file extension, or fail with a clear, actionable message.
    if not (isinstance(artifact, (str, os.PathLike)) and (Path(artifact).suffix or os.path.exists(artifact))):
        raise QualifyError(f"inline input needs an explicit modality (one of {modalities()})")
    ext = Path(str(artifact)).suffix.lower()
    if ext in _INFER:
        return GATES[_INFER[ext]]
    if ext in _AMBIGUOUS:
        raise QualifyError(f"{ext!r} is ambiguous — pass modality= one of {list(_AMBIGUOUS[ext])}")
    raise QualifyError(f"cannot infer modality from {ext!r}; pass modality= (one of {modalities()})")


def _require_extras(gate: Gate) -> None:
    import importlib
    for mod in gate.extras:
        try:
            importlib.import_module(mod)
        except Exception:                              # noqa: BLE001
            raise QualifyError(f"modality {gate.modality!r} needs {mod!r} — "
                               f"{_EXTRA_PIP.get(mod, f'pip install {mod}')}")


def qualify(artifact, modality: str | None = None, **opts) -> QualifyResult:
    """Qualify an artifact through the appropriate karyon gate.

    `artifact` is a file path, a directory or glob of files, or (for mol/dna/promoter) an inline string.
    `modality` is one of `modalities()`; it is REQUIRED for structure files (cofold↔complex) and DNA
    files/strings (dna↔promoter), and inferred for unambiguous extensions (`.sdf`→pose, `.smi`→mol).
    Gate-specific options ride in `opts` (cofold: `ligand`, `ligand_resname`; complex: `chain_a`,
    `chain_b`). Returns a `QualifyResult` whose `.to_dict()` is the stable JSON schema.
    """
    gate = _resolve_gate(artifact, modality)
    _require_extras(gate)
    items = gate.load(artifact, opts)
    if not items:
        raise QualifyError(f"no {gate.modality} artifacts found in {artifact!r}")
    verdicts = tuple((name, gate.check(parsed)) for name, parsed in items)
    batch = gate.batch(items) if gate.batch else None
    return QualifyResult(gate.modality, verdicts, batch)
