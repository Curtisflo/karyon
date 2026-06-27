"""structure_io — a tiny stdlib reader for macromolecular heavy-atom coordinates (cofold-QC support).

The co-folding intermolecular DRC (`cofold_validity.py`) needs the PROTEIN's heavy-atom coordinates +
elements in the same frame as the ligand. The existing pose code only ever parsed the *ligand* SDF (via
RDKit) and consumed the protein verdict from a reference; there is no protein reader in the repo. This
module is that reader — deliberately minimal and **stdlib-only** (no gemmi/biopython/biotite), because all
we need is `(element, x, y, z)` per heavy atom, not a full structure model.

Two formats:
  * PDB / PDBQT — the cached crystal complexes (`{target}_protein.pdb`) and any PDB co-folding output.
  * mmCIF `_atom_site` loop — the format Boltz / AF3 / Chai co-folding tools emit (added for Tier 2).

A co-folding output bundles protein + ligand in ONE file; `split_protein_ligand()` separates them by the
standard convention (polymer ATOM records with standard residues = protein; HETATM / non-standard residue
= ligand), so the intermolecular check runs with no coordinate-frame plumbing.

stdlib-only; no sibling-package imports. RDKit (for vdW radii / ligand parsing) lives in the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

# The 20 standard amino acids (+ common variants) — anything else among the polymer is treated as ligand.
_STANDARD_AA = frozenset((
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL", "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN",
))
# Solvent / ion residues that are neither the protein of interest nor the docked ligand.
_SOLVENT = frozenset(("HOH", "WAT", "DOD", "TIP", "SOL"))


@dataclass(frozen=True)
class Atom:
    """One heavy atom: element symbol + Cartesian coordinates (Å). `is_hetero`/`resname` retained so a
    co-folding structure can be split into protein vs ligand without a second pass. `chain`/`atom_name`/
    `resnum` (added for the protein-complex DRC) let a multi-chain complex be split into interacting chain
    groups and let a heavy↔heavy clash be matched to a wwPDB validation `<clash>` record by chain+atom.
    They default empty so every existing caller (and `cofold_validity.translate`'s positional construction)
    is untouched; use `dataclasses.replace` to move an atom and keep its chain."""

    element: str
    x: float
    y: float
    z: float
    resname: str = ""
    is_hetero: bool = False
    chain: str = ""
    atom_name: str = ""
    resnum: int = 0


def _int_or(s: str, default: int = 0) -> int:
    """Parse a residue-number field (tolerant of blanks / insertion codes / hybrid-36 — default on junk)."""
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def _element_from_pdb(line: str) -> str:
    """Element symbol from a PDB/PDBQT ATOM line: cols 77-78 when present, else inferred from the atom
    name (cols 13-16). Returns '' for hydrogen/deuterium (heavy-atom-only convention) and unknowns."""
    elem = line[76:78].strip() if len(line) >= 78 else ""
    if not elem:
        # infer from the atom-name field: strip leading digits/spaces, take the leading letters
        name = line[12:16].strip()
        elem = "".join(c for c in name if c.isalpha())[:2]
        # a 2-letter all-caps token like "CA" is usually C-alpha (element C), not calcium; for protein
        # atom names the element is the first letter unless it's a known 2-letter metal — keep it simple:
        if len(elem) == 2 and elem[1].islower() is False and name[:1].isalpha():
            elem = elem[0]
    return elem.capitalize()


def read_pdb_atoms(text: str, *, heavy_only: bool = True) -> list[Atom]:
    """Parse ATOM/HETATM records from PDB or PDBQT text → heavy `Atom`s. Tolerant of PDBQT (extra trailing
    charge/type columns don't shift the fixed coordinate columns 31-54)."""
    atoms: list[Atom] = []
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        elem = _element_from_pdb(line)
        if heavy_only and elem in ("H", "D", ""):
            continue
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        atoms.append(Atom(elem, x, y, z, resname=line[17:20].strip(), is_hetero=(rec == "HETATM"),
                          chain=line[21:22].strip(), atom_name=line[12:16].strip(),
                          resnum=_int_or(line[22:26])))
    return atoms


def _split_cif_loop_header(lines: list[str], start: int) -> tuple[list[str], int]:
    """From a `loop_` at index `start`, collect the `_atom_site.<tag>` column tags and return (tags, first
    data-row index)."""
    tags: list[str] = []
    i = start + 1
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("_atom_site."):
            tags.append(s.split(".", 1)[1])
            i += 1
        elif s.startswith("_"):           # a different loop's columns — not atom_site
            return [], i
        else:
            break
    return tags, i


def read_cif_atoms(text: str, *, heavy_only: bool = True) -> list[Atom]:
    """Parse the `_atom_site` loop of an mmCIF file → heavy `Atom`s. Minimal: handles the single-line
    whitespace-delimited row form that Boltz/AF3/Chai emit (no multi-line quoted values in atom_site)."""
    lines = text.splitlines()
    atoms: list[Atom] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            tags, data_start = _split_cif_loop_header(lines, i)
            if tags and any(t.startswith("Cartn_") for t in tags):
                idx = {t: k for k, t in enumerate(tags)}
                col_e = idx.get("type_symbol")
                col_x, col_y, col_z = idx.get("Cartn_x"), idx.get("Cartn_y"), idx.get("Cartn_z")
                col_grp = idx.get("group_PDB")
                col_res = idx.get("label_comp_id") if "label_comp_id" in idx else idx.get("auth_comp_id")
                col_chain = idx.get("auth_asym_id") if "auth_asym_id" in idx else idx.get("label_asym_id")
                col_atom = idx.get("label_atom_id") if "label_atom_id" in idx else idx.get("auth_atom_id")
                col_seq = idx.get("auth_seq_id") if "auth_seq_id" in idx else idx.get("label_seq_id")
                j = data_start
                while j < len(lines):
                    s = lines[j].strip()
                    if not s or s.startswith(("#", "loop_", "_", "data_")):
                        break
                    f = s.split()
                    if len(f) < len(tags):
                        j += 1
                        continue
                    elem = (f[col_e] if col_e is not None else "").capitalize()
                    if heavy_only and elem in ("H", "D", ""):
                        j += 1
                        continue
                    try:
                        x = float(f[col_x]); y = float(f[col_y]); z = float(f[col_z])
                    except (ValueError, TypeError):
                        j += 1
                        continue
                    resname = f[col_res].strip().upper() if col_res is not None else ""
                    is_het = (f[col_grp] == "HETATM") if col_grp is not None else False
                    chain = f[col_chain].strip() if col_chain is not None else ""
                    aname = f[col_atom].strip().strip('"') if col_atom is not None else ""
                    resnum = _int_or(f[col_seq]) if col_seq is not None else 0
                    atoms.append(Atom(elem, x, y, z, resname=resname, is_hetero=is_het,
                                      chain=chain, atom_name=aname, resnum=resnum))
                    j += 1
                i = j
                continue
        i += 1
    return atoms


def read_atoms(text: str, *, fmt: str, heavy_only: bool = True) -> list[Atom]:
    """Dispatch on `fmt` ('pdb' | 'pdbqt' | 'cif' | 'mmcif')."""
    f = fmt.lower()
    if f in ("pdb", "pdbqt", "ent"):
        return read_pdb_atoms(text, heavy_only=heavy_only)
    if f in ("cif", "mmcif"):
        return read_cif_atoms(text, heavy_only=heavy_only)
    raise ValueError(f"unknown structure format {fmt!r}; have pdb/pdbqt/cif")


def split_protein_ligand(atoms: list[Atom], *, ligand_resname: str | None = None
                         ) -> tuple[list[Atom], list[Atom]]:
    """Separate a one-frame complex into (protein_atoms, ligand_atoms).

    Convention: standard-amino-acid records = protein; the docked small molecule = HETATM / non-standard
    residue that is not solvent/ion. When `ligand_resname` is given, the ligand is exactly that residue
    (the robust path for co-folding output, where the ligand's residue id is known from the target)."""
    protein, ligand = [], []
    for a in atoms:
        rn = a.resname.upper()
        if ligand_resname is not None:
            (ligand if rn == ligand_resname.upper() else protein).append(a)
            continue
        if rn in _STANDARD_AA and not a.is_hetero:
            protein.append(a)
        elif rn in _SOLVENT:
            continue
        else:
            ligand.append(a)
    return protein, ligand


# --------------------------------------------------------------------------- #
# Chain-aware helpers — for the protein-complex interface DRC (protein↔protein).
# A multi-chain complex (predicted binder, AF-Multimer, a deposited heterodimer) is split into two
# interacting groups of chains; the interface is the geometry BETWEEN them. Solvent/ion HETATM are dropped
# so the interface is protein↔protein. `chain` rides on each `Atom` (read_pdb/read_cif populate it).
# --------------------------------------------------------------------------- #
def _polymer_only(atoms: list[Atom]) -> list[Atom]:
    """Drop solvent/ion records — keep only what forms a protein chain (standard AA, or any non-solvent
    polymer ATOM). Hetero ligands/cofactors are excluded so a chain-vs-chain interface stays protein↔protein."""
    return [a for a in atoms if a.resname.upper() not in _SOLVENT and not a.is_hetero]


def chain_ids(atoms: list[Atom], *, polymer_only: bool = True) -> list[str]:
    """The distinct chain identifiers present, in first-seen order (the deposited/author chain order)."""
    src = _polymer_only(atoms) if polymer_only else atoms
    seen: list[str] = []
    for a in src:
        if a.chain not in seen:
            seen.append(a.chain)
    return seen


def group_by_chain(atoms: list[Atom], *, polymer_only: bool = True) -> dict[str, list[Atom]]:
    """Partition atoms by chain id (polymer only by default)."""
    out: dict[str, list[Atom]] = {}
    for a in (_polymer_only(atoms) if polymer_only else atoms):
        out.setdefault(a.chain, []).append(a)
    return out


def split_by_chain(atoms: list[Atom], chains_a, chains_b, *, polymer_only: bool = True
                   ) -> tuple[list[Atom], list[Atom]]:
    """Separate a complex into (group_A_atoms, group_B_atoms) by chain id. `chains_a`/`chains_b` are
    iterables of chain ids (a single str is accepted). The robust path when receptor vs binder chains are
    known (e.g. a designed binder is chain B against target chain A)."""
    a_ids = {chains_a} if isinstance(chains_a, str) else set(chains_a)
    b_ids = {chains_b} if isinstance(chains_b, str) else set(chains_b)
    a, b = [], []
    for atom in (_polymer_only(atoms) if polymer_only else atoms):
        if atom.chain in a_ids:
            a.append(atom)
        elif atom.chain in b_ids:
            b.append(atom)
    return a, b


def select_heavy(atoms: list[Atom]) -> list[Atom]:
    """Heavy atoms only (drop H/D and unknowns) — the readers already do this by default, but a complex
    parsed with `heavy_only=False` (or assembled from another source) can be normalized here."""
    return [a for a in atoms if a.element not in ("H", "D", "")]
