"""cofold_data — loaders for the co-folding intermolecular DRC (cofold-QC).

Two substrates, both protein+ligand **in one coordinate frame** (the property that makes the intermolecular
axis ownable — see `cofold_validity.py`):

  * **Tier 1 — crystal complexes (CACHED, no download).** `pb_paper_data.zip` (the PoseBusters/Astex
    reference sets) — `{target}_protein.pdb` + `{target}_ligand.sdf` per target, the native (valid-by-
    construction) complex in the crystal frame. The clean-pass + instrument-decoy arm (PI-1). Already in
    `~/.cache/karyon`. NB: the cached DiffDock/Vina dirs hold ligand poses only (no receptor), so faithfulness
    cannot be measured on them — Tier 2 is needed for that.

  * **Tier 2 — co-folding output, one tarball per method (Zenodo 19138652).** Each method predicts
    protein+ligand jointly in one frame (no alignment), bundled with a `bust_results.csv` (the real
    PoseBusters package over those poses). The faithfulness headline (PI-2/PI-4): our owned intermolecular
    verdict vs PoseBusters' reference inter columns, per pose. Registered methods (`_METHODS`): **boltz**
    (~400 MB, the BioNeMo tool), **rfaa** (RoseTTAFold-All-Atom, ~109 MB), **af3** (AlphaFold-3, ~850 MB),
    **neuralplexer** (~1.6 GB). Internal layouts differ — flat `{TARGET}_model_0_protein.pdb` (Boltz) vs
    per-target subdirs `{TARGET}/{TARGET}_protein.pdb` (RFAA), PDB or mmCIF protein — so the loader
    DISCOVERS the pairing (`_load_method_from`) rather than hard-coding it. Offline-skip until fetched.

Cache/offline-skip plumbing + the `bust_results.csv` inter/intra column groups are reused from `pose_data`.

    python -m karyon.cofold_data                 # smoke: summarize cached crystal complexes
    python -m karyon.cofold_data --fetch rfaa    # fetch + summarize a co-folding set
"""

from __future__ import annotations

import csv
import re
import shutil
import socket
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import structure_io as sio
from .paths import network_allowed
from .pose_data import INTER_COLS, INTRA_COLS, PoseUnavailable, _as_bool, _cache_dir

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False

_UA = "karyon/0.1 (+https://github.com/Curtisflo/karyon)"
_TIMEOUT_S = 600
_ZENODO = "https://zenodo.org/records/19138652/files"
_CRYSTAL_ZIP = "pb_paper_data.zip"


# --------------------------------------------------------------------------- #
# Co-folding method registry. Each method's tarball (Zenodo 19138652) holds a
# `forks/<tool>/inference/<tool>_posebusters_benchmark_outputs_1/` raw run with a `bust_results.csv`
# (the real PoseBusters package over those poses). Internal layouts differ per method — flat
# `{TARGET}_model_0_protein.pdb` (Boltz) vs per-target subdirs `{TARGET}/{TARGET}_protein.pdb` (RFAA) —
# so the loader DISCOVERS the protein/ligand pairing rather than hard-coding it (see `_load_method_from`).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CofoldMethod:
    key: str             # cli/source token
    tarball: str         # Zenodo file name
    extract_dir: str     # cache subdir under ~/.cache/karyon
    label: str           # human label
    size_mb: int         # approx tarball size, for the fetch log
    pairing: str = "suffix"   # how protein↔ligand files pair: "suffix" or "ranked" (see _load_method_from)


_METHODS = {
    # "suffix": protein `*_protein.{pdb,cif}` + sibling `{stem}_ligand.sdf` (Boltz flat, RFAA/AF3 subdirs).
    "boltz": CofoldMethod("boltz", "boltz_benchmark_method_predictions.tar.gz",
                          "cofold_boltz", "Boltz-2", 400),
    "af3": CofoldMethod("af3", "af3_benchmark_method_predictions.tar.gz",
                        "cofold_af3", "AlphaFold-3", 850),
    "rfaa": CofoldMethod("rfaa", "rfaa_benchmark_method_predictions.tar.gz",
                         "cofold_rfaa", "RoseTTAFold-All-Atom", 109),
    # "ranked": NeuralPLexer names by rank+plddt — `{TARGET}/prot_rank1_plddt*.pdb` + `lig_rank1_plddt*.sdf`.
    "neuralplexer": CofoldMethod("neuralplexer", "neuralplexer_benchmark_method_predictions.tar.gz",
                                 "cofold_neuralplexer", "NeuralPLexer", 1571, pairing="ranked"),
}


def cofold_methods() -> list[str]:
    """The co-folding methods the gate can score (registry keys)."""
    return list(_METHODS)


# --------------------------------------------------------------------------- #
# Records.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Complex:
    """A protein + ligand in one frame. `protein` = parsed heavy atoms; `ligand_sdf` = raw SDF (parse with
    `ligand_mol`). For Tier-2 poses the reference PoseBusters verdict rides along; for crystal it's None."""

    target: str
    source: str                       # "crystal" | a co-folding method key
    protein: list                     # list[structure_io.Atom]
    ligand_sdf: str
    rmsd_le_2: bool | None = None
    ref_intra_valid: bool | None = None
    ref_inter_valid: bool | None = None
    ref_checks: dict = field(default_factory=dict)
    # PoseBusters' own numeric ligand↔protein closest distance — lets the harness check the reference
    # actually describes the raw structure we loaded (vs a relaxed copy; see cofold_honesty PI-4).
    ref_min_dist_A: float | None = None


def ligand_mol(block: str, *, primary: bool = True):
    """Parse an SDF/mol block to a sanitized RDKit Mol (or None). The C0 sanitize gate of the DRC.

    `primary=True` returns the **largest organic fragment** — the binder. A co-folding model's combined
    ligand record bundles the primary ligand *plus* co-folded ions / cofactors (PoseBusters classifies
    those as `*_cofactors` / `*_waters`, NOT as the ligand, and excludes them from `minimum_distance_to_
    protein`). Keeping the whole blob inflated the ligand↔protein clash rate ~10× (false ion-in-its-own-site
    contacts); the largest fragment reproduces PoseBusters' ligand-protein distance to within 0.1 Å on
    239/262 Boltz poses (median Δ 0). Verified — the faithfulness fix, not fit-to-accuracy."""
    if not _HAVE_RDKIT:
        return None
    m = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    if m is None:
        return None
    if primary:
        frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
        if frags:
            m = max(frags, key=lambda f: f.GetNumHeavyAtoms())   # the binder; drops co-folded ions/cofactors
    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Tier 1 — cached crystal complexes (no download).
# --------------------------------------------------------------------------- #
def load_crystal_complexes(*, dataset: str = "posebusters_benchmark_set",
                           limit: int | None = None) -> list[Complex]:
    """Native crystal complexes from the cached `pb_paper_data.zip` — protein PDB + native ligand SDF, same
    frame, valid by construction. Offline-skip via `PoseUnavailable` if the zip is absent."""
    zpath = _cache_dir() / _CRYSTAL_ZIP
    if not zpath.exists():
        raise PoseUnavailable(f"{_CRYSTAL_ZIP} not cached at {zpath} (crystal complexes unavailable)")

    out: list[Complex] = []
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        prots = sorted(n for n in names if n.endswith("_protein.pdb") and n.startswith(dataset + "/"))
        for pn in prots:
            base = pn[:-len("_protein.pdb")]
            ln = base + "_ligand.sdf"
            if ln not in names:
                continue
            target = base.rsplit("/", 1)[-1]
            atoms = sio.read_pdb_atoms(z.read(pn).decode(errors="replace"))
            protein, _ = sio.split_protein_ligand(atoms)
            if not protein:
                continue
            out.append(Complex(target=target, source="crystal", protein=protein,
                               ligand_sdf=z.read(ln).decode(errors="replace")))
            if limit and len(out) >= limit:
                break
    if not out:
        raise PoseUnavailable(f"{_CRYSTAL_ZIP} present but no '{dataset}' protein+ligand pairs found")
    return out


# --------------------------------------------------------------------------- #
# Tier 2 — co-folding output (one tarball per method). Layout is DISCOVERED on extract.
# --------------------------------------------------------------------------- #
def _method_cfg(method: str) -> CofoldMethod:
    try:
        return _METHODS[method]
    except KeyError:
        raise ValueError(f"unknown co-folding method {method!r}; known: {list(_METHODS)}") from None


def _fetch_tarball(cfg: CofoldMethod) -> Path:
    """Stream the method tarball to disk (chunked — a `r.read()` of a 400 MB+ body buffers to memory and
    stalls). For the large sets, fetch out-of-band with `curl -C -L` (resumable) — this is the fallback."""
    tar_path = _cache_dir() / cfg.tarball
    if tar_path.exists():
        return tar_path
    url = f"{_ZENODO}/{cfg.tarball}?download=1"
    if not network_allowed():
        raise PoseUnavailable("network disabled via KARYON_NO_NETWORK")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    tmp = tar_path.with_suffix(tar_path.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r, tmp.open("wb") as fh:
            shutil.copyfileobj(r, fh, length=1 << 20)   # 1 MB chunks, never the whole body in RAM
        tmp.replace(tar_path)
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise PoseUnavailable(f"cannot fetch {cfg.tarball} (~{cfg.size_mb} MB): {e}") from e
    return tar_path


def _ensure_extracted(cfg: CofoldMethod) -> Path:
    out = _cache_dir() / cfg.extract_dir
    if out.exists() and any(out.rglob("bust_results.csv")):
        return out
    tar_path = _fetch_tarball(cfg)
    try:
        with tarfile.open(tar_path) as tf:
            tf.extractall(out)                          # noqa: S202 — trusted Zenodo archive, our cache dir
    except (tarfile.TarError, OSError) as e:
        raise PoseUnavailable(f"cannot extract {tar_path.name}: {e}") from e
    if not any(out.rglob("bust_results.csv")):
        raise PoseUnavailable(f"{cfg.tarball} extracted but no bust_results.csv (layout drift?)")
    return out


def load_cofold_poses(method: str = "boltz", *, limit: int | None = None) -> list[Complex]:
    """Co-folding poses for `method` joined to the reference `bust_results.csv`. Each pose is protein +
    ligand in ONE frame (the property that makes the intermolecular axis ownable). Offline-skip via
    `PoseUnavailable` if the method's tarball is not cached."""
    cfg = _method_cfg(method)
    root = _ensure_extracted(cfg)
    return _load_method_from(root, cfg, limit=limit)


def _axis_valid(row: dict, cols: list[str]) -> bool | None:
    """AND over the known reference columns for an axis; None if the row carries none of them."""
    vals = [_as_bool(row.get(c, "")) for c in cols]
    known = [v for v in vals if v is not None]
    return all(known) if known else None


def _as_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _find_raw_output_dir(root: Path) -> Path | None:
    """The RAW posebusters-benchmark output (replicate 1). `_relaxed` is force-field-minimized — it repairs
    the very violations the DRC is about (the raw-pose posture) — and `_ss_` is the single-sequence
    ablation; exclude both so protein / ligand / bust all come from one consistent unrelaxed run."""
    cands = [d for d in sorted(root.rglob("*posebusters_benchmark_outputs_1"))
             if d.is_dir() and "_relaxed" not in d.name and "_ss_" not in d.name]
    return cands[0] if cands else None


def _pair_prefix(name: str) -> str | None:
    """The shared stem of a protein file, used to find its sibling ligand (`{prefix}_ligand.sdf`)."""
    for suf in ("_protein.pdb", "_protein.cif", "_protein.ent"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return None


# (target, protein_file, ligand_file) tuples — one tuple per scored pose, the pairing differing by method.
def _pairs_suffix(base: Path):
    """Boltz/RFAA/AF3: protein `*_protein.{pdb,cif}` paired with its sibling `{stem}_ligand.sdf`. Works flat
    (Boltz: `{T}_model_0_protein.pdb`) or in per-target subdirs (RFAA: `{T}/{T}_protein.pdb`, AF3:
    `{T}/{T}_model_protein.pdb`) via rglob. Target = stem minus a trailing `_model` / `_model_N`."""
    prot_files = [p for p in sorted(base.rglob("*_protein.pdb")) + sorted(base.rglob("*_protein.cif"))
                  if "_aligned" not in p.name and "_tmp" not in p.name]
    for prot_f in prot_files:
        prefix = _pair_prefix(prot_f.name)
        if prefix is None:
            continue
        lig_f = prot_f.with_name(f"{prefix}_ligand.sdf")     # the raw co-folded ligand, NOT *_aligned/_LG/_LIG
        if not lig_f.exists():
            continue
        target = re.sub(r"_model(_\d+)?$", "", prefix)       # Boltz `_model_0`, AF3 `_model`, RFAA none
        yield target, prot_f, lig_f


def _pairs_ranked(base: Path):
    """NeuralPLexer: per-target subdir, names carry rank + plddt — `prot_rank1_plddt*.pdb` is the top pose
    PoseBusters scores, paired with `lig_rank1_plddt*.sdf` (skip `_aligned`, `lig_ref`, `lig_all`)."""
    for tdir in sorted(d for d in base.iterdir() if d.is_dir()):
        prots = sorted(p for p in tdir.glob("prot_rank1_plddt*.pdb") if "_aligned" not in p.name)
        ligs = sorted(p for p in tdir.glob("lig_rank1_plddt*.sdf") if "_aligned" not in p.name)
        if prots and ligs:
            yield tdir.name, prots[0], ligs[0]


def _load_method_from(root: Path, cfg: CofoldMethod, *, limit: int | None) -> list[Complex]:
    """Discover protein+ligand pairs in the raw output dir (each in ONE frame), join `bust_results.csv` on
    `mol_id`. The per-file pairing differs by method (`cfg.pairing`); everything downstream is shared."""
    base = _find_raw_output_dir(root)
    if base is None:
        raise PoseUnavailable(f"{cfg.label} extracted but no raw *_posebusters_benchmark_outputs_1/ at {root}")
    bust: dict[str, dict] = {}
    bust_csv = base / "bust_results.csv"
    if bust_csv.exists():
        with bust_csv.open(newline="") as fh:
            bust = {row.get("mol_id", ""): row for row in csv.DictReader(fh)}

    pairs = _pairs_ranked(base) if cfg.pairing == "ranked" else _pairs_suffix(base)

    out: list[Complex] = []
    for target, prot_f, lig_f in pairs:
        fmt = "cif" if prot_f.suffix == ".cif" else "pdb"
        protein, _ = sio.split_protein_ligand(sio.read_atoms(prot_f.read_text(errors="replace"), fmt=fmt))
        if not protein:
            continue
        row = bust.get(target) or {}
        out.append(Complex(
            target=target, source=cfg.key, protein=protein,
            ligand_sdf=lig_f.read_text(errors="replace"),
            rmsd_le_2=_as_bool(row.get("rmsd_≤_2å", "")) if row else None,
            ref_intra_valid=_axis_valid(row, INTRA_COLS) if row else None,
            ref_inter_valid=_axis_valid(row, INTER_COLS) if row else None,
            ref_checks={c: _as_bool(row.get(c, "")) for c in INTRA_COLS + INTER_COLS} if row else {},
            ref_min_dist_A=_as_float(row.get("smallest_distance_protein", "")) if row else None))
        if limit and len(out) >= limit:
            break
    if not out:
        raise PoseUnavailable(f"{cfg.label} extracted at {root} but no protein+ligand pairs found (layout drift?)")
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="cofold-QC data loaders (crystal complexes + co-folding methods).")
    ap.add_argument("--fetch", metavar="METHOD", choices=list(_METHODS),
                    help=f"fetch + summarize a co-folding set ({'/'.join(_METHODS)})")
    ap.add_argument("--limit", type=int, default=8)
    cli = ap.parse_args()

    if cli.fetch:
        cfg = _method_cfg(cli.fetch)
        try:
            poses = load_cofold_poses(cli.fetch, limit=cli.limit)
            print(f"{cfg.label}: {len(poses)} poses loaded (limit {cli.limit})")
            for p in poses[:cli.limit]:
                m = ligand_mol(p.ligand_sdf)
                nlig = m.GetNumHeavyAtoms() if m else "parse-fail"
                print(f"  {p.target:14} protein {len(p.protein):5} atoms | ligand {nlig} atoms "
                      f"| ref inter-valid {p.ref_inter_valid}")
        except PoseUnavailable as e:
            print("SKIP —", e)
        raise SystemExit(0)

    try:
        cx = load_crystal_complexes(limit=cli.limit)
    except PoseUnavailable as e:
        print("SKIP —", e)
        raise SystemExit(0)
    print(f"crystal complexes (cached, no download): {len(cx)} shown")
    for c in cx:
        m = ligand_mol(c.ligand_sdf)
        nlig = m.GetNumHeavyAtoms() if m else "parse-fail"
        print(f"  {c.target:14} protein {len(c.protein):5} heavy atoms | ligand {nlig} heavy atoms")
