"""ppi_data — loaders for the protein-complex interface DRC (complex-QC).

Two substrates, both two-or-more protein chains **in one coordinate frame** (the property that makes the
interface axis ownable — see `protein_interface_validity.py`):

  * **Natives — deposited PDB complexes + the wwPDB validation reference (no heavyweight install).** A curated
    list of real protein-protein complexes. For each, three login-free fetches: the coordinates (RCSB
    `files.rcsb.org`), the deposited **MolProbity clashscore** (PDBe `global-percentiles` JSON), and the
    wwPDB **validation report** XML (PDBe `entry-files`) whose `<clash>` records are the field's gold-standard
    steric-clash reference. The faithfulness arm compares the owned inter-chain verdict to that reference —
    the cofold-QC pattern (validate the reimplementation, don't strawman it), restored on native complexes.
    NB: the deposited *scalar* clashscore is whole-structure & all-atom-with-H, a DIFFERENT axis from the
    owned inter-chain heavy↔heavy gate (disclosed); the like-for-like reference is the `<clash>` records,
    restricted to inter-chain heavy↔heavy pairs.

  * **Predicted — multimer model outputs (the effect arm).** Predicted/designed complexes (e.g. CASP15
    multimer submissions) carry interface clashes the gate flags. There is no deposited validity reference for
    these (they are not in the PDB), so faithfulness is established on the natives and the *effect* (clash
    prevalence by method) is reported here. Layout discovered on arrival.

stdlib-only (urllib + json + xml.etree); cache-first under `~/.cache/karyon/ppi/`, offline-skip via `PoseUnavailable`.

    python -m karyon.ppi_data            # smoke: fetch+summarize a few native complexes
"""

from __future__ import annotations

import json
import socket
import tarfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from . import structure_io as sio
from .paths import network_allowed
from .pose_data import PoseUnavailable, _cache_dir

_UA = "karyon/0.1 (+https://github.com/Curtisflo/karyon)"
_TIMEOUT_S = 60
_RCSB_PDB = "https://files.rcsb.org/download/{id}.pdb"
_PDBE_CLASHSCORE = "https://www.ebi.ac.uk/pdbe/api/validation/global-percentiles/entry/{id}"
_PDBE_VALIDATION_XML = "https://www.ebi.ac.uk/pdbe/entry-files/download/{id}_validation.xml"
_CASP15_OLIGO = "https://predictioncenter.org/download_area/CASP15/predictions/oligo/{target}.tar.gz"

# CASP15 heteromeric (H-series) targets — multi-chain protein complexes; each tarball bundles every group's
# submitted models. The effect arm: predicted complexes carry interface clashes a deposited structure does not.
PREDICTED_TARGETS = ["H1106", "H1129", "H1134", "H1137", "H1140", "H1142", "H1143", "H1144", "H1166", "H1172"]


# --------------------------------------------------------------------------- #
# A curated set of deposited protein-protein complexes (login-free from the PDB). Protease-inhibitor,
# antibody-antigen, enzyme-regulator, signalling pairs — spanning a real deposited-clashscore gradient so the
# faithfulness arm has signal (a clean interface and a clashy one both occur). Two-partner complexes; a few
# carry >2 chains (antibody H+L), handled by the all-inter-chain audit.
# --------------------------------------------------------------------------- #
NATIVE_PDB_IDS = [
    "1brs", "1ay7", "1ppe", "2ptc", "1tgs", "2sni", "2sic", "1acb", "1cho", "1stf",
    "1tab", "1avx", "1d6r", "1eaw", "1ezu", "1fle", "1gl1", "1hia", "1mct", "1ppf",
    "1r0r", "2tgp", "4sgb", "1bvn", "1cgi", "1dfj", "1oph", "1udi", "1us7", "7cei",
    "1e6e", "1ewy", "1gpw", "1kxp", "1mah", "2hle", "2mta", "2pcc", "1wq1", "1he1",
    "1ahw", "1bvk", "1dqj", "1fbi", "1mlc", "1nca", "1nsn", "1qfw", "1vfb", "2vis",
    "3hfm", "1jps", "1iqd", "1kxq", "1bj1", "1fcc", "2jel", "1nmb", "1e6j", "1bgx",
]


@dataclass(frozen=True)
class NativeComplex:
    """A deposited protein complex + its wwPDB validation reference. `atoms` = all polymer heavy atoms (chain
    populated). `ref_interchain_clashes` = the count of inter-chain heavy↔heavy clash PAIRS in the wwPDB
    validation report (the like-for-like reference). `clashscore` = the deposited whole-structure MolProbity
    clashscore (a different axis — secondary, disclosed)."""

    pdb_id: str
    atoms: list                            # list[structure_io.Atom]
    chains: list                           # distinct chain ids present
    clashscore: float | None = None        # deposited MolProbity whole-structure clashscore
    ref_interchain_clashes: int = 0        # wwPDB inter-chain heavy↔heavy clash pairs (the reference)
    ref_clash_detail: tuple = ()           # a few (chainA,resA,atomA | chainB,resB,atomB) for legibility


# --------------------------------------------------------------------------- #
# Fetch + cache (~/.cache/karyon/ppi/, gitignored). Each artefact cached by name; offline-skips per entry.
# --------------------------------------------------------------------------- #
def _ppi_cache() -> Path:
    d = _cache_dir() / "ppi"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(url: str, cache_name: str) -> str | None:
    """Fetch a text artefact, caching to ~/.cache/karyon/ppi/<cache_name>. Returns None on any network/HTTP error
    (the entry is skipped, not fatal — the loader is offline-tolerant per entry). A cached empty file (a prior
    404) is treated as 'unavailable' without refetching."""
    path = _ppi_cache() / cache_name
    if path.exists():
        text = path.read_text(errors="replace")
        return text or None
    if not network_allowed():
        raise PoseUnavailable("network disabled via KARYON_NO_NETWORK")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
            text = r.read().decode(errors="replace")
        path.write_text(text)
        return text
    except urllib.error.HTTPError as e:
        if e.code == 404:
            path.write_text("")                 # cache the miss so we don't refetch a known-absent artefact
        return None
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
        return None


def _clashscore_of(pdb_id: str) -> float | None:
    """The deposited MolProbity whole-structure clashscore (PDBe global-percentiles). None if absent."""
    txt = _fetch(_PDBE_CLASHSCORE.format(id=pdb_id), f"{pdb_id}.clashscore.json")
    if not txt:
        return None
    try:
        return float(json.loads(txt)[pdb_id]["clashscore"]["rawvalue"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# wwPDB validation-report clash parsing — the like-for-like reference.
# A clash is two <clash> records sharing a `cid`, each under a <ModelledSubgroup chain= resnum= resname=>.
# We keep the INTER-CHAIN, HEAVY↔HEAVY subset — exactly the axis the owned gate measures.
# --------------------------------------------------------------------------- #
def _atom_is_heavy(name: str) -> bool:
    """Element of a PDB atom NAME (strip a leading digit, take the leading letter). Heavy ⇔ not H/D."""
    s = name.strip().lstrip("0123456789")
    return bool(s) and s[0].upper() not in ("H", "D")


def parse_interchain_clashes(xml_text: str) -> tuple[int, tuple]:
    """From a wwPDB validation XML, the number of INTER-CHAIN HEAVY↔HEAVY clash pairs + a few identified ones.
    Pairs are linked by `cid`; a pair counts iff its two atoms are in different chains and both are heavy."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return 0, ()
    by_cid: dict[str, list[tuple]] = {}
    for sg in root.iter("ModelledSubgroup"):
        chain = sg.get("chain") or sg.get("said") or ""
        resnum = sg.get("resnum") or ""
        resname = sg.get("resname") or ""
        for cl in sg.findall("clash"):
            cid = cl.get("cid")
            atom = cl.get("atom") or ""
            if cid is None:
                continue
            by_cid.setdefault(cid, []).append((chain, resnum, resname, atom))
    n = 0
    detail: list[tuple] = []
    for cid, ends in by_cid.items():
        if len(ends) != 2:
            continue                               # a clash links exactly two atoms
        (ca, ra, na, aa), (cb, rb, nb, ab) = ends
        if ca == cb:
            continue                               # intra-chain — not the owned inter-chain axis
        if not (_atom_is_heavy(aa) and _atom_is_heavy(ab)):
            continue                               # heavy↔heavy only (the gate is heavy-atom)
        n += 1
        if len(detail) < 8:
            detail.append((ca, ra, na, aa, cb, rb, nb, ab))
    return n, tuple(detail)


def _ref_clashes_of(pdb_id: str) -> tuple[int, tuple] | None:
    txt = _fetch(_PDBE_VALIDATION_XML.format(id=pdb_id), f"{pdb_id}.validation.xml")
    if not txt:
        return None
    return parse_interchain_clashes(txt)


# --------------------------------------------------------------------------- #
# The native loader.
# --------------------------------------------------------------------------- #
def _load_coords(pdb_id: str) -> list | None:
    txt = _fetch(_RCSB_PDB.format(id=pdb_id), f"{pdb_id}.pdb")
    if not txt:
        return None
    atoms = sio.read_pdb_atoms(txt)
    polymer = sio._polymer_only(atoms)            # protein chains only (drop solvent/ligand HETATM)
    return polymer or None


def load_native_complexes(*, limit: int | None = None, ids: list | None = None) -> list[NativeComplex]:
    """Deposited protein complexes joined to their wwPDB validation reference. Each entry needs all three
    fetches (coords + clashscore + validation XML) and ≥2 chains; entries missing any are skipped. Offline-skip
    via `PoseUnavailable` if NOTHING loads."""
    pool = ids if ids is not None else NATIVE_PDB_IDS
    out: list[NativeComplex] = []
    for pid in pool:
        atoms = _load_coords(pid)
        if not atoms:
            continue
        chains = sio.chain_ids(atoms)
        if len(chains) < 2:
            continue                               # need an inter-chain interface
        ref = _ref_clashes_of(pid)
        if ref is None:
            continue                               # no validation reference ⇒ can't measure faithfulness
        n_clash, detail = ref
        out.append(NativeComplex(pdb_id=pid, atoms=atoms, chains=chains,
                                 clashscore=_clashscore_of(pid),
                                 ref_interchain_clashes=n_clash, ref_clash_detail=detail))
        if limit and len(out) >= limit:
            break
    if not out:
        raise PoseUnavailable("no native complexes could be fetched (coords + clashscore + validation XML) — "
                              "offline? (~/.cache/karyon/ppi/ is empty)")
    return out


# --------------------------------------------------------------------------- #
# Predicted complexes — CASP15 multimer submissions (the effect arm; no deposited validity reference).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PredictedComplex:
    target: str                            # CASP target id (the grouping axis; group→tool isn't deanonymized)
    model: str                             # submission file stem
    atoms: list                            # list[structure_io.Atom], one predicted model
    chains: list


def _fetch_binary(url: str, cache_name: str) -> Path | None:
    """Fetch a binary artefact (a tarball) to ~/.cache/karyon/ppi/<cache_name>. None on any network/HTTP error."""
    path = _ppi_cache() / cache_name
    if path.exists():
        return path
    if not network_allowed():
        raise PoseUnavailable("network disabled via KARYON_NO_NETWORK")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r, tmp.open("wb") as fh:
            import shutil
            shutil.copyfileobj(r, fh, length=1 << 20)
        tmp.replace(path)
        return path
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
        tmp.unlink(missing_ok=True)
        return None


def _ensure_casp_target(target: str) -> Path | None:
    """Fetch + extract one CASP15 oligo target's submissions to ~/.cache/karyon/ppi/casp_<target>/."""
    out = _ppi_cache() / f"casp_{target}"
    if out.exists() and any(out.iterdir()):
        return out
    tar = _fetch_binary(_CASP15_OLIGO.format(target=target), f"{target}.tar.gz")
    if tar is None:
        return None
    try:
        with tarfile.open(tar) as tf:
            tf.extractall(out)                      # noqa: S202 — trusted predictioncenter archive, our cache
    except (tarfile.TarError, OSError):
        return None
    return out if any(out.iterdir()) else None


def _first_model(text: str) -> str:
    """A CASP TS file may hold up to 5 MODELs; keep the first (truncate at the first ENDMDL)."""
    lines = text.splitlines()
    end = next((i for i, ln in enumerate(lines) if ln.startswith("ENDMDL")), len(lines))
    return "\n".join(lines[:end])


def load_predicted_complexes(*, limit: int | None = None, targets: list | None = None,
                             per_target: int = 25) -> list[PredictedComplex]:
    """CASP15 multimer submissions as predicted complexes (one top model per submission). Multi-chain models
    only (an interface needs ≥2 chains). Offline-skip via `PoseUnavailable` if nothing fetches."""
    pool = targets if targets is not None else PREDICTED_TARGETS
    out: list[PredictedComplex] = []
    for tgt in pool:
        root = _ensure_casp_target(tgt)
        if root is None:
            continue
        files = sorted(p for p in root.rglob("*") if p.is_file() and "TS" in p.name)
        kept = 0
        for f in files:
            if kept >= per_target:
                break
            try:
                atoms = sio._polymer_only(sio.read_pdb_atoms(_first_model(f.read_text(errors="replace"))))
            except (OSError, ValueError):
                continue
            chains = sio.chain_ids(atoms)
            if len(chains) < 2 or len(atoms) < 50:
                continue                            # need a real multi-chain model
            out.append(PredictedComplex(target=tgt, model=f.name, atoms=atoms, chains=chains))
            kept += 1
            if limit and len(out) >= limit:
                return out
    if not out:
        raise PoseUnavailable("no CASP15 predicted complexes could be fetched/parsed (offline, or layout drift)")
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="complex-QC native loader (PDB complexes + wwPDB validation ref).")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--predicted", action="store_true", help="summarize a few CASP15 predicted complexes instead")
    cli = ap.parse_args()
    if cli.predicted:
        try:
            preds = load_predicted_complexes(limit=cli.limit)
        except PoseUnavailable as e:
            print("SKIP —", e)
            raise SystemExit(0)
        print(f"predicted complexes (CASP15 oligo): {len(preds)} shown")
        for p in preds:
            print(f"  {p.target} {p.model:24} chains {''.join(p.chains):8} {len(p.atoms):5} atoms")
        raise SystemExit(0)
    try:
        cx = load_native_complexes(limit=cli.limit)
    except PoseUnavailable as e:
        print("SKIP —", e)
        raise SystemExit(0)
    print(f"native complexes (fetched + cached): {len(cx)} shown")
    for c in cx:
        cls = f"{c.clashscore:.1f}" if c.clashscore is not None else "n/a"
        print(f"  {c.pdb_id}  chains {''.join(c.chains):8} {len(c.atoms):5} atoms | "
              f"deposited clashscore {cls:>5} | wwPDB inter-chain heavy clashes {c.ref_interchain_clashes}")
