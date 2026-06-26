"""pose_data — cached loader for deposited docking poses + the reference PoseBusters verdicts (avenue 7).

The substrate for the **docking physical-validity honesty** probe. Source: the PoseBench reproducibility
deposit (Zenodo 19138652, login-free), two small per-method prediction tarballs of the **PoseBusters
benchmark set** (308 post-2021 PDB complexes):
  * `diffdock_benchmark_method_predictions.tar.gz` (11 MB) — DiffDock (deep-learning docking) top poses.
  * `vina_benchmark_method_predictions.tar.gz` (19 MB) — AutoDock Vina (classical docking) top poses.

Each tarball bundles a `bust_results.csv` — the **real PoseBusters package** run over those same poses, with
a True/False per physical-validity check plus `rmsd_≤_2å` (the benchmark "success" label). We use it two ways:
  1. as the **faithfulness oracle** for our from-scratch legible *intramolecular* DRC (`pose_validity.py`),
     the screen-QC → real-MAGeCK precedent (validate the reimplementation, don't strawman it);
  2. as the **consumed intermolecular (protein) verdict** — reproducing the protein-ligand checks end-to-end
     needs each method's pose+receptor in a shared coordinate frame, which the deposit does not cleanly ship
     (Vina's bundled "protein" files are cofactor groups; poses are not in the crystal frame). That frame
     bookkeeping is exactly what PoseBench already did to produce these columns, so we consume the
     intermolecular verdict and disclose it as not-reimplemented (the honest split). The large effect lives
     on this axis; our legible contribution is the faithful intra DRC + the per-pose decomposition/audit.

We evaluate the **raw** (non-relaxed) `_output_2` / `_outputs_2` poses — the canonical PoseBusters posture
(relaxed poses are force-field-minimized, which repairs the very violations the probe is about). Cache-first
(extract each ~tarball once into `~/.cache/karyon/`), offline-skip via `PoseUnavailable` → SKIP. Mirrors
`uspto_data.py` / `molnet_data.py`.

    cd karyon/probe && python pose_data.py        # smoke: fetch/extract + summarize both methods
"""

from __future__ import annotations
from .paths import cache_dir

import csv
import socket
import tarfile
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_UA = "karyon-bio/1 (+https://github.com/BioinfoMachineLearning/PoseBench)"
_TIMEOUT_S = 180
_ZENODO = "https://zenodo.org/records/19138652/files"

# PoseBusters check columns, grouped by axis (the names in bust_results.csv).
INTRA_COLS = ["bond_lengths", "bond_angles", "internal_steric_clash",
              "aromatic_ring_flatness", "double_bond_flatness", "internal_energy"]
INTER_COLS = ["minimum_distance_to_protein", "volume_overlap_with_protein",
              "protein-ligand_maximum_distance"]
# map a bust intra column -> the legible pose_validity contract it corresponds to (for per-contract agreement)
INTRA_TO_CONTRACT = {
    "bond_lengths": "BOND_LENGTH_OUTLIER",
    "bond_angles": "BOND_ANGLE_OUTLIER",
    "internal_steric_clash": "INTERNAL_STERIC_CLASH",
    "aromatic_ring_flatness": "AROMATIC_RING_NONPLANAR",
    "double_bond_flatness": "DOUBLE_BOND_NONPLANAR",
    "internal_energy": "INTERNAL_STRAIN_ENERGY",
}


@dataclass(frozen=True)
class Method:
    key: str            # "diffdock" | "vina"
    tarball: str        # Zenodo filename
    local: str          # cached tarball name
    bust_rel: str       # bust_results.csv path inside the tar
    pose_dir: str       # per-target pose directory inside the tar
    pose_name: str      # "rank1.sdf" (DiffDock) or "{target}.sdf" (Vina)


METHODS = {
    "diffdock": Method(
        "diffdock", "diffdock_benchmark_method_predictions.tar.gz", "diffdock_predictions.tar.gz",
        "forks/DiffDock/inference/diffdock_posebusters_benchmark_output_2/bust_results.csv",
        "forks/DiffDock/inference/diffdock_posebusters_benchmark_output_2", "rank1.sdf"),
    "vina": Method(
        "vina", "vina_benchmark_method_predictions.tar.gz", "vina_predictions.tar.gz",
        "forks/Vina/inference/vina_p2rank_posebusters_benchmark_outputs_2/bust_results.csv",
        "forks/Vina/inference/vina_p2rank_posebusters_benchmark_outputs_2", "{target}.sdf"),
}


class PoseUnavailable(RuntimeError):
    """A prediction tarball could not be fetched/extracted and is not cached → SKIP."""


@dataclass(frozen=True)
class PredictedPose:
    method: str                 # "diffdock" | "vina"
    target: str                 # PDB id of the benchmark target (e.g. "7K0V_VQP")
    block: str                  # the SDF mol block (the predicted ligand pose) — for our intra DRC
    rmsd_le_2: bool             # bust: the benchmark "success" label (RMSD ≤ 2 Å to crystal)
    ref_intra_valid: bool       # bust: AND of the intramolecular columns (our DRC's faithfulness target)
    ref_inter_valid: bool       # bust: AND of the intermolecular (protein) columns (the consumed verdict)
    ref_checks: dict            # per-column bust booleans (for the per-contract faithfulness breakdown)


# --------------------------------------------------------------------------- #
# Cache plumbing (~/.cache/karyon/, gitignored — mirrors uspto_data.py).
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    return here.parents[2]


def _cache_dir() -> Path:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_extracted(m: Method) -> Path:
    """Fetch the tarball if absent, extract once into ~/.cache/karyon/pose_<key>/, return the extract root."""
    out = _cache_dir() / f"pose_{m.key}"
    marker = out / m.bust_rel
    if marker.exists():
        return out
    tar_path = _cache_dir() / m.local
    if not tar_path.exists():
        url = f"{_ZENODO}/{m.tarball}?download=1"
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            data = urllib.request.urlopen(req, timeout=_TIMEOUT_S).read()
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            raise PoseUnavailable(f"cannot reach {url}: {e}") from e
        tar_path.write_bytes(data)
    try:
        with tarfile.open(tar_path) as tf:
            tf.extractall(out)                          # noqa: S202 — trusted Zenodo archive, our cache dir
    except (tarfile.TarError, OSError) as e:
        raise PoseUnavailable(f"cannot extract {tar_path.name}: {e}") from e
    if not marker.exists():
        raise PoseUnavailable(f"{m.local} extracted but {m.bust_rel} missing (layout drift?)")
    return out


def _as_bool(v: str) -> bool | None:
    if v in ("True", "true", "1"):
        return True
    if v in ("False", "false", "0"):
        return False
    return None                                         # "", "nan" → unknown (skipped in the AND)


def load_poses(method: str, *, limit: int | None = None) -> list[PredictedPose]:
    """The raw PoseBusters poses for `method`, joined to the reference bust_results row. Cache-first,
    offline-skip. One top pose per target where both the pose SDF and a bust row exist."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; have {list(METHODS)}")
    m = METHODS[method]
    root = _ensure_extracted(m)

    with (root / m.bust_rel).open(newline="") as fh:
        bust = {row["mol_id"]: row for row in csv.DictReader(fh)}

    def axis_valid(row: dict, cols: list[str]) -> bool:
        vals = [_as_bool(row.get(c, "")) for c in cols]
        return all(v for v in vals if v is not None)    # AND over the known columns

    poses: list[PredictedPose] = []
    for target, row in bust.items():
        sdf = root / m.pose_dir / target / m.pose_name.format(target=target)
        if not sdf.exists():
            continue
        rmsd = _as_bool(row.get("rmsd_≤_2å", ""))
        if rmsd is None:
            continue
        poses.append(PredictedPose(
            method=method, target=target, block=sdf.read_text(),
            rmsd_le_2=rmsd,
            ref_intra_valid=axis_valid(row, INTRA_COLS),
            ref_inter_valid=axis_valid(row, INTER_COLS),
            ref_checks={c: _as_bool(row.get(c, "")) for c in INTRA_COLS + INTER_COLS}))
        if limit and len(poses) >= limit:
            break
    if not poses:
        raise PoseUnavailable(f"{method}: extracted but 0 poses joined to bust_results (layout drift?)")
    return poses


if __name__ == "__main__":
    print("PoseBusters deposited-pose loader (DiffDock + Vina, raw)\n")
    for key in METHODS:
        try:
            poses = load_poses(key)
        except PoseUnavailable as e:
            print(f"  SKIP {key} — {e}")
            continue
        n = len(poses)
        succ = sum(p.rmsd_le_2 for p in poses)
        intra = sum(p.ref_intra_valid for p in poses)
        inter = sum(p.ref_inter_valid for p in poses)
        succ_invalid = sum(1 for p in poses if p.rmsd_le_2 and not (p.ref_intra_valid and p.ref_inter_valid))
        print(f"  {key:9} {n:4} poses | RMSD≤2 'success' {succ:3} ({succ/n:.0%}) | "
              f"intra-valid {intra/n:.0%} | inter-valid {inter/n:.0%}")
        if succ:
            print(f"            of the {succ} successes, {succ_invalid} are physically INVALID "
                  f"({succ_invalid/succ:.0%}) — the inflation the legible audit surfaces")
        fire = Counter()
        for p in poses:
            for c in INTRA_COLS + INTER_COLS:
                if p.ref_checks.get(c) is False:
                    fire[c] += 1
        worst = ", ".join(f"{c}={fire[c]}" for c in sorted(fire, key=lambda c: -fire[c])[:4])
        print(f"            top reference failures: {worst}")
