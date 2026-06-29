"""perturbseq_data — cached loader for a real single-cell Perturb-seq screen (the SCEPTRE/Replogle avenue).

The substrate for the **single-cell screen-reliability QC** probe — the in-domain heavyweight the bulk
screen-QC probe ([SCREEN_QC_RESULT.md](./SCREEN_QC_RESULT.md)) named but did not take ("the SCEPTRE/Replogle
single-cell analog (heavier posture)"). Genome-scale Perturb-seq is the modern CRISPR-screen format; its
per-perturbation single-cell transcriptome readout gives the one thing a bulk dropout screen cannot — a
**direct on-target knockdown** measurement, i.e. a *ground-truth* silent-failure label (a guide that never
suppressed its target ⇒ any "no phenotype" call for it is an artifact).

Substrate — **Replogle et al. 2022** genome-scale Perturb-seq (Cell), the K562 **essential-gene** experiment,
gemgroup-Z-normalized **pseudobulk** (`K562_essential_normalized_bulk_01.h5ad`, ~80 MB on Figshare+). The
single-cell `.h5ad` is 8–65 GB and off the desk posture; the **pseudobulk** carries everything this probe
needs in the `obs` table, one row per perturbation:
  * `gene_transcript` — the perturbation id (`<id>_<GENE>_<protospacers>_<ENSG>`); target = the `<GENE>` field
    (controls are `non-targeting`);
  * `fold_expr` — **on-target knockdown** = residual target-gene expression (lower = stronger KD; the
    silent-failure ground truth; NaN where unmeasurable);
  * `energy_test_p_value` — Replogle's **deposited calibrated** phenotype-significance (the energy-distance
    test over the transcriptome): the credible incumbent "did this perturbation do anything" caller (the
    role SCEPTRE plays for single-cell screens), consumed not reimplemented;
  * `anderson_darling_counts` — number of differentially-expressed genes (a continuous phenotype strength);
  * `num_cells_unfiltered` / `num_cells_filtered` — cells per perturbation (the single-cell power axis);
  * `core_control` — the non-targeting-control set (the NT calibration the QC layer needs).

karyon's role here is the **legible QC layer that qualifies the commodity tool's output** — so the phenotype
caller is *consumed* from the deposit (energy-test p-value, primary; AD DE-count, secondary), and the owned,
legible part is the knockdown/cell qualification layer in `perturbseq_qc.py`. (A pseudobulk-thresholded owned
DE statistic was tried and dropped: averaging to pseudobulk is far weaker than Replogle's cell-level AD test,
so it is not a credible second caller — the two deposited callers carry that robustness instead.)

**Dependencies:** reading `.h5ad` (HDF5) needs **h5py**, an optional dependency (`pip install karyon[singlecell]`);
the whole QC layer stays stdlib + numpy. SKIP cleanly if h5py or the file is absent. Cache-first under the
karyon cache (override with ``$KARYON_CACHE``).

    python -m karyon.perturbseq_data     # smoke: fetch + summarize + control calibration
"""

from __future__ import annotations

import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from .paths import cache_dir, network_allowed

try:
    import h5py            # the optional reader dep; h5py pulls numpy, used implicitly via array methods
    _HAVE_H5PY = True
except Exception:
    _HAVE_H5PY = False

_UA = "chalkeon-bio/1 (+https://chalkeon.local/karyon perturbseq-qc)"
_TIMEOUT_S = 600
# Replogle et al. 2022 — K562 essential-gene gemgroup-Z pseudobulk (Figshare+ article 20029387).
_URL = "https://ndownloader.figshare.com/files/35780870"
_FILE = "K562_essential_normalized_bulk_01.h5ad"
_MD5 = "30496767641cd2e660ee6ecb5baee132"


class DatasetUnavailable(RuntimeError):
    """The Perturb-seq pseudobulk could not be fetched/read (or h5py absent) and is not cached → SKIP."""


@dataclass(frozen=True)
class Perturbation:
    pid: str                 # full gene_transcript id
    target: str              # parsed target gene symbol, or "non-targeting"
    is_control: bool         # core_control — the NT calibration set
    knockdown_resid: float   # fold_expr — residual on-target expression (lower = stronger KD); nan if unmeasured
    energy_p: float          # deposited calibrated phenotype-significance (the primary incumbent caller)
    de_count: int            # anderson_darling_counts — deposited # DE genes (secondary caller / strength)
    n_cells: int             # num_cells_unfiltered (clean int power axis)
    n_cells_filtered: float  # num_cells_filtered (may be nan)

    @property
    def knockdown_measured(self) -> bool:
        return self.knockdown_resid == self.knockdown_resid  # not NaN


# --------------------------------------------------------------------------- #
# Cache plumbing (cache_dir(); override with $KARYON_CACHE — mirrors molnet_data.py).
# --------------------------------------------------------------------------- #
def _cache_path() -> Path:
    return cache_dir() / _FILE


def _fetch(refresh: bool) -> Path:
    path = _cache_path()
    if path.exists() and not refresh:
        return path
    if not network_allowed():
        raise DatasetUnavailable("network disabled via KARYON_NO_NETWORK")
    req = urllib.request.Request(_URL, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r, path.open("wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
        if path.exists():
            path.unlink()
        raise DatasetUnavailable(f"cannot fetch the Perturb-seq pseudobulk and not cached: {e}") from e
    return path


def _decode(arr) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def _target_of(label: str) -> str:
    """`<id>_<GENE>_<protospacers>_<ENSG>` → `<GENE>` (or 'non-targeting')."""
    parts = label.split("_")
    return parts[1] if len(parts) >= 2 else label


def load_perturbations(*, refresh: bool = False) -> list[Perturbation]:
    """One row per perturbation from the Replogle K562-essential pseudobulk (obs metadata only — the matrix
    is not needed). Cache-first, offline-skip; requires h5py to read the HDF5 (the approved reader dep)."""
    if not _HAVE_H5PY:
        raise DatasetUnavailable("h5py not importable (the approved .h5ad reader for this avenue)")
    path = _fetch(refresh)
    try:
        with h5py.File(path, "r") as f:
            o = f["obs"]
            labels = _decode(o["gene_transcript"][:])
            core = o["core_control"][:].astype(bool)
            fold = o["fold_expr"][:].astype(float)
            ep = o["energy_test_p_value"][:].astype(float)
            de = o["anderson_darling_counts"][:].astype(int)
            ncu = o["num_cells_unfiltered"][:].astype(int)
            ncf = o["num_cells_filtered"][:].astype(float)
    except (KeyError, OSError) as e:
        raise DatasetUnavailable(f"unexpected pseudobulk layout ({e}); format drift?") from e

    out: list[Perturbation] = []
    for i, lab in enumerate(labels):
        out.append(Perturbation(
            pid=lab, target=_target_of(lab), is_control=bool(core[i]),
            knockdown_resid=float(fold[i]), energy_p=float(ep[i]), de_count=int(de[i]),
            n_cells=int(ncu[i]), n_cells_filtered=float(ncf[i])))
    return out


def controls(perts: list[Perturbation]) -> list[Perturbation]:
    return [p for p in perts if p.is_control]


def targeting(perts: list[Perturbation]) -> list[Perturbation]:
    return [p for p in perts if not p.is_control]


if __name__ == "__main__":
    import statistics
    print("Loading Replogle K562-essential Perturb-seq pseudobulk\n")
    try:
        perts = load_perturbations()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    tgt, ctl = targeting(perts), controls(perts)
    measured = [p for p in tgt if p.knockdown_measured]
    print(f"  perturbations            : {len(perts)}  ({len(tgt)} targeting / {len(ctl)} non-targeting controls)")
    print(f"  on-target KD measured    : {len(measured)}/{len(tgt)} targeting "
          f"(median residual expr {statistics.median(p.knockdown_resid for p in measured):.3f})")
    print(f"  deposited phenotype call : energy_p<0.05  targeting {sum(p.energy_p < 0.05 for p in tgt) / len(tgt):.0%}"
          f"  vs controls {sum(p.energy_p < 0.05 for p in ctl) / len(ctl):.0%}  (calibration)")
    print(f"  cells / perturbation     : median {int(statistics.median(p.n_cells for p in perts))} "
          f"(min {min(p.n_cells for p in perts)}, max {max(p.n_cells for p in perts)})")
