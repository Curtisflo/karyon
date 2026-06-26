"""molnet_data — cached loader for MoleculeNet molecular-property benchmarks (avenue 6, ADMET-honesty).

The substrate for the **molecular-property benchmark-honesty** probe — the cross-domain *property-prediction*
sibling of the retrosynthesis probe (avenue 5). Same cross-cutting diagnostic (STRATEGY §3: "data leakage
inflates ML accuracy"), here in its canonical chemistry-property form: models report on **random** splits,
which leak (the same Bemis-Murcko scaffolds land in train and test), so the honest eval is a **scaffold-disjoint
split** — and reported random-split accuracy is inflated against it. This is the documented ADMET-benchmark
honesty issue and a real market (Inductive Bio, ADMET).

Two MoleculeNet datasets (login-free DeepChem S3 CSVs), one classification + one regression, to show the
audit ports across task types:
  * **BBBP** — blood-brain-barrier penetration (2,050 mols; binary `p_np`); metric = AUROC.
  * **ESOL** (delaney) — aqueous solubility (1,128 mols; `measured log solubility`); metric = Spearman ρ.

Each record carries the SMILES, the label, and its **Bemis-Murcko scaffold** (RDKit; the scaffold-split key +
the `SCAFFOLD_SEEN_IN_TRAIN` leakage signal). rdkit-gated (chemistry substrate) + offline once `~/.cache/karyon/`
is warm; degrades to `DatasetUnavailable` → SKIP. Mirrors `uspto_data.py` / `crispr_qc_data.py`.

    cd karyon/probe && python molnet_data.py            # smoke: fetch + summarize + scaffold-split disjointness
"""

from __future__ import annotations
from .paths import cache_dir

import csv
import io
import random
import socket
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False

_UA = "karyon-bio/1 (+https://moleculenet.org)"
_TIMEOUT_S = 120
_S3 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"


class DatasetUnavailable(RuntimeError):
    """A MoleculeNet CSV could not be fetched/parsed (or rdkit absent) and is not cached → SKIP."""


@dataclass(frozen=True)
class Spec:
    url: str
    smiles_col: str
    label_col: str
    classification: bool


DATASETS = {
    "bbbp": Spec(f"{_S3}/BBBP.csv", "smiles", "p_np", True),
    "esol": Spec(f"{_S3}/delaney-processed.csv", "smiles", "measured log solubility in mols per litre", False),
}


@dataclass(frozen=True)
class Molecule:
    smiles: str                 # canonical SMILES (the similarity / near-dup key)
    label: float                # 0/1 for classification, continuous for regression
    scaffold: str               # Bemis-Murcko scaffold SMILES ("" for acyclic mols) — scaffold-split key


# --------------------------------------------------------------------------- #
# Cache plumbing (~/.cache/karyon/, gitignored — mirrors uspto_data.py).
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    return here.parents[2]


def _cache_path(name: str) -> Path:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"molnet_{name}.csv"


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        return urllib.request.urlopen(req, timeout=_TIMEOUT_S).read().decode("utf-8", "replace")
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        raise DatasetUnavailable(f"cannot reach {url}: {e}") from e


def _parse(text: str, spec: Spec) -> list[Molecule]:
    out: list[Molecule] = []
    for row in csv.DictReader(io.StringIO(text)):
        smi = (row.get(spec.smiles_col) or "").strip()
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        try:
            label = float(row[spec.label_col])
        except (KeyError, ValueError, TypeError):
            continue
        out.append(Molecule(Chem.MolToSmiles(m), label,
                            MurckoScaffold.MurckoScaffoldSmiles(mol=m)))
    return out


_FIELDS = ["smiles", "label", "scaffold"]


def _write_cache(path: Path, mols: list[Molecule]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_FIELDS)
        for m in mols:
            w.writerow([m.smiles, m.label, m.scaffold])


def _read_cache(path: Path) -> list[Molecule]:
    out = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Molecule(row["smiles"], float(row["label"]), row.get("scaffold", "")))
    return out


def load_dataset(name: str, *, refresh: bool = False) -> list[Molecule]:
    """The MoleculeNet dataset `name` (canonical SMILES + label + Murcko scaffold). Cache-first, offline-skip.
    Requires rdkit on a cold cache (to canonicalize + compute scaffolds)."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; have {list(DATASETS)}")
    path = _cache_path(name)
    if path.exists() and not refresh:
        mols = _read_cache(path)
        print(f"  [cache] {len(mols)} {name} molecules from {path.name}")
        return mols
    if not _HAVE_RDKIT:
        raise DatasetUnavailable(f"{name} not cached and rdkit absent (needed to canonicalize + scaffold)")
    mols = _parse(_fetch(DATASETS[name].url), DATASETS[name])
    if not mols:
        raise DatasetUnavailable(f"parsed 0 usable {name} molecules (format drift?)")
    _write_cache(path, mols)
    print(f"  [cache] wrote {len(mols)} {name} molecules -> {path.name}")
    return mols


# --------------------------------------------------------------------------- #
# Splits: random (the leaky one everyone reports on) + scaffold-disjoint (the honest one).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Split:
    name: str
    train: list[Molecule]
    test: list[Molecule]

    @property
    def sizes(self) -> tuple[int, int]:
        return (len(self.train), len(self.test))


def random_split(mols: list[Molecule], *, seed: int = 0, test_frac: float = 0.2) -> Split:
    idx = list(range(len(mols)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1.0 - test_frac))
    return Split(f"random(seed={seed})", [mols[i] for i in idx[:cut]], [mols[i] for i in idx[cut:]])


def scaffold_split(mols: list[Molecule], *, seed: int = 0, test_frac: float = 0.2) -> Split:
    """Assign whole Bemis-Murcko scaffolds to train/test so no scaffold straddles — MoleculeNet's honest
    split. Acyclic molecules (scaffold "") are pooled under one bucket. Largest scaffolds go to train first
    (the standard deterministic recipe), then the tail fills test up to test_frac."""
    by_scaffold: dict[str, list[Molecule]] = {}
    for m in mols:
        by_scaffold.setdefault(m.scaffold, []).append(m)
    # canonical recipe: big scaffold-sets fill train first; the rare/singleton tail becomes the (hard,
    # genuinely novel) test set. Whole scaffolds stay on one side, so no scaffold straddles.
    groups = sorted(by_scaffold.values(), key=lambda g: (-len(g), g[0].scaffold))
    target_train = len(mols) - int(len(mols) * test_frac)
    train: list[Molecule] = []
    test: list[Molecule] = []
    for g in groups:
        (train if len(train) + len(g) <= target_train else test).extend(g)
    return Split(f"scaffold(seed={seed})", train, test)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MoleculeNet loader (BBBP / ESOL) + scaffold-split disjointness.")
    ap.add_argument("--dataset", default="bbbp", choices=list(DATASETS))
    cli = ap.parse_args()
    print(f"Loading MoleculeNet '{cli.dataset}'\n")
    try:
        mols = load_dataset(cli.dataset)
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)

    scaffs = Counter(m.scaffold for m in mols)
    acyclic = scaffs.get("", 0)
    rs, ss = random_split(mols), scaffold_split(mols)
    rep_scaf = sum(c - 1 for s, c in scaffs.items() if s and c > 1)
    print(f"\n  molecules               : {len(mols)}")
    print(f"  distinct scaffolds      : {len(scaffs)}  ({rep_scaf} mols share a scaffold with another; "
          f"{acyclic} acyclic)")
    if DATASETS[cli.dataset].classification:
        pos = sum(1 for m in mols if m.label >= 0.5)
        print(f"  positives               : {pos} ({pos / len(mols):.1%})")
    else:
        labs = [m.label for m in mols]
        print(f"  label min/max           : {min(labs):.2f} / {max(labs):.2f}")
    print(f"\n  random split   train/test: {rs.sizes}")
    print(f"  scaffold split train/test: {ss.sizes}")
    straddle = {m.scaffold for m in ss.train if m.scaffold} & {m.scaffold for m in ss.test if m.scaffold}
    print(f"  scaffolds straddling the scaffold split: {len(straddle)}  (must be 0)")
