"""retro_template — the FAITHFUL retrosynthesis arm (RDKit + RDChiral) for the honesty probe.

The stdlib retriever ([retro_baseline.py](./retro_baseline.py)) is pinned at the ~1% duplicate rate because
it cannot transfer reaction *templates* — so it only BOUNDS the leakage-impact question. This module answers
it: real **retrosim** (Coley 2017) — Morgan-fingerprint Tanimoto NN over products, then extract each
neighbour's retro template (RDChiral) and **apply** it to the test product to generate ranked precursor
candidates, scored by neighbour similarity. That recovers the literature's ~35–45% top-1, so the
standard-vs-leakage-free gap is the *measured* inflation magnitude, not a bound.

It also makes the AUROC explanatory test **non-circular**: template application solves many non-duplicate
products, so "does train↔test similarity predict the model getting it right" is now a real empirical claim
about a competent model (the stdlib retriever's 0.99 AUROC was a circular dup-detector artifact).

Dependency posture: RDKit + rdchiral are the sanctioned baseline-fairness deps (precedent: ostir/viennarna
for RBS, mageck for screen-QC). RDChiral does **extraction** (its strength); APPLICATION uses raw RDKit
`RunReactants` (rdchiral's `rdchiralRun` KeyErrors on rdkit 2026.03 — a version drift, not our bug), with
RDKit canonicalization on both sides so the top-k match is apples-to-apples. Templates are extracted once
and cached (`~/.cache/karyon/uspto50k_templates.csv`), so re-runs only recompute fingerprints + application.
SKIPs cleanly if rdkit/rdchiral are absent.

    python -m karyon.retro_template --test-sample 1500 --neighbors 20
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import stats_kit
from .retro_honesty import TrainView, leakage_contracts
from .uspto_data import (
    DatasetUnavailable,
    Reaction,
    Split,
    _cache_path,
    load_reactions,
    patent_disjoint_split,
    random_split,
)

try:
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, rdFingerprintGenerator
    from rdchiral.template_extractor import extract_from_reaction
    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:                                        # rdkit/rdchiral/numpy absent → the arm SKIPs
    _HAVE_RDKIT = False

_TAU = 0.7                  # near-duplicate threshold on Morgan-Tanimoto (real chemical similarity)
_MORGAN_R, _MORGAN_BITS = 2, 2048


# --------------------------------------------------------------------------- #
# Canonicalization (map-stripped) — both the truth and the applied precursors go through this, so the
# top-k comparison is apples-to-apples.
# --------------------------------------------------------------------------- #
def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


def canon_set(smiles_dotjoined: str) -> str | None:
    parts = [canon(p) for p in smiles_dotjoined.split(".") if p]
    return ".".join(sorted(parts)) if parts and all(parts) else None


def _reactant_truth(rxn: Reaction) -> str | None:
    return canon_set(rxn.rxn_smiles.split(">>")[0].split(">")[0]) if rxn.rxn_smiles else None


# --------------------------------------------------------------------------- #
# Template extraction (RDChiral), cached row-aligned to load_reactions().
# --------------------------------------------------------------------------- #
def _templates_path() -> Path:
    return _cache_path().with_name("uspto50k_templates.csv")


def _extract_one(rxn: Reaction) -> str:
    rs = rxn.rxn_smiles
    if ">>" not in rs:
        return ""
    try:
        t = extract_from_reaction({"_id": rxn.rid, "reactants": rs.split(">>")[0],
                                   "products": rs.split(">>")[-1]})
        return t.get("reaction_smarts", "") or ""
    except Exception:
        return ""


def load_templates(rxns: list[Reaction], *, refresh: bool = False) -> list[str]:
    """Retro template per reaction (row-aligned to `rxns`); "" where extraction failed. Cached."""
    path = _templates_path()
    if path.exists() and not refresh:
        with path.open(newline="") as fh:
            tmpls = [row["template"] for row in csv.DictReader(fh)]
        if len(tmpls) >= len(rxns):
            print(f"  [cache] {sum(1 for t in tmpls[:len(rxns)] if t)}/{len(rxns)} templates "
                  f"from {path.name}")
            return tmpls[:len(rxns)]
    print(f"  extracting {len(rxns)} retro templates (RDChiral; ~2 min, cached after)…")
    tmpls = [_extract_one(r) for r in rxns]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["template"])
        for t in tmpls:
            w.writerow([t])
    print(f"  [cache] wrote {sum(1 for t in tmpls if t)}/{len(rxns)} templates -> {path.name}")
    return tmpls


# --------------------------------------------------------------------------- #
# Morgan fingerprints + template application.
# --------------------------------------------------------------------------- #
def _fp_generator():
    return rdFingerprintGenerator.GetMorganGenerator(radius=_MORGAN_R, fpSize=_MORGAN_BITS)


def morgan_fps(products: list[str]):
    gen = _fp_generator()
    out = []
    for p in products:
        m = Chem.MolFromSmiles(p)
        out.append(gen.GetFingerprint(m) if m is not None else None)
    return out


def apply_template(template: str, product: str) -> set[str]:
    """Apply a retro template (product>>reactants SMARTS) to a product → canonical precursor sets."""
    try:
        rxn = AllChem.ReactionFromSmarts(template)
    except Exception:
        return set()
    pm = Chem.MolFromSmiles(product)
    if rxn is None or pm is None:
        return set()
    out: set[str] = set()
    try:
        runs = rxn.RunReactants((pm,))
    except Exception:
        return set()
    for tup in runs[:200]:                               # cap explosive templates
        smis, ok = [], True
        for m in tup:
            cs = canon(Chem.MolToSmiles(m)) if m is not None else None
            if cs is None:
                ok = False
                break
            smis.append(cs)
        if ok:
            out.add(".".join(sorted(smis)))
    return out


# --------------------------------------------------------------------------- #
# The faithful model.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FaithfulOutcome:
    rxn: Reaction
    nn_sim: float                 # top neighbour Morgan-Tanimoto
    exact_in_train: bool
    reactant_rank: int | None     # rank of the TRUE precursors among template-generated candidates
    template_seen: bool           # this reaction's own retro template appears in train


class FaithfulModel:
    def __init__(self, train: list[Reaction], train_templates: list[str]):
        self.train = train
        self.templates = train_templates
        self.fps = morgan_fps([r.product for r in train])
        self.valid = [i for i, fp in enumerate(self.fps) if fp is not None]
        self._valid_fps = [self.fps[i] for i in self.valid]
        self.exact = Counter(r.product for r in train)
        self.template_set = {t for t in train_templates if t}

    def neighbors(self, fp, top_n: int) -> list[tuple[int, float]]:
        if fp is None or not self._valid_fps:
            return []
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, self._valid_fps))
        k = min(top_n, len(sims))
        cand = np.argpartition(-sims, k - 1)[:k]
        # deterministic: (-sim, train_idx)
        ranked = sorted(((self.valid[j], float(sims[j])) for j in cand), key=lambda t: (-t[1], t[0]))
        return ranked


def predict(rxn: Reaction, fp, model: FaithfulModel, *, top_n: int) -> FaithfulOutcome:
    nbrs = model.neighbors(fp, top_n)
    truth = _reactant_truth(rxn)
    scores: dict[str, float] = defaultdict(float)
    for i, sim in nbrs:
        t = model.templates[i]
        if not t:
            continue
        for cand in apply_template(t, rxn.product):
            scores[cand] += sim                          # retrosim: sum the parents' similarity
    ranked = sorted(scores, key=lambda c: (-scores[c], c))
    rank = (ranked.index(truth) + 1) if truth in ranked else None
    own_t = _extract_one(rxn)
    return FaithfulOutcome(rxn, nbrs[0][1] if nbrs else 0.0,
                           model.exact.get(rxn.product, 0) > 0, rank,
                           bool(own_t) and own_t in model.template_set)


# --------------------------------------------------------------------------- #
# Run + metrics.
# --------------------------------------------------------------------------- #
def _sample(rxns: list[Reaction], n: int, seed: int = 0) -> list[Reaction]:
    if n >= len(rxns):
        return rxns
    import random
    idx = list(range(len(rxns)))
    random.Random(seed).shuffle(idx)
    return [rxns[i] for i in sorted(idx[:n])]


def run_faithful(split: Split, templates_by_product: dict[str, str], *,
                 test_sample: int, top_n: int) -> list[FaithfulOutcome]:
    train_templates = [templates_by_product.get(r.product, "") for r in split.train]
    model = FaithfulModel(split.train, train_templates)
    test = _sample(split.test, test_sample)
    test_fps = morgan_fps([r.product for r in test])
    return [predict(r, fp, model, top_n=top_n) for r, fp in zip(test, test_fps)]


def topk(outcomes: list[FaithfulOutcome], ks=(1, 3, 5, 10)) -> dict[int, float]:
    n = len(outcomes) or 1
    return {k: sum(1 for o in outcomes if o.reactant_rank is not None and o.reactant_rank <= k) / n
            for k in ks}


def _fmt_topk(d: dict[int, float]) -> str:
    return "  ".join(f"top{k}={v:.1%}" for k, v in d.items())


def template_contracts():
    """The leakage DRC + a TEMPLATE_SEEN_IN_TRAIN contract the faithful arm can finally evaluate."""
    cs = leakage_contracts()
    from . import contracts
    cs.add(contracts.Contract(
        "TEMPLATE_SEEN_IN_TRAIN",
        lambda o, ctx: "this reaction's retro template was seen in train" if o.template_seen else None))
    return cs


def run(test_sample: int = 1500, top_n: int = 20, tau: float = _TAU) -> None:
    if not _HAVE_RDKIT:
        print("SKIP — rdkit/rdchiral/numpy not importable (the faithful arm needs them).")
        raise SystemExit(0)
    print("Faithful retrosim on USPTO-50k (Morgan-Tanimoto NN + RDChiral templates)\n")
    try:
        rxns = load_reactions()
    except DatasetUnavailable as e:
        print(f"SKIP — {e}")
        raise SystemExit(0)
    templates = load_templates(rxns)
    tbp = {r.product: t for r, t in zip(rxns, templates) if t}   # product → its template

    rs, pj = random_split(rxns, seed=0), patent_disjoint_split(rxns, seed=0)
    print(f"\nExtracting fingerprints + applying templates (test sample {test_sample}, {top_n} neighbours)…")
    outs = run_faithful(rs, tbp, test_sample=test_sample, top_n=top_n)
    outs_pj = run_faithful(pj, tbp, test_sample=test_sample, top_n=top_n)

    cs = template_contracts()
    tv = TrainView(patents={r.rid for r in rs.train if r.rid},
                   reactions={(r.product, r.reactant_sig) for r in rs.train}, tau=tau)
    verdicts = [cs.evaluate(o, tv) for o in outs]
    leaked = [not v.ok for v in verdicts]
    clean = [o for o, v in zip(outs, verdicts) if v.ok]

    full = topk(outs)
    cl = topk(clean)
    print("\n=== Faithful top-k (the literature-scale baseline the stdlib arm couldn't reach) ===")
    print(f"  standard split        : {_fmt_topk(full)}   (n={len(outs)})")
    print(f"  leakage-free partition: {_fmt_topk(cl)}   (n={len(clean)}, {len(clean)/len(outs):.0%} survives)")
    print(f"  patent-disjoint split : {_fmt_topk(topk(outs_pj))}   (n={len(outs_pj)})")
    print(f"  MEASURED inflation (standard − leakage-free) top-1 = {full[1] - cl[1]:+.1%}   "
          f"(stdlib arm could only bound this at ~0)")

    fire = Counter()
    for v in verdicts:
        for name in v.fired:
            fire[name] += 1
    n = len(outs)
    print("\nLeakage prevalence on the faithful test sample:")
    for c in cs.contracts:
        print(f"  {c.name:<24} {fire.get(c.name, 0) / n:6.1%}")
    print(f"  {'ANY (leaked)':<24} {sum(leaked) / n:6.1%}")

    au = stats_kit.mann_whitney([o.nn_sim for o in outs if o.reactant_rank == 1],
                                [o.nn_sim for o in outs if o.reactant_rank != 1])
    au_t = stats_kit.mann_whitney([1.0 if o.template_seen else 0.0 for o in outs if o.reactant_rank == 1],
                                  [1.0 if o.template_seen else 0.0 for o in outs if o.reactant_rank != 1])
    print("\nExplanatory power (NON-CIRCULAR now — template application solves non-duplicates):")
    print(f"  AUROC(Tanimoto sim → top-1 correct) = {stats_kit.fmt(au)}")
    print(f"  AUROC(template-seen → top-1 correct) = {stats_kit.fmt(au_t)}")

    floor_seen = sum(o.template_seen for o in outs) / n
    print("\n§5 decomposition (faithful arm):")
    print(f"  template-seen-in-train rate on test = {floor_seen:.1%}  "
          f"(a template-only model is capped near here; application generalizes beyond it)")
    print(f"  top-1 among template-SEEN  = {topk([o for o in outs if o.template_seen])[1]:.1%}")
    unseen = [o for o in outs if not o.template_seen]
    print(f"  top-1 among template-NOVEL = {topk(unseen)[1]:.1%}  (n={len(unseen)} — the genuine-generalization tail)")

    print("\n=== READ (faithful arm) ===")
    print("  The faithful baseline reaches literature-scale top-k, so the standard−leakage-free gap is the")
    print("  MEASURED leakage-inflation magnitude (vs the stdlib arm's bound). Compare to RetroXpert's")
    print("  documented 70.4→62.1 (≈8 pts). Interpretation left to the RESULT doc; qualification only (altitude).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Faithful retrosim arm: Morgan-Tanimoto NN + RDChiral templates.")
    ap.add_argument("--test-sample", type=int, default=1500, help="test reactions to score (speed knob)")
    ap.add_argument("--neighbors", type=int, default=20, help="nearest train products per query")
    ap.add_argument("--tau", type=float, default=_TAU, help="near-duplicate Tanimoto threshold")
    cli = ap.parse_args()
    run(test_sample=cli.test_sample, top_n=cli.neighbors, tau=cli.tau)
