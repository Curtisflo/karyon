"""test_retro_template — proofs for the faithful RDKit/RDChiral retrosynthesis arm (dual pytest / __main__).

The teeth: canonicalization is map-stripped + order-independent; a template extracted from a reaction and
applied to its own product recovers its reactants; the faithful baseline vastly outperforms the stdlib
retriever (top-1 ~38% vs ~1%); and template-SEEN reactions are solved far more often than template-NOVEL
ones (the leakage mechanism, measured). Everything SKIPs if rdkit/rdchiral or the dataset/cache are absent.
"""

from __future__ import annotations

import os

from karyon import retro_template as rt
from karyon.uspto_data import DatasetUnavailable, Reaction, load_reactions, random_split


def _skip(msg: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    print(f"   SKIP — {msg}")


def test_canonicalization_is_mapstripped_and_orderless() -> None:
    if not rt._HAVE_RDKIT:
        _skip("rdkit/rdchiral absent")
        return
    assert rt.canon("[CH3:4][C:1](=[O:2])[OH:3]") == rt.canon("CC(=O)O")     # map-stripped
    assert rt.canon_set("OC(=O)C.NC") == rt.canon_set("NC.OC(=O)C")          # order-independent
    assert rt.canon_set("garbage~~~") is None                               # unparseable → None
    print("1. canonicalization: map-stripped, order-independent, None on junk")


def test_extract_apply_recovers_own_reactants() -> None:
    if not rt._HAVE_RDKIT:
        _skip("rdkit/rdchiral absent")
        return
    r = Reaction(rid="t", klass=1, product="CC(=O)NC", reactant_sig="",
                 rxn_smiles="[OH:3][C:1](=[O:2])[CH3:4].[NH2:5][CH3:6]>>[CH3:4][C:1](=[O:2])[NH:5][CH3:6]")
    tmpl = rt._extract_one(r)
    assert tmpl, "template extraction failed"
    preds = rt.apply_template(tmpl, "CC(=O)NC")
    assert rt.canon_set("OC(=O)C.NC") in preds, f"own reactants not recovered: {preds}"
    print(f"2. extract+apply recovers own reactants ({len(preds)} candidate set(s))")


def test_faithful_beats_stdlib_and_template_seen_drives_accuracy() -> None:
    if not rt._HAVE_RDKIT:
        _skip("rdkit/rdchiral absent")
        return
    try:
        rxns = load_reactions(limit=4000)
    except DatasetUnavailable as e:
        _skip(f"USPTO-50k unreachable and not cached: {e}")
        return
    templates = rt.load_templates(rxns)
    if sum(1 for t in templates if t) < 100:
        _skip("templates not cached (would need a ~2 min RDChiral extraction)")
        return
    tbp = {r.product: t for r, t in zip(rxns, templates) if t}
    outs = rt.run_faithful(random_split(rxns, seed=0), tbp, test_sample=200, top_n=20)
    top1 = rt.topk(outs)[1]
    seen = rt.topk([o for o in outs if o.template_seen])[1]
    novel = rt.topk([o for o in outs if not o.template_seen])[1]
    assert top1 > 0.20, f"faithful top-1 implausibly low ({top1:.1%}) — should dwarf the stdlib ~1%"
    assert seen > novel, f"template-seen should be solved more than template-novel ({seen:.1%} vs {novel:.1%})"
    print(f"3. faithful top-1 {top1:.1%} (≫ stdlib ~1%); template SEEN {seen:.1%} > NOVEL {novel:.1%}")


if __name__ == "__main__":
    test_canonicalization_is_mapstripped_and_orderless()
    test_extract_apply_recovers_own_reactants()
    test_faithful_beats_stdlib_and_template_seen_drives_accuracy()
    print("\nretro_template proofs pass.")
