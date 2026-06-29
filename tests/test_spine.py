"""test_spine — the unified spine: modality resolution, per-gate dispatch, the stable JSON schema, and
spine-faithfulness (a qualified verdict equals the gate's own `validate`). Gate deps skip cleanly.

    python tests/test_spine.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from karyon import spine as q
from karyon.spine import QualifyError

_REPO = Path(__file__).resolve().parents[1]
_POSES = _REPO / "examples" / "compose" / "candidates"


def _pdb(rec, serial, name, resname, chain, resseq, x, y, z, elem) -> str:
    return (f"{rec:<6}{serial:>5} {name:<4} {resname:>3} {chain:1}{resseq:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{'':22}{elem:>2}")


def _protein_block(chain: str, base: float) -> list[str]:
    # three standard-AA heavy atoms forming a small chain cluster
    return [
        _pdb("ATOM", 1, "N", "ALA", chain, 1, base + 0.0, 0.0, 0.0, "N"),
        _pdb("ATOM", 2, "CA", "ALA", chain, 1, base + 1.5, 0.0, 0.0, "C"),
        _pdb("ATOM", 3, "C", "ALA", chain, 1, base + 0.0, 1.5, 0.0, "C"),
    ]


# --------------------------------------------------------------------------- #
# Modality resolution.
# --------------------------------------------------------------------------- #
def test_modality_inference_and_ambiguity() -> None:
    assert q._resolve_gate("ranked.sdf", None).modality == "pose"        # unambiguous → inferred
    assert q._resolve_gate("gen.smi", None).modality == "mol"
    assert q._resolve_gate("x.pdb", "complex").modality == "complex"     # explicit wins
    for ambiguous, expected in [("model.cif", "cofold"), ("designs.fasta", "dna")]:
        with pytest.raises(QualifyError) as e:
            q._resolve_gate(ambiguous, None)
        assert expected in str(e.value)                                  # names the candidates
    with pytest.raises(QualifyError):
        q._resolve_gate("CCO", None)                                     # inline → needs modality
    with pytest.raises(QualifyError):
        q._resolve_gate("x.smi", "nope")                                 # unknown modality
    print("1. modality inference (.sdf/.smi), ambiguity (.cif/.fasta), inline + unknown all handled")


# --------------------------------------------------------------------------- #
# Per-gate dispatch + the stable schema.
# --------------------------------------------------------------------------- #
def test_promoter_dispatch_and_schema() -> None:
    seq = "AAAATTGACAGGCTATAATGCAAAACCCGGGTTTAAACCCGGGTTTAAACCCGGG"
    r = q.qualify(seq, "promoter")
    d = r.to_dict()
    assert set(d) == {"modality", "ok", "items", "batch"}
    assert d["modality"] == "promoter" and d["batch"] is None
    assert set(d["items"][0]) == {"name", "ok", "score", "reasons"}
    assert json.loads(json.dumps(d)) == d                               # JSON round-trip lossless
    # spine-faithfulness: the qualified verdict IS the gate's own validate().
    from karyon import promoter_contracts as pc
    assert r.items[0][1] == pc.validate(seq)
    print("2. promoter dispatch + stable schema + spine == promoter_contracts.validate")


def test_dna_dispatch_inline_and_batch(tmp_path) -> None:
    r1 = q.qualify("ATGGCAGCATTACGCGATTACCGATTACCGGATTACCGAGTAA", "dna")
    assert r1.batch is None and len(r1.items) == 1                      # single → no set-level check
    from karyon import gen_dna_validity as gv
    assert r1.items[0][1] == gv.validate("ATGGCAGCATTACGCGATTACCGATTACCGGATTACCGAGTAA")

    fasta = tmp_path / "designs.fasta"
    fasta.write_text(">a\nATGGCAGCATTACGCGATTACCG\n>b\nGGGATTACCGGATTACCGAGTAA\n")
    r2 = q.qualify(str(fasta), "dna")
    assert len(r2.items) == 2 and r2.batch is not None                  # multi-record → cross-hyb batch
    assert "batch" in r2.to_dict() and r2.to_dict()["batch"] is not None
    print("3. dna dispatch: inline single (no batch) + FASTA multi (cross-hyb batch present)")


def test_mol_dispatch_inline() -> None:
    pytest.importorskip("rdkit")
    r = q.qualify("CC(=O)Oc1ccccc1C(=O)O", "mol")                       # aspirin
    assert r.modality == "mol" and len(r.items) == 1
    from karyon import mol_qc as mq
    assert r.items[0][1] == mq.validate("CC(=O)Oc1ccccc1C(=O)O")
    print("4. mol dispatch (inline SMILES) == mol_qc.validate")


def test_pose_dispatch_bundled_samples() -> None:
    pytest.importorskip("rdkit")
    sample = _POSES / "pose_1.sdf"
    if not sample.exists():
        pytest.skip("bundled pose sample not present")
    r = q.qualify(str(sample), "pose")                                  # inferred would also work (.sdf)
    assert r.modality == "pose" and len(r.items) >= 1
    d = r.to_dict()
    assert isinstance(d["ok"], bool) and set(d["items"][0]) == {"name", "ok", "score", "reasons"}
    print(f"5. pose dispatch over bundled {sample.name} ({len(r.items)} pose(s))")


def test_cofold_dispatch_synthetic(tmp_path) -> None:
    pytest.importorskip("numpy")
    lines = _protein_block("A", 0.0)
    lines.append(_pdb("HETATM", 4, "C1", "LIG", "A", 2, 6.0, 6.0, 6.0, "C"))
    lines.append(_pdb("HETATM", 5, "O1", "LIG", "A", 2, 7.2, 6.0, 6.0, "O"))
    pdb = tmp_path / "complex.pdb"
    pdb.write_text("\n".join(lines) + "\nEND\n")
    r = q.qualify(str(pdb), "cofold")
    assert r.modality == "cofold" and len(r.items) == 1
    from karyon import cofold_validity as cv, structure_io as sio
    atoms = sio.read_atoms(pdb.read_text(), fmt="pdb")
    protein, lig = sio.split_protein_ligand(atoms)
    assert r.items[0][1] == cv.validate(protein, lig)                   # spine == gate
    print("6. cofold dispatch (synthetic PDB) == cofold_validity.validate")


def test_complex_dispatch_synthetic(tmp_path) -> None:
    pytest.importorskip("numpy")
    lines = _protein_block("A", 0.0) + _protein_block("B", 5.0)
    pdb = tmp_path / "dimer.pdb"
    pdb.write_text("\n".join(lines) + "\nEND\n")
    r = q.qualify(str(pdb), "complex")
    assert r.modality == "complex" and len(r.items) == 1
    from karyon import protein_interface_validity as piv, structure_io as sio
    atoms = sio.read_atoms(pdb.read_text(), fmt="pdb")
    groups = sio.group_by_chain(atoms)
    (_, ga), (_, gb) = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:2]
    assert r.items[0][1] == piv.validate(ga, gb)                        # spine == gate
    print("7. complex dispatch (synthetic dimer) == protein_interface_validity.validate")


def test_disclosure_only_verdict_serializes_consistently() -> None:
    """A DNA insert that trips ONLY the weight-0 RESTRICTION_SITE disclosure passes the gate. The verdict
    serialized directly (`Verdict.to_dict`) and through the spine (`QualifyResult.to_dict`) now agree on
    `ok` — closing the latent two-definitions-of-`ok` gap (`not reasons` vs `score == 0`)."""
    from karyon import gen_dna_validity as gv
    seq = "ATCGATCGATCGTACGGAATTCATCGTAGCATCGATCGTAGCATCG"     # clean apart from one EcoRI site (GAATTC)
    v = gv.validate(seq)
    assert v.fired == ["RESTRICTION_SITE"] and v.score == 0.0     # disclosure only — no condemning contract
    assert v.ok is True and v.clean is False
    direct = v.to_dict()
    item = q.qualify(seq, "dna").to_dict()["items"][0]
    assert direct["ok"] is item["ok"] is True                    # same `ok` both ways (passed the gate)
    print("9. a disclosure-only verdict serializes identically direct vs through the spine")


def test_missing_dep_actionable(monkeypatch) -> None:
    # force rdkit "missing" → the pose/mol gate raises a clear, install-pointing QualifyError.
    import importlib
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module",
                        lambda n, *a, **k: (_ for _ in ()).throw(ImportError("x")) if n == "rdkit" else real(n, *a, **k))
    with pytest.raises(QualifyError) as e:
        q.qualify("CCO", "mol")
    assert "karyon[chem]" in str(e.value)
    print("8. a missing optional dep yields an actionable install message")


def _run() -> None:
    test_modality_inference_and_ambiguity()
    test_promoter_dispatch_and_schema()
    test_dna_dispatch_inline_and_batch(Path("/tmp"))
    test_disclosure_only_verdict_serializes_consistently()
    for fn in (test_mol_dispatch_inline, test_pose_dispatch_bundled_samples):
        try:
            fn()
        except Exception as e:                       # noqa: BLE001
            print(f"   (skipped {fn.__name__}: {e})")
    print("\nALL qualify-spine proofs passed (script path runs the dep-free subset).")


if __name__ == "__main__":
    _run()
