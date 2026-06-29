"""test_repair — the agent self-repair loop converges, and each fix is targeted by the NAMED reason.

The DNA path is pure stdlib; the molecule path is rdkit-gated (SKIP without `karyon[chem]`).
"""

from __future__ import annotations

import pytest

from karyon import (Agent, DnaRepairAgent, DnaSpec, MolRepairAgent, MolSpec,
                    RepairTrajectory, qualify, repair_loop)
from karyon import gen_dna_validity as gv
from karyon.repair import _rng_for


def _fired(seq: str) -> set[str]:
    return set(gv.validate(seq).fired)


def _clean_dna_base() -> str:
    """A genuinely clean DNA sequence (no contract fires at all) — built by the loop, then asserted clean."""
    seq = repair_loop(DnaSpec(seed=7), DnaRepairAgent(), "dna",
                      clear_disclosures=("RESTRICTION_SITE", "POLY_G_RUN"), max_rounds=20).final
    assert not _fired(seq), f"expected a clean base, got {sorted(_fired(seq))}"
    return seq


# --------------------------------------------------------------------------- #
# DNA — convergence + targeted reason→fix.
# --------------------------------------------------------------------------- #
def test_dna_loop_converges_and_final_is_clean():
    traj = repair_loop(DnaSpec(seed=1), DnaRepairAgent(), "dna",
                       clear_disclosures=("RESTRICTION_SITE",))
    assert traj.converged
    v = qualify(traj.final, modality="dna").items[0][1]
    assert v.score == 0.0                                    # no condemning contract
    assert "RESTRICTION_SITE" not in v.fired                 # the demanded disclosure was cleared too
    assert traj.rounds >= 2                                  # the rough draft really needed work


def test_dna_revise_targets_the_homopolymer():
    base = _clean_dna_base()
    seq = base[:30] + "A" * 15 + base[45:]                   # inject one named defect
    fired = _fired(seq)
    assert "HOMOPOLYMER_RUN" in fired
    assert DnaRepairAgent()._PRIORITY[1] == "HOMOPOLYMER_RUN"   # it is the top-priority defect present
    fixed, action = DnaRepairAgent().revise(seq, gv.validate(seq), DnaSpec())
    assert "HOMOPOLYMER_RUN" not in _fired(fixed)            # the reason was cleared
    assert "homopolymer" in action.lower()


def test_dna_revise_targets_the_restriction_site():
    base = _clean_dna_base()
    seq = base[:30] + "GAATTC" + base[36:]                   # an EcoRI site on an otherwise clean seq
    assert _fired(seq) == {"RESTRICTION_SITE"}               # the ONLY thing wrong
    fixed, action = DnaRepairAgent().revise(seq, gv.validate(seq), DnaSpec())
    assert "GAATTC" not in fixed and "RESTRICTION_SITE" not in _fired(fixed)
    assert "EcoRI" in action


def test_dna_revise_rebalances_gc():
    seq = "CAAATAAAT" * 8                                    # ~11% GC, runs ≤3, no GC-pairing hairpin (no G)
    fired = _fired(seq)
    assert "GC_OUT_OF_BAND" in fired and "HOMOPOLYMER_RUN" not in fired and "STRONG_HAIRPIN" not in fired
    fixed, action = DnaRepairAgent().revise(seq, gv.validate(seq), DnaSpec())
    assert "GC_OUT_OF_BAND" not in _fired(fixed)
    assert "GC" in action


def test_clear_disclosures_controls_convergence():
    # Without demanding it, the disclosed RESTRICTION_SITE never blocks convergence (it's weight 0).
    plain = repair_loop(DnaSpec(seed=3), DnaRepairAgent(), "dna")
    assert plain.converged
    # Demanding it makes the loop keep going until no restriction site remains.
    strict = repair_loop(DnaSpec(seed=3), DnaRepairAgent(), "dna",
                         clear_disclosures=("RESTRICTION_SITE",))
    assert strict.converged
    assert "RESTRICTION_SITE" not in _fired(strict.final)


def test_non_regression_clean_artifact_converges_immediately():
    clean = _clean_dna_base()

    class _Echo:
        def propose(self, spec):
            return clean
        def revise(self, artifact, verdict, spec):
            raise AssertionError("revise must not be called on a clean artifact")

    traj = repair_loop(None, _Echo(), "dna", clear_disclosures=("RESTRICTION_SITE", "POLY_G_RUN"))
    assert traj.converged and traj.rounds == 1 and traj.steps[0].action == ""


def test_budget_exhausted_reports_honestly():
    traj = repair_loop(DnaSpec(seed=1), DnaRepairAgent(), "dna", max_rounds=0)
    assert not traj.converged
    assert traj.steps[-1].action == "(budget exhausted)"


def test_trajectory_to_dict_schema():
    traj = repair_loop(DnaSpec(seed=2), DnaRepairAgent(), "dna")
    d = traj.to_dict()
    assert set(d) == {"modality", "converged", "rounds", "final", "steps"}
    s0 = d["steps"][0]
    assert set(s0) == {"round", "artifact", "ok", "score", "reasons", "action"}
    assert all(set(r) == {"contract", "message", "weight"} for r in s0["reasons"])


def test_agents_satisfy_the_protocol():
    assert isinstance(DnaRepairAgent(), Agent)
    assert isinstance(MolRepairAgent(), Agent)


def test_determinism():
    a = repair_loop(DnaSpec(seed=5), DnaRepairAgent(), "dna", clear_disclosures=("RESTRICTION_SITE",))
    b = repair_loop(DnaSpec(seed=5), DnaRepairAgent(), "dna", clear_disclosures=("RESTRICTION_SITE",))
    assert a.final == b.final and a.rounds == b.rounds


# --------------------------------------------------------------------------- #
# Molecule — rdkit-gated.
# --------------------------------------------------------------------------- #
def _need_rdkit():
    try:
        import rdkit  # noqa: F401
    except Exception:
        pytest.skip("rdkit absent (pip install 'karyon[chem]')")


def test_mol_loop_converges_to_a_passing_molecule():
    _need_rdkit()
    traj = repair_loop(MolSpec(seed=0), MolRepairAgent(), "mol")
    assert traj.converged
    assert qualify(traj.final, modality="mol").ok                 # the final molecule passes the gate
    assert traj.steps[0].score > 0.0                              # …and the draft genuinely did not


def test_mol_revise_cites_the_cleared_reason():
    _need_rdkit()
    agent = MolRepairAgent()
    bad = agent.propose(MolSpec(seed=0))
    v = qualify(bad, modality="mol").items[0][1]
    assert v.score > 0.0
    fixed, action = agent.revise(bad, v, MolSpec(seed=0))
    assert qualify(fixed, modality="mol").ok
    assert "cleared" in action
