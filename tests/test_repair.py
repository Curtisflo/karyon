"""test_repair — the agent self-repair loop converges, and each fix is targeted by the NAMED reason.

The DNA path is pure stdlib; the molecule path is rdkit-gated (SKIP without `karyon[chem]`).
"""

from __future__ import annotations

import pytest

from karyon import (Agent, AntibodyRepairAgent, AntibodySpec, DnaRepairAgent, DnaSpec,
                    MolRepairAgent, MolSpec, RepairTrajectory, qualify, repair_loop)
from karyon import antibody_developability as ab
from karyon import gen_dna_validity as gv
from karyon.repair import _rng_for


def _fired(seq: str) -> set[str]:
    return set(gv.validate(seq).fired)


def _phi(step, clear: tuple[str, ...]) -> tuple[float, int]:
    """The loop's lexicographic potential (Σ condemning weight, demanded disclosures firing), reconstructed
    from a recorded step — used to assert strict per-round descent for the reference agent."""
    fired = {r.contract for r in step.reasons}
    return (step.score, sum(1 for c in clear if c in fired))


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
    assert traj.stop_reason == "budget"
    assert traj.steps[-1].action == "(budget exhausted)"


def test_rejects_degenerate_params():
    """`stall_window=0` would declare 'stalled' at round 0 regardless of the agent (it sets a new best, then
    `0 >= 0` fires); a negative `max_rounds` yields an empty trajectory. Both are rejected with an actionable
    error rather than silently mis-classifying. `max_rounds=0` stays valid (qualify-only, no edits)."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="stall_window"):
            repair_loop(DnaSpec(seed=1), DnaRepairAgent(), "dna", stall_window=bad)
    with pytest.raises(ValueError, match="max_rounds"):
        repair_loop(DnaSpec(seed=1), DnaRepairAgent(), "dna", max_rounds=-1)


def test_trajectory_to_dict_schema():
    traj = repair_loop(DnaSpec(seed=2), DnaRepairAgent(), "dna")
    d = traj.to_dict()
    assert set(d) == {"modality", "converged", "stop_reason", "rounds", "final", "steps"}
    assert d["stop_reason"] == "converged" and d["converged"] is True
    s0 = d["steps"][0]
    assert set(s0) == {"round", "artifact", "ok", "score", "reasons", "action"}
    assert all(set(r) == {"contract", "message", "weight"} for r in s0["reasons"])


def test_agents_satisfy_the_protocol():
    assert isinstance(DnaRepairAgent(), Agent)
    assert isinstance(MolRepairAgent(), Agent)
    assert isinstance(AntibodyRepairAgent(), Agent)


# --------------------------------------------------------------------------- #
# Antibody — convergence + targeted, residue-class-preserving reason→fix (pure stdlib).
# --------------------------------------------------------------------------- #
def test_antibody_loop_converges_to_a_developable_fv():
    traj = repair_loop(AntibodySpec(), AntibodyRepairAgent(), "antibody",
                       clear_disclosures=AntibodyRepairAgent.DEMANDED, max_rounds=12)
    assert traj.converged
    v = qualify(traj.final, modality="antibody").items[0][1]
    assert v.score == 0.0                                            # no condemning contract on the final Fv
    assert not (set(AntibodyRepairAgent.DEMANDED) & set(v.fired))    # the demanded chemistry hotspots were cleared
    assert traj.steps[0].score > 0.0                                 # the planted draft genuinely needed work


def test_antibody_revise_clears_the_unpaired_cysteine():
    agent = AntibodyRepairAgent()
    draft = agent.propose(AntibodySpec())
    v = qualify(draft, modality="antibody").items[0][1]
    assert "UNPAIRED_CYSTEINE" in v.fired                            # the planted free thiol is the top defect
    fixed, action = agent.revise(draft, v, AntibodySpec())
    assert "UNPAIRED_CYSTEINE" not in qualify(fixed, modality="antibody").items[0][1].fired
    assert "Cys" in action


def test_antibody_revise_breaks_the_cdr_sequon():
    h = ab.TRASTUZUMAB_VH.replace("WGGDGFYAMDY", "WGGDNISYAMDY")     # only a CDR N-glyc sequon
    artifact = f"{h}:{ab.TRASTUZUMAB_VL}"
    v = qualify(artifact, modality="antibody").items[0][1]
    assert v.fired and "N_GLYCOSYLATION_SEQUON_CDR" in v.fired
    fixed, action = AntibodyRepairAgent().revise(artifact, v, AntibodySpec())
    assert "N_GLYCOSYLATION_SEQUON_CDR" not in qualify(fixed, modality="antibody").items[0][1].fired
    assert "sequon" in action


def test_antibody_repair_is_deterministic():
    a = repair_loop(AntibodySpec(), AntibodyRepairAgent(), "antibody",
                    clear_disclosures=AntibodyRepairAgent.DEMANDED)
    b = repair_loop(AntibodySpec(), AntibodyRepairAgent(), "antibody",
                    clear_disclosures=AntibodyRepairAgent.DEMANDED)
    assert a.final == b.final and a.rounds == b.rounds


def test_determinism():
    a = repair_loop(DnaSpec(seed=5), DnaRepairAgent(), "dna", clear_disclosures=("RESTRICTION_SITE",))
    b = repair_loop(DnaSpec(seed=5), DnaRepairAgent(), "dna", clear_disclosures=("RESTRICTION_SITE",))
    assert a.final == b.final and a.rounds == b.rounds


# --------------------------------------------------------------------------- #
# DNA — provable descent + honest non-convergence (the convergence-guarantee work).
# --------------------------------------------------------------------------- #
def test_potential_strictly_decreases_each_round():
    """The provable-descent property: every defect-safe round strictly lowers the lexicographic potential
    (Σ condemning weight, then demanded disclosures firing), across many seeds — so the loop converges by a
    well-founded descent, not the magnitude accident the un-guarded loop relied on."""
    clear = ("RESTRICTION_SITE",)
    for seed in range(30):
        traj = repair_loop(DnaSpec(seed=seed), DnaRepairAgent(), "dna", clear_disclosures=clear)
        assert traj.converged, f"seed {seed}: stopped {traj.stop_reason}"
        phis = [_phi(s, clear) for s in traj.steps]
        for a, b in zip(phis, phis[1:]):
            assert b < a, f"seed {seed}: potential not strictly decreasing {a} -> {b}"
        assert phis[-1] == (0.0, 0)


def test_reference_agent_converges_within_bound_all_seeds():
    for seed in range(50):
        traj = repair_loop(DnaSpec(seed=seed), DnaRepairAgent(), "dna",
                           clear_disclosures=("RESTRICTION_SITE",))
        assert traj.converged and traj.rounds <= 8, f"seed {seed}: {traj.stop_reason} in {traj.rounds} rounds"


class _ThrashAgent:
    """Deliberately non-converging: each revise re-emits a DISTINCT but equivalently-broken sequence, so the
    potential never reaches a new best (and no state exactly repeats) — the canonical thrash → STALL."""

    def propose(self, spec):
        return "A" * 30                       # GC 0% (out of band) + a 30-mer homopolymer → score 3.0, no hairpin

    def revise(self, artifact, verdict, spec):
        return "A" * (len(artifact) + 1), "thrash: re-emit an equivalently-broken sequence"


def test_stall_is_detected_and_named():
    traj = repair_loop(None, _ThrashAgent(), "dna", max_rounds=20, stall_window=3)
    assert not traj.converged
    assert traj.stop_reason == "stalled"                      # caught by the potential guard, not the budget
    action = traj.steps[-1].action
    assert "stalled" in action
    assert "GC_OUT_OF_BAND" in action or "HOMOPOLYMER_RUN" in action   # the named, unresolved contracts


class _CycleAgent:
    """Alternates between two distinct failing sequences — the loop revisits a state exactly → CYCLED."""

    def __init__(self):
        self._flip = False

    def propose(self, spec):
        return "A" * 30

    def revise(self, artifact, verdict, spec):
        self._flip = not self._flip
        return ("C" * 30 if self._flip else "A" * 30), "cycle"


def test_cycle_is_detected():
    traj = repair_loop(None, _CycleAgent(), "dna", max_rounds=20)
    assert not traj.converged
    assert traj.stop_reason == "cycled"
    assert "cycle" in traj.steps[-1].action


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
