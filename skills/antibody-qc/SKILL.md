---
name: antibody-qc
description: >
  Deterministic developability/liability gate for GENERATED antibody Fv sequences (e.g. an AlphaFold-Multimer / RFdiffusion+ProteinMPNN designed binder, or an antibody language model) with karyon. Use AFTER designing an antibody/binder and BEFORE expressing or advancing it — answers "is this Fv developable, or does it carry a liability that makes it undruggable?" with a pass/fail verdict and legible per-reason explanations (an unpaired cysteine, an N-glycosylation sequon in a CDR, an extreme CDR-H3 length or isoelectric point — and, disclosed, the deamidation/isomerization/oxidation hotspots, fragmentation sites, and hydrophobicity/charge proxies even approved antibodies carry). Pure stdlib — no GPU, no network, no numpy. Input is the VH (+VL) amino-acid sequence; single-domain VHH/nanobodies are supported.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "karyon (stdlib only)"
allowed-tools: Bash, Read, Write
---

# antibody-qc — a legible developability gate for designed antibodies

Antibody- and binder-design tools (AlphaFold-Multimer, RFdiffusion + ProteinMPNN, antibody language models)
emit an Fv that scores well on the model's own confidence yet can carry **developability liabilities** the
model never optimized against: a free cysteine that scrambles disulfides, an N-glycosylation sequon in a CDR
that glycosylates the binding site, a chemically labile deamidation/isomerization hotspot in a CDR, an extreme
isoelectric point. These are the manufacturability/stability axis a generative model is blind to.

`antibody-qc` is the programmatic version of that check: a **deterministic design-rule check (DRC)** on the Fv
sequence that emits a pass/fail verdict with a human-readable reason for every flag — the "unroutable net"
report ported to biologics. It is the **complement** to a designer, not a competitor — it qualifies the output,
it does not design antibodies. It is faithful to the **Therapeutic Antibody Profiler** (Raybould et al., *PNAS*
2019) and the clinical-stage developability survey (Jain et al., *PNAS* 2017): the rare/severe liabilities
**fail** the gate, the common chemistry flags that even approved antibodies carry are **disclosed**.

| contract | tier | catches |
|---|---|---|
| `UNPAIRED_CYSTEINE` | **fails** | an odd cysteine count — a free thiol drives disulfide scrambling / aggregation |
| `N_GLYCOSYLATION_SEQUON_CDR` | **fails** | an N-X-[S/T] sequon inside a CDR — variable glycosylation of the binding site |
| `CDR_LENGTH_OUT_OF_RANGE` | **fails** | a CDR-H3 length outside the typical window — a developability/expression outlier |
| `EXTREME_FV_CHARGE` | **fails** | an Fv isoelectric point outside the band — solubility / viscosity / clearance risk |
| `DEAMIDATION_HOTSPOT_CDR` | discloses | Asn deamidation motif (NG/NS) in a CDR — charge heterogeneity on storage |
| `ISOMERIZATION_HOTSPOT_CDR` | discloses | Asp isomerization motif (DG/DS) in a CDR — backbone isomerization |
| `OXIDATION_PRONE_CDR` | discloses | Met/Trp in a CDR — oxidation risk |
| `FRAGMENTATION_DP` | discloses | an acid-labile Asp-Pro bond — low-pH fragmentation |
| `FRAMEWORK_GLYCOSYLATION` | discloses | an N-glyc sequon in framework — usually benign heterogeneity |
| `N_TERMINAL_PYROGLUTAMATE` | discloses | N-terminal Gln/Glu — pyroglutamate (usually benign) |
| `CHARGE_ASYMMETRY` / `HYDROPHOBICITY_HIGH` | discloses | sequence proxies for the structural SFvCSP / PSH TAP metrics |

The verdict separates **disclosure** from **condemnation**: every hazard is *reported*, but only the
disqualifying ones *fail* the Fv — a deamidation hotspot or an N-terminal Gln informs without condemning,
because a real approved antibody carries them too. Thresholds are developability-literature constants (TAP /
Jain) — **zero parameters fitted to accuracy**.

## Install
```bash
pip install karyon          # no extras needed — pure stdlib
```

## Usage
`--modality antibody` is required (a `.fasta` could equally be DNA or a promoter). Provide the heavy (VH) and
light (VL) chains as a 2-record FASTA, an inline `HEAVY:LIGHT` string, or a single VH/VHH chain:
```bash
# An inline Fv (heavy:light):
karyon qualify "EVQLVESGG...:DIQMTQSPS..." --modality antibody

# A designed Fv as a 2-record FASTA (>heavy / >light), e.g. an RFdiffusion+ProteinMPNN binder:
karyon qualify designed_fv.fasta --modality antibody

# A single-domain VHH / nanobody (heavy only):
karyon qualify "EVQLVESGG..." --modality antibody

# JSON verdict for piping into an agent / pipeline:
karyon qualify designed_fv.fasta --modality antibody --json
```

Output is a `PASS` / `FAIL` verdict plus one line per fired contract — a `·` for a disclosed hazard, an `✗`
for a condemning one (e.g. *"N-glycosylation sequon in a CDR (H:NIS@102(H3)) — variable glycosylation of the
binding site"*). Exit code is non-zero on `FAIL` so it gates a pipeline directly. `--json` emits the stable
spine schema — `{modality, ok, items:[...], batch}` — and an Fv passes iff `score == 0`.

From Python:
```python
from karyon import qualify
r = qualify("designed_fv.fasta", modality="antibody")    # or qualify("VH:VL", modality="antibody")
v = r.items[0][1]
print("developable" if v.ok else f"REJECT — {v.messages}")
```

## Self-repair: fix the named liabilities and converge
Because every rejection **names its reason**, an agent can fix it and re-check. `karyon repair -m antibody`
drives a deterministic reference agent that applies the textbook residue-class-preserving developability fixes
(Cys→Ser, break the sequon, Asn→Gln, Asp→Glu) until the gate passes:
```bash
karyon repair -m antibody                          # a self-contained demo (planted liabilities → clean Fv)
python examples/agent_loop/repair_antibody.py      # the same loop, narrated
```
In real use the agent is *your harness* — Claude Code reads the named reason and edits the structure directly.

## Composition with NVIDIA BioNeMo
Install alongside the biologics-design skills (`rfdiffusion`, `proteinmpnn`, AlphaFold-Multimer): the model
designs the binder; this skill gates developability, so the agent only advances Fvs that won't fail on a free
thiol, a CDR glycosite, or an extreme pI — and can explain every rejection. The model proposes, karyon qualifies.

## Scope (honest)
A fast, legible *sequence-determined* developability gate: the chemistry that the sequence fixes (cysteine
pairing, glyc sequons, deamidation/isomerization/oxidation motifs, fragmentation) plus the TAP charge/length
flags and **sequence proxies** for the structural metrics. The true spatial TAP metrics — patches of surface
hydrophobicity / charge (PSH / PPC / PNC) and the structural Fv charge symmetry parameter (SFvCSP) — need a 3D
Fv model and are out of scope here (the CDR-GRAVY and VH/VL-asymmetry flags are coarse stand-ins). CDR
boundaries are located from conserved framework anchors (no ANARCI); when they don't resolve, the CDR-scoped
checks stand down and say so. It does not predict affinity, expression titer, or immunogenicity — pair it with
the generative toolkit for the soft, quantitative axis.
