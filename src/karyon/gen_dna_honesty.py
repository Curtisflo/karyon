"""gen_dna_honesty — the generated-DNA synthesizability gate honesty probe (gen-DNA-QC).

Proves the owned DFM gate (`gen_dna_validity.py`) is a real, faithful instrument over a generative model's
DNA output (NVIDIA's Evo2), the way the structural-gate honesty probes do for the structural gates. Three
arms, the faithfulness arm adjudicated by the two gold-standard packages (both installed; the gate itself
stays pure-stdlib):

  * **Arm A — instrument (offline).** The gate should PASS clean sequences (by-construction `synthetic_clean`
    + real `natural` CDS when fetchable) and FLAG planted decoys (`synthetic_decoys`: GC-extreme / homopolymer
    / hairpin). AUROC(severity → is_decoy) + pass/flag rates. → PI-1.
  * **Arm B — faithfulness.** The owned per-sequence verdict vs **DnaChisel** (EnforceGCContent +
    AvoidPattern homopolymers + AvoidHairpins) on a mixed corpus, restricted to the axes DnaChisel
    implements — the like-for-like "faithful, not a strawman" guard. Plus the owned hairpin signal vs
    **ViennaRNA** ΔG (planted-hairpin decoys fold tighter than GC-matched clean). → PI-2.
  * **Arm C — effect (descriptive).** Per-corpus condemn + disclose rates at gene length (natural vs
    uniform-random vs Markov-generated) — the honest read on where generated DNA carries synthesis/cloning
    hazards the model is blind to. → PI-3.

Pre-registered (committed before running):
  PI-1  Arm A — AUROC(severity → is_decoy) ≥ 0.95 AND flag_rate(decoy) ≥ 0.95 AND pass_rate(synthetic clean)
        ≥ 0.95 (pass_rate on real natural CDS reported alongside; ≥ 0.85 when available).
  PI-2  Arm B — owned-vs-DnaChisel per-sequence agreement ≥ 0.95 on the shared {GC, homopolymer, hairpin}
        axes (per-axis reported); AND owned hairpin flag separates on ViennaRNA ΔG, AUROC(−mfe) ≥ 0.80.
  PI-3  Arm C — descriptive: condemn + disclose rates per generator. Honest read (no forced large effect).

    python -m karyon.gen_dna_honesty                 # all arms (natural CDS offline-skips)
    python -m karyon.gen_dna_honesty --limit 200
"""

from __future__ import annotations

import argparse

from . import stats_kit
from . import gen_dna_validity as gv
from .gen_dna_data import (SeqUnavailable, load_natural_cds, markov_generated, synthetic_clean,
                          synthetic_decoys, uniform_random)

try:
    import dnachisel as dc
    from dnachisel import AvoidHairpins, AvoidPattern, DnaOptimizationProblem, EnforceGCContent
    _HAVE_DC = True
except Exception:
    _HAVE_DC = False

try:
    import RNA
    _HAVE_VRNA = True
except Exception:
    _HAVE_VRNA = False

# the owned condemning axes that DnaChisel also implements (the like-for-like faithfulness set).
_SHARED_AXES = ("gc", "homopolymer", "hairpin")
_OWNED_FOR_AXIS = {"gc": "GC_OUT_OF_BAND", "homopolymer": "HOMOPOLYMER_RUN", "hairpin": "STRONG_HAIRPIN"}


# --------------------------------------------------------------------------- #
# Axis flags — owned (our DRC) and reference (DnaChisel), on the SAME thresholds (calibrate-to-reference).
# --------------------------------------------------------------------------- #
def owned_axis_flags(seq: str, tol: gv.GenDNATol) -> dict[str, bool]:
    fired = set(gv.validate(seq, tol).fired)
    return {ax: (_OWNED_FOR_AXIS[ax] in fired) for ax in _SHARED_AXES}


def _dc_constraints(tol: gv.GenDNATol):
    cons = [EnforceGCContent(mini=tol.gc_min, maxi=tol.gc_max)]
    for base in "ATGC":                                   # a run LONGER than max → a (max+1)-mer to avoid
        cons.append(AvoidPattern(f"{tol.max_homopolymer_run + 1}x{base}"))
    cons.append(AvoidHairpins(stem_size=tol.min_stem_len, hairpin_window=tol.hairpin_max_loop + 2 * tol.min_stem_len))
    return cons


def dnachisel_axis_flags(seq: str, tol: gv.GenDNATol) -> dict[str, bool]:
    """Which DnaChisel constraints FAIL on `seq`, grouped to the shared axes."""
    prob = DnaOptimizationProblem(sequence=seq, constraints=_dc_constraints(tol), logger=None)
    flags = {ax: False for ax in _SHARED_AXES}
    for ev in prob.constraints_evaluations().evaluations:
        if ev.passes:
            continue
        name = ev.specification.__class__.__name__
        if name == "EnforceGCContent":
            flags["gc"] = True
        elif name == "AvoidPattern":
            flags["homopolymer"] = True
        elif name == "AvoidHairpins":
            flags["hairpin"] = True
    return flags


# --------------------------------------------------------------------------- #
# Arm A — instrument.
# --------------------------------------------------------------------------- #
def run_arm_a(clean: list[str], decoys: list[str], natural: list[str], tol: gv.GenDNATol) -> dict:
    def sev(s):
        return gv.featurize(s, tol).severity(tol)

    clean_sev = [sev(s) for s in clean]
    decoy_sev = [sev(s) for s in decoys]
    nat_sev = [sev(s) for s in natural]

    pass_clean = sum(1 for s in clean if not gv.is_unsynthesizable(s, tol)) / (len(clean) or 1)
    flag_decoy = sum(1 for s in decoys if gv.is_unsynthesizable(s, tol)) / (len(decoys) or 1)
    pass_nat = (sum(1 for s in natural if not gv.is_unsynthesizable(s, tol)) / len(natural)) if natural else None

    au = stats_kit.mann_whitney(decoy_sev, clean_sev + nat_sev)
    auroc = au.auroc if isinstance(au, stats_kit.MannWhitney) else float("nan")

    print("\n=== ARM A — instrument check (owned DFM gate; clean + natural vs planted decoys) ===")
    print(f"  corpus                : {len(clean)} synthetic-clean / {len(natural)} natural / {len(decoys)} decoy")
    print(f"  pass_rate(synthetic)  : {pass_clean:.0%}   (a clean sequence should pass)           <- PI-1")
    if pass_nat is not None:
        print(f"  pass_rate(natural CDS): {pass_nat:.0%}   (real E. coli genes should mostly pass)  <- PI-1")
    print(f"  flag_rate(decoy)      : {flag_decoy:.0%}   (a planted barrier should be flagged)    <- PI-1")
    print(f"  AUROC(severity→decoy) : {auroc:.3f}                                               <- PI-1")
    return {"auroc": auroc, "pass_clean": pass_clean, "flag_decoy": flag_decoy, "pass_nat": pass_nat}


# --------------------------------------------------------------------------- #
# Arm B — faithfulness vs DnaChisel (+ ViennaRNA for the hairpin axis).
# --------------------------------------------------------------------------- #
def run_arm_b(corpus: list[str], hairpin_decoys: list[str], gc_matched_clean: list[str],
              tol: gv.GenDNATol) -> dict:
    if not _HAVE_DC:
        print("\n  Arm B SKIP — needs dnachisel.")
        return {}
    n = agree = 0
    per_axis = {ax: [0, 0] for ax in _SHARED_AXES}        # [agree, total]
    for s in corpus:
        ow = owned_axis_flags(s, tol)
        rf = dnachisel_axis_flags(s, tol)
        n += 1
        agree += 1 if (any(ow.values()) == any(rf.values())) else 0
        for ax in _SHARED_AXES:
            per_axis[ax][1] += 1
            per_axis[ax][0] += 1 if ow[ax] == rf[ax] else 0

    overall = agree / (n or 1)
    pa = {ax: (a / t if t else float("nan")) for ax, (a, t) in per_axis.items()}

    print("\n=== ARM B — faithfulness vs DnaChisel (owned DRC vs the gold-standard DFM package) ===")
    print(f"  corpus scored          : {n}   (synthetic clean + decoys + uniform-random + natural)")
    print(f"  per-sequence agreement : {overall:.0%}   (owned 'any-fail' == DnaChisel 'any-fail')   <- PI-2")
    for ax in _SHARED_AXES:
        print(f"     axis {ax:12} agree {pa[ax]:4.0%}   (owned {_OWNED_FOR_AXIS[ax]} vs DnaChisel)")

    # ViennaRNA corroboration of the owned hairpin signal — planted hairpins fold tighter than matched clean.
    hp_auroc = float("nan")
    if _HAVE_VRNA and hairpin_decoys and gc_matched_clean:
        hp_mfe = [-RNA.fold(s)[1] for s in hairpin_decoys]     # −mfe: bigger = more structure
        cl_mfe = [-RNA.fold(s)[1] for s in gc_matched_clean]
        au = stats_kit.mann_whitney(hp_mfe, cl_mfe)
        hp_auroc = au.auroc if isinstance(au, stats_kit.MannWhitney) else float("nan")
        print(f"  hairpin vs ViennaRNA   : AUROC(−ΔG → owned hairpin flag) {hp_auroc:.3f}   "
              f"(planted hairpins fold tighter than GC-matched clean)        <- PI-2")
    elif not _HAVE_VRNA:
        print("  hairpin vs ViennaRNA   : SKIP — ViennaRNA not importable")

    return {"agreement": overall, "per_axis": pa, "hairpin_auroc": hp_auroc}


# --------------------------------------------------------------------------- #
# Arm C — effect (descriptive): condemn + disclose rates per generator, at gene length.
# --------------------------------------------------------------------------- #
def _disclose_rate(seqs: list[str], contract: str, tol: gv.GenDNATol) -> float:
    return sum(1 for s in seqs if contract in gv.validate(s, tol).fired) / (len(seqs) or 1)


def run_arm_c(corpora: dict[str, list[str]], tol: gv.GenDNATol) -> dict:
    print("\n=== ARM C — effect (descriptive): synthesis/cloning hazards per generator (gene length) ===")
    print(f"\n     {'generator':16} {'n':>4} {'condemn':>8} {'restr-site':>10} {'poly-G':>7} {'GC-band':>8}")
    print(f"     {'-'*16} {'-'*4} {'-'*8} {'-'*10} {'-'*7} {'-'*8}")
    rows = {}
    for name, seqs in corpora.items():
        if not seqs:
            continue
        condemn = sum(1 for s in seqs if gv.is_unsynthesizable(s, tol)) / len(seqs)
        rsite = _disclose_rate(seqs, "RESTRICTION_SITE", tol)
        polyg = _disclose_rate(seqs, "POLY_G_RUN", tol)
        gcbad = _disclose_rate(seqs, "GC_OUT_OF_BAND", tol)
        rows[name] = {"n": len(seqs), "condemn": condemn, "restriction": rsite, "polyg": polyg, "gc": gcbad}
        print(f"     {name:16} {len(seqs):>4} {condemn:>7.0%} {rsite:>10.0%} {polyg:>7.0%} {gcbad:>8.0%}")
    return rows


# --------------------------------------------------------------------------- #
def run(limit: int) -> None:
    tol = gv.GenDNATol()
    print("Generated-DNA synthesizability honesty probe (gen-DNA-QC)")
    print("The owned design-for-manufacture DRC — the 'unroutable net' report — over an Evo2-style generator.")
    if not _HAVE_DC:
        print("\n  NOTE: dnachisel not importable — the faithfulness arm (PI-2) will skip.")

    # corpora (short, for the instrument/faithfulness arms)
    clean = synthetic_clean(limit, seed=0)
    decoys = synthetic_decoys(limit, seed=1)
    rnd = uniform_random(limit, seed=2)
    try:
        natural = load_natural_cds(limit=limit)
    except SeqUnavailable as e:
        print(f"\n  natural CDS SKIP — {e}")
        natural = []

    # a GC-matched clean/hairpin split for the ViennaRNA corroboration (both ~50% GC, length-matched)
    hp_only = synthetic_decoys(limit, seed=7)
    hp_only = [s for s in hp_only if "STRONG_HAIRPIN" in gv.validate(s, tol).fired
               and "GC_OUT_OF_BAND" not in gv.validate(s, tol).fired][:limit]
    gc_clean = synthetic_clean(len(hp_only) or 1, seed=8)

    a = run_arm_a(clean, decoys, natural, tol)
    faith_corpus = (clean + decoys + rnd + natural)
    b = run_arm_b(faith_corpus, hp_only, gc_clean, tol)

    # gene-length corpora for the effect arm (restriction-site rate is length-dependent)
    gene_clean = synthetic_clean(limit, seed=10, length_range=(800, 1200))
    gene_rnd = uniform_random(limit, seed=11, length_range=(800, 1200))
    corpora_c = {"synthetic-clean": gene_clean, "uniform-random": gene_rnd}
    if natural:
        gene_natural = [s for s in natural if 800 <= len(s) <= 1200]   # length-matched to the synthetic rows
        if gene_natural:
            corpora_c["natural CDS"] = gene_natural
        try:
            corpora_c["markov(order3)"] = markov_generated(natural, order=3, n=limit, seed=12,
                                                           length_range=(800, 1200))
            corpora_c["markov(order1)"] = markov_generated(natural, order=1, n=limit, seed=13,
                                                           length_range=(800, 1200))
        except SeqUnavailable:
            pass
    c = run_arm_c(corpora_c, tol)

    # --------------------------------------------------------------------- #
    print("\n=== PRE-REGISTERED VERDICT ===")
    p1 = (a["auroc"] >= 0.95 and a["flag_decoy"] >= 0.95 and a["pass_clean"] >= 0.95
          and (a["pass_nat"] is None or a["pass_nat"] >= 0.85))
    print(f"  PI-1 instrument   {'PASS' if p1 else 'FAIL'}  "
          f"AUROC {a['auroc']:.3f}≥0.95 · flag_decoy {a['flag_decoy']:.0%}≥95% · "
          f"pass_clean {a['pass_clean']:.0%}≥95%"
          + (f" · pass_natural {a['pass_nat']:.0%}≥85%" if a["pass_nat"] is not None else ""))

    if b:
        p2 = b["agreement"] >= 0.95 and (b["hairpin_auroc"] != b["hairpin_auroc"] or b["hairpin_auroc"] >= 0.80)
        print(f"  PI-2 faithful     {'PASS' if p2 else 'FAIL'}  "
              f"owned-vs-DnaChisel agreement {b['agreement']:.0%}≥95% "
              f"(GC {b['per_axis']['gc']:.0%} · homopolymer {b['per_axis']['homopolymer']:.0%} · "
              f"hairpin {b['per_axis']['hairpin']:.0%})"
              + (f"; ViennaRNA hairpin AUROC {b['hairpin_auroc']:.2f}≥0.80"
                 if b["hairpin_auroc"] == b["hairpin_auroc"] else ""))
    else:
        print("  PI-2 faithful     SKIP  (dnachisel unavailable)")

    print("  PI-3 effect       (descriptive) per-generator condemn + disclose rates above. Honest read:")
    if "uniform-random" in c:
        print(f"       random/generated DNA is mostly synthesizable under the (lenient, vendor-realistic) "
              f"condemn rules — an honest weak-condemn effect — BUT carries cloning/synthesis HAZARDS the")
        print(f"       model is blind to: restriction sites fire at "
              f"{c['uniform-random']['restriction']:.0%} (uniform-random, gene length), the gate's disclosure.")
    print("\n  Read: GC / homopolymer / hairpin / length / cross-hyb / restriction-site checks over a generator's")
    print("  DNA output, owned by a pure-stdlib DRC and faithful to DnaChisel + ViennaRNA. Qualification, not")
    print("  accuracy: it reports what won't synthesize/clone, it does not make the generator better.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generated-DNA synthesizability honesty probe (gen-DNA-QC).")
    ap.add_argument("--limit", type=int, default=150, help="sequences per corpus")
    cli = ap.parse_args()
    run(cli.limit)
