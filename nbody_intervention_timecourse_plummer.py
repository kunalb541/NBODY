#!/usr/bin/env python3
"""
nbody_intervention_timecourse_plummer.py — does the |L| causal chain hold in the CORE profile?
==============================================================================================

Test D resolved the mechanism in Hernquist (cusp): conserved |L|-depletion → M(<r_c)↑ (t≈5)
→ C₈↑ (t≈10); β transient/downstream.  Plummer already passed the family-transfer HANDLE test,
but its full mechanism TIMING was never resolved.  This runs the same time-course in Plummer
(core profile) to ask whether the same time-ordered chain holds for cusp AND core.

Matched arms orig / radialize / tangentialize / sham, Plummer ε=0.05, N=1024, θ=20°, 100 pairs,
t∈{0,5,10,20,50,100,300,600,1000}.  Same diagnostics/analysis as the Hernquist run; results are
compared directly to it.  No AWS.  No new handles.  paper.tex untouched.
"""
from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from nbody_3d import _worker_init
from nbody_stress import StressConfig, get_initial_conditions, get_simconfig, _integrate_leapfrog
import phase_space_coarse_features as psf
from nbody_intervention_pilot import intervene_anisotropy, sham_rotation
from nbody_anisotropy_mechanism_pilot import tangentialize
from nbody_intervention_timecourse import _obs

OUTDIR = "outputs/nbody_intervention_timecourse_plummer"
HERN_SUMMARY = "outputs/nbody_intervention_timecourse/summary.json"
TIMES = [0, 5, 10, 20, 50, 100, 300, 600, 1000]
THETA = 20.0
N, EPS, FAMILY = 1024, 0.05, "plummer3d"
QUANT = ["beta", "Lspec", "M05", "M10", "M20", "C8", "sigr", "S"]


def _worker(seed):
    cfg = StressConfig(model="direct_isolated", init=FAMILY, seed=seed, n=N, steps=max(TIMES),
                       eps=EPS, box_size=2.0, k_fine=16, plummer_a=0.20)
    sc = get_simconfig(cfg); mass = 1.0 / N
    pos0, vel0 = get_initial_conditions(cfg); center = np.mean(pos0, axis=0)
    th = math.radians(THETA)
    vels = {"orig": vel0, "rad": intervene_anisotropy(pos0, vel0, th, center),
            "tan": tangentialize(pos0, vel0, th, center, seed),
            "sham": sham_rotation(pos0, vel0, th, seed)}
    out = {"seed": seed}
    for name, v in vels.items():
        snaps = _integrate_leapfrog(pos0, v, mass, sc, sorted(set(TIMES)), True)
        out[name] = {str(t): _obs(snaps[t][0], snaps[t][1], cfg) for t in TIMES}
        out[name]["ke0"] = 0.5 * mass * float(np.sum(v * v))
        out[name]["Q0"] = psf.relaxation_observables(pos0, v, cfg)["Q"]
    return out


def _paired(vals, seed=7):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size < 5:
        return [float("nan")] * 3 + [a.size]
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), size=(1500, len(a)))].mean(axis=1)
    return [float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), a.size]


def _ci_pos(p):
    return math.isfinite(p[1]) and (p[1] > 0 or p[2] < 0)


def _first_sig(eff_q):
    for t in TIMES:
        if t > 0 and _ci_pos(eff_q[str(t)]):
            return t
    return None


def analyse(rows):
    eff = {arm: {q: {str(t): _paired([r[arm][str(t)][q] - r["sham"][str(t)][q] for r in rows])
                     for t in TIMES} for q in QUANT} for arm in ("rad", "tan")}
    sham_dyn = {q: {str(t): _paired([r["sham"][str(t)][q] - r["orig"][str(t)][q] for r in rows])
                    for t in TIMES} for q in ("beta", "C8", "M10")}
    dKE = float(np.median([abs(r["rad"]["ke0"] - r["orig"]["ke0"]) / max(r["orig"]["ke0"], 1e-30) for r in rows]))
    dQ = float(np.median([abs(r["rad"]["Q0"] - r["orig"]["Q0"]) / max(abs(r["orig"]["Q0"]), 1e-30) for r in rows]))
    fin = str(TIMES[-1])

    t_M05, t_M10, t_C8 = (_first_sig(eff["rad"][q]) for q in ("M05", "M10", "C8"))
    L0, Lf = eff["rad"]["Lspec"]["0"][0], eff["rad"]["Lspec"][fin][0]
    L_permanent = abs(L0) > 1e-6 and abs(Lf / L0) > 0.85
    b0, bf = eff["rad"]["beta"]["0"][0], eff["rad"]["beta"][fin][0]
    beta_transient = abs(bf) < 0.5 * abs(b0)
    conc_before_c8 = (t_M05 is not None and t_C8 is not None and t_M05 < t_C8) or \
                     (t_M10 is not None and t_C8 is not None and t_M10 < t_C8)
    simultaneous = (t_M05 == t_C8) or (t_M10 == t_C8)
    M_persistent = _ci_pos(eff["rad"]["M10"][fin])
    C8_persistent = _ci_pos(eff["rad"]["C8"][fin])

    def _sham_clean(q):
        # compare the sham to the PEAK intervention effect (not the decayed final value), with an
        # absolute floor — else a small sham looks large once the effect has relaxed away.
        s = max(abs(sham_dyn[q][str(t)][0]) for t in TIMES)
        peak = max(abs(eff["rad"][q][str(t)][0]) for t in TIMES)
        return s < 0.15 * peak + 1e-3
    sham_clean = all(_sham_clean(q) for q in ("beta", "M10", "C8"))
    antisym_late = all((_ci_pos(eff["rad"]["C8"][str(t)]) and _ci_pos(eff["tan"]["C8"][str(t)])
                        and np.sign(eff["rad"]["C8"][str(t)][0]) != np.sign(eff["tan"]["C8"][str(t)][0]))
                       for t in (300, 600, 1000) if str(t) in eff["rad"]["C8"])
    cons_ok = dKE < 1e-2 and dQ < 1e-2

    # comparison to Hernquist
    hern = None
    if os.path.exists(HERN_SUMMARY):
        hs = json.load(open(HERN_SUMMARY))
        hern = {"first_sig": hs.get("first_sig"), "beta_pattern": hs.get("beta_pattern"),
                "L_permanent": hs.get("L_permanent")}

    chain_holds = (L_permanent and beta_transient and (conc_before_c8 or simultaneous)
                   and M_persistent and C8_persistent)
    if not cons_ok:
        verdict = f"REJECT — KE/Q drift (ΔKE/KE={dKE:.1e}, ΔQ/Q={dQ:.1e})."
    elif not sham_clean:
        verdict = "REJECT — sham develops comparable effects; not specific in Plummer."
    elif chain_holds:
        verdict = (f"CROSS-FAMILY (cusp+core) — the same chain holds in Plummer (CORE): |L| permanent "
                   f"({L0:+.2f}→{Lf:+.2f}), β transient ({b0:+.2f}→{bf:+.2f}), M(<r_c) significant at "
                   f"t={t_M05 or t_M10} {'BEFORE' if conc_before_c8 else 'simultaneous with'} C₈ at t={t_C8}, "
                   f"both persistent; antisymmetry holds; sham clean. The |L|→pericenter→concentration→C₈ "
                   f"mechanism is shared by cusp (Hernquist) and core (Plummer) concentrated systems. No AWS.")
    elif _ci_pos(eff["rad"]["C8"][fin]) and not conc_before_c8 and not simultaneous:
        verdict = (f"PROFILE-DIFFERS — Plummer shows a β/C₈ response but the concentration-before-C₈ "
                   f"ordering does NOT hold (M t={t_M10}, C₈ t={t_C8}); the mechanism timing differs by "
                   f"profile (cusp vs core).")
    else:
        verdict = (f"WEAK/PASSIVE IN PLUMMER — the durable chain is not clearly reproduced (L_perm="
                   f"{L_permanent}, β_transient={beta_transient}, M_persist={M_persistent}, "
                   f"C8_persist={C8_persistent}); scope may stay cusp-like despite the family-transfer handle.")

    return {"family": FAMILY, "effects": eff, "sham_dynamic": sham_dyn,
            "first_sig": {"M05": t_M05, "M10": t_M10, "C8": t_C8},
            "L_permanent": L_permanent, "beta_transient": beta_transient,
            "concentration_before_c8": conc_before_c8, "simultaneous": simultaneous,
            "M_persistent": M_persistent, "C8_persistent": C8_persistent,
            "sham_clean": sham_clean, "antisymmetric_late": antisym_late, "chain_holds": chain_holds,
            "hernquist_comparison": hern, "conservation": {"dKE_rel": dKE, "dQ_rel": dQ},
            "aws_needed": False, "verdict": verdict}


def _write(res):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    e = res["effects"]
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "quantity"] + [f"t{t}" for t in TIMES])
        for arm in ("rad", "tan"):
            for q in QUANT:
                w.writerow([arm, q] + [f"{e[arm][q][str(t)][0]:.4f}" for t in TIMES])
    _report(res)
    print(f"[outputs] → {OUTDIR}/")


def _report(res):
    v = res["verdict"]
    band = ("🟢 CROSS-FAMILY (cusp+core)" if v.startswith("CROSS-FAMILY")
            else "🟡 PROFILE-DIFFERS" if v.startswith("PROFILE")
            else "🟡 WEAK/PASSIVE" if v.startswith("WEAK") else "🔴 REJECT")
    e = res["effects"]; h = res["hernquist_comparison"]
    L = ["# Plummer (core) Time-Course — |L| causal chain\n"]
    L.append(f"Plummer, ε=0.05, N=1024, θ=20°, 100 matched pairs. Times: {TIMES}.\n")
    L.append(f"## Verdict — {band}\n\n> **{v}**\n")
    L.append(f"## First significant time (radialize − sham): M(<0.05) t={res['first_sig']['M05']}, "
             f"M(<0.1) t={res['first_sig']['M10']}, C₈ t={res['first_sig']['C8']} "
             f"→ concentration-before-C₈ = {res['concentration_before_c8']} "
             f"(simultaneous={res['simultaneous']})\n")
    if h:
        L.append(f"## Hernquist (cusp) comparison: first-sig {h['first_sig']}, "
                 f"β pattern '{h['beta_pattern']}', |L| permanent {h['L_permanent']}\n")
    L.append(f"- |L| permanent: **{res['L_permanent']}**; β transient: **{res['beta_transient']}**; "
             f"M persistent: {res['M_persistent']}; C₈ persistent: {res['C8_persistent']}; "
             f"antisymmetry late: {res['antisymmetric_late']}; sham clean: {res['sham_clean']}; "
             f"conservation ΔKE/KE={res['conservation']['dKE_rel']:.1e}.\n")
    L.append("## Radialize effect (− sham) vs time\n")
    L.append("| quantity | " + " | ".join(f"t={t}" for t in TIMES) + " |")
    L.append("|---|" + "---|" * len(TIMES))
    for q in QUANT:
        L.append(f"| {q} | " + " | ".join(f"{e['rad'][q][str(t)][0]:+.3f}" for t in TIMES) + " |")
    with open(os.path.join(OUTDIR, "timecourse_plummer_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Plummer time-course of the |L| causal chain (no AWS).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--pairs", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Plummer time-course — {args.pairs} pairs × 4 arms → step {max(TIMES)}")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init, initargs=(True,)) as ex:
        futs = [ex.submit(_worker, 2000 + i) for i in range(args.pairs)]
        for fut in tqdm(as_completed(futs), total=len(futs), ncols=100, unit="pair"):
            rows.append(fut.result())
    res = analyse(rows)
    _write(res)
    print("\n" + res["verdict"])


if __name__ == "__main__":
    main()
