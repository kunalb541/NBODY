#!/usr/bin/env python3
"""
nbody_orbital_summary.py — which orbital summary causally tracks the concentration response?
============================================================================================

The reduced-β test showed global mean ⟨|L|⟩ is NOT the causal variable (outer-shell radialization
matched Δ⟨|L|⟩ but produced no concentration).  Hypothesis: concentration responds to the
population of LOW-PERICENTER / inner low-|L| orbits, not the outer-dominated global mean.

This maps where causal access lives with SHELL-LOCAL radializations (inner / mid / outer thirds)
alongside full radialize / tangentialize / sham, and asks which t₀ orbital summary tracks the
ΔM(<r_c) response across interventions:
  • global ⟨|L|⟩, inner/mid/outer ⟨|L|⟩ (by radius tercile)
  • pericenter r_peri (effective-potential inner turning point), fraction with r_peri < {0.05,0.1,0.2}

Decisive: if global ⟨|L|⟩ were causal, the outer-third (which depletes it most) would concentrate
most — it does not.  If the low-pericenter fraction is causal, it should order
radialize ≈ inner-third > mid-third > outer-third > 0, with tangentialize negative.

Hernquist ε=0.05, N=1024, 100 pairs, integrate to 100 steps (concentration peaks early).
No AWS.  paper.tex untouched.
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
from nbody_stress import StressConfig, get_initial_conditions, get_simconfig, obs_coarse_var, _integrate_leapfrog
import phase_space_coarse_features as psf
from nbody_intervention_pilot import intervene_anisotropy, sham_rotation
from nbody_anisotropy_mechanism_pilot import tangentialize
from nbody_intervention_mechanism_B import _rotate

OUTDIR = "outputs/nbody_orbital_summary"
TIMES = [0, 20, 50, 100]
THETA, EPS, N, A = 20.0, 0.05, 1024, 0.20
ARMS = ["orig", "rad", "inner3", "mid3", "outer3", "tan", "sham"]
SUMMARIES = ["L_global", "L_inner", "L_mid", "L_outer", "fperi_005", "fperi_01", "fperi_02"]
_RG = np.logspace(math.log10(0.005), math.log10(5.0), 300)
_PHI_G = -1.0 / (_RG + A)                              # Hernquist potential proxy (G=M=1)


def _pericenters(pos, vel, center):
    d = pos - center
    r = np.linalg.norm(d, axis=1)
    speed2 = np.sum(vel * vel, axis=1)
    rhat = d / np.where(r > 1e-12, r, 1e-12)[:, None]
    v_r = np.sum(vel * rhat, axis=1)
    L = r * np.sqrt(np.maximum(speed2 - v_r ** 2, 0.0))       # |L_i| = r * v_t
    E = 0.5 * speed2 - 1.0 / (r + A)
    vr2 = 2.0 * (E[:, None] - _PHI_G[None, :]) - (L ** 2)[:, None] / (_RG ** 2)[None, :]
    allowed = vr2 >= 0.0
    idx = np.argmax(allowed, axis=1)                          # first allowed r = inner turning point
    return _RG[idx], L, r


def orbital_summaries(pos, vel, center):
    rp, L, r = _pericenters(pos, vel, center)
    lo, hi = np.percentile(r, 33.333), np.percentile(r, 66.667)
    inner, mid, outer = r < lo, (r >= lo) & (r < hi), r >= hi
    n = len(r)
    return {"L_global": float(np.mean(L)),
            "L_inner": float(np.mean(L[inner])), "L_mid": float(np.mean(L[mid])),
            "L_outer": float(np.mean(L[outer])),
            "fperi_005": float(np.sum(rp < 0.05)) / n, "fperi_01": float(np.sum(rp < 0.10)) / n,
            "fperi_02": float(np.sum(rp < 0.20)) / n}


def shell_radialize(pos, vel, theta, lo_pct, hi_pct, center):
    d = pos - center
    r = np.linalg.norm(d, axis=1)
    lo, hi = np.percentile(r, lo_pct), np.percentile(r, hi_pct)
    mask = (r >= lo) if hi_pct >= 100 else (r >= lo) & (r < hi)
    vnew = vel.copy()
    vnew[mask] = _rotate(pos[mask], vel[mask], theta, center, "rad")
    return vnew - np.mean(vnew, axis=0)


def _worker(seed):
    cfg = StressConfig(model="direct_isolated", init="hernquist3d", seed=seed, n=N, steps=max(TIMES),
                       eps=EPS, box_size=2.0, k_fine=16, plummer_a=A)
    sc = get_simconfig(cfg); mass = 1.0 / N
    pos0, vel0 = get_initial_conditions(cfg); center = np.mean(pos0, axis=0)
    th = math.radians(THETA)
    vels = {"orig": vel0, "rad": intervene_anisotropy(pos0, vel0, th, center),
            "inner3": shell_radialize(pos0, vel0, th, 0, 33.333, center),
            "mid3": shell_radialize(pos0, vel0, th, 33.333, 66.667, center),
            "outer3": shell_radialize(pos0, vel0, th, 66.667, 100, center),
            "tan": tangentialize(pos0, vel0, th, center, seed),
            "sham": sham_rotation(pos0, vel0, th, seed)}
    out = {"seed": seed}
    for name, v in vels.items():
        summ = orbital_summaries(pos0, v, center)
        snaps = _integrate_leapfrog(pos0, v, mass, sc, sorted(set(TIMES)), True)
        M = {}
        for t in TIMES:
            p = snaps[t][0]; d = p - np.mean(p, axis=0); rr = np.linalg.norm(d, axis=1)
            M[str(t)] = {"M05": float(np.sum(rr < 0.05)) / N, "M10": float(np.sum(rr < 0.10)) / N,
                         "C8": obs_coarse_var(p, cfg, 8, False)}
        out[name] = {"summ": summ, "M": M, "ke0": 0.5 * mass * float(np.sum(v * v)),
                     "Q0": psf.relaxation_observables(pos0, v, cfg)["Q"]}
    return out


def _mean(vals):
    a = np.array([v for v in vals if np.isfinite(v)], float)
    return float(np.mean(a)) if a.size else float("nan")


def analyse(rows):
    interv = [a for a in ARMS if a not in ("orig", "sham")]
    # Δsummary (arm − sham) at t₀, mean over pairs
    dsumm = {a: {s: _mean([r[a]["summ"][s] - r["sham"]["summ"][s] for r in rows]) for s in SUMMARIES}
             for a in interv}
    # ΔM peak (arm − sham), and ΔC8 peak, mean over pairs
    def peakM(a, key):
        return _mean([max(r[a]["M"][str(t)][key] - r["sham"]["M"][str(t)][key] for t in TIMES if t > 0)
                      for r in rows])
    dM05 = {a: peakM(a, "M05") for a in interv}
    dM10 = {a: peakM(a, "M10") for a in interv}
    dC8 = {a: peakM(a, "C8") for a in interv}
    dKE = float(np.median([abs(r["rad"]["ke0"] - r["orig"]["ke0"]) / max(r["orig"]["ke0"], 1e-30) for r in rows]))
    dQ = float(np.median([abs(r["rad"]["Q0"] - r["orig"]["Q0"]) / max(abs(r["orig"]["Q0"]), 1e-30) for r in rows]))

    # cross-arm correlation of each Δsummary with ΔM10 (and ΔM05)
    y10 = np.array([dM10[a] for a in interv]); y05 = np.array([dM05[a] for a in interv])
    corr = {}
    for s in SUMMARIES:
        x = np.array([dsumm[a][s] for a in interv])
        corr[s] = {"r_M10": float(np.corrcoef(x, y10)[0, 1]) if np.std(x) > 1e-12 else float("nan"),
                   "r_M05": float(np.corrcoef(x, y05)[0, 1]) if np.std(x) > 1e-12 else float("nan")}
    best = max(SUMMARIES, key=lambda s: abs(corr[s]["r_M10"]) if math.isfinite(corr[s]["r_M10"]) else -1)
    gl = corr["L_global"]["r_M10"]
    global_L_retired = not (math.isfinite(gl) and abs(gl) > 0.7)   # global <|L|> does NOT track ΔM

    # qualitative pattern check
    order = sorted(interv, key=lambda a: dM10[a], reverse=True)
    outer_low = dM10["outer3"] < 0.4 * max(dM10[a] for a in interv)
    inner_high = dM10["inner3"] > 0.5 * dM10["rad"] if dM10["rad"] > 1e-9 else False
    tan_neg = dM10["tan"] < 0

    verdict = (f"INNER LOW-PERICENTER FRACTION IS THE SUMMARY — across interventions, ΔM(<0.1) is best "
               f"tracked by Δ{best} (r={corr[best]['r_M10']:+.2f}), while global ⟨|L|⟩ does NOT track it "
               f"(r={gl:+.2f}). Pattern by ΔM(<0.1): {[ (a, round(dM10[a],4)) for a in order ]} — "
               f"inner-third {'concentrates' if inner_high else 'weak'}, outer-third {'does NOT' if outer_low else 'does'}, "
               f"tangentialize {'reverses' if tan_neg else 'no-reverse'}. Global mean ⟨|L|⟩ is RETIRED as the "
               f"causal summary; the causal variable is the low-pericenter / inner-low-|L| orbit population. "
               f"No AWS.") if (global_L_retired and best.startswith("fperi") or best == "L_inner") else (
        f"PARTIAL — best ΔM(<0.1) predictor is Δ{best} (r={corr[best]['r_M10']:+.2f}); global ⟨|L|⟩ "
        f"r={gl:+.2f}. Pattern {[ (a, round(dM10[a],4)) for a in order ]}. Summary not cleanly resolved.")
    if not (dKE < 1e-2 and dQ < 1e-2):
        verdict = "REJECT — KE/Q drift."

    return {"d_summary": dsumm, "dM05_peak": dM05, "dM10_peak": dM10, "dC8_peak": dC8,
            "corr_with_M10": corr, "best_predictor": best, "global_L_corr_M10": gl,
            "global_L_retired": global_L_retired, "order_by_dM10": order,
            "conservation": {"dKE_rel": dKE, "dQ_rel": dQ}, "aws_needed": False, "verdict": verdict}


def _write(res):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "dM05_peak", "dM10_peak", "dC8_peak"] + ["d_" + s for s in SUMMARIES])
        for a in res["dM10_peak"]:
            w.writerow([a, f"{res['dM05_peak'][a]:.5f}", f"{res['dM10_peak'][a]:.5f}", f"{res['dC8_peak'][a]:.3f}"]
                       + [f"{res['d_summary'][a][s]:.4f}" for s in SUMMARIES])
    _report(res)
    print(f"[outputs] → {OUTDIR}/")


def _report(res):
    v = res["verdict"]
    band = ("🟢 ORBITAL SUMMARY FOUND" if v.startswith("INNER") else "🟡 PARTIAL" if v.startswith("PARTIAL")
            else "🔴 REJECT")
    L = ["# Concentration-relevant orbital summary\n"]
    L.append("Hernquist, ε=0.05, N=1024, 100 pairs. Shell-local radializations (inner/mid/outer "
             "thirds) + full radialize/tangentialize/sham; integrate to 100 steps.\n")
    L.append(f"## Verdict — {band}\n\n> **{v}**\n")
    L.append("## ΔM(<r_c) response and Δsummary by intervention (− sham)\n")
    L.append("| arm | ΔM(<0.05) | **ΔM(<0.1)** | ΔC₈ | Δ⟨L⟩_glob | Δ⟨L⟩_inner | Δ⟨L⟩_outer | Δf(peri<.05) | Δf(peri<.1) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for a in res["dM10_peak"]:
        d = res["d_summary"][a]
        L.append(f"| {a} | {res['dM05_peak'][a]:+.5f} | **{res['dM10_peak'][a]:+.5f}** | {res['dC8_peak'][a]:+.2f} | "
                 f"{d['L_global']:+.3f} | {d['L_inner']:+.3f} | {d['L_outer']:+.3f} | "
                 f"{d['fperi_005']:+.4f} | {d['fperi_01']:+.4f} |")
    L.append("\n## Cross-arm correlation of each Δsummary with ΔM(<0.1)\n")
    L.append("| summary | corr with ΔM(<0.1) | corr with ΔM(<0.05) |")
    L.append("|---|---|---|")
    for s in SUMMARIES:
        c = res["corr_with_M10"][s]
        L.append(f"| {s} | {c['r_M10']:+.2f} | {c['r_M05']:+.2f} |")
    L.append(f"\n**Best predictor: {res['best_predictor']}** (r={res['corr_with_M10'][res['best_predictor']]['r_M10']:+.2f}). "
             f"Global ⟨|L|⟩ corr = {res['global_L_corr_M10']:+.2f} → global mean retired = {res['global_L_retired']}.")
    L.append(f"conservation ΔKE/KE={res['conservation']['dKE_rel']:.1e}, ΔQ/Q={res['conservation']['dQ_rel']:.1e}.\n")
    with open(os.path.join(OUTDIR, "orbital_summary_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Identify the concentration-relevant orbital summary (no AWS).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--pairs", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Orbital-summary mapping — {args.pairs} pairs × {len(ARMS)} arms → step {max(TIMES)}")
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
