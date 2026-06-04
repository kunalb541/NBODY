#!/usr/bin/env python3
"""
nbody_intervention_betafree_L.py — change |L| with a SMALLER β cost (outer-shell radialization)
===============================================================================================

Highest-value remaining mechanism test: can we drive the concentration response with a handle that
depletes |L| but changes β much LESS than full radialization?

Geometric fact: |L_i| = r_i·v_t is radius-weighted; the outermost particles carry most of ⟨|L|⟩.
So radializing ONLY the outer shell depletes |L| at a smaller GLOBAL β cost (fewer particles
touched), and — unlike Test B — it adds NO inner tangentialization, so it avoids the core-puffing
that reversed C₈ there.  It is NOT β-free (a residual β change remains — the coupling cannot be
fully removed with a velocity handle), but it cleanly varies the |L|/β ratio.

Matched arms: orig / full-radialize / outer-radialize (tuned to match Δ|L| at min Δβ) / sham.
Hernquist ε=0.05, N=1024, 100 pairs, t∈{0,5,10,20,50,100,300,600}.

Decision: if outer-radialize reproduces the concentration response at a much SMALLER Δβ →
evidence favors |L| (β largely dispensable for concentration).  If concentration tracks Δβ
instead → β regains importance.  No AWS.  paper.tex untouched.
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
from nbody_intervention_mechanism_B import _rotate
from nbody_intervention_timecourse import _obs

OUTDIR = "outputs/nbody_intervention_betafree_L"
TIMES = [0, 5, 10, 20, 50, 100, 300, 600]
THETA_FULL, EPS, N = 20.0, 0.05, 1024
QUANT = ["beta", "Lspec", "M05", "M10", "M20", "C8", "sigr"]


def radialize_outer(pos, vel, theta_rad, p, center):
    """Radialize (speed-preserving) only the outermost fraction p of particles by radius."""
    d = pos - center
    r = np.linalg.norm(d, axis=1)
    outer = r >= np.percentile(r, 100.0 * (1.0 - p))
    vnew = vel.copy()
    vnew[outer] = _rotate(pos[outer], vel[outer], theta_rad, center, "rad")
    return vnew - np.mean(vnew, axis=0)


def _beta_L(pos, vel, center):
    d = pos - center
    r = np.linalg.norm(d, axis=1)
    rhat = d / np.where(r > 1e-12, r, 1e-12)[:, None]
    v_r = np.sum(vel * rhat, axis=1)
    v_t = np.linalg.norm(vel - v_r[:, None] * rhat, axis=1)
    sigr = float(np.std(v_r)); sigt = float(np.sqrt(np.mean(v_t ** 2)))
    beta = 1.0 - sigt ** 2 / (2.0 * sigr ** 2) if sigr > 1e-9 else float("nan")
    return beta, float(np.mean(np.linalg.norm(np.cross(d, vel), axis=1)))


def tune(ics):
    th = math.radians(THETA_FULL)
    dL_ref = float(np.mean([(_beta_L(p, intervene_anisotropy(p, v, th, c), c)[1] - _beta_L(p, v, c)[1])
                            for p, v, c in ics]))
    db_ref = float(np.mean([(_beta_L(p, intervene_anisotropy(p, v, th, c), c)[0] - _beta_L(p, v, c)[0])
                            for p, v, c in ics]))
    best = None
    for frac in (0.15, 0.20, 0.25, 0.35, 0.50):
        for thd in (20, 30, 40, 50, 60, 70):
            dbs, dLs = [], []
            for p, v, c in ics:
                b0, L0 = _beta_L(p, v, c)
                b1, L1 = _beta_L(p, radialize_outer(p, v, math.radians(thd), frac, c), c)
                dbs.append(b1 - b0); dLs.append(L1 - L0)
            db, dL = float(np.mean(dbs)), float(np.mean(dLs))
            if abs(dL - dL_ref) <= 0.20 * abs(dL_ref):     # matched |L|
                score = abs(db)                            # minimize β change among matched
                if best is None or score < best["abs_db"]:
                    best = {"frac": frac, "theta": thd, "db": db, "dL": dL, "abs_db": abs(db)}
    if best is None:                                       # fallback: closest |L|
        best = {"frac": 0.25, "theta": 50, "db": float("nan"), "dL": float("nan"), "abs_db": 1e9}
    best.update({"dL_ref": dL_ref, "db_ref": db_ref})
    return best


def _worker(task):
    seed, frac, theta = task
    cfg = StressConfig(model="direct_isolated", init="hernquist3d", seed=seed, n=N, steps=max(TIMES),
                       eps=EPS, box_size=2.0, k_fine=16, plummer_a=0.20)
    sc = get_simconfig(cfg); mass = 1.0 / N
    pos0, vel0 = get_initial_conditions(cfg); center = np.mean(pos0, axis=0)
    vels = {"orig": vel0, "rad": intervene_anisotropy(pos0, vel0, math.radians(THETA_FULL), center),
            "outer": radialize_outer(pos0, vel0, math.radians(theta), frac, center),
            "sham": sham_rotation(pos0, vel0, math.radians(THETA_FULL), seed)}
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


def analyse(rows, tune_info):
    eff = {arm: {q: {str(t): _paired([r[arm][str(t)][q] - r["sham"][str(t)][q] for r in rows])
                     for t in TIMES} for q in QUANT} for arm in ("rad", "outer")}
    dKE = float(np.median([abs(r["outer"]["ke0"] - r["orig"]["ke0"]) / max(r["orig"]["ke0"], 1e-30) for r in rows]))
    dQ = float(np.median([abs(r["outer"]["Q0"] - r["orig"]["Q0"]) / max(abs(r["orig"]["Q0"]), 1e-30) for r in rows]))
    imp = {a: {"beta": eff[a]["beta"]["0"][0], "L": eff[a]["Lspec"]["0"][0]} for a in ("rad", "outer")}
    l_matched = abs(imp["outer"]["L"] - imp["rad"]["L"]) <= 0.25 * abs(imp["rad"]["L"])
    beta_ratio = imp["outer"]["beta"] / imp["rad"]["beta"] if abs(imp["rad"]["beta"]) > 1e-9 else float("nan")

    pk = lambda a, q: max(abs(eff[a][q][str(t)][0]) for t in TIMES)
    m10_repro = pk("outer", "M10") / pk("rad", "M10") if pk("rad", "M10") > 1e-9 else float("nan")
    m05_repro = pk("outer", "M05") / pk("rad", "M05") if pk("rad", "M05") > 1e-9 else float("nan")
    fin = str(TIMES[-1])
    c8_repro = eff["outer"]["C8"][fin][0] / eff["rad"]["C8"][fin][0] if abs(eff["rad"]["C8"][fin][0]) > 1e-9 else float("nan")
    t_M = _first_sig(eff["outer"]["M10"]); t_C8 = _first_sig(eff["outer"]["C8"])
    conc_responds = _ci_pos(eff["outer"]["M10"][str(20)]) or _ci_pos(eff["outer"]["M05"][str(20)])
    c8_no_reversal = np.sign(eff["outer"]["C8"][fin][0]) == np.sign(eff["rad"]["C8"][fin][0])
    cons_ok = dKE < 1e-2 and dQ < 1e-2

    if not cons_ok:
        verdict = "REJECT — KE/Q drift."
    elif not l_matched:
        verdict = f"L NOT MATCHED — outer Δ|L|={imp['outer']['L']:+.2f} vs full {imp['rad']['L']:+.2f}; retune."
    elif beta_ratio < 0.6 and conc_responds and m10_repro > 0.5 and c8_no_reversal:
        verdict = (f"FAVORS |L| — outer-shell radialization depletes |L| as much as full radialization "
                   f"(Δ|L|={imp['outer']['L']:+.2f} vs {imp['rad']['L']:+.2f}) at only {beta_ratio:.0%} of "
                   f"the global β change (Δβ₀={imp['outer']['beta']:+.2f} vs {imp['rad']['beta']:+.2f}), yet "
                   f"reproduces {m10_repro:.0%} of the M(<0.1) and {m05_repro:.0%} of the M(<0.05) "
                   f"concentration response (significant at t={t_M}), C₈ same sign (no reversal). With far "
                   f"LESS β change, the concentration response largely survives → evidence favors |L| as the "
                   f"driver, β as a smaller contributor. (Not β-free: a residual β change remains — the "
                   f"coupling can be reduced but not removed with a velocity handle.) No AWS.")
    elif conc_responds and abs(m10_repro - beta_ratio) < 0.2:
        verdict = (f"TRACKS β — the concentration response scales with the β change "
                   f"(M(<0.1) reproduced {m10_repro:.0%} ≈ β ratio {beta_ratio:.0%}); reducing β reduces the "
                   f"effect proportionally → β regains importance; |L| alone is not sufficient.")
    elif not conc_responds:
        verdict = (f"CONCENTRATION LOST — at reduced β ({beta_ratio:.0%}) the matched-|L| outer handle does "
                   f"NOT produce a significant concentration response; the global anisotropy change mattered.")
    else:
        verdict = (f"MIXED — outer handle (β {beta_ratio:.0%} of full, |L| matched) reproduces "
                   f"{m10_repro:.0%} of concentration; partial — both |L| and β contribute.")

    return {"tune": tune_info, "effects": eff, "imposed": imp, "l_matched": l_matched,
            "beta_ratio_outer_to_full": beta_ratio, "m10_reproduced": m10_repro, "m05_reproduced": m05_repro,
            "c8_reproduced": c8_repro, "c8_no_reversal": bool(c8_no_reversal),
            "outer_first_sig": {"M10": t_M, "C8": t_C8}, "conc_responds": conc_responds,
            "conservation": {"dKE_rel": dKE, "dQ_rel": dQ}, "aws_needed": False, "verdict": verdict}


def _write(res):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    e = res["effects"]
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "quantity"] + [f"t{t}" for t in TIMES])
        for arm in ("rad", "outer"):
            for q in QUANT:
                w.writerow([arm, q] + [f"{e[arm][q][str(t)][0]:.4f}" for t in TIMES])
    _report(res)
    print(f"[outputs] → {OUTDIR}/")


def _report(res):
    v = res["verdict"]
    band = ("🟢 FAVORS |L|" if v.startswith("FAVORS") else "🟡 TRACKS β" if v.startswith("TRACKS")
            else "🟡 MIXED" if v.startswith("MIXED") else "🟠 RETUNE" if v.startswith("L NOT")
            else "🔴 " + v.split(" ")[0])
    e = res["effects"]; im = res["imposed"]; t = res["tune"]
    L = ["# Reduced-β |L| Handle (outer-shell radialization)\n"]
    L.append(f"Hernquist, ε=0.05, N=1024, 100 pairs. Outer handle: radialize outermost {t['frac']:.0%} "
             f"at θ={t['theta']}°. Times: {TIMES}.\n")
    L.append(f"## Verdict — {band}\n\n> **{v}**\n")
    L.append(f"## Construction (t₀)\n- full radialize: Δβ₀={im['rad']['beta']:+.3f}, Δ|L|={im['rad']['L']:+.3f}")
    L.append(f"- **outer radialize: Δβ₀={im['outer']['beta']:+.3f}** "
             f"({res['beta_ratio_outer_to_full']:.0%} of full), **Δ|L|={im['outer']['L']:+.3f}** "
             f"(matched={res['l_matched']})")
    L.append(f"- concentration reproduced: M(<0.1) {res['m10_reproduced']:.0%}, M(<0.05) {res['m05_reproduced']:.0%}; "
             f"C₈ no-reversal={res['c8_no_reversal']}; conservation ΔKE/KE={res['conservation']['dKE_rel']:.1e}\n")
    L.append("## Effect vs time (− sham)\n")
    for arm in ("rad", "outer"):
        L.append(f"### {arm}")
        L.append("| quantity | " + " | ".join(f"t={t_}" for t_ in TIMES) + " |")
        L.append("|---|" + "---|" * len(TIMES))
        for q in QUANT:
            L.append(f"| {q} | " + " | ".join(f"{e[arm][q][str(t_)][0]:+.3f}" for t_ in TIMES) + " |")
        L.append("")
    with open(os.path.join(OUTDIR, "betafree_L_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Reduced-β |L| handle (outer-shell radialization).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--pairs", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    print("tuning outer-shell handle to match Δ|L| at minimal Δβ (t₀ ensemble)...")
    ics = []
    for i in range(args.pairs):
        cfg = StressConfig(model="direct_isolated", init="hernquist3d", seed=2000 + i, n=N, steps=max(TIMES),
                           eps=EPS, box_size=2.0, k_fine=16, plummer_a=0.20)
        p, vv = get_initial_conditions(cfg); ics.append((p, vv, np.mean(p, axis=0)))
    t = tune(ics)
    print(f"  full: Δβ={t['db_ref']:+.2f} Δ|L|={t['dL_ref']:+.2f} | outer({t['frac']:.0%}@{t['theta']}°): "
          f"Δβ={t['db']:+.2f} Δ|L|={t['dL']:+.2f}  (β ratio {t['db']/t['db_ref']:.0%})")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init, initargs=(True,)) as ex:
        futs = [ex.submit(_worker, (2000 + i, t["frac"], t["theta"])) for i in range(args.pairs)]
        for fut in tqdm(as_completed(futs), total=len(futs), ncols=100, unit="pair"):
            rows.append(fut.result())
    res = analyse(rows, t)
    _write(res)
    print("\n" + res["verdict"])


if __name__ == "__main__":
    main()
