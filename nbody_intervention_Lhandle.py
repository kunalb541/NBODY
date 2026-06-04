#!/usr/bin/env python3
"""
nbody_intervention_Lhandle.py — non-rotational |L| handle (confirm the resolved mechanism)
=========================================================================================

Test D resolved the mechanism: conserved |L_i|-depletion → smaller pericenters → M(<r_c)↑ (t≈5)
→ C₈↑ (t≈10); β is a transient downstream signature.  This confirms it with a DIFFERENT-FORM
|L| handle (not the fixed-angle radialization rotation):

  v_t-magnitude rescale — reduce the tangential speed v_t → f·v_t (f<1) and raise |v_r| to keep
  the per-particle speed (so KE/E/Q are conserved).  |L_i| = r_i·v_t drops directly; f is tuned
  on the ensemble to match radialization's Δ⟨|L|⟩.

Key comparison: if this different-form |L| depletion reproduces the SAME M(<r_c)-then-C₈ time
ordering — especially if its Δβ DIFFERS from radialization's — the causal variable is |L|, not the
rotation geometry.  β change is reported honestly (this is a non-rotational *L*-confirmation, not a
β-free claim).

Hernquist, ε=0.05, N=1024, 100 matched pairs, t∈{0,5,10,50,100,300,600}.  No AWS.  paper.tex untouched.
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
from nbody_intervention_timecourse import _obs

OUTDIR = "outputs/nbody_intervention_Lhandle"
TIMES = [0, 5, 10, 50, 100, 300, 600]
THETA = 20.0
N, EPS = 1024, 0.05
QUANT = ["beta", "Lspec", "M05", "M10", "M20", "C8", "sigr"]


def vt_scale(pos, vel, f, center):
    """Reduce tangential speed v_t → f·v_t and raise |v_r| to preserve per-particle speed.
    Depletes |L_i| = r·v_t directly; conserves KE (speed) and positions (so initial concentration
    is unchanged — concentration must DEVELOP from the altered orbits)."""
    d = pos - center
    r = np.linalg.norm(d, axis=1)
    rhat = d / np.where(r > 1e-12, r, 1e-12)[:, None]
    v_r = np.sum(vel * rhat, axis=1)
    vt_vec = vel - v_r[:, None] * rhat
    v_t = np.linalg.norm(vt_vec, axis=1)
    speed = np.linalg.norm(vel, axis=1)
    that = vt_vec / np.where(v_t > 1e-12, v_t, 1.0)[:, None]
    v_t_new = f * v_t
    v_r_mag_new = np.sqrt(np.maximum(speed ** 2 - v_t_new ** 2, 0.0))
    sgn = np.where(v_r >= 0, 1.0, -1.0)
    vnew = (sgn * v_r_mag_new)[:, None] * rhat + v_t_new[:, None] * that
    return vnew - np.mean(vnew, axis=0)


def _mean_L(pos, vel, center):
    return float(np.mean(np.linalg.norm(np.cross(pos - center, vel), axis=1)))


def tune_f(ics):
    """Pick f so the v_t-rescale matches radialization's mean Δ⟨|L|⟩."""
    L_orig, dL_rad = [], []
    th = math.radians(THETA)
    for pos, vel, c in ics:
        L0 = _mean_L(pos, vel, c)
        L_orig.append(L0)
        dL_rad.append(_mean_L(pos, intervene_anisotropy(pos, vel, th, c), c) - L0)
    mean_L0, mean_dL = float(np.mean(L_orig)), float(np.mean(dL_rad))
    f = max(0.0, 1.0 + mean_dL / mean_L0)
    # achieved Δ|L| for the v_t-rescale at this f
    dL_vt = [(_mean_L(p, vt_scale(p, v, f, c), c) - _mean_L(p, v, c)) for p, v, c in ics]
    return {"f": f, "mean_L_orig": mean_L0, "dL_rad": mean_dL, "dL_vt": float(np.mean(dL_vt))}


def _worker(task):
    seed, f = task
    cfg = StressConfig(model="direct_isolated", init="hernquist3d", seed=seed, n=N, steps=max(TIMES),
                       eps=EPS, box_size=2.0, k_fine=16, plummer_a=0.20)
    sc = get_simconfig(cfg); mass = 1.0 / N
    pos0, vel0 = get_initial_conditions(cfg); center = np.mean(pos0, axis=0)
    th = math.radians(THETA)
    vels = {"orig": vel0, "rad": intervene_anisotropy(pos0, vel0, th, center),
            "vt": vt_scale(pos0, vel0, f, center), "sham": sham_rotation(pos0, vel0, th, seed)}
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


def analyse(rows, tune):
    eff = {arm: {q: {str(t): _paired([r[arm][str(t)][q] - r["sham"][str(t)][q] for r in rows])
                     for t in TIMES} for q in QUANT} for arm in ("rad", "vt")}
    dKE = float(np.median([abs(r["vt"]["ke0"] - r["orig"]["ke0"]) / max(r["orig"]["ke0"], 1e-30) for r in rows]))
    dQ = float(np.median([abs(r["vt"]["Q0"] - r["orig"]["Q0"]) / max(abs(r["orig"]["Q0"]), 1e-30) for r in rows]))

    # imposed at t0
    imp = {arm: {"beta": eff[arm]["beta"]["0"][0], "L": eff[arm]["Lspec"]["0"][0]} for arm in ("rad", "vt")}
    l_matched = abs(imp["vt"]["L"] - imp["rad"]["L"]) <= 0.25 * abs(imp["rad"]["L"])
    cons_ok = dKE < 1e-2 and dQ < 1e-2

    t_M05 = {a: _first_sig(eff[a]["M05"]) for a in ("rad", "vt")}
    t_M10 = {a: _first_sig(eff[a]["M10"]) for a in ("rad", "vt")}
    t_C8 = {a: _first_sig(eff[a]["C8"]) for a in ("rad", "vt")}
    fin = str(TIMES[-1])
    vt_conc_before_c8 = ((t_M05["vt"] is not None and t_C8["vt"] is not None and t_M05["vt"] < t_C8["vt"])
                         or (t_M10["vt"] is not None and t_C8["vt"] is not None and t_M10["vt"] < t_C8["vt"]))
    vt_reproduces = (_ci_pos(eff["vt"]["M10"][fin]) and _ci_pos(eff["vt"]["C8"][fin])
                     and np.sign(eff["vt"]["M10"][fin][0]) == np.sign(eff["rad"]["M10"][fin][0])
                     and np.sign(eff["vt"]["C8"][fin][0]) == np.sign(eff["rad"]["C8"][fin][0]))
    # β comparison: does v_t-rescale move M/C8 with LESS β than radialize?
    beta_ratio = imp["vt"]["beta"] / imp["rad"]["beta"] if abs(imp["rad"]["beta"]) > 1e-9 else float("nan")
    c8_ratio = eff["vt"]["C8"][fin][0] / eff["rad"]["C8"][fin][0] if abs(eff["rad"]["C8"][fin][0]) > 1e-9 else float("nan")
    less_beta_same_effect = (beta_ratio < 0.8) and (c8_ratio > 0.5)   # less β but comparable C8 → L-driven

    if not cons_ok:
        verdict = f"REJECT (bulk) — KE/Q drift (ΔKE/KE={dKE:.1e}, ΔQ/Q={dQ:.1e})."
    elif not l_matched:
        verdict = (f"L NOT MATCHED — v_t-rescale Δ|L|={imp['vt']['L']:+.2f} vs radialize {imp['rad']['L']:+.2f}; "
                   "retune f before interpreting.")
    elif vt_reproduces and vt_conc_before_c8:
        extra = (f" AND with LESS β (v_t Δβ₀={imp['vt']['beta']:+.2f} = {beta_ratio:.0%} of radialize's "
                 f"{imp['rad']['beta']:+.2f}, yet {c8_ratio:.0%} of its C₈) → the effect tracks |L|, "
                 f"not the rotation/β." if less_beta_same_effect else
                 f" (v_t Δβ₀={imp['vt']['beta']:+.2f} = {beta_ratio:.0%} of radialize's; β still changes — "
                 f"this is a non-rotational L-confirmation, not a β-free one).")
        verdict = (f"MECHANISM CONFIRMED — a non-rotational, different-form |L|-depletion (v_t rescale, "
                   f"Δ|L|={imp['vt']['L']:+.2f}≈ref {imp['rad']['L']:+.2f}) reproduces the SAME ordering: "
                   f"M(<r_c) significant at t={t_M05['vt'] or t_M10['vt']} BEFORE C₈ at t={t_C8['vt']}, same "
                   f"signs as radialization." + extra + " The causal variable is the angular-momentum "
                   f"distribution. No AWS.")
    elif vt_reproduces:
        verdict = (f"CONFIRMED (ordering unresolved) — v_t-rescale reproduces M/C₈ sign at matched Δ|L|, but "
                   f"the M-before-C₈ ordering was not cleanly resolved (M t={t_M10['vt']}, C₈ t={t_C8['vt']}).")
    else:
        verdict = (f"DOES NOT REPRODUCE — matched Δ|L| via v_t-rescale did NOT reproduce radialize's M/C₈ "
                   f"effect (vt C₈={eff['vt']['C8'][fin][0]:+.2f} vs rad {eff['rad']['C8'][fin][0]:+.2f}); "
                   f"the radialization geometry mattered beyond |L| alone.")

    return {"tune": tune, "effects": eff, "imposed": imp, "l_matched": l_matched,
            "first_sig": {"rad": {"M05": t_M05["rad"], "M10": t_M10["rad"], "C8": t_C8["rad"]},
                          "vt": {"M05": t_M05["vt"], "M10": t_M10["vt"], "C8": t_C8["vt"]}},
            "vt_concentration_before_c8": vt_conc_before_c8, "vt_reproduces": vt_reproduces,
            "beta_ratio_vt_to_rad": beta_ratio, "c8_ratio_vt_to_rad": c8_ratio,
            "less_beta_same_effect": less_beta_same_effect,
            "conservation": {"dKE_rel": dKE, "dQ_rel": dQ}, "aws_needed": False, "verdict": verdict}


def _write(rows, res):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "quantity"] + [f"t{t}" for t in TIMES])
        for arm in ("rad", "vt"):
            for q in QUANT:
                w.writerow([arm, q] + [f"{res['effects'][arm][q][str(t)][0]:.4f}" for t in TIMES])
    _report(res)
    print(f"[outputs] → {OUTDIR}/")


def _report(res):
    v = res["verdict"]
    band = ("🟢 MECHANISM CONFIRMED" if v.startswith("MECHANISM CONFIRMED")
            else "🟡 CONFIRMED (partial)" if v.startswith("CONFIRMED")
            else "🔴 DOES NOT REPRODUCE" if v.startswith("DOES NOT")
            else "🟠 RETUNE" if v.startswith("L NOT") else "🔴 REJECT")
    e = res["effects"]; im = res["imposed"]
    L = ["# Non-Rotational |L| Handle — confirmation of the resolved mechanism\n"]
    L.append(f"Hernquist, ε=0.05, N=1024, 100 matched pairs. Handle: v_t-magnitude rescale "
             f"(f={res['tune']['f']:.3f}, speed-preserving). Times: {TIMES}.\n")
    L.append(f"## Verdict — {band}\n\n> **{v}**\n")
    L.append(f"## Construction\n- radialize: Δβ₀={im['rad']['beta']:+.3f}, Δ|L|={im['rad']['L']:+.3f}")
    L.append(f"- **v_t-rescale: Δβ₀={im['vt']['beta']:+.3f}** ({res['beta_ratio_vt_to_rad']:.0%} of radialize), "
             f"**Δ|L|={im['vt']['L']:+.3f}** (matched={res['l_matched']})")
    L.append(f"- conservation ΔKE/KE={res['conservation']['dKE_rel']:.1e}, ΔQ/Q={res['conservation']['dQ_rel']:.1e}\n")
    L.append(f"## Time ordering (first significant, − sham)\n"
             f"- radialize: M(<0.05) t={res['first_sig']['rad']['M05']}, M(<0.1) t={res['first_sig']['rad']['M10']}, "
             f"C₈ t={res['first_sig']['rad']['C8']}")
    L.append(f"- **v_t-rescale: M(<0.05) t={res['first_sig']['vt']['M05']}, M(<0.1) t={res['first_sig']['vt']['M10']}, "
             f"C₈ t={res['first_sig']['vt']['C8']}** → concentration-before-C₈ = {res['vt_concentration_before_c8']}\n")
    L.append("## Effect vs time (− sham)\n")
    for arm in ("rad", "vt"):
        L.append(f"### {arm}")
        L.append("| quantity | " + " | ".join(f"t={t}" for t in TIMES) + " |")
        L.append("|---|" + "---|" * len(TIMES))
        for q in QUANT:
            L.append(f"| {q} | " + " | ".join(f"{e[arm][q][str(t)][0]:+.3f}" for t in TIMES) + " |")
        L.append("")
    with open(os.path.join(OUTDIR, "Lhandle_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Non-rotational |L| handle confirmation (no AWS).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--pairs", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    print("L-handle — tuning f to match radialize's Δ|L| (t₀ ensemble)...")
    ics = []
    for i in range(args.pairs):
        cfg = StressConfig(model="direct_isolated", init="hernquist3d", seed=2000 + i, n=N, steps=max(TIMES),
                           eps=EPS, box_size=2.0, k_fine=16, plummer_a=0.20)
        p, vv = get_initial_conditions(cfg)
        ics.append((p, vv, np.mean(p, axis=0)))
    t = tune_f(ics)
    print(f"  f={t['f']:.3f}: Δ|L| radialize={t['dL_rad']:+.3f}, v_t-rescale={t['dL_vt']:+.3f}")
    print("L-handle — running matched quadruples (time-course)...")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init, initargs=(True,)) as ex:
        futs = [ex.submit(_worker, (2000 + i, t["f"])) for i in range(args.pairs)]
        for fut in tqdm(as_completed(futs), total=len(futs), ncols=100, unit="pair"):
            rows.append(fut.result())
    res = analyse(rows, t)
    _write(rows, res)
    print("\n" + res["verdict"])


if __name__ == "__main__":
    main()
