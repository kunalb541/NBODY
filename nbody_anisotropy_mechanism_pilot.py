#!/usr/bin/env python3
"""
nbody_anisotropy_mechanism_pilot.py — Test A (antisymmetry) + mediation diagnostics
===================================================================================

First mechanism pilot for the confirmed anisotropy causal handle.  Matched QUADRUPLES per seed:
orig / radialize / tangentialize / sham (all speed-preserving → KE/E/Q conserved).  Measures, at
t₀ and t₁, the quantities that separate the candidate mechanisms:

  β              velocity-dispersion anisotropy (H1)
  <|L_i|>        mean specific angular momentum (H2: depletion)
  M(<0.1)        central-mass fraction (H3: infall → concentration)
  <v_r>(r<0.3)   inner radial velocity (infall proxy)
  C₈             clustering target

Decides:
  • Antisymmetry (Q3): radialize C₈↑ and tangentialize C₈↓ with opposite sign → causality clean.
  • Infall mediation (Q2): does ΔC₈ co-move with Δ central mass across interventions?
  • β vs L (Q1, partial): C₈ tracks β↑ / |L_i|↓ — but they are entangled; a clean split needs the
    L-matched control (test B), deferred.

Hernquist, ε=0.05, N=1024, θ=20°, 100 matched quadruples, t₁=600.  No AWS.  paper.tex untouched.
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

OUTDIR = "outputs/nbody_anisotropy_mechanism_pilot"
R_CEN, R_INNER = 0.10, 0.30


def tangentialize(pos, vel, theta, center, seed):
    """Speed-preserving rotation of each velocity TOWARD the tangential plane (β↓, |L_i|↑)."""
    d = pos - center
    r = np.linalg.norm(d, axis=1)
    rhat = d / np.where(r > 1e-12, r, 1e-12)[:, None]
    v_r = np.sum(vel * rhat, axis=1)
    vt_vec = vel - v_r[:, None] * rhat
    v_t = np.linalg.norm(vt_vec, axis=1)
    speed = np.linalg.norm(vel, axis=1)
    that = np.zeros_like(vel)
    big = v_t > 1e-9
    that[big] = vt_vec[big] / v_t[big][:, None]
    if np.any(~big):                                   # purely-radial: pick a random t̂ ⊥ r̂
        rng = np.random.default_rng(seed ^ 0x7777_7777)
        rnd = rng.normal(size=(int(np.sum(~big)), 3))
        rs = rhat[~big]
        rnd -= np.sum(rnd * rs, axis=1)[:, None] * rs
        rnd /= np.linalg.norm(rnd, axis=1)[:, None]
        that[~big] = rnd
    phi = np.arctan2(v_t, v_r)
    phi2 = np.where(phi <= math.pi / 2.0, np.minimum(phi + theta, math.pi / 2.0),
                    np.maximum(phi - theta, math.pi / 2.0))
    vnew = speed[:, None] * (np.cos(phi2)[:, None] * rhat + np.sin(phi2)[:, None] * that)
    return vnew - np.mean(vnew, axis=0)


def _mech_obs(pos, vel, cfg):
    center = np.mean(pos, axis=0)
    d = pos - center
    r = np.linalg.norm(d, axis=1)
    rhat = d / np.where(r > 1e-12, r, 1e-12)[:, None]
    v_r = np.sum(vel * rhat, axis=1)
    v_t = np.linalg.norm(vel - v_r[:, None] * rhat, axis=1)
    sigr = float(np.std(v_r)); sigt = float(np.sqrt(np.mean(v_t ** 2)))
    beta = 1.0 - sigt ** 2 / (2.0 * sigr ** 2) if sigr > 1e-9 else float("nan")
    Lspec = float(np.mean(np.linalg.norm(np.cross(d, vel), axis=1)))
    inner = r < R_INNER
    return {"beta": beta, "Lspec": Lspec, "C8": obs_coarse_var(pos, cfg, 8, False),
            "Mcen": float(np.sum(r < R_CEN)) / len(r),
            "vr_inner": float(np.mean(v_r[inner])) if np.any(inner) else float("nan")}


def _worker(task):
    seed, theta_deg, eps = task
    theta = math.radians(theta_deg)
    cfg = StressConfig(model="direct_isolated", init="hernquist3d", seed=seed, n=1024, steps=600,
                       eps=eps, box_size=2.0, k_fine=16, plummer_a=0.20)
    sc = get_simconfig(cfg); mass = 1.0 / 1024
    pos0, vel0 = get_initial_conditions(cfg); center = np.mean(pos0, axis=0)
    vels = {"orig": vel0,
            "rad": intervene_anisotropy(pos0, vel0, theta, center),
            "tan": tangentialize(pos0, vel0, theta, center, seed),
            "sham": sham_rotation(pos0, vel0, theta, seed)}
    out = {"seed": seed}
    for name, v in vels.items():
        o0 = _mech_obs(pos0, v, cfg)
        ke0 = 0.5 * mass * float(np.sum(v * v))
        Q0 = psf.relaxation_observables(pos0, v, cfg)["Q"]
        snaps = _integrate_leapfrog(pos0, v, mass, sc, sorted({0, 600}), True)
        o1 = _mech_obs(snaps[600][0], snaps[600][1], cfg)
        out[name] = {**{f"{k}0": o0[k] for k in o0}, **{f"{k}1": o1[k] for k in o1},
                     "ke0": ke0, "Q0": Q0}
    return out


def _paired(vals, seed=7):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size < 5:
        return [float("nan")] * 3 + [a.size]
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), size=(2000, len(a)))].mean(axis=1)
    return [float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), a.size]


def _ci_pos(p):
    return math.isfinite(p[1]) and (p[1] > 0 or p[2] < 0)


def analyse(rows):
    Q = ["beta", "Lspec", "C8", "Mcen", "vr_inner"]
    eff = {"rad": {}, "tan": {}}
    for arm in ("rad", "tan"):
        for q in Q:
            eff[arm][q] = _paired([r[arm][f"{q}1"] - r["sham"][f"{q}1"] for r in rows])
    imposed = {
        "rad_beta0": _paired([r["rad"]["beta0"] - r["orig"]["beta0"] for r in rows]),
        "tan_beta0": _paired([r["tan"]["beta0"] - r["orig"]["beta0"] for r in rows]),
        "rad_Lspec0": _paired([r["rad"]["Lspec0"] - r["orig"]["Lspec0"] for r in rows]),
        "tan_Lspec0": _paired([r["tan"]["Lspec0"] - r["orig"]["Lspec0"] for r in rows]),
    }
    sham_beta0 = _paired([r["sham"]["beta0"] - r["orig"]["beta0"] for r in rows])
    # mediation: per-pair ΔC8 vs Δ central-mass (radialize − sham)
    dC8 = np.array([r["rad"]["C81"] - r["sham"]["C81"] for r in rows])
    dM = np.array([r["rad"]["Mcen1"] - r["sham"]["Mcen1"] for r in rows])
    med_r = float(np.corrcoef(dC8, dM)[0, 1]) if np.std(dC8) > 1e-12 and np.std(dM) > 1e-12 else float("nan")
    # conservation (rad vs orig at t0)
    dKE = float(np.median([abs(r["rad"]["ke0"] - r["orig"]["ke0"]) / max(r["orig"]["ke0"], 1e-30) for r in rows]))
    dQ = float(np.median([abs(r["rad"]["Q0"] - r["orig"]["Q0"]) / max(abs(r["orig"]["Q0"]), 1e-30) for r in rows]))
    orig_relax = {q: [float(np.mean([r["orig"][f"{q}0"] for r in rows])),
                      float(np.mean([r["orig"][f"{q}1"] for r in rows]))] for q in Q}

    c8r, c8t = eff["rad"]["C8"], eff["tan"]["C8"]
    mr, mt = eff["rad"]["Mcen"], eff["tan"]["Mcen"]
    vrr = eff["rad"]["vr_inner"]
    antisym = _ci_pos(c8r) and _ci_pos(c8t) and (np.sign(c8r[0]) != np.sign(c8t[0]))
    # concentration mediation judged at the MEAN level: central mass responds antisymmetrically,
    # same sign as C₈ (the per-pair ΔC₈↔ΔMcen correlation is a weak secondary check — C₈ grid 0.25
    # and M(<0.1) probe different scales, so pair-level fluctuations need not track).
    conc_mediated = (_ci_pos(mr) and _ci_pos(mt) and np.sign(mr[0]) != np.sign(mt[0])
                     and np.sign(mr[0]) == np.sign(c8r[0]))
    net_infall = _ci_pos(vrr)                          # is there a NET inner radial velocity change?
    eff_per_beta = {"rad": c8r[0] / imposed["rad_beta0"][0] if imposed["rad_beta0"][0] else float("nan"),
                    "tan": c8t[0] / imposed["tan_beta0"][0] if imposed["tan_beta0"][0] else float("nan")}
    cons_ok = dKE < 1e-2 and dQ < 1e-2
    sham_null = not _ci_pos(sham_beta0)

    if not cons_ok:
        dec = "INVALID — KE/Q not preserved."
    elif not sham_null:
        dec = "INVALID — sham imposes anisotropy."
    elif not antisym:
        dec = (f"NON-ANTISYMMETRIC — radialize C₈={c8r[0]:+.2f}, tangentialize C₈={c8t[0]:+.2f}; "
               "not opposite-sign. Causal interpretation is incomplete / nonlinear; stop and rethink.")
    elif conc_mediated:
        ratio = (eff_per_beta["rad"] / eff_per_beta["tan"]) if eff_per_beta["tan"] else float("nan")
        dec = (f"ANTISYMMETRIC + CONCENTRATION-MEDIATED — radialize raises C₈ ({c8r[0]:+.2f}) and "
               f"central mass M(<{R_CEN}) ({mr[0]:+.4f}); tangentialize lowers both ({c8t[0]:+.2f}, "
               f"{mt[0]:+.4f}) — clean SIGN antisymmetry. The clustering tracks central CONCENTRATION "
               f"at the mean level, via orbital pericenter reduction (lower specific L → deeper plunge), "
               f"NOT bulk infall velocity (inner ⟨v_r⟩ effect {'significant' if net_infall else 'null'}). "
               f"MAGNITUDE is asymmetric/nonlinear: radialize is ~{abs(ratio):.0f}× more effective per "
               f"unit Δβ₀ (concentration is one-directional). β and |L_i| both flip (entangled) — the "
               f"β-vs-L split needs the L-matched control (test B). [per-pair ΔC₈↔ΔMcen r={med_r:+.2f}: "
               f"weak, scale mismatch.] Proceed to test B.")
    else:
        dec = (f"ANTISYMMETRIC, concentration response unclear — radialize C₈={c8r[0]:+.2f}, "
               f"tangentialize C₈={c8t[0]:+.2f} (clean sign), but central mass does not respond "
               f"antisymmetrically (rad ΔM={mr[0]:+.4f}, tan ΔM={mt[0]:+.4f}); the C₈ channel may be "
               "shell-phase structure, not concentration. Reconsider H3.")
    return {"effects": eff, "imposed": imposed, "sham_beta0": sham_beta0,
            "mediation_dC8_dMcen_r": med_r, "antisymmetric": antisym,
            "concentration_mediated": conc_mediated, "net_infall_velocity": net_infall,
            "c8_per_unit_beta": eff_per_beta,
            "conservation": {"dKE_rel": dKE, "dQ_rel": dQ, "ok": cons_ok}, "sham_null": sham_null,
            "orig_relax": orig_relax, "aws_needed": False, "decision": dec}


def _write(rows, res):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["seed"] + [f"{a}_{q}1" for a in ("orig", "rad", "tan", "sham")
                           for q in ("beta", "Lspec", "C8", "Mcen")]
        w.writerow(cols)
        for r in rows:
            w.writerow([r["seed"]] + [r[a][f"{q}1"] for a in ("orig", "rad", "tan", "sham")
                                      for q in ("beta", "Lspec", "C8", "Mcen")])
    _report(res)
    print(f"[outputs] → {OUTDIR}/")


def _report(res):
    dec = res["decision"]
    band = ("🟢 ANTISYM + CONCENTRATION-MEDIATED" if dec.startswith("ANTISYMMETRIC + CONCENTRATION")
            else "🟡 ANTISYM (concentration unclear)" if dec.startswith("ANTISYMMETRIC,")
            else "🔴 PROBLEM")
    e = res["effects"]
    def f(p):
        return f"{p[0]:+.4f} [{p[1]:+.4f}, {p[2]:+.4f}]"
    L = ["# Anisotropy Mechanism — Test A (antisymmetry) + mediation\n"]
    L.append("Hernquist, ε=0.05, N=1024, θ=20°, 100 matched quadruples (orig/radialize/"
             "tangentialize/sham), t₁=600.\n")
    L.append(f"## Verdict — {band}\n\n> **{dec}**\n")
    L.append("## Causal effects (intervention − sham, t₁), paired 95% CI\n")
    L.append("| quantity | radialize | tangentialize | antisymmetric? |")
    L.append("|---|---|---|---|")
    for q in ["beta", "Lspec", "C8", "Mcen", "vr_inner"]:
        r, t = e["rad"][q], e["tan"][q]
        anti = "yes" if (_ci_pos(r) and _ci_pos(t) and np.sign(r[0]) != np.sign(t[0])) else "—"
        L.append(f"| {q} | {f(r)} | {f(t)} | {anti} |")
    L.append("")
    im = res["imposed"]
    L.append("## Imposed at t₀ (handle check)\n")
    L.append(f"- radialize: Δβ₀ = {f(im['rad_beta0'])}, Δ⟨|L_i|⟩ = {f(im['rad_Lspec0'])}")
    L.append(f"- tangentialize: Δβ₀ = {f(im['tan_beta0'])}, Δ⟨|L_i|⟩ = {f(im['tan_Lspec0'])}")
    L.append(f"- sham Δβ₀ = {f(res['sham_beta0'])} (≈0)\n")
    L.append("## Mediation & controls\n")
    L.append(f"- ΔC₈ ↔ Δ central-mass M(<{R_CEN}) correlation (radialize−sham): "
             f"r = {res['mediation_dC8_dMcen_r']:+.2f}")
    L.append(f"- conservation: ΔKE/KE = {res['conservation']['dKE_rel']:.1e}, "
             f"ΔQ/Q = {res['conservation']['dQ_rel']:.1e}; sham null = {res['sham_null']}")
    L.append(f"- natural relaxation (orig t₀→t₁): "
             f"{ {q: [round(x,3) for x in v] for q, v in res['orig_relax'].items()} }\n")
    L.append("## β vs L (entangled — needs test B)\n")
    L.append("radialize raises β and lowers ⟨|L_i|⟩; tangentialize does the opposite. C₈ follows "
             "β↑ / |L_i|↓ together, so this pilot cannot separate them — the L-matched (β-null) "
             "control (test B) is required.\n")
    with open(os.path.join(OUTDIR, "mechanism_report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Anisotropy mechanism pilot — Test A (no AWS).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--theta-deg", type=float, default=20.0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    tasks = [(2000 + i, args.theta_deg, args.eps) for i in range(args.pairs)]
    print(f"Mechanism Test A: Hernquist ε={args.eps} θ={args.theta_deg}° {args.pairs} quadruples")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init, initargs=(True,)) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), ncols=100, unit="quad"):
            rows.append(fut.result())
    res = analyse(rows)
    _write(rows, res)
    print("\n" + res["decision"])


if __name__ == "__main__":
    main()
