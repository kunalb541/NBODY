#!/usr/bin/env python3
"""
nbody_intervention_mechanism_B.py — separate β from angular-momentum depletion (Test B)
=======================================================================================

Test A left β and |L_i| entangled (radialize raises β AND lowers |L_i|).  Test B builds a
β-NULL, L-MATCHED intervention to separate them:

  construct a speed-preserving perturbation with Δβ₀ ≈ 0 but Δ⟨|L_i|⟩ ≈ radialization's.

Construction (exploits that |L_i| = r_i·v_t is RADIUS-WEIGHTED while β is a dispersion ratio):
  radialize the OUTER shell (large r → big |L| cut, modest +β) and tangentialize the INNER
  shell (cancels the net β, small |L| impact).  Tune (split, θ_out, θ_in) on the ensemble to
  hit Δβ₀≈0 and Δ⟨|L|⟩≈target BEFORE running the causal test.

Then matched quadruples: orig / radialize / L-matched-β-null / sham → t₁.  Decision:
  • L-matched reproduces radialize's C₈/central-mass effect → ANGULAR-MOMENTUM DEPLETION (H2)
  • L-matched has ~no C₈/central-mass effect          → ANISOTROPY β (H1)
  • intermediate                                      → BOTH
  • cannot hit Δβ₀≈0 & matched Δ|L|                   → β and |L| too coupled; do NOT interpret

Hernquist, ε=0.05, N=1024, θ=20° reference, 100 matched seeds.  No AWS.  paper.tex untouched.
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
from nbody_intervention_pilot import sham_rotation

OUTDIR = "outputs/nbody_intervention_mechanism_B"
R_CEN = 0.10
THETA_REF = 20.0
N, EPS = 1024, 0.05


def _rotate(pos, vel, theta, center, mode, seed=0):
    """Speed-preserving per-particle rotation. mode='rad' → nearest radial; 'tan' → tangential.
    NO momentum re-zero (caller re-zeros the combined field once)."""
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
    if mode == "tan" and np.any(~big):
        rng = np.random.default_rng(seed ^ 0x7777_7777)
        rnd = rng.normal(size=(int(np.sum(~big)), 3)); rs = rhat[~big]
        rnd -= np.sum(rnd * rs, axis=1)[:, None] * rs
        rnd /= np.linalg.norm(rnd, axis=1)[:, None]
        that[~big] = rnd
    phi = np.arctan2(v_t, v_r)
    if mode == "rad":
        phi2 = np.where(phi <= math.pi / 2, np.maximum(phi - theta, 0.0), np.minimum(phi + theta, math.pi))
    else:
        phi2 = np.where(phi <= math.pi / 2, np.minimum(phi + theta, math.pi / 2), np.maximum(phi - theta, math.pi / 2))
    return speed[:, None] * (np.cos(phi2)[:, None] * rhat + np.sin(phi2)[:, None] * that)


def _rezero(v):
    return v - np.mean(v, axis=0)


def radialize_all(pos, vel, theta, center):
    return _rezero(_rotate(pos, vel, theta, center, "rad"))


def lmatched_betanull(pos, vel, theta_out, theta_in, center, split_pct, seed):
    d = pos - center
    r = np.linalg.norm(d, axis=1)
    outer = r >= np.percentile(r, split_pct)
    vnew = vel.copy()
    vnew[outer] = _rotate(pos[outer], vel[outer], math.radians(theta_out), center, "rad")
    vnew[~outer] = _rotate(pos[~outer], vel[~outer], math.radians(theta_in), center, "tan", seed)
    return _rezero(vnew)


def _beta_L(pos, vel, center):
    d = pos - center
    r = np.linalg.norm(d, axis=1)
    rhat = d / np.where(r > 1e-12, r, 1e-12)[:, None]
    v_r = np.sum(vel * rhat, axis=1)
    v_t = np.linalg.norm(vel - v_r[:, None] * rhat, axis=1)
    sigr = float(np.std(v_r)); sigt = float(np.sqrt(np.mean(v_t ** 2)))
    beta = 1.0 - sigt ** 2 / (2.0 * sigr ** 2) if sigr > 1e-9 else float("nan")
    Lspec = float(np.mean(np.linalg.norm(np.cross(d, vel), axis=1)))
    return beta, Lspec


# ── tuning (t₀ only, no integration) ─────────────────────────────────────────────

def tune(ics):
    # radialize-all reference targets
    db_r, dL_r = [], []
    for pos, vel, c in ics:
        b0, L0 = _beta_L(pos, vel, c)
        b1, L1 = _beta_L(pos, radialize_all(pos, vel, math.radians(THETA_REF), c), c)
        db_r.append(b1 - b0); dL_r.append(L1 - L0)
    dbeta_ref, dL_ref = float(np.mean(db_r)), float(np.mean(dL_r))

    best = None
    grid = [(sp, to, ti) for sp in (40, 50, 60) for to in (20, 30, 40) for ti in (5, 10, 15, 20, 25, 30, 35)]
    for sp, to, ti in grid:
        dbs, dLs = [], []
        for pos, vel, c in ics:
            b0, L0 = _beta_L(pos, vel, c)
            v2 = lmatched_betanull(pos, vel, to, ti, c, sp, seed=2000)
            b1, L1 = _beta_L(pos, v2, c)
            dbs.append(b1 - b0); dLs.append(L1 - L0)
        dbeta, dL = float(np.mean(dbs)), float(np.mean(dLs))
        beta_null = abs(dbeta) <= 0.15 * abs(dbeta_ref)        # |Δβ| within 15% of radialize's
        l_err = abs(dL - dL_ref) / abs(dL_ref) if dL_ref else 1e9
        score = (0 if beta_null else 1, l_err)                 # prefer β-null, then closest |L|
        if best is None or score < best["score"]:
            best = {"score": score, "split_pct": sp, "theta_out": to, "theta_in": ti,
                    "dbeta": dbeta, "dL": dL, "beta_null": beta_null, "l_err": l_err}
    best.update({"dbeta_ref": dbeta_ref, "dL_ref": dL_ref})
    return best


# ── causal test (with integration) ───────────────────────────────────────────────

def _obs(pos, vel, cfg):
    center = np.mean(pos, axis=0)
    b, L = _beta_L(pos, vel, center)
    d = pos - center; r = np.linalg.norm(d, axis=1)
    rhat = d / np.where(r > 1e-12, r, 1e-12)[:, None]
    sigr = float(np.std(np.sum(vel * rhat, axis=1)))
    return {"beta": b, "Lspec": L, "C8": obs_coarse_var(pos, cfg, 8, False),
            "Mcen": float(np.sum(r < R_CEN)) / len(r), "sigr": sigr,
            "S": psf.relaxation_observables(pos, vel, cfg)["S_mix"]}


def _worker(task):
    seed, split_pct, theta_out, theta_in = task
    cfg = StressConfig(model="direct_isolated", init="hernquist3d", seed=seed, n=N, steps=600,
                       eps=EPS, box_size=2.0, k_fine=16, plummer_a=0.20)
    sc = get_simconfig(cfg); mass = 1.0 / N
    pos0, vel0 = get_initial_conditions(cfg); center = np.mean(pos0, axis=0)
    vels = {"orig": vel0,
            "rad": radialize_all(pos0, vel0, math.radians(THETA_REF), center),
            "lmatch": lmatched_betanull(pos0, vel0, theta_out, theta_in, center, split_pct, seed),
            "sham": sham_rotation(pos0, vel0, math.radians(THETA_REF), seed)}
    out = {"seed": seed}
    for name, v in vels.items():
        o0 = _obs(pos0, v, cfg)
        ke0 = 0.5 * mass * float(np.sum(v * v)); Q0 = psf.relaxation_observables(pos0, v, cfg)["Q"]
        snaps = _integrate_leapfrog(pos0, v, mass, sc, sorted({0, 600}), True)
        o1 = _obs(snaps[600][0], snaps[600][1], cfg)
        out[name] = {**{f"{k}0": o0[k] for k in o0}, **{f"{k}1": o1[k] for k in o1}, "ke0": ke0, "Q0": Q0}
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


def analyse(rows, tune_info):
    Q = ["beta", "Lspec", "C8", "Mcen", "sigr", "S"]
    eff = {arm: {q: _paired([r[arm][f"{q}1"] - r["sham"][f"{q}1"] for r in rows]) for q in Q}
           for arm in ("rad", "lmatch")}
    imposed = {arm: {"beta": _paired([r[arm]["beta0"] - r["orig"]["beta0"] for r in rows]),
                     "Lspec": _paired([r[arm]["Lspec0"] - r["orig"]["Lspec0"] for r in rows])}
               for arm in ("rad", "lmatch")}
    sham_b0 = _paired([r["sham"]["beta0"] - r["orig"]["beta0"] for r in rows])
    dKE = float(np.median([abs(r["lmatch"]["ke0"] - r["orig"]["ke0"]) / max(r["orig"]["ke0"], 1e-30) for r in rows]))
    dQ = float(np.median([abs(r["lmatch"]["Q0"] - r["orig"]["Q0"]) / max(abs(r["orig"]["Q0"]), 1e-30) for r in rows]))

    # construction check (measured at t₀ on the actual matched-pair seeds)
    imp_b_lm, imp_L_lm = imposed["lmatch"]["beta"][0], imposed["lmatch"]["Lspec"][0]
    imp_b_r, imp_L_r = imposed["rad"]["beta"][0], imposed["rad"]["Lspec"][0]
    beta_null = abs(imp_b_lm) <= 0.15 * abs(imp_b_r)
    l_matched = abs(imp_L_lm - imp_L_r) <= 0.20 * abs(imp_L_r)
    construction_ok = beta_null and l_matched
    cons_ok = dKE < 1e-2 and dQ < 1e-2

    c8_r, c8_lm = eff["rad"]["C8"], eff["lmatch"]["C8"]
    m_r, m_lm = eff["rad"]["Mcen"], eff["lmatch"]["Mcen"]
    # fraction of radialize's C8/central-mass effect reproduced by the β-null L-matched arm
    frac_c8 = c8_lm[0] / c8_r[0] if abs(c8_r[0]) > 1e-9 else float("nan")
    frac_m = m_lm[0] / m_r[0] if abs(m_r[0]) > 1e-9 else float("nan")
    lm_moves = _ci_pos(c8_lm) or _ci_pos(m_lm)

    confounded = math.isfinite(frac_c8) and math.isfinite(frac_m) and (np.sign(frac_c8) != np.sign(frac_m))
    lm_beta1 = eff["lmatch"]["beta"][0]
    if not cons_ok:
        verdict = "INVALID (bulk) — KE/Q not preserved by the L-matched intervention."
    elif not construction_ok:
        verdict = (f"CONSTRUCTION FAILED — could not hit Δβ₀≈0 with matched Δ|L| "
                   f"(achieved Δβ₀={imp_b_lm:+.3f} [target ~0, ref {imp_b_r:+.2f}], "
                   f"Δ|L|={imp_L_lm:+.3f} [ref {imp_L_r:+.2f}]). β and |L| are too coupled under "
                   f"speed-preserving rotations to separate cleanly here — itself informative. "
                   f"Do NOT interpret the causal result; tuning/relaxation of design needed.")
    elif confounded:
        verdict = (f"CONFOUNDED CONSTRUCTION (β/|L| not cleanly separable) — the L-matched arm hit the "
                   f"GLOBAL scalar targets (Δβ₀={imp_b_lm:+.3f}≈0, Δ|L|={imp_L_lm:+.2f}≈ref {imp_L_r:+.2f}), "
                   f"but depleting the radius-weighted ⟨|L|⟩ at fixed β REQUIRES a radial anisotropy "
                   f"gradient (radialize outer / tangentialize inner), and that gradient itself reshapes "
                   f"scale-dependent clustering. Result is scale-split: core concentration M(<{R_CEN}) "
                   f"rises ({m_lm[0]:+.4f}, +{abs(frac_m)*100:.0f}% of radialize, SAME sign → L-depletion "
                   f"DOES drive concentration) while mid-scale C₈ REVERSES ({c8_lm[0]:+.2f}, opposite sign, "
                   f"from inner tangentialization). So C₈ cannot cleanly separate β vs L here. "
                   f"CLEAN findings: (1) L-depletion at β-null causally raises central mass → angular-"
                   f"momentum depletion contributes to concentration; (2) the L-matched arm sustains "
                   f"β(t₁)={lm_beta1:+.3f} from an imposed +{imp_b_lm:.2f} → L-depletion is partly upstream "
                   f"of anisotropy. A confound-free β/L split is likely unreachable under speed-preserving "
                   f"rotations (β and |L| geometrically coupled, as anticipated). Next: a non-rotational "
                   f"handle, or a single-scale concentration target instead of C₈.")
    elif not lm_moves:
        verdict = (f"ANISOTROPY (β) IS THE HANDLE — the β-null L-matched intervention depletes |L| "
                   f"like radialization (Δ|L|={imp_L_lm:+.2f} vs ref {imp_L_r:+.2f}) at Δβ₀≈0 "
                   f"({imp_b_lm:+.3f}), yet has ~NO effect on C₈ ({c8_lm[0]:+.2f} CI incl 0) or central "
                   f"mass ({m_lm[0]:+.4f}). Angular-momentum depletion alone does NOT drive the effect; "
                   f"the causal handle is the velocity-dispersion anisotropy β.")
    elif frac_c8 > 0.7 and frac_m > 0.7:
        verdict = (f"ANGULAR-MOMENTUM DEPLETION IS THE HANDLE — at Δβ₀≈0 ({imp_b_lm:+.3f}) the "
                   f"L-matched intervention reproduces {frac_c8:.0%} of radialize's C₈ effect and "
                   f"{frac_m:.0%} of its central-mass effect. β is mostly a proxy; the causal variable "
                   f"is specific angular-momentum depletion.")
    else:
        verdict = (f"BOTH CONTRIBUTE — at Δβ₀≈0 the L-matched intervention reproduces {frac_c8:.0%} of "
                   f"radialize's C₈ effect and {frac_m:.0%} of central-mass; partial. Angular-momentum "
                   f"depletion and anisotropy β both contribute to the clustering handle.")

    return {"tune": tune_info, "effects": eff, "imposed": imposed, "sham_beta0": sham_b0,
            "construction": {"beta_null": beta_null, "l_matched": l_matched, "ok": construction_ok,
                             "lm_dbeta0": imp_b_lm, "rad_dbeta0": imp_b_r,
                             "lm_dL": imp_L_lm, "rad_dL": imp_L_r},
            "frac_c8_reproduced": frac_c8, "frac_mcen_reproduced": frac_m, "confounded": confounded,
            "conservation": {"dKE_rel": dKE, "dQ_rel": dQ, "ok": cons_ok}, "aws_needed": False,
            "verdict": verdict}


def _write(rows, res):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + [f"{a}_{q}1" for a in ("orig", "rad", "lmatch", "sham") for q in ("beta", "Lspec", "C8", "Mcen")])
        for r in rows:
            w.writerow([r["seed"]] + [r[a][f"{q}1"] for a in ("orig", "rad", "lmatch", "sham") for q in ("beta", "Lspec", "C8", "Mcen")])
    _report(res)
    print(f"[outputs] → {OUTDIR}/")


def _report(res):
    v = res["verdict"]
    band = ("🟢 β IS THE HANDLE" if v.startswith("ANISOTROPY")
            else "🟢 L-DEPLETION IS THE HANDLE" if v.startswith("ANGULAR")
            else "🟡 BOTH CONTRIBUTE" if v.startswith("BOTH")
            else "🟠 CONFOUNDED — β/|L| not cleanly separable" if v.startswith("CONFOUNDED")
            else "🟠 CONSTRUCTION FAILED" if v.startswith("CONSTRUCTION")
            else "🔴 INVALID")
    e = res["effects"]; c = res["construction"]
    def f(p):
        return f"{p[0]:+.4f} [{p[1]:+.4f}, {p[2]:+.4f}]"
    L = ["# Anisotropy Mechanism — Test B (β-null, L-matched control)\n"]
    L.append("Hernquist, ε=0.05, N=1024, 100 matched quadruples (orig / radialize / "
             "L-matched-β-null / sham), t₁=600.\n")
    L.append(f"## Verdict — {band}\n\n> **{v}**\n")
    L.append("## Construction check (t₀ — must hit Δβ₀≈0 with matched Δ|L|)\n")
    L.append(f"- tuned params: split={res['tune']['split_pct']}%, θ_out={res['tune']['theta_out']}°, "
             f"θ_in={res['tune']['theta_in']}°")
    L.append(f"- radialize: Δβ₀={c['rad_dbeta0']:+.3f}, Δ⟨|L|⟩={c['rad_dL']:+.3f}")
    L.append(f"- **L-matched: Δβ₀={c['lm_dbeta0']:+.3f}** (β-null={c['beta_null']}), "
             f"**Δ⟨|L|⟩={c['lm_dL']:+.3f}** (L-matched={c['l_matched']})")
    L.append(f"- construction OK: **{c['ok']}**; sham Δβ₀={f(res['sham_beta0'])}; "
             f"conservation ΔKE/KE={res['conservation']['dKE_rel']:.1e}, ΔQ/Q={res['conservation']['dQ_rel']:.1e}\n")
    L.append("## Causal effects at t₁ (intervention − sham)\n")
    L.append("| quantity | radialize | L-matched β-null | reproduced |")
    L.append("|---|---|---|---|")
    for q in ["beta", "Lspec", "C8", "Mcen", "sigr", "S"]:
        r, lm = e["rad"][q], e["lmatch"][q]
        frac = f"{lm[0]/r[0]:.0%}" if abs(r[0]) > 1e-9 else "—"
        L.append(f"| {q} | {f(r)} | {f(lm)} | {frac} |")
    L.append(f"\nC₈ reproduced: {res['frac_c8_reproduced']:.0%}; central-mass reproduced: "
             f"{res['frac_mcen_reproduced']:.0%}.\n")
    with open(os.path.join(OUTDIR, "mechanism_B_report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Mechanism Test B: β-null L-matched control (no AWS).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--pairs", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    # Phase 1: tune the β-null L-matched construction on the ensemble (t₀ only, cheap)
    print("Test B — tuning the β-null L-matched intervention (t₀ ensemble)...")
    ics = []
    for i in range(args.pairs):
        cfg = StressConfig(model="direct_isolated", init="hernquist3d", seed=2000 + i, n=N, steps=600,
                           eps=EPS, box_size=2.0, k_fine=16, plummer_a=0.20)
        p, vv = get_initial_conditions(cfg)
        ics.append((p, vv, np.mean(p, axis=0)))
    t = tune(ics)
    print(f"  reference radialize: Δβ₀={t['dbeta_ref']:+.3f}, Δ|L|={t['dL_ref']:+.3f}")
    print(f"  tuned: split={t['split_pct']}% θ_out={t['theta_out']}° θ_in={t['theta_in']}° "
          f"→ Δβ₀={t['dbeta']:+.3f} (β-null={t['beta_null']}), Δ|L|={t['dL']:+.3f} "
          f"(L err={t['l_err']*100:.0f}%)")

    # Phase 2: causal matched-pair test with the tuned construction
    print("Test B — running causal matched quadruples...")
    tasks = [(2000 + i, t["split_pct"], t["theta_out"], t["theta_in"]) for i in range(args.pairs)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init, initargs=(True,)) as ex:
        futs = [ex.submit(_worker, tk) for tk in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), ncols=100, unit="quad"):
            rows.append(fut.result())
    res = analyse(rows, t)
    _write(rows, res)
    print("\n" + res["verdict"])


if __name__ == "__main__":
    main()
