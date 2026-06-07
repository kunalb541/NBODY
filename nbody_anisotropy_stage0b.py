#!/usr/bin/env python3
"""
nbody_anisotropy_stage0b.py — Stage 0B: M(<0.2)=4eps aperture + short pre-relaxation.
Stage 0 blocked the battery: the M(<0.1)=2eps aperture is not stationary (unsoftened DF in softened
potential). Stage 0B applies the approved fix and reports the no-prerelax vs with-prerelax comparison
so the aperture effect and the pre-relaxation effect are separately attributable. Local; no science run.
"""
from __future__ import annotations
import json
import numpy as np
import nbody_anisotropic_ic as ic
from nbody_stress import StressConfig, get_simconfig, _integrate_leapfrog
from nbody_3d import _worker_init

A, EPS, N, BOX = 0.20, 0.05, 2048, 2.0
RA = 1.5 * A
PRERELAX = 300                                     # steps under the softened potential before measuring
TIMES = [0, 5, 10, 20, 50, 100, 300, 600, 1000]    # measurement snapshots (after pre-relax)
SEEDS = list(range(5))
SHELLS = np.array([0.20, 0.30, 0.45, 0.70, 1.0])   # resolved beta shells (r > 4eps)


def _center(pos):
    c = np.median(pos, axis=0)
    for _ in range(10):
        r = np.linalg.norm(pos - c, axis=1); m = r < np.percentile(r, 70)
        if m.sum() < 50:
            break
        c = np.mean(pos[m], axis=0)
    return c


def _obs(pos, vel):
    c = _center(pos); d = pos - c; r = np.linalg.norm(d, axis=1)
    rhat = d / np.maximum(r, 1e-30)[:, None]
    vr = np.sum(vel * rhat, axis=1); vt = np.linalg.norm(vel - vr[:, None] * rhat, axis=1)
    res = r > 4 * EPS
    sr2 = np.mean(vr[res] ** 2); st2 = np.mean(vt[res] ** 2)
    M02 = float(np.mean(r < 0.2)); M01 = float(np.mean(r < 0.1))
    lag = np.percentile(r, [25, 50, 90])
    rh = float(np.percentile(r, 50)); inner = r < rh; X = d[inner]
    S = X.T @ X / len(X); ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    # beta in shells
    bsh = []
    for lo, hi in zip(SHELLS[:-1], SHELLS[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() < 100:
            bsh.append(np.nan); continue
        bsh.append(1.0 - np.mean(vt[m] ** 2) / (2.0 * np.mean(vr[m] ** 2)))
    return {"M02": M02, "M01": M01, "beta": float(1 - st2 / (2 * sr2)), "lag": lag.tolist(),
            "ca": float(np.sqrt(ev[2] / ev[0])), "ba": float(np.sqrt(ev[1] / ev[0])), "bsh": bsh}


def _target_beta(aniso, rc):
    if aniso == "radial":
        return rc ** 2 / (rc ** 2 + RA ** 2)
    if aniso == "tangential":
        return -0.5
    return 0.0


def _measure(profile, aniso, prerelax):
    cfg = StressConfig(model="direct_isolated", init="hernquist3d", seed=0, n=N, steps=PRERELAX + max(TIMES),
                       eps=EPS, box_size=BOX, k_fine=16, plummer_a=A)
    sc = get_simconfig(cfg); mass = 1.0 / N
    series = []
    for s in SEEDS:
        pos, vel = ic.sample(profile, N, A, aniso, np.random.default_rng(1000 + s), r_a=RA); pos += BOX / 2
        if prerelax:
            p0, v0 = _integrate_leapfrog(pos, vel, mass, sc, [PRERELAX], True)[PRERELAX]
        else:
            p0, v0 = pos, vel
        snaps = _integrate_leapfrog(p0, v0, mass, sc, TIMES, True)
        series.append({t: _obs(snaps[t][0], snaps[t][1]) for t in TIMES})

    def mt(t, k): return float(np.mean([series[s][t][k] for s in SEEDS]))
    def mlag(t): return np.mean([series[s][t]["lag"] for s in SEEDS], axis=0)
    M02_0 = mt(0, "M02"); M01_0 = mt(0, "M01"); b0 = mt(0, "beta"); lag0 = mlag(0)
    res = {
        "M02_drift": float(np.median([abs(mt(t, "M02") - M02_0) / M02_0 for t in TIMES if t > 0])),
        "M01_drift": float(np.median([abs(mt(t, "M01") - M01_0) / max(M01_0, 1e-6) for t in TIMES if t > 0])),
        "beta_drift": float(max(abs(mt(t, "beta") - b0) for t in TIMES if t > 0)),
        "breathing": float(max(np.max(np.abs(mlag(t) - lag0) / lag0) for t in TIMES if t > 0)),
        "ca_min": float(min(mt(t, "ca") for t in TIMES)), "ba_min": float(min(mt(t, "ba") for t in TIMES)),
        "M02_0": M02_0,
    }
    # beta(r) profile match at measurement t=0 (post pre-relax)
    bsh0 = np.nanmean([series[s][0]["bsh"] for s in SEEDS], axis=0)
    rcs = 0.5 * (SHELLS[:-1] + SHELLS[1:])
    devs = [abs(b - _target_beta(aniso, rc)) for b, rc in zip(bsh0, rcs) if not np.isnan(b)]
    res["beta_profile_maxdev"] = float(max(devs)) if devs else float("nan")
    return res


def main():
    import sys
    profs = sys.argv[1:] or ["hernquist", "plummer"]
    _worker_init(True)
    out = {}
    print(f"Stage 0B | N={N} eps={EPS} r_a={RA} aperture M(<0.2)=4eps | pre-relax={PRERELAX} steps | {len(SEEDS)} seeds\n")
    print(f"{'family':22s} {'M01drift_np':>11} {'M02drift_np':>11} {'M02drift_PR':>11} {'breath_PR':>9} "
          f"{'bdrift_PR':>9} {'bprof_PR':>8} {'shape_PR':>9}  PASS")
    for profile in profs:
        for aniso in ["isotropic", "radial", "tangential"]:
            pr = _measure(profile, aniso, True)
            npre = _measure(profile, aniso, False) if aniso == "isotropic" else {"M01_drift": float("nan"), "M02_drift": float("nan")}
            g_M = pr["M02_drift"] < 0.10; g_b = pr["beta_drift"] < 0.05
            g_breath = pr["breathing"] < 0.15
            g_shape = pr["ca_min"] > 0.85 and pr["ba_min"] > 0.85
            g_bprof = (aniso == "isotropic") or (pr["beta_profile_maxdev"] < 0.12)
            PASS = bool(g_M and g_b and g_breath and g_shape and g_bprof)
            out[f"{profile}_{aniso}"] = {"no_prerelax": npre, "prerelax": pr, "PASS": PASS}
            print(f"{profile+'_'+aniso:22s} {npre['M01_drift']:11.3f} {npre['M02_drift']:11.3f} "
                  f"{pr['M02_drift']:11.3f} {pr['breathing']:9.3f} {pr['beta_drift']:9.3f} "
                  f"{pr['beta_profile_maxdev']:8.3f} {min(pr['ca_min'],pr['ba_min']):9.3f}  {'PASS' if PASS else 'FAIL'}")
    json.dump(out, open("/tmp/stage0b_results.json", "w"), indent=2, default=float)
    aniso_only = {k: v for k, v in out.items() if "isotropic" not in k}
    npass = sum(v["PASS"] for v in aniso_only.values())
    print(f"\nanisotropic families passing Stage 0B: {npass}/4   (hernquist_radial: "
          f"{out['hernquist_radial']['PASS']})")
    print("[written] /tmp/stage0b_results.json")


if __name__ == "__main__":
    main()
