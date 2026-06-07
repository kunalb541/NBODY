#!/usr/bin/env python3
"""
nbody_anisotropy_stage0.py — Stage 0 equilibrium-hold gates (dynamic) for the anisotropic ICs.
Integrates each family's UN-intervened IC under the SOFTENED production potential (eps=0.05) to the
1000-step science horizon and checks gates 4-6 (M(<0.1) drift, beta drift, breathing + shape/ROI).
Static gates 1-3 (density/beta/virial) and 7 (iso anchor) are validated separately. Local, cheap.
"""
from __future__ import annotations
import json
import numpy as np
import nbody_anisotropic_ic as ic
from nbody_stress import StressConfig, get_simconfig, _integrate_leapfrog
from nbody_3d import _worker_init

A, EPS, N, BOX = 0.20, 0.05, 2048, 2.0
RA = 1.5 * A
TIMES = [0, 5, 10, 20, 50, 100, 300, 600, 1000]
SEEDS = list(range(5))


def _center(pos):
    c = np.median(pos, axis=0)                          # robust start (immune to far outliers)
    for _ in range(10):
        r = np.linalg.norm(pos - c, axis=1)
        m = r < np.percentile(r, 70)
        if m.sum() < 50:
            break
        c = np.mean(pos[m], axis=0)
    return c


def _obs(pos, vel):
    c = _center(pos); d = pos - c; r = np.linalg.norm(d, axis=1)
    rhat = d / np.maximum(r, 1e-30)[:, None]
    vr = np.sum(vel * rhat, axis=1); vt = np.linalg.norm(vel - vr[:, None] * rhat, axis=1)
    res = r > 2 * EPS                                    # resolved region for beta
    sr2 = np.mean(vr[res] ** 2); st2 = np.mean(vt[res] ** 2)
    beta = 1.0 - st2 / (2.0 * sr2)
    M01 = float(np.mean(r < 0.1))
    lag = np.percentile(r, [10, 50, 90])
    rh = float(np.percentile(r, 50)); inner = r < rh; X = d[inner]
    S = X.T @ X / len(X); ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    return {"M01": M01, "beta": float(beta), "lag": lag.tolist(),
            "ba": float(np.sqrt(ev[1] / ev[0])), "ca": float(np.sqrt(ev[2] / ev[0]))}


def run_family(profile, aniso):
    cfg = StressConfig(model="direct_isolated", init="hernquist3d", seed=0, n=N, steps=max(TIMES),
                       eps=EPS, box_size=BOX, k_fine=16, plummer_a=A)
    sc = get_simconfig(cfg); mass = 1.0 / N
    series = []
    for s in SEEDS:
        pos, vel = ic.sample(profile, N, A, aniso, np.random.default_rng(1000 + s), r_a=RA)
        pos = pos + BOX / 2.0
        snaps = _integrate_leapfrog(pos, vel, mass, sc, TIMES, True)
        series.append({t: _obs(snaps[t][0], snaps[t][1]) for t in TIMES})

    def mt(t, k): return float(np.mean([series[s][t][k] for s in SEEDS]))
    def mlag(t): return np.mean([series[s][t]["lag"] for s in SEEDS], axis=0)
    M0, b0, lag0 = mt(0, "M01"), mt(0, "beta"), mlag(0)
    Mdrift = float(np.median([abs(mt(t, "M01") - M0) / M0 for t in TIMES if t > 0]))
    bdrift = float(max(abs(mt(t, "beta") - b0) for t in TIMES if t > 0))
    breath = float(max(np.max(np.abs(mlag(t) - lag0) / lag0) for t in TIMES if t > 0))
    ca_min = float(min(mt(t, "ca") for t in TIMES)); ba_min = float(min(mt(t, "ba") for t in TIMES))
    g4 = Mdrift < 0.10; g5 = bdrift < 0.05; g6 = (breath < 0.15 and ca_min > 0.85 and ba_min > 0.85)
    return {"M0": M0, "beta0": b0, "Mdrift": Mdrift, "beta_drift": bdrift, "breathing": breath,
            "ca_min": ca_min, "ba_min": ba_min, "g4_Mstable": g4, "g5_betastable": g5,
            "g6_shape": g6, "PASS": bool(g4 and g5 and g6)}


def main():
    _worker_init(True)
    out = {}
    print(f"Stage 0 equilibrium-hold | N={N} eps={EPS} r_a={RA} | {len(SEEDS)} seeds, horizon {max(TIMES)} steps\n")
    print(f"{'family':22s} {'M0':>6} {'Mdrift':>7} {'b0':>6} {'bdrift':>7} {'breath':>7} {'ca_min':>6} {'ba_min':>6}  gates")
    for profile in ["hernquist", "plummer"]:
        for aniso in ["isotropic", "radial", "tangential"]:
            d = run_family(profile, aniso); out[f"{profile}_{aniso}"] = d
            g = f"4{'+' if d['g4_Mstable'] else 'x'}5{'+' if d['g5_betastable'] else 'x'}6{'+' if d['g6_shape'] else 'x'}"
            print(f"{profile+'_'+aniso:22s} {d['M0']:6.3f} {d['Mdrift']:7.3f} {d['beta0']:+6.2f} "
                  f"{d['beta_drift']:7.3f} {d['breathing']:7.3f} {d['ca_min']:6.3f} {d['ba_min']:6.3f}  "
                  f"{g} {'PASS' if d['PASS'] else 'FAIL'}")
    json.dump(out, open("/tmp/stage0_results.json", "w"), indent=2)
    aniso_only = {k: v for k, v in out.items() if "isotropic" not in k}
    npass = sum(v["PASS"] for v in aniso_only.values())
    print(f"\nanisotropic families passing Stage 0: {npass}/4")
    print("hernquist_radial PASS:", out["hernquist_radial"]["PASS"], "(stop-rule family)")
    print("[written] /tmp/stage0_results.json")


if __name__ == "__main__":
    main()
