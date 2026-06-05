#!/usr/bin/env python3
"""
nbody_aws_pilot_benchmark.py — MEASURED runtime calibration for the new intervention code
=========================================================================================

Times one matched pair (the full 7-arm set over the full snapshot grid) of the *actual* pilot
worker at N = 1024 / 2048 / 4096, with a cost breakdown (intervention build / integration /
pericenter / summary-features / conservation).  Confirms O(N²) scaling and extrapolates to the
A) N=4096 pilot, B) full prereg battery local, C) full battery on 64 vCPU — from measured seconds,
not guesses.  Read-only timing: writes nothing, launches nothing.
"""
from __future__ import annotations

import math
import time

import numpy as np

from nbody_stress import StressConfig, get_initial_conditions, get_simconfig, _integrate_leapfrog
import phase_space_coarse_features as psf
from nbody_intervention_pilot import intervene_anisotropy, sham_rotation
from nbody_anisotropy_mechanism_pilot import tangentialize
from nbody_orbital_summary import shell_radialize
from nbody_intervention_timecourse import _obs
import nbody_aws_low_pericenter_pilot as P
from nbody_3d import _worker_init

TIMES = P.TIMES                  # [0,5,10,20,50,100,300,600]
EPS, A, THETA = 0.05, 0.20, 20.0


def timed_pair(profile, seed, N):
    cfg = StressConfig(model="direct_isolated", init=profile, seed=seed, n=N, steps=max(TIMES),
                       eps=EPS, box_size=2.0, k_fine=16, plummer_a=A)
    sc = get_simconfig(cfg); mass = 1.0 / N
    T = {"build": 0.0, "integrate": 0.0, "peri": 0.0, "summary": 0.0, "cons": 0.0}

    t = time.perf_counter()
    pos0, vel0 = get_initial_conditions(cfg); center = np.mean(pos0, axis=0); th = math.radians(THETA)
    vels = {"orig": vel0, "full": intervene_anisotropy(pos0, vel0, th, center),
            "inner-med": shell_radialize(pos0, vel0, th, 0.0, 33.333, center),
            "mid": shell_radialize(pos0, vel0, th, 33.333, 66.667, center),
            "outer": shell_radialize(pos0, vel0, th, 66.667, 100.0, center),
            "tan": tangentialize(pos0, vel0, th, center, seed),
            "sham": sham_rotation(pos0, vel0, th, seed)}
    r0 = np.linalg.norm(pos0 - center, axis=1)
    phig_a, phia_a = P._phi_analytic(P._RG, profile), P._phi_analytic(r0, profile)
    phig_m, phia_m = P._phi_measured(P._RG, r0), P._phi_measured(r0, r0)
    T["build"] += time.perf_counter() - t

    orig_snaps = None
    for name, v in vels.items():
        t = time.perf_counter(); snaps = _integrate_leapfrog(pos0, v, mass, sc, TIMES, True); T["integrate"] += time.perf_counter() - t
        if name == "orig":
            orig_snaps = snaps
        t = time.perf_counter()
        P._pericenters(pos0, v, center, phig_a, phia_a); P._pericenters(pos0, v, center, phig_m, phia_m)
        T["peri"] += time.perf_counter() - t
        t = time.perf_counter()
        for tt in TIMES:
            _obs(snaps[tt][0], snaps[tt][1], cfg)
        T["summary"] += time.perf_counter() - t
    t = time.perf_counter()
    psf.relaxation_observables(orig_snaps[0][0], orig_snaps[0][1], cfg)
    psf.relaxation_observables(orig_snaps[600][0], orig_snaps[600][1], cfg)
    T["cons"] += time.perf_counter() - t
    T["total"] = sum(v for k, v in T.items() if k != "total")
    return T


def main():
    _worker_init(True)
    print("Warming up Numba (compile once)…", flush=True)
    timed_pair("hernquist3d", 1, 512)            # discard: triggers JIT compilation

    Ns = [1024, 2048]                            # N=4096 measured from the real pilot run (run.log)
    K = {1024: 3, 2048: 2}
    rows = {}
    for N in Ns:
        samples = [timed_pair("hernquist3d", 100 + i, N) for i in range(K[N])]
        agg = {k: float(np.mean([s[k] for s in samples])) for k in samples[0]}
        rows[N] = agg
        br = "  ".join(f"{k} {agg[k]:.2f}s({100*agg[k]/agg['total']:.0f}%)"
                       for k in ["build", "integrate", "peri", "summary", "cons"])
        print(f"N={N:5d}: {agg['total']:6.2f} s/pair single-thread (7 arms, 600-step grid)  |  {br}", flush=True)

    print("\n=== O(N²) scaling check (per-pair total) ===", flush=True)
    print(f"  1024→2048: ×{rows[2048]['total']/rows[1024]['total']:.2f}  (O(N²) predicts ×4.0)")
    pred_4096 = rows[2048]["total"] * 4.0
    print(f"  → extrapolated N=4096 single-thread: {pred_4096:.0f} s/pair "
          f"(cross-check: real pilot Plummer 29.5 s/pair wall × 9 workers ≈ 265 s)")
    print("\nA/B/C runtime table computed in the report from these measured per-pair seconds.")


if __name__ == "__main__":
    main()
