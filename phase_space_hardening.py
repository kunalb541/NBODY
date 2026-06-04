#!/usr/bin/env python3
"""
phase_space_hardening.py — baseline-sharing control for the phase-space pilot
=============================================================================

The pilot's "ψ_ℓ beyond bulk" looked strong for Δβ and Δσ_r.  But ψ_ℓ contains the initial
β(r) and σ_r(r) profiles, and Δβ = β(t₁)−β(t₀) shares −β(t₀) — the same baseline-sharing
trap that killed the C₈ branch.  The pilot's bulk control {Q₀,E,radial,C₈} does NOT include
β₀/σ_r₀/S₀.

This script recomputes the increment with a HARDENED control = bulk ∪ {initial value of the
target's own quantity}: β₀ for Δβ, σ_r₀ for Δσ_r, comoving-entropy S₀ for ΔS.  The initial
scalars are computed from the spatial pilot's cached t₀ snapshots (identical ICs by seed),
aligned to each phase-space cell by seed — no re-simulation.

If a strong "ψ beyond bulk" signal SURVIVES the hardened control, it is genuine.  If it
collapses (like C₈ did), it was baseline-sharing.
"""
from __future__ import annotations

import json
import os
import numpy as np

from nbody_stress import StressConfig
import phase_space_coarse_features as psf
from coarse_grain_pilot import _fit_predict, _r2_boot, _ci, _stable_seed
from phase_space_coarse_pilot import PSConfig, cell_key, OUTDIR
from sklearn.metrics import r2_score

SNAP_DIR = "outputs/coarse_grain_pilot/cache/snapshots"   # spatial pilot t₀ snapshots (same seeds)
CELL_DIR = os.path.join(OUTDIR, "cache", "cells")

# target → initial-quantity control to add on top of bulk
SELF0 = {"dS_late": "S0", "dS_early": "S0",
         "dSigr_late": "sigr0", "dSigr_early": "sigr0",
         "dBeta_late": "beta0", "dBeta_early": "beta0"}


def _initial_scalars(pos: np.ndarray, vel: np.ndarray) -> dict:
    center = np.mean(pos, axis=0)
    r, v_r, v_t = psf.decompose(pos.astype(np.float64), vel.astype(np.float64), center)
    inside = r <= psf.PS_RMAX * np.median(r)   # generous; matches relaxation_observables intent
    sigr = float(np.std(v_r))
    sigt = float(np.sqrt(np.mean(v_t ** 2)))
    beta = 1.0 - sigt ** 2 / (2.0 * sigr ** 2) if sigr > 1e-9 else np.nan
    return {"S0": psf._phase_entropy(r, v_r), "sigr0": sigr, "beta0": beta}


def main() -> None:
    pc = PSConfig()
    # initial scalars per (cell, seed) from the spatial snapshots
    init_by_cell: dict = {}
    for fam in pc.families:
        for eps in pc.eps:
            sp = os.path.join(SNAP_DIR, f"{fam}_eps{eps:g}.npz")
            snap = np.load(sp, allow_pickle=True)
            pos, vel, seeds = snap["pos"], snap["vel"], snap["seed"]
            m = {}
            for i in range(len(seeds)):
                m[int(seeds[i])] = _initial_scalars(pos[i], vel[i])
            init_by_cell[cell_key(fam, eps)] = m

    print(f"{'cell':22s} {'target':6s}  R2(bulk) R2(bulk+self0)  ψ–bulk[CI]            ψ–(bulk+self0)[CI]      survives?")
    results = {}
    for fam in pc.families:
        for eps in pc.eps:
            key = cell_key(fam, eps)
            npz = np.load(os.path.join(CELL_DIR, f"{fam}_eps{eps:g}.npz"), allow_pickle=True)
            psi = npz["psi"].astype(np.float64)
            bulk = npz["bulk"].astype(np.float64)
            tnames = list(npz["target_names"]); tgts = npz["targets"].astype(np.float64)
            seeds = [int(s) for s in npz["seed"]]
            scales = list(npz["scales"])
            imap = init_by_cell[key]

            for tname, selfkey in SELF0.items():
                if tname not in tnames:
                    continue
                y = tgts[:, tnames.index(tname)]
                self0 = np.array([imap[s][selfkey] for s in seeds], float)
                finite = np.isfinite(y) & np.isfinite(self0) \
                    & np.isfinite(psi.reshape(len(y), -1)).all(1) & np.isfinite(bulk).all(1)
                idx = np.where(finite)[0]; n = len(idx)
                if n < 30:
                    continue
                perm = np.random.default_rng(_stable_seed(key + tname)).permutation(n)
                nt = max(int(round(pc.test_frac * n)), 5)
                te, tr = idx[perm[:nt]], idx[perm[nt:]]
                yte = y[te]
                boot = np.random.default_rng(_stable_seed(key + tname + "|b")).integers(0, nt, size=(pc.n_boot, nt))

                # best ψ scale on this split
                r2s = []
                for s in range(psi.shape[1]):
                    p, _ = _fit_predict(psi[tr, s, :], y[tr], psi[te, s, :])
                    r2s.append(r2_score(yte, p))
                psi_best = psi[:, int(np.argmax(r2s)), :]

                def incr(baseX):
                    pb, _ = _fit_predict(baseX[tr], y[tr], baseX[te])
                    pj, _ = _fit_predict(np.column_stack([baseX, psi_best])[tr], y[tr],
                                         np.column_stack([baseX, psi_best])[te])
                    bb = _r2_boot(yte, pb, boot); bj = _r2_boot(yte, pj, boot)
                    lo, hi = _ci(bj - bb)
                    return r2_score(yte, pb), float(r2_score(yte, pj) - r2_score(yte, pb)), lo, hi

                r2_bulk, inc_b, lo_b, hi_b = incr(bulk)
                control = np.column_stack([bulk, self0])
                r2_ctrl, inc_c, lo_c, hi_c = incr(control)
                survives = lo_c is not None and lo_c > 0
                print(f"{key:22s} {tname:6s}  {r2_bulk:+.2f}    {r2_ctrl:+.2f}          "
                      f"{inc_b:+.3f}[{lo_b:+.3f},{hi_b:+.3f}]   {inc_c:+.3f}[{lo_c:+.3f},{hi_c:+.3f}]   "
                      f"{'YES' if survives else 'no'}")
                results[f"{key}|{tname}"] = {
                    "r2_bulk": r2_bulk, "r2_bulk_self0": r2_ctrl,
                    "psi_beyond_bulk": inc_b, "psi_beyond_bulk_ci": [lo_b, hi_b],
                    "psi_beyond_bulk_self0": inc_c, "psi_beyond_bulk_self0_ci": [lo_c, hi_c],
                    "survives_hardened": bool(survives),
                }

    # summary by target
    print("\n=== survival counts (ψ beyond bulk+self0, CI>0) ===")
    by_t = {}
    for k, v in results.items():
        t = k.split("|")[-1]
        by_t.setdefault(t, [0, 0])
        by_t[t][1] += 1
        by_t[t][0] += int(v["survives_hardened"])
    for t, (s, n) in sorted(by_t.items()):
        print(f"  {t:12s}: {s}/{n} cells survive the hardened control")

    out = {"results": results, "survival_by_target": {t: {"survive": s, "n": n} for t, (s, n) in by_t.items()}}
    with open(os.path.join(OUTDIR, "hardening_check.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWritten → {os.path.join(OUTDIR, 'hardening_check.json')}")


if __name__ == "__main__":
    main()
