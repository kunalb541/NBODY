#!/usr/bin/env python3
"""
coarse_grain_late_horizon.py — late-horizon closure test for the coarse-graining pilot
======================================================================================

The primary pilot tested the EARLY target ΔC8 (step 100) and returned RED.  This module
closes the one remaining loophole — the integration horizon — by re-running ℓ*(ε) + the
hardened bulk + C8(t₀) control for the MID (300) and LATE (600) ΔC8 targets (and absolute
C8 at late time).

Speed + correctness design:
  • Workers are INTEGRATION-ONLY: regenerate a replicate's ICs from its seed (float64,
    identical to the primary run) and integrate to step 600, returning C8 at each horizon.
    No feature recompute — the cached φ_ℓ is reused.
  • Alignment is by an EXACT float32 match on the early target: the cached feature row
    stores ΔC8-early(=float32); the integration returns c1-c0 for a known seed, and
    np.float32(c1-c0) reproduces that stored value bit-for-bit (same computation).  This
    BOTH aligns each late target to the correct φ_ℓ row AND proves the pairing is right
    (n_matched must equal n_rows).  This fixes the row-order bug of the first attempt.
  • Cusps are processed first and results are checkpointed after every cell, so a kill
    cannot waste the run.

Closure rule (pre-registered):
  • Late horizon ALSO fails → branch CLOSED (law retired across early/mid/late ΔC8).
  • Late horizon shows something → NEW target-specific hypothesis, not a revival of the law.

No AWS.  Does not touch paper.tex.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np
from tqdm import tqdm

from nbody_3d import _worker_init
from nbody_stress import StressConfig, get_simconfig, get_initial_conditions, \
    obs_coarse_var, _integrate_leapfrog
import coarse_grain_features as cgf
from coarse_grain_pilot import OUTDIR, PilotConfig, cell_key, _scale_fit, _spearman

HORIZONS = [100, 300, 600]   # early (self-test), mid, late

TARGET_SPECS = [
    ("early_delta", lambda c: c["c1"] - c["c0"]),   # self-test vs primary
    ("mid_delta",   lambda c: c["c3"] - c["c0"]),
    ("late_delta",  lambda c: c["c6"] - c["c0"]),
    ("late_abs",    lambda c: c["c6"]),
]


def _int_worker(task: tuple) -> tuple:
    fam, eps, seed, pc_dict = task
    pc = PilotConfig(**pc_dict)
    cfg = StressConfig(model="direct_isolated", init=fam, seed=int(seed), n=pc.n,
                       steps=max(HORIZONS), eps=float(eps), box_size=pc.box_size,
                       pm_grid=pc.pm_grid, k_fine=pc.k_fine, plummer_a=pc.plummer_a)
    sc = get_simconfig(cfg)
    pos0, vel0 = get_initial_conditions(cfg)
    snaps = _integrate_leapfrog(pos0, vel0, 1.0 / pc.n, sc, sorted({0, *HORIZONS}), True)
    c = {h: obs_coarse_var(snaps[h][0], cfg, 8, False) for h in [0, *HORIZONS]}
    return (fam, eps, int(seed), c[0], c[100], c[300], c[600])


def run_closure(pc: PilotConfig, workers: int) -> dict:
    seeds = [pc.seed0 + i for i in range(pc.replicates)]
    feat_dir = os.path.join(OUTDIR, "cache", "features")
    ckpt_path = os.path.join(OUTDIR, "late_horizon_closure.json")
    # cusps first so the rule-relevant results are saved earliest
    cells = ([(f, e) for f in ("hernquist3d", "plummer3d") if f in pc.families
              for e in sorted(pc.eps)]
             + [(f, e) for f in pc.families if f not in ("hernquist3d", "plummer3d")
                for e in sorted(pc.eps)])

    out: dict = {"horizons": HORIZONS, "by_target": {t: {} for t, _ in TARGET_SPECS},
                 "alignment": {}}

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(True,)) as ex:
        for fam, eps in cells:
            key = cell_key(fam, eps)
            futs = [ex.submit(_int_worker, (fam, eps, s, asdict(pc))) for s in seeds]
            results: dict = {}
            for fut in tqdm(as_completed(futs), total=len(futs), ncols=100,
                            desc=f"{fam[:9]} eps={eps:g}", unit="sim"):
                f, e, s, c0, c1, c3, c6 = fut.result()
                results[s] = (c0, c1, c3, c6)

            # exact float32 early-target → (c0,c1,c3,c6) map for alignment + validation
            early_to = {np.float32(c1 - c0).item(): (c0, c1, c3, c6)
                        for (c0, c1, c3, c6) in results.values()}
            n_unique = len(early_to)   # < n_seeds ⇒ float32 collision (would mispair)

            npz = np.load(os.path.join(feat_dir, f"{fam}_eps{eps:g}.npz"), allow_pickle=True)
            phi = npz["phi"].astype(np.float64)
            scal = npz["scal"].astype(np.float64)
            cols = list(npz["scal_cols"])
            early_cached = npz["target"].astype(np.float32)
            bl = {g: cgf.baseline_matrix(
                      [dict(zip(cols, row)) for row in scal], g)[0]
                  for g in cgf.BASELINE_GROUPS}

            cv = {"c0": [], "c1": [], "c3": [], "c6": []}
            n_match = 0
            for j in range(len(early_cached)):
                d = early_to.get(np.float32(early_cached[j]).item())
                if d is None:
                    cv["c0"].append(np.nan); cv["c1"].append(np.nan)
                    cv["c3"].append(np.nan); cv["c6"].append(np.nan)
                else:
                    n_match += 1
                    cv["c0"].append(d[0]); cv["c1"].append(d[1])
                    cv["c3"].append(d[2]); cv["c6"].append(d[3])
            out["alignment"][key] = {"n_rows": int(len(early_cached)), "n_matched": int(n_match),
                                     "n_seeds": len(results), "n_unique_keys": int(n_unique)}
            cvals = {k: np.array(v, float) for k, v in cv.items()}

            for tname, tfun in TARGET_SPECS:
                r = _scale_fit(phi, tfun(cvals), bl, key, pc)
                fr = out["by_target"][tname].setdefault(fam, {"ell_star": [], "rows": []})
                fr["ell_star"].append(r["ell_star"])
                fr["rows"].append({"eps": eps, **r})

            with open(ckpt_path, "w") as fh:          # checkpoint after every cell
                json.dump(out, fh, indent=2, default=float)

    # spearman + control-survival per (target, family)
    for tname in out["by_target"]:
        for fam, fr in out["by_target"][tname].items():
            fr["spearman_eps_ellstar"] = _spearman(np.array(sorted(pc.eps)),
                                                    np.array(fr["ell_star"]))
            fr["phi_beyond_ctrl_any_pos"] = any(r["pbc_lo"] > 0 for r in fr["rows"])

    cusps = [f for f in pc.families if f in ("hernquist3d", "plummer3d")]
    late, late_abs = out["by_target"]["late_delta"], out["by_target"]["late_abs"]
    cusp_sp = {f: late[f]["spearman_eps_ellstar"] for f in cusps if f in late}
    cusp_sp_abs = {f: late_abs[f]["spearman_eps_ellstar"] for f in cusps if f in late_abs}
    pbc_any = any(late[f]["phi_beyond_ctrl_any_pos"] for f in cusps if f in late) or \
        any(late_abs[f]["phi_beyond_ctrl_any_pos"] for f in cusps if f in late_abs)
    ellstar_increases = bool(cusp_sp) and all(s > 0 for s in cusp_sp.values())
    late_shows_something = ellstar_increases and pbc_any

    out["closure"] = {
        "cusp_spearman_late_delta": cusp_sp,
        "cusp_spearman_late_abs": cusp_sp_abs,
        "phi_beyond_ctrl_any_pos_late": pbc_any,
        "ellstar_increases_with_eps": ellstar_increases,
        "late_shows_something": late_shows_something,
        "verdict": (
            "NEW TARGET-SPECIFIC HYPOTHESIS — late horizon shows an ε-increasing, "
            "control-surviving ℓ* in BOTH cusps. NOT a revival of the general "
            "force-resolution law; flag as a fresh late-target-specific question."
            if late_shows_something else
            "BRANCH CLOSED — the late horizon also fails (no consistent ε-increasing, "
            "control-surviving ℓ* in cusps). The force-resolution coarse-graining law "
            "is retired across early, mid, and late ΔC8 targets."
        ),
    }
    with open(ckpt_path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Late-horizon closure test (no AWS).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    pc = PilotConfig()
    out = run_closure(pc, args.workers)

    print("\n=== ALIGNMENT (n_matched must equal n_rows) ===")
    for key, a in out["alignment"].items():
        print(f"  {key:24s} {a['n_matched']}/{a['n_rows']} "
              f"{'OK' if a['n_matched'] == a['n_rows'] else 'INCOMPLETE'}")

    prim = json.load(open(os.path.join(OUTDIR, "summary.json")))["cells"]
    print("\n=== SELF-TEST: early_delta ℓ* must match the primary run ===")
    ok = True
    for fam in pc.families:
        got = out["by_target"]["early_delta"][fam]["ell_star"]
        exp = [prim[cell_key(fam, e)]["ell_star"] for e in sorted(pc.eps)]
        ok &= (got == exp)
        print(f"  {fam:12s} closure={got}  primary={exp}  {'OK' if got == exp else 'MISMATCH'}")
    print(f"  → self-test {'PASSED' if ok else 'FAILED'}")

    print("\n================ LATE-HORIZON CLOSURE ================")
    for tname in [t for t, _ in TARGET_SPECS]:
        print(f"\n[{tname}]")
        for fam, r in out["by_target"][tname].items():
            print(f"  {fam:12s} ell*(eps)={r['ell_star']}  "
                  f"spearman={r['spearman_eps_ellstar']:+.2f}  "
                  f"phi>ctrl anywhere={r['phi_beyond_ctrl_any_pos']}")
    c = out["closure"]
    print(f"\ncusp Spearman (late ΔC8):   {c['cusp_spearman_late_delta']}")
    print(f"cusp Spearman (late absC8): {c['cusp_spearman_late_abs']}")
    print(f"phi beyond control (late):  {c['phi_beyond_ctrl_any_pos_late']}")
    print(f"\n>>> {c['verdict']}")


if __name__ == "__main__":
    main()
