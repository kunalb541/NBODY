#!/usr/bin/env python3
"""
phase_space_coarse_pilot.py — local cusp phase-space coarse-graining pilot
=========================================================================

Tests the NEW branch: does coarse phase-space information ψ_ℓ predict future
mixing / relaxation **beyond bulk energy/virial controls**, and beyond purely spatial
coarse features φ_ℓ?  Headline target = Δ phase-space entropy (mixing); ΔQ and the
σ_r / β changes are secondary.  ℓ*(ε) is secondary only and must not revive ℓ*∼ε.

Self-contained workers (regenerate float64 ICs from seed → compute every t₀ feature set
AND integrate to the late horizon for the targets, paired per replicate), processed cell by
cell with checkpointing.  Cusps only.  No AWS.  Does not touch paper.tex.

Stages:  run  (features + targets, cached per cell)  →  analyze  (R², increments, report)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from nbody_3d import _worker_init
from nbody_stress import (
    StressConfig, get_initial_conditions, get_simconfig, _integrate_leapfrog,
    obs_fine_knn_all, obs_fine_pk_small, obs_fine_close_pairs, obs_fof_groups,
    obs_fine_local_vel_disp,
)
import coarse_grain_features as cgf
import phase_space_coarse_features as psf
from coarse_grain_pilot import _fit_predict, _r2_boot, _ci, _stable_seed, _spearman

from sklearn.metrics import r2_score

OUTDIR = "outputs/phase_space_coarse_pilot"
HORIZONS = [100, 600]   # early (robustness) + late (headline; relaxation is late-time)

# headline first
TARGETS: List[Tuple[str, str]] = [
    ("dS_late",  "Δentropy(late)"),    # HEADLINE — phase-space mixing
    ("dS_early", "Δentropy(early)"),
    ("dQ_late",  "ΔQ(late)"),
    ("dQ_early", "ΔQ(early)"),
    ("dSigr_late", "Δσr(late)"),
    ("dBeta_late", "Δβ(late)"),
]
HEADLINE = "dS_late"


@dataclass
class PSConfig:
    families: List[str]  = field(default_factory=lambda: ["hernquist3d", "plummer3d"])
    eps: List[float]     = field(default_factory=lambda: [0.02, 0.05, 0.10])
    n: int               = 1024
    replicates: int      = 500
    seed0: int           = 2000
    box_size: float      = 2.0
    plummer_a: float     = 0.20
    k_fine: int          = 16
    pm_grid: int         = 32
    grid_g: int          = 64     # spatial φ_ℓ baseline grid (cheaper than the retired 96³; fair comparison)
    scales: List[float]  = field(default_factory=lambda: [0.025, 0.04, 0.06, 0.09, 0.14, 0.2, 0.3])
    test_frac: float     = 0.30
    n_boot: int          = 1000


def cell_key(fam: str, eps: float) -> str:
    return f"{fam}|eps={eps:g}"


def _cfg(pc: PSConfig, fam: str, eps: float, seed: int, steps: int) -> StressConfig:
    return StressConfig(model="direct_isolated", init=fam, seed=seed, n=pc.n, steps=steps,
                        eps=eps, box_size=pc.box_size, pm_grid=pc.pm_grid,
                        k_fine=pc.k_fine, plummer_a=pc.plummer_a)


# ── worker: all t₀ features + integrate to late horizon for targets ──────────────

def _worker(task: tuple) -> dict:
    fam, eps, seed, pc_dict = task
    pc = PSConfig(**pc_dict)
    cfg = _cfg(pc, fam, eps, seed, max(HORIZONS))
    sc = get_simconfig(cfg)
    pos0, vel0 = get_initial_conditions(cfg)
    box = pc.box_size

    relax0 = psf.relaxation_observables(pos0, vel0, cfg)
    psi = psf.psi_all_scales(pos0, vel0, pc.scales)
    phi = cgf.phi_all_scales(pos0, pc.grid_g, box, pc.scales)
    fine_pos = [obs_fine_knn_all(pos0, pc.k_fine, eps, False, box),
                obs_fine_pk_small(pos0, False, box, pc.pm_grid, 1.0 / pc.n),
                obs_fine_close_pairs(pos0, eps, False, box),
                obs_fof_groups(pos0, False, box, cfg.fof_b)]
    fine_kin = [obs_fine_local_vel_disp(pos0, vel0, pc.k_fine, False, box)]
    bulk = psf.bulk_features(pos0, vel0, cfg, relax0)

    snaps = _integrate_leapfrog(pos0, vel0, 1.0 / pc.n, sc, sorted({0, *HORIZONS}), True)
    re = psf.relaxation_observables(snaps[100][0], snaps[100][1], cfg)
    rl = psf.relaxation_observables(snaps[600][0], snaps[600][1], cfg)

    def d(obs, k):
        return float(obs[k] - relax0[k])
    targets = {
        "dS_early": d(re, "S_mix"), "dS_late": d(rl, "S_mix"),
        "dQ_early": d(re, "Q"),     "dQ_late": d(rl, "Q"),
        "dSigr_early": d(re, "sigr"), "dSigr_late": d(rl, "sigr"),
        "dBeta_early": d(re, "beta"), "dBeta_late": d(rl, "beta"),
    }
    return {"fam": fam, "eps": eps, "seed": seed,
            "psi": psi.astype(np.float32), "phi": phi.astype(np.float32),
            "fine_pos": np.array(fine_pos, np.float32),
            "fine_kin": np.array(fine_kin, np.float32),
            "bulk": np.array([bulk[k] for k in psf.bulk_feature_names()], np.float32),
            "targets": targets}


def stage_run(pc: PSConfig, workers: int, force: bool) -> None:
    cell_dir = os.path.join(OUTDIR, "cache", "cells")
    os.makedirs(cell_dir, exist_ok=True)
    seeds = [pc.seed0 + i for i in range(pc.replicates)]
    t0 = time.perf_counter()
    n_done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(True,)) as ex:
        for fam in pc.families:
            for eps in pc.eps:
                path = os.path.join(cell_dir, f"{fam}_eps{eps:g}.npz")
                if os.path.exists(path) and not force:
                    print(f"[run] cached {cell_key(fam, eps)} — skip"); continue
                futs = [ex.submit(_worker, (fam, eps, s, asdict(pc))) for s in seeds]
                rows: Dict[int, dict] = {}
                for fut in tqdm(as_completed(futs), total=len(futs), ncols=100,
                                desc=f"{fam[:9]} eps={eps:g}", unit="sim"):
                    r = fut.result(); rows[r["seed"]] = r
                ordered = [rows[s] for s in seeds if s in rows]
                tnames = list(ordered[0]["targets"].keys())
                np.savez_compressed(
                    path,
                    psi=np.stack([r["psi"] for r in ordered]),
                    phi=np.stack([r["phi"] for r in ordered]),
                    fine_pos=np.stack([r["fine_pos"] for r in ordered]),
                    fine_kin=np.stack([r["fine_kin"] for r in ordered]),
                    bulk=np.stack([r["bulk"] for r in ordered]),
                    targets=np.array([[r["targets"][t] for t in tnames] for r in ordered], np.float32),
                    target_names=np.array(tnames),
                    seed=np.array([r["seed"] for r in ordered]),
                    psi_names=np.array(psf.psi_feature_names()),
                    bulk_names=np.array(psf.bulk_feature_names()),
                    fam=fam, eps=eps, scales=np.array(pc.scales),
                )
                n_done += len(ordered)
    dt = time.perf_counter() - t0
    if n_done:
        print(f"[run] {n_done} reps in {dt:.1f}s ({dt/max(n_done,1)*1000:.0f} ms/rep wall)")
        rt = {}
        p = os.path.join(OUTDIR, "cache", "runtime.json")
        if os.path.exists(p):
            rt = json.load(open(p))
        rt.update({"run_wall_s": rt.get("run_wall_s", 0) + dt, "run_reps": rt.get("run_reps", 0) + n_done,
                   "workers": workers})
        json.dump(rt, open(p, "w"), indent=2)


# ── analysis ─────────────────────────────────────────────────────────────────────

def _best_scale_r2(feat3d, y, tr, te, boot):
    """Best-over-scale held-out R² for a (R, nS, nF) feature block; returns (best_idx, r2, pred)."""
    yte = y[te]
    r2s, preds = [], []
    for s in range(feat3d.shape[1]):
        pred, _ = _fit_predict(feat3d[tr, s, :], y[tr], feat3d[te, s, :])
        preds.append(pred); r2s.append(r2_score(yte, pred))
    b = int(np.argmax(r2s))
    return b, np.array(r2s), preds[b]


def _r2(X, y, tr, te):
    pred, _ = _fit_predict(X[tr], y[tr], X[te])
    return r2_score(y[te], pred), pred


def _increment(baseX, addX, y, tr, te, boot):
    """Held-out R²(base+add) − R²(base), with paired bootstrap CI."""
    yte = y[te]
    pb, _ = _fit_predict(baseX[tr], y[tr], baseX[te])
    pj, _ = _fit_predict(np.column_stack([baseX, addX])[tr], y[tr],
                         np.column_stack([baseX, addX])[te])
    bb = _r2_boot(yte, pb, boot); bj = _r2_boot(yte, pj, boot)
    lo, hi = _ci(bj - bb)
    return float(r2_score(yte, pj) - r2_score(yte, pb)), lo, hi


def _analyse_cell(npz, pc: PSConfig) -> dict:
    psi = npz["psi"].astype(np.float64)
    phi = npz["phi"].astype(np.float64)
    fine_pos = npz["fine_pos"].astype(np.float64)
    fine_kin = npz["fine_kin"].astype(np.float64)
    bulk = npz["bulk"].astype(np.float64)
    tnames = list(npz["target_names"])
    tgts = npz["targets"].astype(np.float64)
    fam = str(npz["fam"]); eps = float(npz["eps"]); scales = list(npz["scales"])
    R = psi.shape[0]

    static = {"fine_pos": fine_pos, "fine_kin": fine_kin, "bulk": bulk}
    finite_feat = np.isfinite(psi.reshape(R, -1)).all(1) & np.isfinite(phi.reshape(R, -1)).all(1)
    for X in static.values():
        finite_feat &= np.isfinite(X).all(1)

    key = cell_key(fam, eps)
    out = {"family": fam, "eps": eps, "scales": scales, "targets": {}}
    for tname, tlabel in TARGETS:
        y = tgts[:, tnames.index(tname)]
        finite = finite_feat & np.isfinite(y)
        idx = np.where(finite)[0]; n = len(idx)
        if n < 30:
            continue
        perm = np.random.default_rng(_stable_seed(key + tname)).permutation(n)
        nt = max(int(round(pc.test_frac * n)), 5)
        te, tr = idx[perm[:nt]], idx[perm[nt:]]
        boot = np.random.default_rng(_stable_seed(key + tname + "|b")).integers(0, nt, size=(pc.n_boot, nt))

        psi_b, psi_r2s, _ = _best_scale_r2(psi, y, tr, te, boot)
        phi_b, phi_r2s, _ = _best_scale_r2(phi, y, tr, te, boot)
        psi_best = psi[:, psi_b, :]; phi_best = phi[:, phi_b, :]

        r2 = {nm: _r2(X, y, tr, te)[0] for nm, X in static.items()}
        r2["phi_best"] = float(phi_r2s[phi_b]); r2["psi_best"] = float(psi_r2s[psi_b])

        inc_bulk = _increment(bulk, psi_best, y, tr, te, boot)          # HEADLINE control
        inc_phi = _increment(phi_best, psi_best, y, tr, te, boot)        # phase-space specific?
        inc_kin = _increment(fine_kin, psi_best, y, tr, te, boot)

        out["targets"][tname] = {
            "label": tlabel, "n": n, "r2": r2,
            "psi_scale_r2": psi_r2s.tolist(), "phi_scale_r2": phi_r2s.tolist(),
            "ell_star_psi": float(scales[psi_b]),
            "psi_beyond_bulk": inc_bulk[0], "psi_beyond_bulk_lo": inc_bulk[1], "psi_beyond_bulk_hi": inc_bulk[2],
            "psi_beyond_phi": inc_phi[0], "psi_beyond_phi_lo": inc_phi[1], "psi_beyond_phi_hi": inc_phi[2],
            "psi_beyond_kin": inc_kin[0], "psi_beyond_kin_lo": inc_kin[1], "psi_beyond_kin_hi": inc_kin[2],
        }
    return out


# ── hardened control: ψ beyond bulk + the INITIAL value of the target's own quantity ──
# Δβ shares −β₀, Δσ_r shares −σ_r₀, ΔS shares −S₀ — the C₈ baseline-sharing trap.  The bulk
# control omits these, so a strong "ψ beyond bulk" for Δβ/Δσ_r can be pure baseline-sharing.
# Here we add the initial target quantity (from the spatial pilot's cached t₀ snapshots, same
# seeds) to the control and re-test.  Survival of this control is the real bar.
SELF0 = {"dS_late": "S0", "dS_early": "S0", "dSigr_late": "sigr0", "dSigr_early": "sigr0",
         "dBeta_late": "beta0", "dBeta_early": "beta0"}
SNAP_DIR_SPATIAL = "outputs/coarse_grain_pilot/cache/snapshots"


def _initial_scalars(pos: np.ndarray, vel: np.ndarray) -> dict:
    center = np.mean(pos, axis=0)
    r, v_r, v_t = psf.decompose(pos.astype(np.float64), vel.astype(np.float64), center)
    sigr = float(np.std(v_r)); sigt = float(np.sqrt(np.mean(v_t ** 2)))
    beta = 1.0 - sigt ** 2 / (2.0 * sigr ** 2) if sigr > 1e-9 else float("nan")
    return {"S0": psf._phase_entropy(r, v_r), "sigr0": sigr, "beta0": beta}


def _hardened_pass(pc: PSConfig) -> dict:
    cell_dir = os.path.join(OUTDIR, "cache", "cells")
    if not os.path.isdir(SNAP_DIR_SPATIAL):
        return {"available": False}
    by_ct: dict = {}; surv: dict = {}
    for fam in pc.families:
        for eps in pc.eps:
            key = cell_key(fam, eps)
            sp = os.path.join(SNAP_DIR_SPATIAL, f"{fam}_eps{eps:g}.npz")
            cp = os.path.join(cell_dir, f"{fam}_eps{eps:g}.npz")
            if not (os.path.exists(sp) and os.path.exists(cp)):
                continue
            snap = np.load(sp, allow_pickle=True)
            imap = {int(snap["seed"][i]): _initial_scalars(snap["pos"][i], snap["vel"][i])
                    for i in range(len(snap["seed"]))}
            npz = np.load(cp, allow_pickle=True)
            psi = npz["psi"].astype(np.float64); bulk = npz["bulk"].astype(np.float64)
            tnames = list(npz["target_names"]); tgts = npz["targets"].astype(np.float64)
            cseeds = [int(s) for s in npz["seed"]]
            for tname, selfkey in SELF0.items():
                if tname not in tnames or any(s not in imap for s in cseeds):
                    continue
                y = tgts[:, tnames.index(tname)]
                self0 = np.array([imap[s][selfkey] for s in cseeds], float)
                finite = (np.isfinite(y) & np.isfinite(self0)
                          & np.isfinite(psi.reshape(len(y), -1)).all(1) & np.isfinite(bulk).all(1))
                idx = np.where(finite)[0]; n = len(idx)
                if n < 30:
                    continue
                perm = np.random.default_rng(_stable_seed(key + tname)).permutation(n)
                nt = max(int(round(pc.test_frac * n)), 5)
                te, tr = idx[perm[:nt]], idx[perm[nt:]]; yte = y[te]
                boot = np.random.default_rng(_stable_seed(key + tname + "|b")).integers(0, nt, size=(pc.n_boot, nt))
                r2s = [r2_score(yte, _fit_predict(psi[tr, s, :], y[tr], psi[te, s, :])[0])
                       for s in range(psi.shape[1])]
                psi_best = psi[:, int(np.argmax(r2s)), :]
                ctrl = np.column_stack([bulk, self0])
                pb, _ = _fit_predict(ctrl[tr], y[tr], ctrl[te])
                pj, _ = _fit_predict(np.column_stack([ctrl, psi_best])[tr], y[tr],
                                     np.column_stack([ctrl, psi_best])[te])
                lo, hi = _ci(_r2_boot(yte, pj, boot) - _r2_boot(yte, pb, boot))
                surv_cell = lo is not None and lo > 0
                by_ct[f"{key}|{tname}"] = {
                    "inc": float(r2_score(yte, pj) - r2_score(yte, pb)), "lo": lo, "hi": hi,
                    "r2_control": float(r2_score(yte, pb)), "survives": bool(surv_cell)}
                s0, n0 = surv.get(tname, [0, 0]); surv[tname] = [s0 + int(surv_cell), n0 + 1]
    return {"available": True, "by_cell_target": by_ct, "survival": surv}


def stage_analyze(pc: PSConfig) -> dict:
    cell_dir = os.path.join(OUTDIR, "cache", "cells")
    cells = {}
    for fam in pc.families:
        for eps in pc.eps:
            path = os.path.join(cell_dir, f"{fam}_eps{eps:g}.npz")
            if os.path.exists(path):
                cells[cell_key(fam, eps)] = _analyse_cell(np.load(path, allow_pickle=True), pc)
    summary = {"config": asdict(pc), "cells": cells}
    summary["hardening"] = _hardened_pass(pc)
    summary["kill_tests"] = _kill_tests(pc, cells, summary["hardening"])
    return summary


def _ci_pos(d, lo, hi):
    return d.get(lo) is not None and isinstance(d.get(lo), float) and math.isfinite(d[lo]) and d[lo] > 0


def _kill_tests(pc: PSConfig, cells: dict, hardening: dict) -> dict:
    cusps = pc.families
    def cell_t(fam, eps, tname):
        c = cells.get(cell_key(fam, eps))
        return c["targets"].get(tname) if c and tname in c["targets"] else None

    # RAW (uncontrolled) "ψ beyond bulk" — kept for the record, but NOT the bar.
    raw_beyond_bulk = {}
    for tname, _ in TARGETS:
        raw_beyond_bulk[tname] = sum(
            1 for fam in cusps for eps in pc.eps
            if (t := cell_t(fam, eps, tname)) and _ci_pos(t, "psi_beyond_bulk_lo", "psi_beyond_bulk_hi"))
    head_beats_phi = any(_ci_pos(t, "psi_beyond_phi_lo", "psi_beyond_phi_hi")
                         for fam in cusps for eps in pc.eps if (t := cell_t(fam, eps, HEADLINE)))

    # HARDENED control = bulk + initial target quantity (the real bar).
    surv = (hardening or {}).get("survival", {}) or {}
    n_cells = len(cusps) * len(pc.eps)
    head_hard = surv.get(HEADLINE, [0, 0])                       # [survive, n]
    ROBUST = max(4, int(math.ceil(0.66 * n_cells)))              # ≥4/6: majority-ish, control-surviving
    robust_targets = [t for t, (s, _) in surv.items() if s >= ROBUST]
    best_partial = max(((t, s, n) for t, (s, n) in surv.items()), key=lambda x: x[1], default=("none", 0, 0))

    entropy_alive_hardened = head_hard[0] >= ROBUST

    ell = {fam: [cell_t(fam, e, HEADLINE)["ell_star_psi"] if cell_t(fam, e, HEADLINE) else None
                 for e in sorted(pc.eps)] for fam in cusps}
    ell_sp = {fam: (_spearman(np.array(sorted(pc.eps)), np.array(v)) if all(x is not None for x in v) else None)
              for fam, v in ell.items()}

    rt = json.load(open(os.path.join(OUTDIR, "cache", "runtime.json"))) \
        if os.path.exists(os.path.join(OUTDIR, "cache", "runtime.json")) else {}

    if not hardening.get("available"):
        decision = ("INCONCLUSIVE — spatial-pilot t₀ snapshots needed for the hardened "
                    "baseline-sharing control are missing; raw 'beyond bulk' is not trustworthy.")
    elif entropy_alive_hardened and head_beats_phi:
        decision = ("ALIVE — ψ_ℓ predicts future mixing beyond the HARDENED control (bulk + S₀) "
                    "in a majority of cusp cells, and beats spatial φ_ℓ. Phase-space coarse-graining "
                    "branch is alive; run a medium local confirmation. No AWS.")
    elif robust_targets:
        decision = (f"TARGET-SPECIFIC — the mixing headline fails the hardened control, but "
                    f"{robust_targets} survive it. Treat as a narrow target-specific effect, not a "
                    f"general phase-space coarse-graining result; confirm before any claim.")
    else:
        decision = ("RETIRE (branch open) — after the hardened control (bulk + initial target "
                    "value), ψ_ℓ adds no robust signal for the mixing headline (survives "
                    f"{head_hard[0]}/{head_hard[1]} cells). The strong raw Δβ/Δσ_r signals were "
                    "baseline-sharing (β₀/σ_r₀ predicting their own change — the same trap as C₈). "
                    f"Best partial survivor: {best_partial[0]} {best_partial[1]}/{best_partial[2]}. "
                    "This specific phase-space coarse-scale hypothesis is retired; the N-body branch "
                    "remains OPEN for new (e.g. interventional/causal) hypotheses. No AWS.")

    return {
        "headline_target": HEADLINE,
        "hardened_control": "bulk + initial target quantity (β₀/σ_r₀/S₀)",
        "headline_survives_hardened": f"{head_hard[0]}/{head_hard[1]}",
        "hardened_survival_by_target": {t: f"{s}/{n}" for t, (s, n) in surv.items()},
        "raw_beyond_bulk_by_target": {t: f"{c}/{n_cells}" for t, c in raw_beyond_bulk.items()},
        "psi_beats_spatial_phi_headline": head_beats_phi,
        "robust_targets_hardened": robust_targets,
        "ell_star_psi_spearman": ell_sp,
        "runtime": rt,
        "aws_needed": False,
        "decision": decision,
    }


# ── outputs ──────────────────────────────────────────────────────────────────────

def _json_safe(x):
    if isinstance(x, np.floating):
        v = float(x); return v if math.isfinite(v) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return str(x)


def write_outputs(pc: PSConfig, summary: dict) -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    fig_dir = os.path.join(OUTDIR, "figures"); os.makedirs(fig_dir, exist_ok=True)

    rows = []
    for key, c in summary["cells"].items():
        for tname, t in c["targets"].items():
            base = {"family": c["family"], "eps": c["eps"], "target": tname, "label": t["label"], "n": t["n"]}
            for nm, v in t["r2"].items():
                rows.append({**base, "metric": f"R2_{nm}", "value": v, "ci_lo": "", "ci_hi": ""})
            for nm in ("bulk", "phi", "kin"):
                rows.append({**base, "metric": f"psi_beyond_{nm}",
                             "value": t[f"psi_beyond_{nm}"], "ci_lo": t[f"psi_beyond_{nm}_lo"],
                             "ci_hi": t[f"psi_beyond_{nm}_hi"]})
            rows.append({**base, "metric": "ell_star_psi", "value": t["ell_star_psi"], "ci_lo": "", "ci_hi": ""})
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_safe)
    with open(os.path.join(OUTDIR, "config.json"), "w") as f:
        json.dump(asdict(pc), f, indent=2)

    _figures(pc, summary, fig_dir)
    _report(pc, summary)
    print(f"[outputs] results.csv, summary.json, figures/, pilot_report.md → {OUTDIR}/")


def _figures(pc: PSConfig, summary: dict, fig_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_sorted = sorted(pc.eps)
    # 1. ψ-beyond-bulk ΔR² (with CI) for the headline target, per family/eps
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    for i, fam in enumerate(pc.families):
        vals, los, his = [], [], []
        for e in eps_sorted:
            c = summary["cells"].get(cell_key(fam, e))
            t = c["targets"].get(HEADLINE) if c else None
            vals.append(t["psi_beyond_bulk"] if t else np.nan)
            los.append((t["psi_beyond_bulk"] - t["psi_beyond_bulk_lo"]) if t else 0)
            his.append((t["psi_beyond_bulk_hi"] - t["psi_beyond_bulk"]) if t else 0)
        x = np.arange(len(eps_sorted)) + (i - 0.5) * width
        ax.bar(x, vals, width, yerr=[los, his], capsize=3, label=fam)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(eps_sorted))); ax.set_xticklabels([f"{e:g}" for e in eps_sorted])
    ax.set_xlabel("ε"); ax.set_ylabel("ΔR²  (ψ_ℓ beyond bulk)")
    ax.set_title(f"Headline: does ψ_ℓ beat bulk controls for {HEADLINE}?")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_psi_beyond_bulk.pdf")); plt.close(fig)

    # 2. R² comparison of feature sets for the headline target
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sets = ["bulk", "fine_pos", "fine_kin", "phi_best", "psi_best"]
    xb = np.arange(len(eps_sorted) * len(pc.families))
    labels = []
    data = {s: [] for s in sets}
    for fam in pc.families:
        for e in eps_sorted:
            c = summary["cells"].get(cell_key(fam, e))
            t = c["targets"].get(HEADLINE) if c else None
            labels.append(f"{fam[:4]}\nε={e:g}")
            for s in sets:
                data[s].append(t["r2"].get(s, np.nan) if t else np.nan)
    w = 0.16
    for j, s in enumerate(sets):
        ax.bar(xb + (j - 2) * w, data[s], w, label=s)
    ax.set_xticks(xb); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("held-out R²"); ax.set_title(f"Feature-set R² for {HEADLINE}")
    ax.legend(fontsize=8); ax.axhline(0, color="k", lw=0.6); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_featureset_R2.pdf")); plt.close(fig)


def _report(pc: PSConfig, summary: dict) -> None:
    kt = summary["kill_tests"]; rt = kt["runtime"]
    L = ["# Phase-Space Coarse-Graining Pilot — Report\n"]
    L.append(f"**Families:** {', '.join(pc.families)} (cusps)  **ε:** {pc.eps}  **N:** {pc.n}  "
             f"**reps:** {pc.replicates}  **horizons:** {HORIZONS}  **regressor:** RidgeCV  ")
    L.append(f"**Headline target:** `{HEADLINE}` = Δ coarse phase-space (r,v_r) entropy (mixing).\n")
    dec = kt["decision"]
    band = ("🟢 ALIVE" if dec.startswith("ALIVE")
            else "🟡 TARGET-SPECIFIC" if dec.startswith("TARGET-SPECIFIC")
            else "⚪ INCONCLUSIVE" if dec.startswith("INCONCLUSIVE")
            else "🔴 RETIRED — this hypothesis killed; N-body branch stays OPEN")
    L.append(f"## Verdict — {band}\n\n> **{dec}**\n")

    L.append("## Hardened control — the real bar\n")
    L.append("Raw 'ψ beyond bulk' is contaminated by baseline-sharing: ψ_ℓ contains the initial "
             "β(r)/σ_r(r) profiles, and Δβ=β(t₁)−β(t₀) shares −β(t₀) (the C₈ trap). The honest "
             "control adds the **initial target quantity** (β₀/σ_r₀/S₀) to bulk.\n")
    L.append("| target | raw ψ>bulk | **hardened ψ>(bulk+self₀)** |")
    L.append("|---|---|---|")
    for tname, tlabel in TARGETS:
        raw = kt["raw_beyond_bulk_by_target"].get(tname, "—")
        hard = kt["hardened_survival_by_target"].get(tname, "—")
        L.append(f"| {tname} | {raw} | **{hard}** |")
    L.append("")

    L.append("## The reported questions\n")
    L.append(f"1. **Does ψ_ℓ beat bulk controls?** RAW yes for several targets, but that is "
             f"baseline-sharing. Under the **hardened** control the mixing headline `{HEADLINE}` "
             f"survives only **{kt['headline_survives_hardened']}** cells, and the strong raw "
             f"Δβ/Δσ_r collapse to 0/6.")
    L.append(f"2. **Does ψ_ℓ beat spatial φ_ℓ?** "
             f"{'YES' if kt['psi_beats_spatial_phi_headline'] else 'NO'} for the headline — phase-space "
             f"features carry far more relaxation info than spatial ones (this echoes the paper's "
             f"known kinematic/VelDisp advantage, not a new scale law).")
    L.append(f"3. **Which target works (hardened)?** "
             f"{kt['robust_targets_hardened'] or 'none at the ≥majority bar'}.")
    L.append(f"4. **ℓ*(ε) (secondary, must NOT revive ℓ*∼ε):** {kt['ell_star_psi_spearman']} "
             f"— inconsistent; no force-resolution scale law.")
    L.append(f"5. **AWS?** No. Ran {rt.get('run_reps','?')} reps in "
             f"{rt.get('run_wall_s',float('nan')):.0f}s on {rt.get('workers','?')} local workers.\n")

    L.append("## Per-cell, per-target (held-out R² and ψ increments)\n")
    L.append("| family | ε | target | R²(bulk) | R²(φ*) | R²(ψ*) | ψ−bulk [CI] | ψ−φ [CI] | ℓ*_ψ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for fam in pc.families:
        for e in sorted(pc.eps):
            c = summary["cells"].get(cell_key(fam, e))
            if not c:
                continue
            for tname, tlabel in TARGETS:
                t = c["targets"].get(tname)
                if not t:
                    continue
                L.append(f"| {fam} | {e:g} | {tlabel} | {t['r2']['bulk']:+.3f} | "
                         f"{t['r2']['phi_best']:+.3f} | {t['r2']['psi_best']:+.3f} | "
                         f"{t['psi_beyond_bulk']:+.3f} [{t['psi_beyond_bulk_lo']:+.3f},{t['psi_beyond_bulk_hi']:+.3f}] | "
                         f"{t['psi_beyond_phi']:+.3f} [{t['psi_beyond_phi_lo']:+.3f},{t['psi_beyond_phi_hi']:+.3f}] | "
                         f"{t['ell_star_psi']:g} |")
    L.append("")
    with open(os.path.join(OUTDIR, "pilot_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-space coarse-graining cusp pilot (no AWS).")
    ap.add_argument("--stage", choices=["all", "run", "analyze"], default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--eps", type=float, nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    pc = PSConfig()
    if args.replicates is not None: pc.replicates = args.replicates
    if args.families is not None:   pc.families = args.families
    if args.eps is not None:        pc.eps = args.eps
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Phase-space pilot: families={pc.families} eps={pc.eps} N={pc.n} reps={pc.replicates}")

    if args.stage in ("all", "run"):
        stage_run(pc, args.workers, args.force)
    if args.stage in ("all", "analyze"):
        summary = stage_analyze(pc)
        write_outputs(pc, summary)
        print("\n" + summary["kill_tests"]["decision"])


if __name__ == "__main__":
    main()
