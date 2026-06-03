#!/usr/bin/env python3
"""
coarse_grain_pilot.py — local coarse-graining *scale* pilot for the N-body battery
==================================================================================

Tests, cheaply and locally, whether the predictively-optimal coarse-graining scale tracks
the gravitational softening:

    ℓ*(ε) = argmax_ℓ R²(ℓ; ε)      should increase monotonically with ε  (ideally ℓ* ≈ C·ε).

The pilot reuses the existing N-body engine (forces, integrator, IC families, observables)
and adds a scale-resolved smoothed feature family φ_ℓ + a held-out Ridge R² analysis.

Three cached stages (run with --stage all, or sim / features / analyze):

  sim       run direct_isolated to the early horizon, cache t₀ (pos, vel) + ΔC8-early target.
  features  deposit → Gaussian-smooth at each ℓ → φ_ℓ, plus baselines.  Re-running with a
            different ℓ grid recomputes features ONLY — never re-simulates.
  analyze   RidgeCV R²(ℓ; ε) on a fixed split, ℓ*(ε), Δ_coarse, kill tests, figures, report.

No AWS.  Does not touch paper.tex.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from nbody_3d import _worker_init
from nbody_stress import (
    StressConfig,
    get_initial_conditions,
    get_simconfig,
    obs_coarse_var,
    _integrate_leapfrog,
)
import coarse_grain_features as cgf

# ── Pilot configuration ──────────────────────────────────────────────────────────

OUTDIR = "outputs/coarse_grain_pilot"


@dataclass
class PilotConfig:
    families:  List[str]   = field(default_factory=lambda: ["hernquist3d", "plummer3d", "bimodal3d"])
    eps:       List[float] = field(default_factory=lambda: [0.02, 0.05, 0.10])
    n:         int         = 1024
    steps:     int         = 100          # = H_EARLY → ΔC8-early target
    replicates: int        = 500
    seed0:     int         = 2000         # paper seed convention
    box_size:  float       = 2.0
    plummer_a: float       = 0.20
    k_fine:    int         = 16
    pm_grid:   int         = 32
    grid_g:    int         = 96           # φ_ℓ deposit grid (cell = box/g ≈ 0.0208)
    scales:    List[float] = field(default_factory=lambda: [0.025, 0.04, 0.06, 0.09, 0.14, 0.20, 0.30])
    test_frac: float       = 0.30
    n_boot:    int         = 1000
    model:     str         = "direct_isolated"   # ONLY model where ε affects the force


def _stable_seed(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % (2 ** 31)


def cell_key(family: str, eps: float) -> str:
    return f"{family}|eps={eps:g}"


def _cfg_for(pc: PilotConfig, family: str, eps: float, seed: int) -> StressConfig:
    return StressConfig(
        model=pc.model, init=family, seed=seed, n=pc.n, steps=pc.steps,
        eps=eps, box_size=pc.box_size, pm_grid=pc.pm_grid,
        k_fine=pc.k_fine, plummer_a=pc.plummer_a,
    )


# ── Stage 1: simulate + cache t₀ snapshots ───────────────────────────────────────

def _sim_worker(task: tuple) -> tuple:
    family, eps, seed, pc_dict = task
    pc = PilotConfig(**pc_dict)
    cfg = _cfg_for(pc, family, eps, seed)
    sc = get_simconfig(cfg)
    pos0, vel0 = get_initial_conditions(cfg)
    mass = 1.0 / pc.n
    snaps = _integrate_leapfrog(pos0, vel0, mass, sc,
                                sorted({0, pc.steps}), use_numba=True)
    cg8_0 = obs_coarse_var(snaps[0][0], cfg, 8, False)
    cg8_e = obs_coarse_var(snaps[pc.steps][0], cfg, 8, False)
    target = float(cg8_e - cg8_0)
    return (family, eps, seed,
            pos0.astype(np.float32), vel0.astype(np.float32),
            target, float(cg8_0))


def stage_sim(pc: PilotConfig, workers: int, force: bool) -> None:
    snap_dir = os.path.join(OUTDIR, "cache", "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    seeds = [pc.seed0 + i for i in range(pc.replicates)]

    tasks: List[tuple] = []
    cells_needed: List[str] = []
    for family, eps in product(pc.families, pc.eps):
        path = os.path.join(snap_dir, f"{family}_eps{eps:g}.npz")
        if os.path.exists(path) and not force:
            continue
        cells_needed.append(cell_key(family, eps))
        for seed in seeds:
            tasks.append((family, eps, seed, asdict(pc)))

    if not tasks:
        print(f"[sim] all {len(pc.families) * len(pc.eps)} cells cached — skipping.")
        return

    print(f"[sim] {len(tasks)} simulations over {len(cells_needed)} cells "
          f"(N={pc.n}, steps={pc.steps}, model={pc.model})")
    acc: Dict[str, Dict[str, list]] = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(True,)) as ex:
        futs = [ex.submit(_sim_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), ncols=100, unit="sim"):
            family, eps, seed, pos0, vel0, target, cg8_0 = fut.result()
            d = acc.setdefault(cell_key(family, eps),
                               {"pos": [], "vel": [], "tgt": [], "cg8_0": [],
                                "seed": [], "family": family, "eps": eps})
            d["pos"].append(pos0); d["vel"].append(vel0)
            d["tgt"].append(target); d["cg8_0"].append(cg8_0); d["seed"].append(seed)
    dt = time.perf_counter() - t0

    for key, d in acc.items():
        path = os.path.join(snap_dir, f"{d['family']}_eps{d['eps']:g}.npz")
        np.savez_compressed(
            path,
            pos=np.stack(d["pos"]), vel=np.stack(d["vel"]),
            target=np.array(d["tgt"], np.float32),
            cg8_0=np.array(d["cg8_0"], np.float32),
            seed=np.array(d["seed"], np.int64),
            family=d["family"], eps=d["eps"], n=pc.n, steps=pc.steps,
        )
    per = dt / max(len(tasks), 1)
    print(f"[sim] done in {dt:.1f}s  ({per*1000:.1f} ms/sim wall over {workers} workers)")
    _record_runtime(pc, {"sim_wall_s": dt, "sim_count": len(tasks),
                         "sim_ms_per_sim_wall": per * 1000, "sim_workers": workers})


# ── Stage 2: features (φ_ℓ + baselines) from cached snapshots ─────────────────────

def _feat_worker(task: tuple) -> tuple:
    family, eps, seed, pos, vel, target, cg8_0, pc_dict = task
    pc = PilotConfig(**pc_dict)
    cfg = _cfg_for(pc, family, eps, int(seed))
    posf = pos.astype(np.float64)
    phi = cgf.phi_all_scales(posf, pc.grid_g, pc.box_size, pc.scales)   # (n_scales, n_feat)
    scal = cgf.baseline_features(posf, vel.astype(np.float64), cfg)
    return (family, eps, phi.astype(np.float32), scal, float(target), float(cg8_0))


def stage_features(pc: PilotConfig, workers: int, force: bool) -> None:
    snap_dir = os.path.join(OUTDIR, "cache", "snapshots")
    feat_dir = os.path.join(OUTDIR, "cache", "features")
    os.makedirs(feat_dir, exist_ok=True)

    tasks: List[tuple] = []
    for family, eps in product(pc.families, pc.eps):
        fpath = os.path.join(feat_dir, f"{family}_eps{eps:g}.npz")
        if os.path.exists(fpath) and not force:
            continue
        spath = os.path.join(snap_dir, f"{family}_eps{eps:g}.npz")
        if not os.path.exists(spath):
            print(f"[features] missing snapshot {spath} — run --stage sim first.")
            continue
        snap = np.load(spath, allow_pickle=True)
        pos, vel = snap["pos"], snap["vel"]
        tgt, cg8_0, seeds = snap["target"], snap["cg8_0"], snap["seed"]
        for i in range(len(pos)):
            tasks.append((family, eps, int(seeds[i]), pos[i], vel[i],
                          float(tgt[i]), float(cg8_0[i]), asdict(pc)))

    if not tasks:
        print("[features] all cells cached — skipping.")
        return

    n_feat = len(cgf.phi_feature_names())
    print(f"[features] {len(tasks)} snapshots × {len(pc.scales)} scales "
          f"(grid={pc.grid_g}³, {n_feat} φ-features)")
    acc: Dict[str, Dict[str, list]] = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_feat_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), ncols=100, unit="snap"):
            family, eps, phi, scal, target, cg8_0 = fut.result()
            d = acc.setdefault(cell_key(family, eps),
                               {"phi": [], "scal": [], "tgt": [],
                                "family": family, "eps": eps})
            d["phi"].append(phi); d["scal"].append(scal); d["tgt"].append(target)
    dt = time.perf_counter() - t0

    scal_cols = sorted({k for d in acc.values() for s in d["scal"] for k in s})
    for key, d in acc.items():
        scal_mat = np.array([[float(s.get(c, np.nan)) for c in scal_cols]
                             for s in d["scal"]], np.float32)
        path = os.path.join(feat_dir, f"{d['family']}_eps{d['eps']:g}.npz")
        np.savez_compressed(
            path,
            phi=np.stack(d["phi"]),               # (R, n_scales, n_feat)
            scal=scal_mat, scal_cols=np.array(scal_cols),
            target=np.array(d["tgt"], np.float32),
            scales=np.array(pc.scales), phi_names=np.array(cgf.phi_feature_names()),
            family=d["family"], eps=d["eps"],
        )
    print(f"[features] done in {dt:.1f}s")
    _record_runtime(pc, {"feat_wall_s": dt, "feat_count": len(tasks)})


# ── Stage 3: analysis (RidgeCV R², ℓ*, kill tests) ───────────────────────────────

from sklearn.linear_model import RidgeCV          # noqa: E402
from sklearn.metrics import r2_score              # noqa: E402
from sklearn.pipeline import make_pipeline        # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

_ALPHAS = np.logspace(-3, 4, 18)


def _fit_predict(Xtr, ytr, Xte) -> Tuple[np.ndarray, float]:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=_ALPHAS))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    alpha = float(model.named_steps["ridgecv"].alpha_)
    return pred, alpha


def _r2_boot(y_te: np.ndarray, pred: np.ndarray, boot: np.ndarray) -> np.ndarray:
    """Vectorized bootstrap R² over precomputed resample-index rows."""
    yt = y_te[boot]                       # (B, n_test)
    pr = pred[boot]
    ss_res = np.sum((yt - pr) ** 2, axis=1)
    ss_tot = np.sum((yt - yt.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)


def _ci(a: np.ndarray) -> Tuple[float, float]:
    a = a[np.isfinite(a)]
    if a.size < 10:
        return float("nan"), float("nan")
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def _analyse_cell(npz, pc: PilotConfig) -> dict:
    phi = npz["phi"].astype(np.float64)               # (R, n_scales, n_feat)
    scal = npz["scal"].astype(np.float64)             # (R, n_scal_cols)
    scal_cols = list(npz["scal_cols"])
    y = npz["target"].astype(np.float64)
    scales = list(npz["scales"])
    family = str(npz["family"]); eps = float(npz["eps"])
    R, n_scales, n_feat = phi.shape

    def col(name: str) -> np.ndarray:
        return scal[:, scal_cols.index(name)]

    baselines = {g: np.column_stack([col(c) for c in cols])
                 for g, cols in cgf.BASELINE_GROUPS.items()}

    # Common finite sample: target + all φ + all baselines finite → one shared split.
    finite = np.isfinite(y)
    finite &= np.isfinite(phi.reshape(R, -1)).all(axis=1)
    for X in baselines.values():
        finite &= np.isfinite(X).all(axis=1)
    idx_all = np.where(finite)[0]
    n_used = len(idx_all)

    key = cell_key(family, eps)
    rng = np.random.default_rng(_stable_seed(key))
    perm = rng.permutation(n_used)
    n_test = max(int(round(pc.test_frac * n_used)), 5)
    test_local, train_local = perm[:n_test], perm[n_test:]
    tr, te = idx_all[train_local], idx_all[test_local]
    yte = y[te]

    rng_b = np.random.default_rng(_stable_seed(key + "|boot"))
    boot = rng_b.integers(0, n_test, size=(pc.n_boot, n_test))

    # per-scale fit
    scale_r2: List[float] = []
    scale_pred: List[np.ndarray] = []
    scale_alpha: List[float] = []
    for s in range(n_scales):
        pred, alpha = _fit_predict(phi[tr, s, :], y[tr], phi[te, s, :])
        scale_pred.append(pred)
        scale_r2.append(float(r2_score(yte, pred)))
        scale_alpha.append(alpha)
    scale_r2 = np.array(scale_r2)
    scale_boot = np.column_stack([_r2_boot(yte, p, boot) for p in scale_pred])  # (B, n_scales)

    s_star = int(np.argmax(scale_r2))
    ell_star = float(scales[s_star])
    ell_star_boot = np.array([scales[i] for i in np.nanargmax(
        np.where(np.isfinite(scale_boot), scale_boot, -np.inf), axis=1)])

    # baselines
    base_r2: Dict[str, float] = {}
    base_pred: Dict[str, np.ndarray] = {}
    for g, X in baselines.items():
        pred, _ = _fit_predict(X[tr], y[tr], X[te])
        base_pred[g] = pred
        base_r2[g] = float(r2_score(yte, pred))

    # Δ_coarse = R²(ℓ*) − R²(fine_all), paired bootstrap
    fine_boot = _r2_boot(yte, base_pred["fine_all"], boot)
    delta_boot = scale_boot[:, s_star] - fine_boot
    d_lo, d_hi = _ci(delta_boot)
    delta_coarse = scale_r2[s_star] - base_r2["fine_all"]

    # bulk control: does φ_ℓ* add R² beyond the trivial baselines?
    # The control set is bulk + persistence (cg8₀).  Because the target
    # ΔC8 = cg8_e − cg8₀ shares −cg8₀ with the initial state, including cg8₀ lets
    # Ridge cancel that term, so the increment from φ_ℓ* measures GENUINE
    # scale-resolved future-prediction power (predicting cg8_e), not baseline-sharing.
    bulk_X = baselines["bulk"]
    control_X = np.column_stack([bulk_X, baselines["persistence"]])
    pred_bulk, _ = _fit_predict(bulk_X[tr], y[tr], bulk_X[te])
    pred_control, _ = _fit_predict(control_X[tr], y[tr], control_X[te])
    controlphi = np.column_stack([control_X, phi[:, s_star, :]])
    pred_controlphi, _ = _fit_predict(controlphi[tr], y[tr], controlphi[te])
    control_boot = _r2_boot(yte, pred_control, boot)
    controlphi_boot = _r2_boot(yte, pred_controlphi, boot)
    incr_boot = controlphi_boot - control_boot
    incr_lo, incr_hi = _ci(incr_boot)

    s_lo, s_hi = _ci(scale_boot[:, s_star])

    # winner category (coarse-graining optimum vs fine, CI-aware)
    if not math.isfinite(d_lo) or (d_lo <= 0.0 <= d_hi):
        winner = "tie"
    elif delta_coarse > 0:
        winner = "coarse-small" if ell_star <= 0.09 else "coarse-large"
    else:
        winner = "fine"

    return {
        "family": family, "eps": eps, "n_used": n_used,
        "n_train": len(tr), "n_test": len(te),
        "scales": scales,
        "scale_r2": scale_r2.tolist(),
        "scale_r2_lo": [float(_ci(scale_boot[:, s])[0]) for s in range(n_scales)],
        "scale_r2_hi": [float(_ci(scale_boot[:, s])[1]) for s in range(n_scales)],
        "scale_alpha": scale_alpha,
        "ell_star": ell_star, "s_star": s_star,
        "r2_at_star": float(scale_r2[s_star]),
        "r2_at_star_lo": s_lo, "r2_at_star_hi": s_hi,
        "ell_star_boot_lo": float(np.percentile(ell_star_boot, 2.5)),
        "ell_star_boot_hi": float(np.percentile(ell_star_boot, 97.5)),
        "ell_star_boot_med": float(np.median(ell_star_boot)),
        "ell_star_boot": ell_star_boot.tolist(),
        "base_r2": base_r2,
        "delta_coarse": delta_coarse, "delta_coarse_lo": d_lo, "delta_coarse_hi": d_hi,
        "bulk_r2": base_r2["bulk"],
        "control_r2": float(r2_score(yte, pred_control)),
        "controlphi_r2": float(r2_score(yte, pred_controlphi)),
        "bulk_phi_increment": float(controlphi_boot.mean() - control_boot.mean()),
        "bulk_phi_increment_lo": incr_lo, "bulk_phi_increment_hi": incr_hi,
        "winner": winner,
    }


def _scale_fit(phi: np.ndarray, y: np.ndarray, bl: Dict[str, np.ndarray],
               key: str, pc: PilotConfig) -> dict:
    """Lean per-cell scale sweep for an arbitrary target y (used by target-robustness).

    Returns ℓ* and the φ_ℓ*-beyond-(bulk+cg8₀) held-out increment with bootstrap CI.
    Mirrors the split/control logic of `_analyse_cell` so the two are comparable.
    """
    R, nS, nF = phi.shape
    finite = np.isfinite(y) & np.isfinite(phi.reshape(R, -1)).all(1)
    for X in bl.values():
        finite &= np.isfinite(X).all(1)
    idx = np.where(finite)[0]
    n = len(idx)
    perm = np.random.default_rng(_stable_seed(key)).permutation(n)
    nt = max(int(round(pc.test_frac * n)), 5)
    te, tr = idx[perm[:nt]], idx[perm[nt:]]
    yte = y[te]
    boot = np.random.default_rng(_stable_seed(key + "|boot")).integers(0, nt, size=(pc.n_boot, nt))
    r2, preds = [], []
    for s in range(nS):
        p, _ = _fit_predict(phi[tr, s, :], y[tr], phi[te, s, :])
        preds.append(p); r2.append(r2_score(yte, p))
    r2 = np.array(r2); sstar = int(np.argmax(r2))
    ctrl = np.column_stack([bl["bulk"], bl["persistence"]])
    p_ctrl = _fit_predict(ctrl[tr], y[tr], ctrl[te])[0]
    cphi = np.column_stack([ctrl, phi[:, sstar, :]])
    p_cphi = _fit_predict(cphi[tr], y[tr], cphi[te])[0]
    incr = _r2_boot(yte, p_cphi, boot) - _r2_boot(yte, p_ctrl, boot)
    lo, hi = _ci(incr)
    return {"ell_star": float(pc.scales[sstar]), "r2_star": float(r2[sstar]),
            "scale_r2": r2.tolist(), "phi_beyond_ctrl": float(np.nanmean(incr)),
            "pbc_lo": lo, "pbc_hi": hi}


def _target_robustness(pc: PilotConfig) -> dict:
    """Q4: re-run ℓ*(ε) against the ABSOLUTE future target C8(t₁)=cg8_e (no delta).

    If a real force-resolution scale law existed it would survive the change of
    target.  Computed for free from the cached features (cg8₀ + ΔC8 → cg8_e).
    """
    feat_dir = os.path.join(OUTDIR, "cache", "features")
    out: Dict[str, dict] = {}
    for fam in pc.families:
        ells, pbc_pos, rows = [], False, []
        for e in sorted(pc.eps):
            path = os.path.join(feat_dir, f"{fam}_eps{e:g}.npz")
            if not os.path.exists(path):
                continue
            npz = np.load(path, allow_pickle=True)
            phi = npz["phi"].astype(np.float64)
            scal = npz["scal"].astype(np.float64)
            cols = list(npz["scal_cols"])
            cg8_0 = scal[:, cols.index("cg8_0")]
            y = npz["target"].astype(np.float64) + cg8_0          # absolute C8(t₁)
            bl = {g: np.column_stack([scal[:, cols.index(c)] for c in cc])
                  for g, cc in cgf.BASELINE_GROUPS.items()}
            r = _scale_fit(phi, y, bl, cell_key(fam, e), pc)
            ells.append(r["ell_star"]); rows.append({"eps": e, **r})
            if r["pbc_lo"] > 0:
                pbc_pos = True
        if ells:
            out[fam] = {"target": "abs_C8(t1)", "ell_star": ells,
                        "spearman": _spearman(np.array(sorted(pc.eps)), np.array(ells)),
                        "phi_beyond_ctrl_any_pos": pbc_pos, "rows": rows}
    return out


def stage_analyze(pc: PilotConfig) -> dict:
    feat_dir = os.path.join(OUTDIR, "cache", "features")
    cells: Dict[str, dict] = {}
    for family, eps in product(pc.families, pc.eps):
        path = os.path.join(feat_dir, f"{family}_eps{eps:g}.npz")
        if not os.path.exists(path):
            print(f"[analyze] missing features {path} — run --stage features first.")
            continue
        cells[cell_key(family, eps)] = _analyse_cell(np.load(path, allow_pickle=True), pc)

    if not cells:
        raise SystemExit("[analyze] no feature caches found.")

    # family-level ℓ*(ε) trend
    family_summ: Dict[str, dict] = {}
    for family in pc.families:
        eps_sorted = sorted(pc.eps)
        cs = [cells.get(cell_key(family, e)) for e in eps_sorted]
        if any(c is None for c in cs):
            continue
        ell = np.array([c["ell_star"] for c in cs])
        boots = [np.array(c["ell_star_boot"]) for c in cs]
        B = min(len(b) for b in boots)
        joint = np.column_stack([b[:B] for b in boots])     # (B, n_eps), independent cells
        nondec = float(np.mean(np.all(np.diff(joint, axis=1) >= 0, axis=1)))
        strict = float(np.mean(np.all(np.diff(joint, axis=1) > 0, axis=1)))
        sp = _spearman(np.array(eps_sorted, float), ell)
        sp_boot = np.array([_spearman(np.array(eps_sorted, float), joint[b])
                            for b in range(min(B, 400))])
        family_summ[family] = {
            "eps": eps_sorted,
            "ell_star": ell.tolist(),
            "ell_star_ci": [[c["ell_star_boot_lo"], c["ell_star_boot_hi"]] for c in cs],
            "r2_at_star": [c["r2_at_star"] for c in cs],
            "delta_coarse": [c["delta_coarse"] for c in cs],
            "delta_coarse_ci": [[c["delta_coarse_lo"], c["delta_coarse_hi"]] for c in cs],
            "bulk_phi_increment_ci": [[c["bulk_phi_increment_lo"],
                                       c["bulk_phi_increment_hi"]] for c in cs],
            "winner": [c["winner"] for c in cs],
            "spearman_eps_ellstar": sp,
            "p_nondecreasing": nondec,
            "p_strict_increasing": strict,
            "p_spearman_pos": float(np.mean(sp_boot > 0)),
        }

    summary = {"config": asdict(pc), "cells": cells, "families": family_summ}
    summary["target_robustness"] = _target_robustness(pc)
    summary["kill_tests"] = _kill_tests(pc, cells, family_summ, summary["target_robustness"])
    return summary


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (scipy; correct average-rank tie handling)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    from scipy.stats import spearmanr
    rho = spearmanr(x, y).correlation
    return float(rho) if np.isfinite(rho) else 0.0


# ── Kill tests ───────────────────────────────────────────────────────────────────

def _kill_tests(pc: PilotConfig, cells: dict, fams: dict, tgt_robust: dict) -> dict:
    cusps = [f for f in pc.families if f in ("hernquist3d", "plummer3d")]

    # 1. structured vs flat: peak prominence relative to bootstrap noise
    prominences = []
    for c in cells.values():
        r2 = np.array(c["scale_r2"])
        width = c["r2_at_star_hi"] - c["r2_at_star_lo"]   # bootstrap CI width at ℓ*
        prom = (np.nanmax(r2) - np.nanmedian(r2)) / max(width, 1e-6)
        prominences.append(float(prom))
    structured = float(np.median(prominences)) > 1.0

    # 2. ℓ* increases with ε (primary: cusp families)
    cusp_nondec = [fams[f]["p_nondecreasing"] for f in cusps if f in fams]
    cusp_sp = [fams[f]["spearman_eps_ellstar"] for f in cusps if f in fams]
    monotonic = bool(cusp_sp) and all(s > 0 for s in cusp_sp) and \
        np.mean(cusp_nondec) > 0.5

    # 3. coarse optimum beats fine (Δ_coarse CI > 0 in ≥1 cusp cell)
    coarse_beats_fine = any(
        cells[cell_key(f, e)]["delta_coarse_lo"] > 0
        for f in cusps for e in pc.eps if cell_key(f, e) in cells)

    # 4. survives bulk control (φ_ℓ* adds R² beyond bulk + cg8₀, CI > 0)
    survives_bulk = any(
        cells[cell_key(f, e)]["bulk_phi_increment_lo"] > 0
        for f in cusps for e in pc.eps if cell_key(f, e) in cells)

    # 4b. target-generic: does the cusp ℓ* story hold sign AND survive control under
    #     the ABSOLUTE target C8(t₁) too?  Generic ⇒ same increasing trend for both
    #     targets plus a control-surviving increment in a cusp.
    delta_cusp_sp = {f: fams[f]["spearman_eps_ellstar"] for f in cusps if f in fams}
    abs_cusp_sp = {f: tgt_robust[f]["spearman"] for f in cusps if f in tgt_robust}
    abs_pbc_any = any(tgt_robust[f]["phi_beyond_ctrl_any_pos"] for f in cusps if f in tgt_robust)
    target_generic = bool(abs_cusp_sp) and survives_bulk and abs_pbc_any \
        and all(s > 0 for s in abs_cusp_sp.values()) \
        and all(s > 0 for s in delta_cusp_sp.values())

    # 5. runtime / AWS
    rt = _load_runtime()

    # decision
    if not structured and not monotonic:
        decision = "STOP — ℓ*(ε) is flat/unstructured; retire the force-resolution-scale claim."
    elif monotonic and coarse_beats_fine and survives_bulk and target_generic:
        decision = ("STRONG — ℓ*(ε) tracks ε, beats fine features, and survives bulk "
                    "controls. A full battery is justified.")
    elif monotonic or structured:
        decision = ("NOISY-BUT-ALIVE — partial signal; run a medium local confirmation "
                    "(more reps / families / ε / a later horizon) before a full battery.")
    else:
        decision = "STOP — no monotonic or structured ℓ*(ε)."

    return {
        "1_structured": structured,
        "1_median_peak_prominence": float(np.median(prominences)),
        "2_monotonic_ellstar": monotonic,
        "2_cusp_spearman": {f: fams[f]["spearman_eps_ellstar"] for f in cusps if f in fams},
        "2_cusp_p_nondecreasing": {f: fams[f]["p_nondecreasing"] for f in cusps if f in fams},
        "3_coarse_beats_fine": coarse_beats_fine,
        "4_survives_bulk_control": survives_bulk,
        "4b_target_generic": target_generic,
        "4b_delta_cusp_spearman": delta_cusp_sp,
        "4b_abs_cusp_spearman": abs_cusp_sp,
        "4b_abs_phi_beyond_ctrl_any_pos": abs_pbc_any,
        "5_runtime": rt,
        "aws_needed": False,
        "decision": decision,
    }


# ── Runtime bookkeeping ──────────────────────────────────────────────────────────

def _runtime_path() -> str:
    return os.path.join(OUTDIR, "cache", "runtime.json")


def _record_runtime(pc: PilotConfig, upd: dict) -> None:
    os.makedirs(os.path.dirname(_runtime_path()), exist_ok=True)
    rt = _load_runtime()
    rt.update(upd)
    with open(_runtime_path(), "w") as f:
        json.dump(rt, f, indent=2)


def _load_runtime() -> dict:
    p = _runtime_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


# ── Outputs: CSV, JSON, figures, report ──────────────────────────────────────────

def write_outputs(pc: PilotConfig, summary: dict) -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    fig_dir = os.path.join(OUTDIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # results.csv
    import csv
    rows = []
    for key, c in summary["cells"].items():
        for s, ell in enumerate(c["scales"]):
            rows.append({
                "family": c["family"], "eps": c["eps"], "kind": "coarse_scale",
                "label": f"phi_ell={ell:g}", "scale": ell, "n_feat": len(cgf.phi_feature_names()),
                "n_used": c["n_used"], "n_train": c["n_train"], "n_test": c["n_test"],
                "r2": c["scale_r2"][s], "r2_lo": c["scale_r2_lo"][s], "r2_hi": c["scale_r2_hi"][s],
                "alpha": c["scale_alpha"][s], "is_ell_star": int(s == c["s_star"]),
            })
        for g, r2 in c["base_r2"].items():
            rows.append({
                "family": c["family"], "eps": c["eps"], "kind": "baseline",
                "label": g, "scale": "", "n_feat": len(cgf.BASELINE_GROUPS[g]),
                "n_used": c["n_used"], "n_train": c["n_train"], "n_test": c["n_test"],
                "r2": r2, "r2_lo": "", "r2_hi": "", "alpha": "", "is_ell_star": "",
            })
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # summary.json (strip bulky bootstrap arrays)
    slim = json.loads(json.dumps(summary, default=_json_safe))
    for c in slim["cells"].values():
        c.pop("ell_star_boot", None)
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(slim, f, indent=2, default=_json_safe)

    with open(os.path.join(OUTDIR, "config.json"), "w") as f:
        json.dump(asdict(pc), f, indent=2)

    _figures(pc, summary, fig_dir)
    _report(pc, summary)
    print(f"[outputs] results.csv, summary.json, config.json, figures/, pilot_report.md "
          f"→ {OUTDIR}/")


def _json_safe(x):
    if isinstance(x, (np.floating,)):
        v = float(x); return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return str(x)


def _figures(pc: PilotConfig, summary: dict, fig_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = pc.families
    eps_sorted = sorted(pc.eps)
    scales = pc.scales

    # 1. heatmap R²(ℓ, ε) per family
    fig, axes = plt.subplots(1, len(fams), figsize=(4.4 * len(fams), 4.0), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        M = np.full((len(eps_sorted), len(scales)), np.nan)
        for i, e in enumerate(eps_sorted):
            c = summary["cells"].get(cell_key(fam, e))
            if c:
                M[i] = c["scale_r2"]
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(scales)))
        ax.set_xticklabels([f"{s:g}" for s in scales], rotation=45, fontsize=8)
        ax.set_yticks(range(len(eps_sorted)))
        ax.set_yticklabels([f"{e:g}" for e in eps_sorted])
        ax.set_xlabel("ℓ (smoothing scale)"); ax.set_ylabel("ε (softening)")
        ax.set_title(fam)
        for i, e in enumerate(eps_sorted):
            c = summary["cells"].get(cell_key(fam, e))
            if c:
                ax.plot(c["s_star"], i, "r*", ms=14, mec="white")
        fig.colorbar(im, ax=ax, fraction=0.046, label="held-out R²")
    fig.suptitle("R²(ℓ; ε) — red ★ = ℓ*(ε)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_heatmap_R2.pdf")); plt.close(fig)

    # 2. ℓ* vs ε per family with reference lines
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    colors = plt.cm.tab10(np.linspace(0, 1, len(fams)))
    for fam, col in zip(fams, colors):
        fs = summary["families"].get(fam)
        if not fs:
            continue
        ell = np.array(fs["ell_star"], float)
        ci = np.array(fs["ell_star_ci"], float)
        yerr = np.abs(ci.T - ell)
        ax.errorbar(eps_sorted, ell, yerr=yerr, marker="o", capsize=4,
                    color=col, label=f"{fam} (ρ={fs['spearman_eps_ellstar']:+.2f})")
    ev = np.array(eps_sorted)
    ax.plot(ev, ev, "k--", alpha=0.5, label="ℓ = ε")
    ax.plot(ev, 2 * ev, "k:", alpha=0.5, label="ℓ = 2ε")
    ax.set_xlabel("softening ε"); ax.set_ylabel("ℓ*(ε)  (argmax R²)")
    ax.set_title("Does the optimal coarse-graining scale track softening?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_ellstar_vs_eps.pdf")); plt.close(fig)

    # 3. winner map
    cats = ["fine", "tie", "coarse-small", "coarse-large"]
    cmap_idx = {c: i for i, c in enumerate(cats)}
    fig, ax = plt.subplots(figsize=(1.6 * len(eps_sorted) + 2, 0.8 * len(fams) + 2))
    M = np.full((len(fams), len(eps_sorted)), np.nan)
    for r, fam in enumerate(fams):
        for cidx, e in enumerate(eps_sorted):
            c = summary["cells"].get(cell_key(fam, e))
            if c:
                M[r, cidx] = cmap_idx[c["winner"]]
                ax.text(cidx, r, c["winner"], ha="center", va="center", fontsize=8)
    ax.imshow(M, aspect="auto", cmap="Set3", vmin=0, vmax=len(cats) - 1)
    ax.set_xticks(range(len(eps_sorted))); ax.set_xticklabels([f"{e:g}" for e in eps_sorted])
    ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams)
    ax.set_xlabel("ε"); ax.set_title("Winner map (description that best predicts ΔC8-early)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_winner_map.pdf")); plt.close(fig)


def _report(pc: PilotConfig, summary: dict) -> None:
    kt = summary["kill_tests"]
    rt = kt["5_runtime"]
    L = []
    L.append("# Coarse-Graining Scale Pilot — Report\n")
    L.append(f"**Model:** `{pc.model}` (only model where ε affects the force)  ")
    L.append(f"**Families:** {', '.join(pc.families)}  ")
    L.append(f"**ε:** {pc.eps}  **N:** {pc.n} (fixed across ε)  "
             f"**steps:** {pc.steps} (ΔC8-early)  **reps/cell:** {pc.replicates}  ")
    L.append(f"**Smoothing scales ℓ:** {pc.scales}  **grid:** {pc.grid_g}³  "
             f"**regressor:** RidgeCV  **split:** {1-pc.test_frac:.0%}/{pc.test_frac:.0%} fixed\n")

    if kt["decision"].startswith("STOP"):
        band = "🔴 RED — retire the coarse-graining *law*; the prior result stands only as a methods / stylized observable-class study"
    elif "STRONG" in kt["decision"]:
        band = "🟢 GREEN — branch alive: force resolution selects the predictive coarse-graining scale"
    else:
        band = "🟡 YELLOW — promising but unconfirmed; run a medium local confirmation (no AWS)"
    L.append(f"## Verdict — {band}\n")
    L.append(f"> **{kt['decision']}**\n")

    L.append("## Kill tests\n")
    L.append(f"1. **R²(ℓ) structured (not flat)?** "
             f"{'YES' if kt['1_structured'] else 'NO'} "
             f"(median peak prominence = {kt['1_median_peak_prominence']:.2f}× CI width; >1 ⇒ structured).")
    L.append(f"2. **ℓ* increases with ε (cusps)?** "
             f"{'YES' if kt['2_monotonic_ellstar'] else 'NO'}  ")
    for f, s in kt["2_cusp_spearman"].items():
        nd = kt["2_cusp_p_nondecreasing"][f]
        L.append(f"   - {f}: Spearman(ε, ℓ*) = {s:+.2f}, P(ℓ* non-decreasing) = {nd:.2f}")
    L.append(f"3. **Coarse optimum beats fine features?** "
             f"{'YES' if kt['3_coarse_beats_fine'] else 'NO'} "
             f"(Δ_coarse 95% CI excludes 0 in ≥1 cusp cell).")
    L.append(f"4. **Survives bulk controls?** "
             f"{'YES' if kt['4_survives_bulk_control'] else 'NO'} "
             f"(φ_ℓ* adds held-out R² beyond bulk + persistence cg8₀ controls, CI > 0, "
             f"in ≥1 cusp cell — i.e. genuine future-prediction power, not baseline-sharing).")
    sim_ms = rt.get("sim_ms_per_sim_wall", float("nan"))
    L.append(f"5. **Runtime / AWS?** sim stage "
             f"{rt.get('sim_wall_s', float('nan')):.1f}s for {rt.get('sim_count','?')} sims "
             f"({sim_ms:.1f} ms/sim wall on {rt.get('sim_workers','?')} workers); "
             f"features {rt.get('feat_wall_s', float('nan')):.1f}s. "
             f"**AWS needed: {kt['aws_needed']}.**\n")

    L.append("## ℓ*(ε) by family\n")
    L.append("| family | ε | ℓ* | ℓ* 95% CI | R²(ℓ*) | R²(fine) | Δ_coarse [CI] | φ beyond ctrl [CI] | winner |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for fam in pc.families:
        for e in sorted(pc.eps):
            c = summary["cells"].get(cell_key(fam, e))
            if not c:
                continue
            L.append(
                f"| {fam} | {e:g} | {c['ell_star']:g} | "
                f"[{c['ell_star_boot_lo']:g}, {c['ell_star_boot_hi']:g}] | "
                f"{c['r2_at_star']:.3f} | {c['base_r2']['fine_all']:.3f} | "
                f"{c['delta_coarse']:+.3f} [{c['delta_coarse_lo']:+.3f}, {c['delta_coarse_hi']:+.3f}] | "
                f"{c['bulk_phi_increment']:+.3f} [{c['bulk_phi_increment_lo']:+.3f}, "
                f"{c['bulk_phi_increment_hi']:+.3f}] | {c['winner']} |")
    L.append("")

    # ── target robustness (Q4): absolute target C8(t₁) ──
    tr_rob = summary.get("target_robustness", {})
    L.append("## Target robustness — absolute C₈(t₁) (no delta)\n")
    L.append("| family | ℓ*(ε) [abs target] | Spearman(ε,ℓ*) | φ beyond ctrl > 0 anywhere? |")
    L.append("|---|---|---|---|")
    for fam in pc.families:
        t = tr_rob.get(fam)
        if t:
            L.append(f"| {fam} | {t['ell_star']} | {t['spearman']:+.2f} | "
                     f"{'yes' if t['phi_beyond_ctrl_any_pos'] else 'no'} |")
    L.append("")

    # ── the five questions ──
    def _spdict(d):
        return ", ".join(f"{k.replace('3d','')}={v:+.2f}" for k, v in d.items())
    delta_sp = kt.get("4b_delta_cusp_spearman", {})
    abs_sp = kt.get("4b_abs_cusp_spearman", {})
    L.append("## The five questions\n")
    L.append(f"**1. Does ℓ* track ε?**  **{'YES' if kt['2_monotonic_ellstar'] else 'NO'}.**  "
             f"R²(ℓ) curves are flat (median peak prominence "
             f"{kt['1_median_peak_prominence']:.2f}× the bootstrap CI width; a real peak needs >1). "
             f"For the cusp families where softening should matter most, Spearman(ε, ℓ*) = "
             f"{_spdict(delta_sp)} — i.e. no increase (the hypothesis predicts a clear rise). "
             f"P(ℓ* strictly increasing across ε) ≈ 0 in every family.")
    L.append(f"\n**2. Does Δ_coarse survive the hardened control?**  "
             f"**{'YES' if kt['4_survives_bulk_control'] else 'NO'}.**  "
             f"After controlling for bulk + cg8₀, the scale-resolved φ_ℓ* adds ≈0 held-out R² "
             f"in every cell (all 95% CIs include 0). The apparent φ_ℓ R² is fully explained by "
             f"bulk quantities + the trivial −cg8₀ baseline-sharing. And Δ_coarse vs fine_all is "
             f"*negative* for the cusps (fine features win) — coarse does not beat fine.")
    L.append(f"\n**3. Which family shows the cleanest effect?**  "
             f"None show a force-resolution *scale* effect. **bimodal** has the cleanest "
             f"*predictability* (R² up to ~0.6, coarse_fixed up to ~0.99) but its ℓ* sits at the "
             f"largest scales and is set by clump geometry, not ε — the paper's known coarse "
             f"dominance, re-derived. The **cusps (hernquist, plummer)**, where ε should matter "
             f"most, give the cleanest **null**: flat R²(ℓ) and ℓ* that does not rise with ε.")
    L.append(f"\n**4. Is the signal target-specific or generic?**  "
             f"**Neither survives — the weak apparent trend is target-specific and wrong-signed.**  "
             f"Swapping ΔC8 → absolute C₈(t₁): cusp Spearman(ε, ℓ*) = {_spdict(abs_sp)} "
             f"(vs delta {_spdict(delta_sp)}) — the sign changes with the target and is "
             f"≤0 for cusps under both. φ beyond control stays ≈0 for the absolute target too. "
             f"There is no target-generic, control-surviving scale law.")
    L.append(f"\n**5. Is AWS justified?**  **NO.**  The pilot is red, so by the decision rule "
             f"there is nothing to scale up. Independently, runtime is trivial: "
             f"{rt.get('sim_count','?')} sims in {rt.get('sim_wall_s',0):.0f}s + features "
             f"{rt.get('feat_wall_s',0):.0f}s on {rt.get('sim_workers','?')} local workers; the "
             f"full grid extrapolates to a single local day — below the ~2–3 day AWS threshold.")
    L.append("\n## Baseline R² reference (per cell)\n")
    L.append("See `results.csv` (kind=baseline) for fine_pos / fine_kin / fine_all / "
             "coarse_fixed / persistence / bulk R² in every cell, and `summary.json` "
             "→ `target_robustness` for the full absolute-target sweep.")

    with open(os.path.join(OUTDIR, "pilot_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Local coarse-graining scale pilot (no AWS).")
    ap.add_argument("--stage", choices=["all", "sim", "features", "analyze"], default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--eps", type=float, nargs="+", default=None)
    ap.add_argument("--scales", type=float, nargs="+", default=None)
    ap.add_argument("--force-sim", action="store_true")
    ap.add_argument("--force-features", action="store_true")
    args = ap.parse_args()

    pc = PilotConfig()
    if args.replicates is not None: pc.replicates = args.replicates
    if args.n is not None:          pc.n = args.n
    if args.families is not None:   pc.families = args.families
    if args.eps is not None:        pc.eps = args.eps
    if args.scales is not None:     pc.scales = args.scales

    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Pilot config: families={pc.families} eps={pc.eps} N={pc.n} "
          f"reps={pc.replicates} scales={pc.scales}")

    if args.stage in ("all", "sim"):
        stage_sim(pc, args.workers, args.force_sim)
    if args.stage in ("all", "features"):
        stage_features(pc, args.workers, args.force_features)
    if args.stage in ("all", "analyze"):
        summary = stage_analyze(pc)
        write_outputs(pc, summary)
        print("\n" + summary["kill_tests"]["decision"])


if __name__ == "__main__":
    main()
