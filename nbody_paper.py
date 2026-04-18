#!/usr/bin/env python3
"""
nbody_paper.py — hardened figure/table driver for the 3D N-body ODD paper
==========================================================================
CI hardening is complete. This is the flagship grid version.

Changes vs previous version:
  • Expanded flagship grid: N up to 16384, 5 ε values, 500 reps
  • DIRECT_N_MAX=2048: direct_isolated capped (O(N²) constraint);
    pm_periodic runs full N range
  • EPS_LS / EPS_MK cover all 5 ε values
  • load_csv_rows() uses robust type inference (no string whitelist)
  • fig04 panel 2 explicitly uses CoarseG8 as physical anchor
  • fig07 uses log-scale N axis with model-switch shading
  • fig14 title clarifies direct-isolated only
  • write_macros() typo fixed: BimodalBestCoarseR
  • write_exclusion_summary() new output: strict per-class exclusion counts
"""
from __future__ import annotations
import argparse
import csv
import datetime
import json
import math
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from nbody_3d import (
    _HAS_NUMBA,
    _worker_init,
    direct_acc,
    integrate_leapfrog,
    pm_acc_3d,
    SimConfig,
)
from nbody_stress import (
    COARSE_OBS,
    FINE_OBS,
    FINE_POSITIONAL_OBS,
    FINE_KINEMATIC_OBS,
    PRIMARY_TARGET,
    FAMILY_TARGETS,
    StressConfig,
    analyse,
    convergence_analysis,
    get_initial_conditions,
    get_simconfig,
    min_image,
    run_stress,
)

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------

DATA_DIR  = os.path.join("outputs", "data")
TABLE_DIR = os.path.join("outputs", "tables")
FIG_DIR   = os.path.join("outputs", "figures")

# ---------------------------------------------------------------------------
# Flagship battery grid
# ---------------------------------------------------------------------------

PAPER_N       = [256, 512, 1024, 2048, 4096, 8192, 16384]
PAPER_EPS     = [0.02, 0.03, 0.05, 0.07, 0.10]
PAPER_K       = 16
PAPER_MODELS  = ["direct_isolated", "pm_periodic"]
PAPER_INITS   = ["bimodal3d", "hernquist3d", "plummer3d", "cold_clumpy3d"]
# Angular-shuffle null-control ICs: direct_isolated only, excluded from PM cross-check.
# These run at reduced N (≤ DIRECT_N_MAX) alongside the core battery.
# bimodal3d_angshuf uses per-clump shuffle (see _angular_shuffle_bimodal_pos in
# nbody_stress.py) to preserve the coarse two-clump geometry while destroying
# fine structure within each clump.
PAPER_INITS_ANGSHUF = ["bimodal3d_angshuf", "hernquist3d_angshuf", "plummer3d_angshuf"]
PAPER_STEPS   = 600
PAPER_REPS    = 500
H_EARLY       = 100
H_MID         = 300

CHECKPOINT_EVERY = 2000

# direct_isolated is O(N²): cap at this N.  pm_periodic runs the full range.
DIRECT_N_MAX  = 2048

SHOWCASE_N    = 512
SHOWCASE_SEED = 2000

IC_ORDER = ["bimodal3d", "hernquist3d", "plummer3d", "cold_clumpy3d"]
IC_LABELS = {
    "bimodal3d":     "bimodal",
    "hernquist3d":   "Hernquist",
    "plummer3d":     "Plummer",
    "cold_clumpy3d": "cold-clumpy",
}
IC_COLORS = {
    "bimodal3d":     "#1b7837",
    "hernquist3d":   "#762a83",
    "plummer3d":     "#2166ac",
    "cold_clumpy3d": "#d6604d",
}
MODEL_LABELS = {
    "direct_isolated": "direct-isolated",
    "pm_periodic":     "PM-periodic",
}
MODEL_COLORS = {
    "direct_isolated": "#2166ac",
    "pm_periodic":     "#8c510a",
}
PRED_COLORS = {
    "CoarseG8":        "#2166ac",
    "CoarseG4":        "#6baed6",
    "CoarseG16":       "#08306b",
    "CoarseConc":      "#4d9221",   # radial family — concentration proxy
    "CoarseRShellVar": "#a6d96a",   # radial family — shell variance
    "kNN-all":         "#d6604d",
    "ClosePairs":      "#b2182b",
    "Pk-small":        "#f4a582",
    "FoF-groups":      "#762a83",
    "VelDisp":         "#1a9641",
}

# Angular-shuffle null-control IC labels (used in fig17)
IC_ANGSHUF_BASES = ["bimodal3d", "hernquist3d", "plummer3d"]   # ICs we run angshuf for
IC_LABELS_ANGSHUF = {
    f"{b}_angshuf": f"{IC_LABELS[b]} (angshuf)" for b in ["bimodal3d", "hernquist3d",
                                                             "plummer3d", "cold_clumpy3d"]
}

# All 5 ε values covered — no KeyError for 0.03 or 0.07
EPS_LS = {0.02: "-", 0.03: (0, (3, 1)), 0.05: "--", 0.07: (0, (1, 1)), 0.10: ":"}
EPS_MK = {0.02: "o", 0.03: "D",         0.05: "s",  0.07: "v",         0.10: "^"}

STYLE = {
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":   8.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

VERDICT_COLORS = {
    "FINE":         "#1a9641",
    "COARSE":       "#2166ac",
    "TIE":          "#969696",
    "UNDERPOWERED": "#d9a61a",
    "---":          "#cccccc",
}

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in [DATA_DIR, TABLE_DIR, FIG_DIR]:
        os.makedirs(d, exist_ok=True)


def draw_missing(ax: plt.Axes, title: Optional[str] = None,
                 text: str = "Data unavailable") -> None:
    ax.clear()
    if title:
        ax.set_title(title)
    ax.text(0.5, 0.5, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_alpha(0.3)


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def fmt_r(v: Optional[float], nd: int = 3) -> str:
    return "---" if v is None else f"{v:+.{nd}f}"


def fmt_ci(lo: Optional[float], hi: Optional[float], nd: int = 2) -> str:
    if lo is None or hi is None:
        return "[---, ---]"
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _json_safe(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _infer_value(v: str) -> Any:
    """Robust type inference for CSV values — no string whitelist."""
    if v == "":
        return None
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def load_csv_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                parsed[k] = _infer_value(v)
            # Backward-compat: derive coarse_g8_f from already-stored columns so
            # existing CSVs work without a rerun.
            # coarse_g8_f = d_coarse_g8_late + coarse_g8_0  (both always present)
            if parsed.get("coarse_g8_f") is None:
                late = safe_float(parsed.get("d_coarse_g8_late"))
                base = safe_float(parsed.get("coarse_g8_0"))
                if late is not None and base is not None:
                    parsed["coarse_g8_f"] = late + base
            rows.append(parsed)
    return rows


def write_csv_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write.")
    # Union of all row keys — safe when resume merges old rows (missing new
    # columns like coarse_g8_f) with new rows.  rows[0].keys() alone would
    # silently drop columns present only in later rows.
    all_keys: dict = {}
    for r in rows:
        all_keys.update(dict.fromkeys(r.keys()))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_keys),
                           extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


def _paper_done_key(row: Dict[str, Any]) -> tuple:
    """Unique key for a completed run — matches StressConfig identity fields."""
    return (
        int(row.get("n", 0)),
        float(row.get("eps", 0.0)),
        int(row.get("k_fine", 0)),
        str(row.get("model", "")),
        str(row.get("init", "")),
        int(row.get("seed", 0)),
    )


def ok_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("status") == "ok"]


def filter_rows(
    rows: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    init:  Optional[str] = None,
    n:     Optional[int] = None,
    eps:   Optional[float] = None,
) -> List[Dict[str, Any]]:
    out = ok_rows(rows)
    if model is not None:
        out = [r for r in out if r.get("model") == model]
    if init is not None:
        out = [r for r in out if r.get("init") == init]
    if n is not None:
        out = [r for r in out if int(r.get("n", -1)) == int(n)]
    if eps is not None:
        out = [r for r in out
               if abs(float(r.get("eps", -1)) - float(eps)) < 1e-12]
    return out


# ---------------------------------------------------------------------------
# Battery builder
# ---------------------------------------------------------------------------

def build_configs(reps: int = PAPER_REPS,
                  steps: int = PAPER_STEPS) -> List[StressConfig]:
    """
    Build the flagship battery config list.
    direct_isolated is capped at DIRECT_N_MAX (O(N²) constraint).
    pm_periodic runs the full PAPER_N range.

    Angular-shuffle null-control ICs (PAPER_INITS_ANGSHUF) are included
    as direct_isolated only — radial obs are undefined for periodic models.
    They run the same N / eps grid as the core direct_isolated cells.
    """
    seeds = [2000 + i for i in range(reps)]
    configs = []

    # Core IC battery: all models, N capped for direct
    for n, eps, model, init, seed in product(
            PAPER_N, PAPER_EPS, PAPER_MODELS, PAPER_INITS, seeds):
        if model == "direct_isolated" and n > DIRECT_N_MAX:
            continue
        configs.append(StressConfig(
            model=model, init=init, seed=seed, n=n,
            steps=steps, eps=eps, k_fine=PAPER_K,
            h_early=H_EARLY, h_mid=H_MID,
        ))

    # Angular-shuffle null controls: direct_isolated only, all N ≤ DIRECT_N_MAX
    for n, eps, init, seed in product(
            [n for n in PAPER_N if n <= DIRECT_N_MAX],
            PAPER_EPS, PAPER_INITS_ANGSHUF, seeds):
        configs.append(StressConfig(
            model="direct_isolated", init=init, seed=seed, n=n,
            steps=steps, eps=eps, k_fine=PAPER_K,
            h_early=H_EARLY, h_mid=H_MID,
        ))

    return configs


def _config_cost(cfg: StressConfig) -> float:
    """Heuristic cost estimate for a StressConfig, used for scheduling.

    PM runs cost O(N log N) per step; direct runs cost O(N²) per step.
    Cheapest configs first so short jobs fill gaps while expensive ones run.
    """
    if cfg.model == "pm_periodic":
        return cfg.n * math.log2(max(cfg.n, 2)) * cfg.k_fine
    else:
        return cfg.n * cfg.n * cfg.k_fine


def run_battery(workers: int, configs: List[StressConfig],
                use_numba: bool,
                checkpoint: Optional[str] = None) -> List[Dict[str, Any]]:
    # Sort cheapest first so the queue drains evenly and tqdm ETA is accurate.
    configs = sorted(configs, key=_config_cost)

    rows: List[Dict[str, Any]] = []
    total = len(configs)

    # Bounded submission: keep at most MAX_QUEUED tasks in flight so the
    # futures dict doesn't consume gigabytes of memory for 44k configs.
    MAX_QUEUED = max(workers * 4, 256)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(use_numba,),
    ) as ex:
        pending: dict = {}
        cfg_iter = iter(configs)
        submitted = 0

        def _fill_queue():
            nonlocal submitted
            while len(pending) < MAX_QUEUED and submitted < total:
                cfg = next(cfg_iter)
                fut = ex.submit(run_stress, cfg, use_numba)
                pending[fut] = cfg
                submitted += 1

        _fill_queue()  # seed the queue

        with tqdm(total=total, unit="run", ncols=100) as pbar:
            while pending:
                done_futs = set()
                for fut in as_completed(list(pending)):
                    done_futs.add(fut)
                    rows.append(fut.result())
                    pbar.update(1)
                    if checkpoint and len(rows) % CHECKPOINT_EVERY == 0:
                        write_csv_rows(checkpoint, rows)
                    break  # re-fill after each completion to keep queue topped up
                for fut in done_futs:
                    del pending[fut]
                _fill_queue()

    rows.sort(key=lambda r: (r["n"], r["eps"], r["model"], r["init"], r["seed"]))
    return rows


# ---------------------------------------------------------------------------
# Analysis cell accessors
# ---------------------------------------------------------------------------

def make_key(model: str, init: str, n: int, eps: float,
             k: int = PAPER_K) -> str:
    # Key format must stay in sync with nbody_stress._make_key().
    return f"N={n}|eps={eps}|k={k}|model={model}|init={init}"


def get_cell(analysis: Dict, model: str, init: str,
             n: int, eps: float) -> Dict:
    return analysis.get(make_key(model, init, n, eps), {})


def get_metric(cell: Dict, pred: str,
               tgt: str = PRIMARY_TARGET) -> Optional[float]:
    return safe_float(cell.get(f"r_{pred}_{tgt}"))


def get_ci(cell: Dict, pred: str,
           tgt: str = PRIMARY_TARGET) -> Tuple[Optional[float], Optional[float]]:
    return (safe_float(cell.get(f"ci_lo_{pred}_{tgt}")),
            safe_float(cell.get(f"ci_hi_{pred}_{tgt}")))


def get_winner_gap(cell: Dict,
                   tgt: str = PRIMARY_TARGET
                   ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    gm  = safe_float(cell.get(f"winner_gap_mean_{tgt}"))
    glo = safe_float(cell.get(f"winner_gap_ci_lo_{tgt}"))
    ghi = safe_float(cell.get(f"winner_gap_ci_hi_{tgt}"))
    return gm, glo, ghi


def get_verdict(cell: Dict, tgt: str = PRIMARY_TARGET) -> str:
    if not cell:
        return "---"
    # analyse() writes "primary_verdict" / "verdict" for the primary target only;
    # there are no per-target verdict keys for non-primary targets.
    if tgt == PRIMARY_TARGET:
        return cell.get("primary_verdict") or cell.get("verdict") or "---"
    # For other targets return "---" — per-target verdicts are not stored.
    return "---"


def get_best_coarse_name(cell: Dict,
                         tgt: str = PRIMARY_TARGET) -> Optional[str]:
    return cell.get(f"best_coarse_name_{tgt}")


def get_best_fine_name(cell: Dict,
                       tgt: str = PRIMARY_TARGET) -> Optional[str]:
    return cell.get(f"best_fine_name_{tgt}")


def get_best_coarse_abs_r(cell: Dict,
                          tgt: str = PRIMARY_TARGET) -> Optional[float]:
    return safe_float(cell.get(f"best_coarse_abs_r_{tgt}"))


def get_best_fine_abs_r(cell: Dict,
                        tgt: str = PRIMARY_TARGET) -> Optional[float]:
    return safe_float(cell.get(f"best_fine_abs_r_{tgt}"))


def get_pos_gap_ci(cell: Dict, tgt: str = PRIMARY_TARGET
                   ) -> Tuple[Optional[float], Optional[float]]:
    # analyse() stores bootstrap output as {gk}_{tgt_name}, where gk comes
    # from _winner_gap_bootstrap() return dict keys: winner_gap_pos_ci_lo/hi
    return (safe_float(cell.get(f"winner_gap_pos_ci_lo_{tgt}")),
            safe_float(cell.get(f"winner_gap_pos_ci_hi_{tgt}")))


def get_kin_gap_ci(cell: Dict, tgt: str = PRIMARY_TARGET
                   ) -> Tuple[Optional[float], Optional[float]]:
    return (safe_float(cell.get(f"winner_gap_kin_ci_lo_{tgt}")),
            safe_float(cell.get(f"winner_gap_kin_ci_hi_{tgt}")))


# ---------------------------------------------------------------------------
# Showcase sim helpers
# ---------------------------------------------------------------------------

def projected_density_image(pos: np.ndarray, box_size: float = 2.0,
                             grid: int = 128, periodic: bool = False) -> np.ndarray:
    if pos is None or len(pos) == 0:
        return np.zeros((grid, grid))
    x, y = pos[:, 0], pos[:, 1]
    if periodic:
        x, y = np.mod(x, box_size), np.mod(y, box_size)
        mask = np.ones(len(x), dtype=bool)
    else:
        mask = (x >= 0.0) & (x < box_size) & (y >= 0.0) & (y < box_size)
    img, _, _ = np.histogram2d(x[mask], y[mask], bins=grid,
                                range=[[0.0, box_size], [0.0, box_size]])
    return np.log10(img + 0.5)


def _run_showcase_sim(init: str, seed: int, n: int, eps: float,
                      steps: int) -> Dict[Any, np.ndarray]:
    cfg_s = StressConfig(model="direct_isolated", init=init, seed=seed,
                         n=n, steps=steps, eps=eps, k_fine=PAPER_K,
                         h_early=H_EARLY, h_mid=H_MID)
    sc = get_simconfig(cfg_s)
    pos0, vel0 = get_initial_conditions(cfg_s)
    mass = 1.0 / n
    snap_steps = sorted({0, H_EARLY, H_MID, steps})
    snaps = integrate_leapfrog(pos0, vel0, mass, sc, snap_steps, False)
    out = {s: snaps[s][0] for s in snap_steps if s in snaps}
    out["vel0"] = vel0
    return out


def compute_local_veldisp_per_particle(
    pos: np.ndarray, vel: np.ndarray, k: int,
    periodic: bool = False, box_size: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if pos is None or vel is None or len(pos) < 2:
        return np.empty((0, 3)), np.array([])
    inside = (np.ones(len(pos), dtype=bool) if periodic
              else np.all((pos >= 0.0) & (pos < box_size), axis=1))
    pos_in, vel_in = pos[inside], vel[inside]
    if len(pos_in) < 2:
        return np.empty((0, 3)), np.array([])
    k_eff = min(k, len(pos_in) - 1)
    dx = pos_in[:, None, :] - pos_in[None, :, :]
    if periodic:
        dx = min_image(dx, box_size)
    r2 = np.sum(dx * dx, axis=-1)
    np.fill_diagonal(r2, np.inf)
    nn_idx = np.argpartition(r2, kth=k_eff - 1, axis=1)[:, :k_eff]
    v_nn   = vel_in[nn_idx]
    v_mean = np.mean(v_nn, axis=1, keepdims=True)
    local_std = np.sqrt(np.mean(np.sum((v_nn - v_mean) ** 2, axis=-1), axis=1))
    return pos_in, local_std


# ---------------------------------------------------------------------------
# TeX macros
# ---------------------------------------------------------------------------

def write_macros(analysis: Dict, rows: List[Dict], path: str,
                 n_replicates: int = PAPER_REPS, n_boot: int = 1000) -> None:
    bim       = get_cell(analysis, "direct_isolated", "bimodal3d",   1024, 0.05)
    her_small = get_cell(analysis, "direct_isolated", "hernquist3d", 1024, 0.02)
    her_big   = get_cell(analysis, "direct_isolated", "hernquist3d", 1024, 0.10)
    plu_small = get_cell(analysis, "direct_isolated", "plummer3d",   1024, 0.02)

    drift_vals = [safe_float(r.get("energy_rel_drift"))
                  for r in filter_rows(rows, model="direct_isolated")]
    drift_vals = [v for v in drift_vals if v is not None]
    drift_arr  = np.array(drift_vals, dtype=float) if drift_vals else np.array([])
    drift_med = float(np.median(drift_arr)) if len(drift_arr) else 0.0
    drift_max = float(np.max(drift_arr))    if len(drift_arr) else 0.0
    # outlier counts for quality-cut sensitivity appendix
    n_outlier_01 = int(np.sum(drift_arr >= 0.1))   # aggressive cut: drift >= 0.1
    n_outlier_1  = int(np.sum(drift_arr >= 1.0))   # strict cut: drift >= 1

    bim_bc_name = get_best_coarse_name(bim) or "CoarseG8"
    bim_bc_r    = get_best_coarse_abs_r(bim)
    bim_bf_name = get_best_fine_name(bim) or "---"
    bim_bf_r    = get_best_fine_abs_r(bim)
    bim_gm, bim_glo, bim_ghi = get_winner_gap(bim)

    hv          = get_metric(her_small, "VelDisp")
    hvlo, hvhi  = get_ci(her_small, "VelDisp")
    h_bc_name   = get_best_coarse_name(her_small) or "CoarseG8"
    h_kglo, h_kghi = get_kin_gap_ci(her_small)
    h_pglo, h_pghi = get_pos_gap_ci(her_small)

    pv          = get_metric(plu_small, "VelDisp")
    pvlo, pvhi  = get_ci(plu_small, "VelDisp")

    direct_ns = [n for n in PAPER_N if n <= DIRECT_N_MAX]

    mapping = {
        "TotalRuns":       str(len(rows)),
        "NCells":          str(len(analysis)),
        "NReplicates":     str(n_replicates),
        "NBootstrap":      str(n_boot),
        "NSteps":          str(PAPER_STEPS),
        "HEarly":          str(H_EARLY),
        "HMid":            str(H_MID),
        "NParticleMin":    str(min(PAPER_N)),
        "NParticleMax":    str(max(PAPER_N)),
        "DirectNMax":      str(DIRECT_N_MAX),
        "EpsMin":          f"{min(PAPER_EPS):.2f}",
        "EpsMax":          f"{max(PAPER_EPS):.2f}",
        "NEpsValues":      str(len(PAPER_EPS)),
        "NICs":            str(len(PAPER_INITS) + len(PAPER_INITS_ANGSHUF)),
        "NICsCore":        str(len(PAPER_INITS)),
        "NICsAngShuf":     str(len(PAPER_INITS_ANGSHUF)),
        "NModels":         str(len(PAPER_MODELS)),
        "ShowcaseN":       str(SHOWCASE_N),
        "PrimaryTarget":   PRIMARY_TARGET.replace("Δ", "$\\Delta$"),
        # bimodal — actual best coarse comparator name (NOT hardcoded CoarseG8)
        "BimodalBestCoarseName":    bim_bc_name,
        "BimodalBestCoarseR":       fmt_r(bim_bc_r),
        "BimodalBestFineName":      bim_bf_name,
        "BimodalBestFineR":         fmt_r(bim_bf_r),
        "BimodalWinnerGapMean":     fmt_r(bim_gm),
        "BimodalWinnerGapCILo":     "---" if bim_glo is None else f"{bim_glo:+.2f}",
        "BimodalWinnerGapCIHi":     "---" if bim_ghi is None else f"{bim_ghi:+.2f}",
        "BimodalVerdict":           get_verdict(bim),
        # BimodalCoarseG8R removed — unused in paper.tex and causes
        # preamble leak with hyperref.
        "BimodalNRange":            f"{min(direct_ns)}--{max(direct_ns)}",
        "BimodalNScaleMin":         str(min(direct_ns)),
        "BimodalNScaleMax":         str(max(direct_ns)),
        # hernquist
        "HernquistBestCoarseName":  h_bc_name,
        "HernquistVelDispR":        fmt_r(hv),
        "HernquistVelDispCILo":     "---" if hvlo is None else f"{hvlo:+.2f}",
        "HernquistVelDispCIHi":     "---" if hvhi is None else f"{hvhi:+.2f}",
        "HernquistKinGapCILo":      "---" if h_kglo is None else f"{h_kglo:+.2f}",
        "HernquistKinGapCIHi":      "---" if h_kghi is None else f"{h_kghi:+.2f}",
        "HernquistPosGapCILo":      "---" if h_pglo is None else f"{h_pglo:+.2f}",
        "HernquistPosGapCIHi":      "---" if h_pghi is None else f"{h_pghi:+.2f}",
        "HernquistKNNR":            fmt_r(get_metric(her_small, "kNN-all")),
        "HernquistVelDispEpsTen":   fmt_r(get_metric(her_big, "VelDisp")),
        "HernquistVerdict":         get_verdict(her_small),
        # plummer
        "PlummerVelDispR":          fmt_r(pv),
        "PlummerVelDispCILo":       "---" if pvlo is None else f"{pvlo:+.2f}",
        "PlummerVelDispCIHi":       "---" if pvhi is None else f"{pvhi:+.2f}",
        # diagnostics
        "EnergyDriftMedian":   f"{{${drift_med:.2e}$}}",
        "EnergyDriftMax":      f"{{${drift_max:.2e}$}}",
        "NOutlierDriftTenth":  str(n_outlier_01),   # runs with drift >= 0.1
        "NOutlierDriftOne":    str(n_outlier_1),    # runs with drift >= 1
        # family stability: cells whose verdict is consistent across all
        # active prediction-target families (excluding underpowered cells)
        "NFamilyStableCells": str(sum(
            1 for cell in analysis.values()
            if cell.get("family_stable") and not cell.get("underpowered")
        )),
    }
    with open(path, "w") as f:
        for k, v in mapping.items():
            # Strip leading '+' only for standalone numeric macros (r values,
            # means).  Keep explicit sign for CI bound macros (ending in
            # CILo / CIHi) so mixed-sign intervals render as [-0.05, +0.19]
            # rather than the asymmetric [-0.05, 0.19].
            is_ci_bound = k.endswith("CILo") or k.endswith("CIHi")
            if not is_ci_bound and isinstance(v, str) and v.startswith("+"):
                v = v[1:]
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


# ---------------------------------------------------------------------------
# LaTeX table writers
# ---------------------------------------------------------------------------

def write_verdict_summary(analysis: Dict, path: str) -> None:
    with open(path, "w") as f:
        f.write("\\begin{tabular}{lllrrrrrrlr}\n\\toprule\n")
        f.write("Model & IC & $\\epsilon$ & Best coarse & $|r|_{\\rm bc}$ & "
                "Best fine & $|r|_{\\rm bf}$ & "
                "Gap mean & Gap CI & Verdict & $n_{\\rm lw}$\\\\\n\\midrule\n")
        for mi, model in enumerate(PAPER_MODELS):
            for init in IC_ORDER:
                for eps in PAPER_EPS:
                    cell = get_cell(analysis, model, init, 1024, eps)
                    if not cell:
                        f.write(f"{MODEL_LABELS[model]} & {IC_LABELS[init]} & "
                                f"{eps:.2f} & --- & --- & --- & --- & --- & --- & --- & ---\\\\\n")
                        continue
                    bc_name = get_best_coarse_name(cell) or "---"
                    bc_r    = get_best_coarse_abs_r(cell)
                    bf_name = get_best_fine_name(cell) or "---"
                    bf_r    = get_best_fine_abs_r(cell)
                    gm, glo, ghi = get_winner_gap(cell)
                    gap_ci  = fmt_ci(glo, ghi)
                    verdict = get_verdict(cell)
                    n_lw = cell.get(f"n_listwise_{PRIMARY_TARGET}")
                    n_lw_str = str(int(n_lw)) if n_lw is not None else "---"
                    f.write(
                        f"{MODEL_LABELS[model]} & {IC_LABELS[init]} & {eps:.2f} & "
                        f"{bc_name} & {fmt_r(bc_r)} & "
                        f"{bf_name} & {fmt_r(bf_r)} & "
                        f"{fmt_r(gm)} & {gap_ci} & {verdict} & {n_lw_str}\\\\\n"
                    )
            if mi < len(PAPER_MODELS) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_winner_gap_table(analysis: Dict, path: str) -> None:
    with open(path, "w") as f:
        f.write("\\begin{tabular}{llrrrl}\n\\toprule\n")
        f.write("IC & $\\epsilon$ & "
                "Overall gap CI & Positional CI & Kinematic CI & Verdict\\\\\n"
                "\\midrule\n")
        for ii, init in enumerate(IC_ORDER):
            for eps in PAPER_EPS:
                cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
                if not cell:
                    continue
                gm, glo, ghi = get_winner_gap(cell)
                gap_ci = fmt_ci(glo, ghi)
                pglo, pghi = get_pos_gap_ci(cell)
                kglo, kghi = get_kin_gap_ci(cell)
                pos_ci = fmt_ci(pglo, pghi)
                kin_ci = fmt_ci(kglo, kghi)
                verdict = get_verdict(cell)
                f.write(
                    f"{IC_LABELS[init]} & {eps:.2f} & "
                    f"{gap_ci} & {pos_ci} & {kin_ci} & {verdict}\\\\\n"
                )
            if ii < len(IC_ORDER) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_n_scaling(analysis: Dict, path: str) -> None:
    with open(path, "w") as f:
        f.write("\\begin{tabular}{llrrrrrl}\n\\toprule\n")
        f.write("IC & $N$ & "
                "$|r|_{\\rm bc}(0.02)$ & $|r|_{\\rm bc}(0.05)$ & $|r|_{\\rm bc}(0.10)$ & "
                "$|r|_{\\rm bf,max}$ & "
                "Verdict(0.05) & Model\\\\\n\\midrule\n")
        for ii, init in enumerate(IC_ORDER):
            for n in PAPER_N:
                model = "direct_isolated" if n <= DIRECT_N_MAX else "pm_periodic"
                bc_vals = []
                bests   = []
                for eps in [0.02, 0.05, 0.10]:
                    cell = get_cell(analysis, model, init, n, eps)
                    bc_r = get_best_coarse_abs_r(cell)
                    bc_vals.append(fmt_r(bc_r))
                    bf = get_best_fine_abs_r(cell)
                    if bf is not None:
                        bests.append(bf)
                bfmax = f"{max(bests):.3f}" if bests else "---"
                v05   = get_verdict(get_cell(analysis, model, init, n, 0.05))
                mlbl  = "direct" if n <= DIRECT_N_MAX else "PM"
                f.write(
                    f"{IC_LABELS[init]} & {n} & "
                    f"{bc_vals[0]} & {bc_vals[1]} & {bc_vals[2]} & "
                    f"{bfmax} & {v05} & {mlbl}\\\\\n"
                )
            if ii < len(IC_ORDER) - 1:
                f.write("\\midrule\n")
        # Rendered footnote — visible in compiled PDF, not just a LaTeX comment.
        # Model boundary footnote moved to table caption to avoid inflating
        # the table width.
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_cond_fine(analysis: Dict, path: str) -> None:
    with open(path, "w") as f:
        f.write("\\begin{tabular}{llrrrrl}\n\\toprule\n")
        f.write("IC & $\\epsilon$ & "
                "$r_{\\rm VelDisp}$ & $r_{\\rm kNN}$ & "
                "Pos.~gap CI & Kin.~gap CI & Verdict\\\\\n\\midrule\n")
        _cond_inits = ["hernquist3d", "plummer3d"]
        for ii, init in enumerate(_cond_inits):
            for eps in PAPER_EPS:
                cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
                pglo, pghi = get_pos_gap_ci(cell)
                kglo, kghi = get_kin_gap_ci(cell)
                f.write(
                    f"{IC_LABELS[init]} & {eps:.2f} & "
                    f"{fmt_r(get_metric(cell, 'VelDisp'))} & "
                    f"{fmt_r(get_metric(cell, 'kNN-all'))} & "
                    f"{fmt_ci(pglo, pghi)} & {fmt_ci(kglo, kghi)} & "
                    f"{get_verdict(cell)}\\\\\n"
                )
            if ii < len(_cond_inits) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_family_stability(analysis: Dict, path: str) -> None:
    # n_fam is used only as a fallback; the per-row denominator comes from
    # n_family_active stored by analyse(), which is model-dependent (periodic
    # models exclude ΔHMR-* and ΔConc-* targets that are NaN by construction).
    n_fam = len(FAMILY_TARGETS)
    with open(path, "w") as f:
        f.write("\\begin{tabular}{llllrrr}\n\\toprule\n")
        # Column header omits a fixed denominator — the denominator varies by model
        # and is shown inline in each data cell as "#n/denom".
        f.write("Model & IC & $\\epsilon$ & Primary & "
                "\\#COARSE & \\#FINE & Stable\\\\\n\\midrule\n")
        for mi, model in enumerate(PAPER_MODELS):
            for init in IC_ORDER:
                for eps in PAPER_EPS:
                    cell = get_cell(analysis, model, init, 1024, eps)
                    if not cell:
                        continue
                    pv       = get_verdict(cell)
                    n_coarse = int(cell.get("n_family_coarse") or 0)
                    n_fine   = int(cell.get("n_family_fine")   or 0)
                    n_active = int(cell.get("n_family_active") or n_fam)
                    stable   = "yes" if cell.get("family_stable") else "no"
                    f.write(
                        f"{MODEL_LABELS[model]} & {IC_LABELS[init]} & "
                        f"{eps:.2f} & {pv} & "
                        f"{n_coarse}/{n_active} & {n_fine}/{n_active} & {stable}\\\\\n"
                    )
            if mi < len(PAPER_MODELS) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_exclusion_summary(analysis: Dict, path_tex: str,
                             path_json: str) -> None:
    """Strict per-class exclusion counts for the direct_isolated primary target."""
    tgt   = PRIMARY_TARGET
    total = 0
    n_pos = 0   # fine CI entirely above 0 (positional fine beats coarse)
    n_kin = 0   # fine CI entirely above 0 (kinematic fine beats coarse)
    n_overall = 0  # overall gap CI entirely above 0 (fine beats coarse)
    n_coarse_wins = 0  # gap CI entirely below 0 (coarse wins)

    excl_json: Dict[str, Any] = {}

    for key, cell in analysis.items():
        if cell.get("model") != "direct_isolated":
            continue
        if cell.get("underpowered"):
            continue
        total += 1

        pglo, pghi = get_pos_gap_ci(cell, tgt)
        kglo, kghi = get_kin_gap_ci(cell, tgt)
        gm, glo, ghi = get_winner_gap(cell, tgt)

        pos_beats  = (pglo is not None and pglo > 0)
        kin_beats  = (kglo is not None and kglo > 0)
        fine_beats = (glo  is not None and glo  > 0)
        crs_wins   = (ghi  is not None and ghi  < 0)

        if pos_beats:  n_pos  += 1
        if kin_beats:  n_kin  += 1
        if fine_beats: n_overall += 1
        if crs_wins:   n_coarse_wins += 1

        excl_json[key] = {
            "pos_beats":  pos_beats,
            "kin_beats":  kin_beats,
            "fine_beats": fine_beats,
            "crs_wins":   crs_wins,
        }

    with open(path_json, "w") as f:
        json.dump(excl_json, f, indent=2)

    with open(path_tex, "w") as f:
        f.write("\\begin{tabular}{lr}\n\\toprule\n"
                "Exclusion criterion & Count\\\\\n\\midrule\n")
        f.write(f"Positional fine CI entirely above 0 & {n_pos}\\\\\n")
        f.write(f"Kinematic fine CI entirely above 0 & {n_kin}\\\\\n")
        f.write(f"Overall fine CI entirely above 0 & {n_overall}\\\\\n")
        f.write(f"Coarse CI entirely below 0 (coarse wins) & {n_coarse_wins}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"% {total} powered cells evaluated (underpowered excluded)\n")
        f.write(f"% Target: {tgt}\n")
        f.write("% CI threshold: gap\\_CI\\_lo $> 0$ for beats; "
                "gap\\_CI\\_hi $< 0$ for coarse wins\n")

    print(f"  Exclusion summary: {n_pos} pos-beats / {n_kin} kin-beats / "
          f"{n_coarse_wins} coarse-wins out of {total} powered cells")


def write_diagnostics(rows: List[Dict], path: str) -> None:
    direct_rows = filter_rows(rows, model="direct_isolated")
    drifts = [v for v in (safe_float(r.get("energy_rel_drift"))
                          for r in direct_rows) if v is not None]
    vir0 = [v for v in (safe_float(r.get("virial_0"))
                        for r in direct_rows) if v is not None]
    virf = [v for v in (safe_float(r.get("virial_f"))
                        for r in direct_rows) if v is not None]
    with open(path, "w") as f:
        f.write("\\begin{tabular}{lrr}\n\\toprule\n"
                "Diagnostic & Median & Max\\\\\n\\midrule\n")
        if drifts:
            f.write(f"Relative energy drift & "
                    f"{np.median(drifts):.2e} & {np.max(drifts):.2e}\\\\\n")
        else:
            f.write("Relative energy drift & --- & ---\\\\\n")
        if vir0:
            f.write(f"Initial virial ratio & {np.median(vir0):.3f} & "
                    f"{np.min(vir0):.3f}/{np.max(vir0):.3f}\\\\\n")
        else:
            f.write("Initial virial ratio & --- & ---\\\\\n")
        if virf:
            f.write(f"Final virial ratio & {np.median(virf):.3f} & "
                    f"{np.min(virf):.3f}/{np.max(virf):.3f}\\\\\n")
        else:
            f.write("Final virial ratio & --- & ---\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_sensitivity_table(analysis: Dict, path: str,
                             thresholds: Optional[List[float]] = None) -> None:
    """Verdict-threshold sensitivity table.

    Re-evaluates each non-underpowered cell's verdict using a sweep of
    VERDICT_GAP_THRESHOLD substitutes in `thresholds`.  Writes a CSV with
    columns: threshold, n_FINE, n_TIE, n_COARSE, n_underpowered, n_cells,
    frac_flip (fraction of powered cells that differ from the production
    threshold verdict).

    The production threshold is the first value in `thresholds` that equals
    VERDICT_GAP_THRESHOLD (imported from nbody_stress); if absent, the
    reference column is built from the actual stored cell["verdict"] values.
    """
    from nbody_stress import VERDICT_GAP_THRESHOLD as _T_PROD
    if thresholds is None:
        thresholds = [0.00, 0.02, 0.05, 0.08, 0.10]

    def _reval(cell: Dict, T: float) -> str:
        if cell.get("underpowered"):
            return "UNDERPOWERED"
        gm  = cell.get(f"winner_gap_mean_{PRIMARY_TARGET}")
        glo = cell.get(f"winner_gap_ci_lo_{PRIMARY_TARGET}")   # note: ci_lo not lo
        ghi = cell.get(f"winner_gap_ci_hi_{PRIMARY_TARGET}")   # note: ci_hi not hi
        if gm is None or glo is None or ghi is None:
            return "UNDERPOWERED"
        try:
            gm, glo, ghi = float(gm), float(glo), float(ghi)
        except (TypeError, ValueError):
            return "UNDERPOWERED"
        if not (math.isfinite(gm) and math.isfinite(glo) and math.isfinite(ghi)):
            return "UNDERPOWERED"
        if glo > T:
            return "FINE"
        if ghi < -T:
            return "COARSE"
        return "TIE"

    # Build reference verdicts at production threshold
    cells = list(analysis.values())
    ref_verdicts = {id(c): _reval(c, _T_PROD) for c in cells}
    n_powered = sum(1 for v in ref_verdicts.values() if v != "UNDERPOWERED")

    rows_out = []
    for T in thresholds:
        n_fine = n_tie = n_coarse = n_up = 0
        n_flip = 0
        for c in cells:
            v = _reval(c, T)
            if v == "FINE":       n_fine   += 1
            elif v == "TIE":      n_tie    += 1
            elif v == "COARSE":   n_coarse += 1
            else:                 n_up     += 1
            if v != "UNDERPOWERED" and ref_verdicts[id(c)] != "UNDERPOWERED":
                if v != ref_verdicts[id(c)]:
                    n_flip += 1
        frac_flip = (n_flip / n_powered) if n_powered > 0 else float("nan")
        prod_marker = " *" if abs(T - _T_PROD) < 1e-9 else ""
        rows_out.append({
            "threshold":       f"{T:.2f}{prod_marker}",
            "n_FINE":          n_fine,
            "n_TIE":           n_tie,
            "n_COARSE":        n_coarse,
            "n_underpowered":  n_up,
            "n_cells":         len(cells),
            "frac_flip":       f"{frac_flip:.3f}" if math.isfinite(frac_flip) else "---",
        })

    _dir = os.path.dirname(path)
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    fieldnames = ["threshold", "n_FINE", "n_TIE", "n_COARSE",
                  "n_underpowered", "n_cells", "frac_flip"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"  Sensitivity table: {path}  (* = production threshold)")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig01_ic_gallery(showcase_pos0: Dict[str, np.ndarray]) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.0))
    for idx, (ax, init) in enumerate(zip(axes.flat, IC_ORDER)):
        pos = showcase_pos0.get(init)
        if pos is None:
            draw_missing(ax, IC_LABELS[init]); continue
        img = projected_density_image(pos, periodic=False)
        ax.imshow(img.T, origin="lower", extent=[0, 2, 0, 2],
                  cmap="magma", aspect="equal")
        ax.set_title(IC_LABELS[init], fontweight="bold")
        # Only label outer edges
        if idx >= 2:
            ax.set_xlabel("$x$")
        else:
            ax.set_xticklabels([])
        if idx % 2 == 0:
            ax.set_ylabel("$y$")
        else:
            ax.set_yticklabels([])
    fig.suptitle(f"Initial-condition gallery  ($N={SHOWCASE_N}$, projected density)",
                 fontsize=11, fontweight="bold")
    savefig(fig, "fig01_ic_gallery.pdf")


def fig02_snapshots(showcase_snaps: Dict[str, Dict]) -> None:
    plt.rcParams.update(STYLE)
    cases = [("bimodal3d", 0.05), ("hernquist3d", 0.02), ("hernquist3d", 0.10)]
    steps = [0, H_EARLY, H_MID, PAPER_STEPS]
    fig, axes = plt.subplots(len(cases), len(steps), figsize=(12.5, 8.0))
    if len(cases) == 1:
        axes = np.array([axes])
    for i, (init, eps) in enumerate(cases):
        key  = f"{init}_{eps:.2f}"
        snap = showcase_snaps.get(key)
        for j, step in enumerate(steps):
            ax = axes[i, j]
            if snap is None or step not in snap:
                draw_missing(ax, f"{IC_LABELS[init]}, ε={eps:.2f}" if j == 0 else None)
                continue
            img = projected_density_image(snap[step], periodic=False)
            ax.imshow(img.T, origin="lower", extent=[0, 2, 0, 2],
                      cmap="magma", aspect="equal")
            if i == 0: ax.set_title(f"t = {step}")
            if j == 0: ax.set_ylabel(f"{IC_LABELS[init]}\nε={eps:.2f}")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        "Time evolution of projected density in the fiducial box $[0, L)^2$ "
        "(showcase runs; escapers outside the box are not shown)"
    )
    savefig(fig, "fig02_snapshots.pdf")


def fig03_verdict_map(analysis: Dict) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharey=True)
    panel_titles = {
        "direct_isolated": "direct-isolated  (primary evidence)",
        "pm_periodic":     "PM-periodic  (cross-check only)",
    }
    for ax, model in zip(axes, PAPER_MODELS):
        mat    = np.full((len(IC_ORDER), len(PAPER_EPS)), np.nan)
        labels = np.empty((len(IC_ORDER), len(PAPER_EPS)), dtype=object)
        for i, init in enumerate(IC_ORDER):
            for j, eps in enumerate(PAPER_EPS):
                cell = get_cell(analysis, model, init, 1024, eps)
                gm, _, _ = get_winner_gap(cell)
                if gm is not None:
                    mat[i, j] = gm
                v = get_verdict(cell)
                labels[i, j] = v[:2] if v not in ("---", None) else "---"
        im = ax.imshow(mat, origin="upper", aspect="auto",
                       cmap="RdBu_r", vmin=-0.35, vmax=0.35)
        ax.set_title(panel_titles.get(model, model), fontsize=11, pad=8)
        ax.set_xticks(range(len(PAPER_EPS)))
        ax.set_xticklabels([f"{e:.2f}" for e in PAPER_EPS], fontsize=10)
        ax.set_yticks(range(len(IC_ORDER)))
        ax.set_yticklabels([IC_LABELS[i] for i in IC_ORDER], fontsize=11)
        ax.set_xlabel(r"$\epsilon$", fontsize=11)
        for i in range(len(IC_ORDER)):
            for j in range(len(PAPER_EPS)):
                vstr    = str(labels[i, j])
                bg      = mat[i, j]
                txt_col = "white" if (np.isfinite(bg) and abs(bg) > 0.2) else "0.15"
                ax.text(j, i, vstr, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=txt_col)
    # Place colorbar to the right of both panels.
    # Use a manually-placed axes so the colorbar doesn't steal space from
    # the rightmost column cells.  _tight_rect tells savefig to constrain
    # tight_layout to leave room for the colorbar at x=0.91.
    fig._tight_rect = [0, 0, 0.89, 1]
    fig.subplots_adjust(wspace=0.08)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$|r_{\rm best\ fine}| - |r_{\rm best\ coarse}|$", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle("Verdict map at $N=1024$: fine advantage by IC family and softening",
                 fontsize=12, y=0.98)
    savefig(fig, "fig03_verdict_map.pdf")


def fig04_bimodal_anchor(analysis: Dict, rows: List[Dict]) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax1, ax2 = axes

    rs = filter_rows(rows, model="direct_isolated", init="bimodal3d", n=1024, eps=0.05)
    x = np.array([r.get("coarse_g8_0", np.nan) for r in rs], dtype=float)
    y = np.array([r.get("d_coarse_g8_early", np.nan) for r in rs], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 3 and np.std(x[mask]) > 1e-12 and np.std(y[mask]) > 1e-12:
        ax1.scatter(x[mask], y[mask], color=IC_COLORS["bimodal3d"], alpha=0.75, s=18)
        p = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(np.min(x[mask]), np.max(x[mask]), 100)
        ax1.plot(xs, p[0] * xs + p[1], color="0.25", lw=2)
        ax1.set_xlabel(r"Initial coarse density variance $\sigma_\rho^2(G8)$")
        ax1.set_ylabel(r"Future $\Delta C_8^{\rm early}$")
        ax1.set_title("Bimodal anchor: direct-isolated, $N=1024$, $\\epsilon=0.05$")
    else:
        draw_missing(ax1, "Bimodal anchor")

    # Panel 2: CoarseG8 (physical anchor) vs best fine across epsilon
    cg8_vals = []
    fine_vals = []
    for eps in PAPER_EPS:
        cell = get_cell(analysis, "direct_isolated", "bimodal3d", 1024, eps)
        cg8_vals.append(get_metric(cell, "CoarseG8"))
        fine_vals.append(get_best_fine_abs_r(cell))

    ax2.plot(PAPER_EPS,
             [np.nan if v is None else abs(v) for v in cg8_vals],
             marker="o", color=PRED_COLORS["CoarseG8"],
             lw=2, label=r"CoarseG8 $|r|$")
    ax2.plot(PAPER_EPS,
             [np.nan if v is None else v for v in fine_vals],
             marker="s", color="0.4", lw=1.5, ls="--",
             label="best fine $|r|$")
    ax2.set_xlabel(r"$\epsilon$")
    ax2.set_ylabel(r"$|r|$")
    ax2.set_title("Bimodal coarse dominance across softening")
    ax2.legend(frameon=False)
    savefig(fig, "fig04_bimodal_anchor.pdf")


def fig05_cond_fine(analysis: Dict) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    pos_preds = [p for _, p in FINE_POSITIONAL_OBS]
    kin_preds = [p for _, p in FINE_KINEMATIC_OBS]
    for ax, init in zip(axes, ["hernquist3d", "plummer3d"]):
        # Best coarse (varies with eps)
        bc_vals = []
        for eps in PAPER_EPS:
            cell  = get_cell(analysis, "direct_isolated", init, 1024, eps)
            bc_n  = get_best_coarse_name(cell) or "CoarseG8"
            bc_vals.append(get_metric(cell, bc_n))
        ax.plot(PAPER_EPS, [np.nan if v is None else v for v in bc_vals],
                marker="^", lw=2.2, color="#2166ac", label="best coarse", zorder=5)

        # Positional fine class
        for pred in pos_preds:
            vals, lo_arr, hi_arr = [], [], []
            for eps in PAPER_EPS:
                cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
                vals.append(get_metric(cell, pred))
                clo, chi = get_ci(cell, pred)
                lo_arr.append(clo); hi_arr.append(chi)
            col   = PRED_COLORS.get(pred, "0.55")
            yy    = np.array([np.nan if v is None else v for v in vals])
            lo_np = np.array([np.nan if v is None else v for v in lo_arr])
            hi_np = np.array([np.nan if v is None else v for v in hi_arr])
            ax.plot(PAPER_EPS, yy, marker="o", color=col,
                    alpha=0.65, lw=1.2, label=f"{pred} [pos]")
            if np.any(np.isfinite(lo_np)) and np.any(np.isfinite(hi_np)):
                ax.fill_between(PAPER_EPS, lo_np, hi_np, color=col, alpha=0.08)

        # Kinematic fine class
        for pred in kin_preds:
            vals, lo_arr, hi_arr = [], [], []
            for eps in PAPER_EPS:
                cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
                vals.append(get_metric(cell, pred))
                clo, chi = get_ci(cell, pred)
                lo_arr.append(clo); hi_arr.append(chi)
            col   = PRED_COLORS.get(pred, "0.35")
            yy    = np.array([np.nan if v is None else v for v in vals])
            lo_np = np.array([np.nan if v is None else v for v in lo_arr])
            hi_np = np.array([np.nan if v is None else v for v in hi_arr])
            ax.plot(PAPER_EPS, yy, marker="D", color=col,
                    lw=2.2, label=f"{pred} [kin]", zorder=4)
            if np.any(np.isfinite(lo_np)) and np.any(np.isfinite(hi_np)):
                ax.fill_between(PAPER_EPS, lo_np, hi_np, color=col, alpha=0.15)

        ax.axhline(0.0, color="0.7", lw=1, ls="--")
        ax.set_title(IC_LABELS[init])
        ax.set_xlabel(r"$\epsilon$")
    axes[0].set_ylabel(r"Pearson $r$ with primary target")
    for ax in axes:
        ax.legend(frameon=False, loc="lower left", fontsize=7.5, ncol=2)
    fig.suptitle("Conditional fine-leaning candidates: concentrated profiles at $N=1024$")
    savefig(fig, "fig05_cond_fine.pdf")


def fig06_eps_transition(analysis: Dict) -> None:
    plt.rcParams.update(STYLE)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.0, 4.5),
                                             gridspec_kw={"width_ratios": [1, 1]})
    mk_cycle = ["o", "s", "D", "^"]
    for idx, init in enumerate(IC_ORDER):
        vals = []
        for eps in PAPER_EPS:
            cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
            bf = get_best_fine_abs_r(cell); bc = get_best_coarse_abs_r(cell)
            vals.append(np.nan if (bf is None or bc is None) else bf - bc)
        # Bimodal + cold-clumpy on left, concentrated on right
        if init in ("bimodal3d", "cold_clumpy3d"):
            ax_left.plot(PAPER_EPS, vals, marker=mk_cycle[idx], lw=2.2, ms=7,
                         color=IC_COLORS[init], label=IC_LABELS[init], zorder=4)
        else:
            ax_right.plot(PAPER_EPS, vals, marker=mk_cycle[idx], lw=2.2, ms=7,
                          color=IC_COLORS[init], label=IC_LABELS[init], zorder=4)
    for ax in (ax_left, ax_right):
        ax.axhline(0.0, color="0.5", lw=1, ls="--")
        ax.set_xlabel(r"$\epsilon$")
        ax.legend(frameon=False, fontsize=9, loc="best")
    ax_left.set_ylabel(r"Winner gap: $|r_{\rm best\ fine}| - |r_{\rm best\ coarse}|$")
    ax_left.set_title("Multi-cluster ICs", fontweight="bold")
    ax_right.set_title("Concentrated profiles", fontweight="bold")
    fig.suptitle(r"Winner gap vs softening ($N=1024$, direct-isolated)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig06_eps_transition.pdf")


def fig07_n_scaling(analysis: Dict) -> None:
    """
    N-scaling of best coarse vs best fine.
    Log-scale N axis; shading for N > DIRECT_N_MAX where model switches to PM.
    """
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.2), sharey=True)
    n_arr = np.array(PAPER_N)

    # Split N values into direct and PM segments so they are never connected
    # by a line.  Connecting them would imply a continuous N-trend across a
    # model-switch boundary; the two regimes use different force solvers and
    # are separate experiments.
    d_arr = np.array([n for n in PAPER_N if n <= DIRECT_N_MAX])
    p_arr = np.array([n for n in PAPER_N if n >  DIRECT_N_MAX])

    # Show only 3 representative eps values to reduce clutter
    _show_eps = [0.02, 0.05, 0.10]
    for ax, init in zip(axes, IC_ORDER):
        for eps in _show_eps:
            d_cv, d_fv, p_cv, p_fv = [], [], [], []
            for n in PAPER_N:
                model = "direct_isolated" if n <= DIRECT_N_MAX else "pm_periodic"
                cell  = get_cell(analysis, model, init, n, eps)
                cv_   = get_best_coarse_abs_r(cell)
                fv_   = get_best_fine_abs_r(cell)
                (d_cv if n <= DIRECT_N_MAX else p_cv).append(np.nan if cv_ is None else cv_)
                (d_fv if n <= DIRECT_N_MAX else p_fv).append(np.nan if fv_ is None else fv_)
            col = IC_COLORS[init]
            ls_ = EPS_LS[eps]; mk_ = EPS_MK[eps]
            # Direct segment — solid lines
            if len(d_arr):
                ax.plot(d_arr, d_cv, marker=mk_, ls=ls_, color=col,
                        lw=2.0, alpha=0.85)
                ax.plot(d_arr, d_fv, marker=mk_, ls=ls_, color="0.5",
                        lw=1.2, alpha=0.55)
            # PM segment — dashed lines to signal a different experiment
            if len(p_arr):
                ax.plot(p_arr, p_cv, marker=mk_, ls="--", color=col,
                        lw=1.5, alpha=0.70)
                ax.plot(p_arr, p_fv, marker=mk_, ls="--", color="0.5",
                        lw=1.0, alpha=0.40)
        # shade PM region
        ax.axvspan(DIRECT_N_MAX, max(PAPER_N) * 1.15,
                   color="0.93", zorder=0)
        ax.axvline(DIRECT_N_MAX, color="0.6", lw=1.2, ls=":", zorder=1)
        ax.axhline(0.0, color="0.75", lw=1, ls="--")
        ax.set_title(IC_LABELS[init], color=IC_COLORS[init], fontweight="bold")
        ax.set_xlabel(r"$N$")
        ax.set_xscale("log")
        ax.set_xticks(PAPER_N)
        ax.set_xticklabels([str(n) for n in PAPER_N], rotation=45,
                           ha="right", fontsize=7)
    axes[0].set_ylabel(r"$|r|$")
    handles = [
        plt.Line2D([0], [0], color="k",   lw=2.0, ls="-",  label=r"best coarse $|r|$ — direct"),
        plt.Line2D([0], [0], color="0.5", lw=1.2, ls="-",  label=r"best fine $|r|$ — direct"),
        plt.Line2D([0], [0], color="k",   lw=1.5, ls="--", label=r"best coarse $|r|$ — PM"),
        plt.Line2D([0], [0], color="0.5", lw=1.0, ls="--", label=r"best fine $|r|$ — PM"),
        plt.Rectangle((0, 0), 1, 1, fc="0.93", label=f"PM region ($N>{DIRECT_N_MAX}$)"),
    ]
    # ε marker legend (only shown values)
    for eps_val in _show_eps:
        handles.append(plt.Line2D([0], [0], color="0.4", lw=0, marker=EPS_MK[eps_val],
                                  label=rf"$\epsilon={eps_val}$", markersize=5))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        r"$N$-scaling: coarse vs fine $|r|$"
        f"  (solver switch at $N={DIRECT_N_MAX}$)",
        y=1.10, fontsize=10, fontweight="bold")
    savefig(fig, "fig07_n_scaling.pdf")


def fig08_model_comparison(analysis: Dict) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.2), sharey=True)
    for ax, init in zip(axes, IC_ORDER):
        for model in PAPER_MODELS:
            vals = []
            for eps in PAPER_EPS:
                cell = get_cell(analysis, model, init, 1024, eps)
                gm, _, _ = get_winner_gap(cell)
                vals.append(gm)
            ax.plot(PAPER_EPS,
                    [np.nan if v is None else v for v in vals],
                    marker="o", color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model])
        ax.axhline(0.0, color="0.75", lw=1, ls="--")
        ax.set_title(IC_LABELS[init], color=IC_COLORS[init], fontweight="bold")
        ax.set_xlabel(r"$\epsilon$")
    axes[0].set_ylabel(r"Winner gap mean")
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    fig.suptitle("Force-model comparison at $N=1024$")
    savefig(fig, "fig08_model_comparison.pdf")


def fig09_diagnostics(rows: List[Dict]) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))
    ax1, ax2, ax3 = axes
    direct_rows = filter_rows(rows, model="direct_isolated")
    drifts = np.array([safe_float(r.get("energy_rel_drift")) for r in direct_rows], dtype=float)
    drifts = drifts[np.isfinite(drifts)]
    if len(drifts) > 0:
        # Use logarithmic bins so the histogram is readable on a log x-axis
        clipped = drifts[(drifts > 0) & (drifts < 0.1)]
        n_outliers = int(np.sum(drifts >= 0.1))
        log_bins = np.logspace(-6, -1, 40)
        ax1.hist(clipped, bins=log_bins, color="#4c78a8", alpha=0.85)
        ax1.set_xscale("log")
        ax1.set_xlim(1e-6, 1e-1)
        ax1.set_title("Relative energy drift", fontsize=11)
        ax1.set_xlabel(r"$|\Delta E|/|E_0|$", fontsize=11)
        if n_outliers > 0:
            ax1.text(0.97, 0.95, f"{n_outliers} runs with drift $> 0.1$\n(max = {np.max(drifts):.1f})",
                     transform=ax1.transAxes, ha="right", va="top", fontsize=7,
                     bbox=dict(fc="white", alpha=0.8, ec="0.7", boxstyle="round,pad=0.3"))
    else:
        draw_missing(ax1, "Relative energy drift")

    vir0 = np.array([safe_float(r.get("virial_0")) for r in direct_rows], dtype=float)
    vir0 = vir0[np.isfinite(vir0)]
    virf = np.array([safe_float(r.get("virial_f")) for r in direct_rows], dtype=float)
    virf = virf[np.isfinite(virf)]
    if len(vir0) and len(virf):
        ax2.hist(vir0, bins=18, alpha=0.55, label="initial")
        ax2.hist(virf, bins=18, alpha=0.55, label="final")
        ax2.set_title("Virial-ratio distribution", fontsize=11)
        ax2.set_xlabel(r"$Q = 2K/|U|$", fontsize=11)
        ax2.legend(frameon=False, fontsize=10)
    else:
        draw_missing(ax2, "Virial-ratio distribution")

    pts = []   # collect (drift, virial_f, color) as pairwise-complete triples
    for init in IC_ORDER:
        rs = filter_rows(rows, model="direct_isolated", init=init, n=1024, eps=0.05)
        for r in rs:
            xv = safe_float(r.get("energy_rel_drift"))
            yv = safe_float(r.get("virial_f"))
            if xv is not None and yv is not None:
                pts.append((xv, yv, IC_COLORS[init]))
    if pts:
        # Plot per-IC so we get a legend
        for init in IC_ORDER:
            rs = filter_rows(rows, model="direct_isolated", init=init, n=1024, eps=0.05)
            xv = [safe_float(r.get("energy_rel_drift")) for r in rs]
            yv = [safe_float(r.get("virial_f")) for r in rs]
            xv = [x for x, y in zip(xv, yv) if x is not None and y is not None]
            yv = [y for x, y in zip([safe_float(r.get("energy_rel_drift")) for r in rs],
                                     [safe_float(r.get("virial_f")) for r in rs])
                  if x is not None and y is not None]
            if xv:
                ax3.scatter(xv, yv, s=15, alpha=0.6, color=IC_COLORS[init],
                            edgecolors="none", label=IC_LABELS[init])
        ax3.set_xscale("log")
        ax3.set_xlabel(r"$|\Delta E|/|E_0|$", fontsize=11)
        ax3.set_ylabel(r"Final virial ratio $Q_f$", fontsize=11)
        ax3.set_title("Energy drift vs final virial ratio", fontsize=11)
        ax3.legend(frameon=False, fontsize=9, loc="upper left")
    else:
        draw_missing(ax3, "Energy drift vs final virial ratio")
    fig.tight_layout()
    savefig(fig, "fig09_diagnostics.pdf")


def fig10_summary_matrix(analysis: Dict) -> None:
    plt.rcParams.update(STYLE)
    model = "direct_isolated"
    n_ref = 1024
    # Use all coarse and fine observables so VelDisp and CoarseRShellVar are not
    # silently dropped.  The old [:4] slice excluded the 5th entry in each list.
    obs_classes = [(p, p) for _, p in COARSE_OBS] + [(p, p) for _, p in FINE_OBS]
    if not obs_classes:
        obs_classes = [
            ("CoarseG8", r"Coarse$\sigma^2$(G8)"),
            ("kNN-all",  "Fine pos.\nkNN-all"),
            ("VelDisp",  "Fine kin.\nVelDisp"),
        ]
    preds = [p for p, _ in obs_classes]
    mat = np.full((len(IC_ORDER), len(preds) * len(PAPER_EPS)), np.nan)
    for i, init in enumerate(IC_ORDER):
        for j, pred in enumerate(preds):
            for k, eps in enumerate(PAPER_EPS):
                cell = get_cell(analysis, model, init, n_ref, eps)
                v = get_metric(cell, pred)
                if v is not None:
                    mat[i, j * len(PAPER_EPS) + k] = v
    n_cols = len(preds) * len(PAPER_EPS)
    fig, ax = plt.subplots(figsize=(max(18.0, len(preds) * 3.2), 5.5))
    im = ax.imshow(mat, origin="upper", cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(IC_ORDER)))
    ax.set_yticklabels([IC_LABELS[i] for i in IC_ORDER], fontsize=12, fontweight="bold")
    # ε labels under each column — show only on first and last of each group
    col_labels = []
    for g in range(len(preds)):
        for k, eps in enumerate(PAPER_EPS):
            if k == 0 or k == len(PAPER_EPS) - 1:
                col_labels.append(f"{eps:.2f}")
            else:
                col_labels.append("")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=0, ha="center", fontsize=8.5)
    ax.set_xlabel(r"$\epsilon$", fontsize=11, labelpad=30)
    for divider in range(1, len(preds)):
        ax.axvline(divider * len(PAPER_EPS) - 0.5, color="white", lw=3)
    # Group labels below x-axis ticks — use transform for stable placement
    pred_labels = [n for _, n in obs_classes]
    ax.tick_params(axis="x", pad=4)
    for j, name in enumerate(pred_labels):
        mid = j * len(PAPER_EPS) + (len(PAPER_EPS) - 1) / 2
        # Use figure fraction y so labels sit reliably below tick marks
        ax.annotate(name, xy=(mid, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -38), textcoords="offset points",
                    ha="center", va="top", fontsize=8.5, fontweight="bold",
                    annotation_clip=False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(r"Pearson $r$ with primary target", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title(f"Summary matrix at $N={n_ref}$, {MODEL_LABELS[model]}",
                 fontsize=12, fontweight="bold", pad=14)
    fig.subplots_adjust(bottom=0.22)
    savefig(fig, "fig10_summary_matrix.pdf")


def fig11_bimodal_mechanism(showcase_pos0: Dict[str, np.ndarray],
                             analysis: Dict, rows: List[Dict]) -> None:
    plt.rcParams.update(STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    pos0 = showcase_pos0.get("bimodal3d")
    if pos0 is None:
        draw_missing(ax1, "Bimodal initial state")
    else:
        x = pos0[:, 0]; y = pos0[:, 1]
        ax1.scatter(x, y, s=4, alpha=0.6, color=IC_COLORS["bimodal3d"], edgecolors="none")
        grid = 8
        for v in np.linspace(0.0, 2.0, grid + 1):
            ax1.axvline(v, color="0.75", lw=0.6)
            ax1.axhline(v, color="0.75", lw=0.6)
        ax1.set_xlim(0, 2); ax1.set_ylim(0, 2)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_title("Bimodal showcase: particles + 8×8 coarse grid")
        ax1.set_xlabel("x"); ax1.set_ylabel("y")

    for n, marker in [(256, "o"), (1024, "s")]:
        rs = filter_rows(rows, model="direct_isolated",
                         init="bimodal3d", n=n, eps=0.05)
        if rs:
            xx = np.array([r.get("coarse_g8_0", np.nan) for r in rs], dtype=float)
            yy = np.array([r.get("d_coarse_g8_early", np.nan) for r in rs], dtype=float)
            m = np.isfinite(xx) & np.isfinite(yy)
            if m.sum() > 0:
                ax2.scatter(xx[m], yy[m], s=22, alpha=0.65, marker=marker,
                            label=f"N={n}", edgecolors="none")
    ax2.set_xlabel(r"$\sigma_\rho^2(G8)$ at $t=0$")
    ax2.set_ylabel(r"$\Delta C_8^{\rm early}$")
    ax2.set_title("Bimodal anchor across particle count")
    ax2.legend(frameon=False)
    savefig(fig, "fig11_bimodal_mechanism.pdf")


def fig12_veldisp_mechanism(showcase_snaps: Dict[str, Dict],
                            analysis: Optional[Dict] = None) -> None:
    """VelDisp mechanism figure: 1 row with particle image, VelDisp map,
    and overlaid radial profiles at two softening values."""
    plt.rcParams.update(STYLE)
    init = "hernquist3d"
    eps_cases = [0.02, 0.10]
    eps_colors = {0.02: "#1a9641", 0.10: "#d73027"}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Col 1: particle positions (use eps=0.02 showcase)
    key = f"{init}_{eps_cases[0]:.2f}"
    sc = showcase_snaps.get(key)
    if sc is not None and 0 in sc:
        pos0 = np.asarray(sc[0], dtype=float)
        img = projected_density_image(pos0, periodic=False)
        ax1.imshow(img.T, origin="lower", extent=[0, 2, 0, 2],
                   cmap="magma", aspect="equal")
    ax1.set_title("Hernquist IC", fontsize=10, fontweight="bold")
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$")

    # Col 2: VelDisp map (eps=0.02 — where fine wins)
    if sc is not None and 0 in sc and "vel0" in sc:
        pos0 = np.asarray(sc[0], dtype=float)
        vel0 = np.asarray(sc["vel0"], dtype=float)
        pos_in, local_std = compute_local_veldisp_per_particle(
            pos0, vel0, PAPER_K, periodic=False, box_size=2.0)
        if len(local_std) > 0:
            x, y = pos_in[:, 0], pos_in[:, 1]
            gridsz = 28
            xedges = np.linspace(0.0, 2.0, gridsz + 1)
            yedges = np.linspace(0.0, 2.0, gridsz + 1)
            counts, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
            sums, _, _ = np.histogram2d(x, y, bins=[xedges, yedges],
                                        weights=local_std)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_local = np.where(counts > 0, sums / counts, np.nan)
            im = ax2.imshow(mean_local.T, origin="lower", extent=[0, 2, 0, 2],
                            aspect="equal", cmap="viridis")
            cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
            cbar.set_label("local VelDisp", fontsize=8)
    ax2.set_title(rf"VelDisp map ($\epsilon = {eps_cases[0]}$)",
                  fontsize=10, fontweight="bold")
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$y$")

    # Col 3: overlaid radial profiles at both eps values
    for eps in eps_cases:
        key_e = f"{init}_{eps:.2f}"
        sc_e = showcase_snaps.get(key_e)
        if sc_e is None or 0 not in sc_e or "vel0" not in sc_e:
            continue
        pos0 = np.asarray(sc_e[0], dtype=float)
        vel0 = np.asarray(sc_e["vel0"], dtype=float)
        pos_in, local_std = compute_local_veldisp_per_particle(
            pos0, vel0, PAPER_K, periodic=False, box_size=2.0)
        if len(local_std) == 0:
            continue
        x, y = pos_in[:, 0], pos_in[:, 1]
        r_proj = np.sqrt((x - 1.0) ** 2 + (y - 1.0) ** 2)
        nbins = 20
        bins = np.linspace(np.min(r_proj), np.max(r_proj), nbins + 1)
        idx = np.digitize(r_proj, bins) - 1
        rc, mu, sd = [], [], []
        for b in range(nbins):
            m = idx == b
            if np.sum(m) >= 6:
                rc.append(0.5 * (bins[b] + bins[b + 1]))
                mu.append(np.mean(local_std[m]))
                sd.append(np.std(local_std[m]))
        if rc:
            rc = np.asarray(rc); mu = np.asarray(mu); sd = np.asarray(sd)
            col = eps_colors[eps]
            # Build label with r value
            lbl = rf"$\epsilon = {eps}$"
            if analysis is not None:
                cell = get_cell(analysis, "direct_isolated", init,
                                SHOWCASE_N, eps)
                r_vd = get_metric(cell, "VelDisp")
                if r_vd is not None:
                    lbl += rf"  ($r_{{VD}} = {r_vd:.2f}$)"
            ax3.plot(rc, mu, color=col, lw=2.5, label=lbl)
            ax3.fill_between(rc, mu - 0.5*sd, mu + 0.5*sd, color=col, alpha=0.12)

    ax3.set_title("Radial VelDisp profile", fontsize=10, fontweight="bold")
    ax3.set_xlabel("projected radius from centre")
    ax3.set_ylabel("local VelDisp")
    ax3.legend(frameon=True, fancybox=True, fontsize=8.5, loc="best",
               framealpha=0.9, edgecolor="0.7")

    fig.tight_layout()
    savefig(fig, "fig12_veldisp_mechanism.pdf")


def fig13_eps_boundary(analysis: Dict) -> None:
    """Verdict boundary as a function of eps/a.

    Uses IC-specific scale radii so the x-axis is physically meaningful for
    each family:
      hernquist3d / plummer3d : a = plummer_a = 0.20  (Plummer scale radius)
      bimodal3d               : a = plummer_a * 0.5 = 0.10  (clump scale)
      cold_clumpy3d           : no natural Plummer scale; uses a=0.20 as a
                                conventional reference — labelled "(conv.)"

    Using a single fixed a_ref=0.20 for all ICs would misplace the bimodal
    x-values by a factor of 2 and is physically unjustified for cold_clumpy.
    """
    # IC-specific Plummer scale radii (match StressConfig defaults)
    _IC_A_REF = {
        "hernquist3d":   0.20,   # plummer_a
        "plummer3d":     0.20,   # plummer_a
        "bimodal3d":     0.10,   # plummer_a * 0.5  (per-clump Plummer scale)
        "cold_clumpy3d": 0.20,   # conventional reference — not a Plummer IC
    }
    _IC_A_LABEL = {
        "hernquist3d":   r"$a=0.20$",
        "plummer3d":     r"$a=0.20$",
        "bimodal3d":     r"$a=0.10$ (clump)",
        "cold_clumpy3d": r"$a=0.20$ (conv.)",
    }
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for init in IC_ORDER:
        a_ref = _IC_A_REF[init]
        x = np.array([eps / a_ref for eps in PAPER_EPS], dtype=float)
        y = []
        for eps in PAPER_EPS:
            cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
            bf = get_best_fine_abs_r(cell); bc = get_best_coarse_abs_r(cell)
            y.append(np.nan if (bf is None or bc is None) else bf - bc)
        label = f"{IC_LABELS[init]} ({_IC_A_LABEL[init]})"
        ax.plot(x, y, marker="o", color=IC_COLORS[init], label=label)
    ax.axhline(0.0, color="0.55", lw=1, ls="--")
    _x_a020 = [eps / 0.20 for eps in PAPER_EPS]
    ax.axvspan(min(_x_a020), max(_x_a020), color="0.92", zorder=0,
               label=r"data range for $a=0.20$ ICs")
    ax.set_xlabel(r"Softening-to-scale-radius ratio $\epsilon/a$  (IC-specific $a$)")
    ax.set_ylabel(r"Winner gap: $|r_{\rm best\ fine}| - |r_{\rm best\ coarse}|$")
    ax.set_title(
        r"Winner gap vs $\epsilon/a$ ($N=1024$, direct-isolated)"
        "\n(bimodal: clump scale a=0.10; cold-clumpy: conventional a=0.20)"
    )
    ax.legend(frameon=False, loc="best")
    savefig(fig, "fig13_eps_boundary.pdf")


def fig14_winner_gap_ci_map(analysis: Dict) -> None:
    """Winner-gap CI heatmap for direct_isolated at N=1024.
    Shows winner-gap mean with CI bounds as cell annotations."""
    plt.rcParams.update(STYLE)
    model = "direct_isolated"
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), sharey=True)
    for ax, n in zip(axes, [512, 1024]):
        mat    = np.full((len(IC_ORDER), len(PAPER_EPS)), np.nan)
        labels = np.empty((len(IC_ORDER), len(PAPER_EPS)), dtype=object)
        for i, init in enumerate(IC_ORDER):
            for j, eps in enumerate(PAPER_EPS):
                cell = get_cell(analysis, model, init, n, eps)
                gm, glo, ghi = get_winner_gap(cell)
                if gm is not None:
                    mat[i, j] = gm
                v = get_verdict(cell)
                if glo is not None and ghi is not None:
                    labels[i, j] = f"{v[:2]}\n[{glo:+.2f},{ghi:+.2f}]"
                else:
                    labels[i, j] = v[:2] if v not in ("---", None) else "---"
        im = ax.imshow(mat, origin="upper", aspect="auto",
                       cmap="RdBu_r", vmin=-0.35, vmax=0.35)
        ax.set_title(f"direct-isolated, $N={n}$", fontsize=9.5)
        ax.set_xticks(range(len(PAPER_EPS)))
        ax.set_xticklabels([f"{e:.2f}" for e in PAPER_EPS])
        ax.set_yticks(range(len(IC_ORDER)))
        ax.set_yticklabels([IC_LABELS[i] for i in IC_ORDER])
        ax.set_xlabel(r"$\epsilon$")
        for i in range(len(IC_ORDER)):
            for j in range(len(PAPER_EPS)):
                bg = mat[i, j]
                tc = "white" if (np.isfinite(bg) and abs(bg) > 0.2) else "0.15"
                ax.text(j, i, str(labels[i, j]), ha="center", va="center",
                        fontsize=7.5, color=tc)
        plt.colorbar(im, ax=ax, shrink=0.82, label="winner gap mean")
    fig.suptitle(
        "Winner-gap CI heatmap: direct-isolated  "
        "(FINE = fine obs wins; COARSE = coarse obs wins)"
    )
    savefig(fig, "fig14_winner_gap_ci_map.pdf")


def fig15_class_gap(analysis: Dict) -> None:
    """Positional vs kinematic gap CI for hernquist/plummer cells."""
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ax, init in zip(axes, ["hernquist3d", "plummer3d"]):
        pos_lo, pos_hi, kin_lo, kin_hi = [], [], [], []
        for eps in PAPER_EPS:
            cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
            pglo, pghi = get_pos_gap_ci(cell)
            kglo, kghi = get_kin_gap_ci(cell)
            pos_lo.append(pglo); pos_hi.append(pghi)
            kin_lo.append(kglo); kin_hi.append(kghi)
        eps_arr = np.array(PAPER_EPS)

        def _safe(lst):
            return np.array([np.nan if v is None else v for v in lst])

        plo = _safe(pos_lo); phi = _safe(pos_hi)
        klo = _safe(kin_lo); khi = _safe(kin_hi)
        pmid = 0.5 * (plo + phi)
        kmid = 0.5 * (klo + khi)

        ax.plot(eps_arr, pmid, marker="o", color="#2166ac", lw=2, label="pos gap midpoint")
        ax.fill_between(eps_arr, plo, phi, color="#2166ac", alpha=0.18)
        ax.plot(eps_arr, kmid, marker="D", color="#d6604d", lw=2, label="kin gap midpoint")
        ax.fill_between(eps_arr, klo, khi, color="#d6604d", alpha=0.18)
        ax.axhline(0.0, color="0.6", lw=1, ls="--")
        ax.set_title(IC_LABELS[init])
        ax.set_xlabel(r"$\epsilon$")
    axes[0].set_ylabel("Gap CI (fine minus coarse)")
    axes[0].legend(frameon=False)
    fig.suptitle("Positional vs kinematic class gap: Hernquist & Plummer, direct-isolated",
                 fontsize=10)
    savefig(fig, "fig15_class_gap.pdf")


def fig16_convergence(conv_path: str) -> None:
    """
    VelDisp convergence curves for hernquist/plummer frontier cells.
    Shows r_abs_mean ± std and ci_width_mean ± std across repeated subsamples.
    """
    plt.rcParams.update(STYLE)
    if not os.path.exists(conv_path):
        print(f"  fig16_convergence: {conv_path} not found — skipping")
        return
    with open(conv_path) as f:
        conv = json.load(f)
    if not conv:
        print("  fig16_convergence: convergence.json empty — skipping")
        return

    target_keys = {(init, eps, 1024)
                   for init in ["hernquist3d", "plummer3d"]
                   for eps in [0.02, 0.10]}
    rows_to_plot = [(k, v) for k, v in conv.items()
                   if (v.get("init"), v.get("eps"), v.get("n")) in target_keys
                   and v.get("curve")]

    if not rows_to_plot:
        print("  fig16_convergence: no matching cells — skipping")
        return

    ncols = min(len(rows_to_plot), 4)
    nrows = math.ceil(len(rows_to_plot) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.8 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, (key, v) in enumerate(rows_to_plot):
        ax   = axes[idx // ncols][idx % ncols]
        crv  = v["curve"]
        ns   = [p["n_reps"]        for p in crv]
        rm   = [p.get("r_abs_mean")    for p in crv]
        rs_  = [p.get("r_abs_std")     for p in crv]
        wm   = [p.get("ci_width_mean") for p in crv]
        ws   = [p.get("ci_width_std")  for p in crv]

        def _safe(lst):
            return np.array([v if v is not None else np.nan for v in lst], dtype=float)

        rm_a  = _safe(rm);  rs_a  = _safe(rs_)
        wm_a  = _safe(wm);  ws_a  = _safe(ws)
        ns_a  = np.array(ns, dtype=float)

        color_r = PRED_COLORS["VelDisp"]
        ax2 = ax.twinx()

        # r_abs_mean with ± std shading
        ax.plot(ns_a, rm_a, color=color_r, marker="o", ms=5, lw=2,
                label=r"$|r|$ mean")
        mask = np.isfinite(rm_a) & np.isfinite(rs_a)
        if mask.sum() > 1:
            ax.fill_between(ns_a[mask],
                            (rm_a - rs_a)[mask],
                            (rm_a + rs_a)[mask],
                            color=color_r, alpha=0.2)

        # ci_width_mean with ± std shading
        ax2.plot(ns_a, wm_a, color="0.55", marker="s", ms=4, lw=1.5, ls="--",
                 label="CI width mean")
        mask2 = np.isfinite(wm_a) & np.isfinite(ws_a)
        if mask2.sum() > 1:
            ax2.fill_between(ns_a[mask2],
                             (wm_a - ws_a)[mask2],
                             (wm_a + ws_a)[mask2],
                             color="0.55", alpha=0.15)

        init = v.get("init", ""); eps = v.get("eps", 0.0)
        ax.set_title(f"{IC_LABELS.get(init, init)},  $\\epsilon={eps:.2f}$",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Replicates", fontsize=10)
        ax.set_ylabel(r"$|r|$ (VelDisp)", color=color_r, fontsize=10)
        ax2.set_ylabel("95% CI width", color="0.55", fontsize=10)
        ax.tick_params(labelsize=9)
        ax2.tick_params(labelsize=9)
        ax.axhline(0.0, color="0.8", lw=0.8, ls=":")

    for idx in range(len(rows_to_plot), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.suptitle(
        "VelDisp convergence: kinematic frontier cells\n"
        fr"$|r|$ mean ± std (shading) with {PRIMARY_TARGET}; "
        "repeated random subsamples",
        fontsize=10
    )
    savefig(fig, "fig16_convergence.pdf")


def fig17_radial_and_null(analysis: Dict) -> None:
    """Two-panel figure that closes two skeptic attack surfaces.

    Left panel  — Radial coarse observer performance.
      For each core IC (direct_isolated, N=1024), plot:
        |r(CoarseConc)|, |r(CoarseRShellVar)|, |r(best fine)|  vs ε.
      This directly answers the question "does the radial family actually
      compete?" — if |r(CoarseConc)| and |r(CoarseRShellVar)| are
      consistently below |r(best fine)|, the result holds against a
      grid-independent baseline too.

    Right panel — Angular-shuffle null control.
      Winner-gap heatmap for _angshuf IC variants (direct_isolated, N=1024).
      FINE verdict here = marginal-distribution artifact.
      TIE/COARSE verdict = signal is structure-dependent.
      Placeholder shown if angshuf data is absent.
    """
    plt.rcParams.update(STYLE)

    radial_preds  = ["CoarseConc", "CoarseRShellVar"]
    angshuf_inits = [f"{b}_angshuf" for b in IC_ANGSHUF_BASES]

    def _safe_r(cell: Optional[Dict], pred_name: str) -> Optional[float]:
        if cell is None:
            return None
        v = cell.get(f"r_{pred_name}_{PRIMARY_TARGET}")
        if v is None or not math.isfinite(float(v)):
            return None
        return abs(float(v))

    def _safe_bf(cell: Optional[Dict]) -> Optional[float]:
        if cell is None:
            return None
        v = cell.get(f"best_fine_abs_r_{PRIMARY_TARGET}")
        return float(v) if (v is not None and math.isfinite(float(v))) else None

    # Check whether radial obs are present in any cell
    def _has_radial(an: Dict) -> bool:
        return any(
            f"r_CoarseConc_{PRIMARY_TARGET}" in cell
            for cell in an.values()
        )

    has_radial  = _has_radial(analysis)
    has_angshuf = any(
        c.get("model") == "direct_isolated" and c.get("init") in angshuf_inits
        for c in analysis.values()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5))

    # ── Left: radial coarse family performance — line plot ────────────────────
    ax = axes[0]
    if not has_radial:
        ax.text(0.5, 0.5,
                "Radial observers not in analysis.\n"
                "Rerun battery with current code.",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("Radial coarse family — CoarseConc / CoarseRShellVar")
    else:
        color_conc    = PRED_COLORS["CoarseConc"]
        color_rshell  = PRED_COLORS["CoarseRShellVar"]
        color_bf      = "0.25"
        for init in IC_ORDER:
            conc_vals   = []
            rshell_vals = []
            bf_vals     = []
            for eps in PAPER_EPS:
                cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
                conc_vals.append(_safe_r(cell, "CoarseConc"))
                rshell_vals.append(_safe_r(cell, "CoarseRShellVar"))
                bf_vals.append(_safe_bf(cell))
            col = IC_COLORS[init]
            _yc = [np.nan if v is None else v for v in conc_vals]
            _yr = [np.nan if v is None else v for v in rshell_vals]
            _yb = [np.nan if v is None else v for v in bf_vals]
            ax.plot(PAPER_EPS, _yc, marker="o", color=col, ls="-", lw=1.8)
            ax.plot(PAPER_EPS, _yr, marker="s", color=col, ls="--", lw=1.2,
                    alpha=0.7)
        ax.axhline(0.0, color="0.7", lw=1, ls="--")
        ax.set_xlabel(r"$\epsilon$", fontsize=11)
        ax.set_ylabel(r"$|r|$", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.set_title("Radial coarse family: CoarseConc and RShellVar",
                      fontsize=11, fontweight="bold")
        # Two-section legend: line style (top) + IC colour (bottom)
        _h = [plt.Line2D([0], [0], color="0.3", ls="-",  marker="o", lw=1.8,
                          label="CoarseConc  (solid)"),
              plt.Line2D([0], [0], color="0.3", ls="--", marker="s", lw=1.4,
                          label="RShellVar  (dashed)")]
        for init in IC_ORDER:
            _h.append(plt.Line2D([0], [0], color=IC_COLORS[init], lw=2.5,
                                  label=IC_LABELS[init]))
        # Place legend in the open gap between the top pair of lines
        # (bimodal/Plummer solid ~0.27–0.31) and the lower cluster
        # (~0.07–0.17).  bbox_to_anchor in axes fraction; upper-right
        # corner of the legend sits at ~y_axes=0.76 ≈ y_data=0.22.
        ax.legend(handles=_h, frameon=True, framealpha=0.92,
                  edgecolor="0.8", fontsize=9, ncol=2,
                  loc="upper right", bbox_to_anchor=(0.98, 0.76),
                  handlelength=2.0)

    # ── Right: angular-shuffle null control — winner-gap heatmap ─────────────
    ax = axes[1]
    if not has_angshuf:
        ax.text(0.5, 0.5,
                "Angular-shuffle null not yet in analysis.\n"
                "(Included in flagship battery — rerun to populate.)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
                style="italic", color="0.4")
        ax.set_title("Angular-shuffle null control (pending run)")
    else:
        present  = [a for a in angshuf_inits
                    if any(c.get("model") == "direct_isolated" and c.get("init") == a
                           for c in analysis.values())]
        nrows_a  = len(present)
        mat_a    = np.full((nrows_a, len(PAPER_EPS)), np.nan)
        lbl_a    = np.empty((nrows_a, len(PAPER_EPS)), dtype=object)
        for i, init in enumerate(present):
            for j, eps in enumerate(PAPER_EPS):
                cell = get_cell(analysis, "direct_isolated", init, 1024, eps)
                if cell is None:
                    lbl_a[i, j] = "---"
                    continue
                gm = cell.get(f"winner_gap_mean_{PRIMARY_TARGET}")
                if gm is not None and math.isfinite(float(gm)):
                    mat_a[i, j] = float(gm)
                v = get_verdict(cell)
                lbl_a[i, j] = v[:2] if v not in ("---", None) else "---"

        # Mark the bimodal row with a dagger to signal the caveat in the title:
        # the bimodal angshuf preserves the coarse inter-clump separation, so a
        # COARSE/TIE verdict there does NOT confirm that coarse structure is
        # load-bearing — it only confirms that within-clump fine structure is
        # not needed.  The null is informative only for Hernquist and Plummer.
        ylabels = [
            (IC_LABELS_ANGSHUF.get(a, a) + r" $\dagger$"
             if a == "bimodal3d_angshuf"
             else IC_LABELS_ANGSHUF.get(a, a))
            for a in present
        ]
        im = ax.imshow(mat_a, origin="upper", aspect="auto",
                       cmap="RdBu_r", vmin=-0.35, vmax=0.35)
        ax.set_title("Angular-shuffle null control", fontsize=10,
                      fontweight="bold")
        ax.set_xticks(range(len(PAPER_EPS)))
        ax.set_xticklabels([f"{e:.2f}" for e in PAPER_EPS], fontsize=10)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(ylabels, fontsize=10)
        ax.set_xlabel(r"$\epsilon$", fontsize=11)
        for i in range(len(present)):
            for j in range(len(PAPER_EPS)):
                bg = mat_a[i, j]
                tc = "white" if (np.isfinite(bg) and abs(bg) > 0.2) else "0.15"
                ax.text(j, i, str(lbl_a[i, j]), ha="center", va="center",
                        fontsize=11, fontweight="bold", color=tc)
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.06)
        cb.set_label("winner gap", fontsize=9, labelpad=6)
        cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Radial coarse family (left) and angular-shuffle null (right)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95])
    savefig(fig, "fig17_radial_and_null.pdf")


# ---------------------------------------------------------------------------
# savefig and output validation
# ---------------------------------------------------------------------------

def savefig(fig: plt.Figure, name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    try:
        # If the figure has a _tight_rect attribute, use it to prevent
        # tight_layout from expanding subplots into manually-placed axes
        # (e.g. colorbars added via fig.add_axes).
        rect = getattr(fig, "_tight_rect", None)
        if rect is not None:
            fig.tight_layout(rect=rect)
        else:
            fig.tight_layout()
    except Exception:
        pass  # some figures use constrained_layout instead
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


_FLAGSHIP_FIGURES = [
    "fig01_ic_gallery.pdf",
    "fig02_snapshots.pdf",
    "fig03_verdict_map.pdf",
    "fig04_bimodal_anchor.pdf",
    "fig05_cond_fine.pdf",
    "fig06_eps_transition.pdf",
    "fig07_n_scaling.pdf",
    "fig08_model_comparison.pdf",
    "fig09_diagnostics.pdf",
    "fig10_summary_matrix.pdf",
    "fig11_bimodal_mechanism.pdf",
    "fig12_veldisp_mechanism.pdf",
    "fig13_eps_boundary.pdf",
    "fig14_winner_gap_ci_map.pdf",
    "fig15_class_gap.pdf",
    "fig16_convergence.pdf",
    "fig17_radial_and_null.pdf",
]

_FLAGSHIP_TABLES = [
    "paper_macros.tex",
]

_FLAGSHIP_TABLES_DIR = [
    "verdict_summary.tex",
    "winner_gap_table.tex",
    "n_scaling.tex",
    "cond_fine.tex",
    "family_stability.tex",
    "exclusion_summary.tex",
    "diagnostics.tex",
]


def _pm_force_rms_error(n: int = 256, eps: float = 0.05, seed: int = 42,
                        G: float = 1.0, box_size: float = 2.0,
                        pm_grid: int = 32) -> float:
    """RMS relative force error: PM vs direct at the same particle configuration.

    Creates a random Plummer distribution with N particles on a periodic torus,
    evaluates accelerations with both direct-sum (exact) and PM, and returns:

        rms_err = sqrt( mean_i( |a_PM_i - a_dir_i|² / |a_dir_i|² ) )

    A value < 0.30 (30 % RMS relative) is expected for pm_grid=32 at N=256 and
    is consistent with PM being used for a coarse cross-check, not a precision
    integrator.  Values > 1.0 indicate a PM bug (wrong grid normalisation,
    CIC weight leak, etc.) and should be treated as fatal.
    """
    rng = np.random.default_rng(seed)
    a_scale = 0.20
    u = rng.uniform(1e-6, 1.0 - 1e-6, n)
    r = a_scale / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    r = np.clip(r, 0.0, 0.45 * box_size)
    center = np.full(3, box_size / 2.0)
    dirs = rng.standard_normal((n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30
    pos = np.mod(center + r[:, None] * dirs, box_size)

    mass_pp = 1.0 / n   # total mass = 1

    # Direct forces on a periodic torus
    cfg_dir = SimConfig(
        model="direct_periodic", integrator="leapfrog_kdk", init="plummer3d",
        seed=seed, n=n, G=G, eps=eps, box_size=box_size, pm_grid=pm_grid,
    )
    a_dir = direct_acc(pos, mass_pp, cfg_dir, periodic=True, use_numba=False)

    # PM forces
    cfg_pm = SimConfig(
        model="pm_periodic", integrator="leapfrog_kdk", init="plummer3d",
        seed=seed, n=n, G=G, eps=eps, box_size=box_size, pm_grid=pm_grid,
    )
    a_pm = pm_acc_3d(pos, mass_pp, cfg_pm)

    diff = a_pm - a_dir
    ref  = np.linalg.norm(a_dir, axis=1)
    # Exclude near-zero-force particles (avoid division by tiny ref)
    ok = ref > 1e-6 * float(np.median(ref[ref > 0] if np.any(ref > 0) else [1.0]))
    if not np.any(ok):
        return float("nan")
    rel2 = np.sum(diff[ok] ** 2, axis=1) / (ref[ok] ** 2)
    return float(np.sqrt(np.mean(rel2)))


def validate_outputs(analysis: Dict, rows: List[Dict],
                     skip_figures: bool,
                     skip_tables: bool) -> Tuple[List[str], List[str]]:
    """
    Check that the flagship output set is internally consistent.

    Returns (all_warnings, fatal_warnings).  fatal_warnings is a subset of
    all_warnings.  The caller decides whether to abort on fatal.
    """
    all_w:   List[str] = []
    fatal_w: List[str] = []

    def _warn(msg: str, fatal: bool = False) -> None:
        all_w.append(msg)
        if fatal:
            fatal_w.append(msg)

    # ── 1. Required figures exist (fatal when figures were requested) ──────
    if not skip_figures:
        for fname in _FLAGSHIP_FIGURES:
            p = os.path.join(FIG_DIR, fname)
            if not os.path.exists(p):
                _warn(f"MISSING FIGURE: {p}", fatal=True)

    # ── 2. Required table files exist (advisory) ───────────────────────────
    if not skip_tables:
        for fname in _FLAGSHIP_TABLES:
            p = os.path.join(DATA_DIR, fname)
            if not os.path.exists(p):
                _warn(f"MISSING TABLE/MACRO: {p}", fatal=False)
        for fname in _FLAGSHIP_TABLES_DIR:
            p = os.path.join(TABLE_DIR, fname)
            if not os.path.exists(p):
                _warn(f"MISSING TABLE: {p}", fatal=False)

    # ── 3. Analysis is nonempty (fatal) ───────────────────────────────────
    if not analysis:
        _warn("VALIDATION: analysis dict is empty — no cells produced", fatal=True)
        return all_w, fatal_w  # remaining checks would all be vacuous

    # ── 4. Core flagship cells: missing (fatal) vs underpowered (advisory) ─
    flagship_cells_missing:      List[str] = []
    flagship_cells_underpowered: List[str] = []
    for n in [n for n in PAPER_N if n <= DIRECT_N_MAX]:
        for eps in PAPER_EPS:
            for init in PAPER_INITS:
                cell = get_cell(analysis, "direct_isolated", init, n, eps)
                if not cell:
                    flagship_cells_missing.append(
                        f"direct_isolated/{init}/N={n}/eps={eps:.2f}")
                elif cell.get("underpowered"):
                    n_lw = cell.get("n_listwise_primary", "?")
                    flagship_cells_underpowered.append(
                        f"direct_isolated/{init}/N={n}/eps={eps:.2f} "
                        f"(n_ok={cell.get('n_ok')}, n_listwise={n_lw})")
    if flagship_cells_missing:
        _warn(
            f"MISSING FLAGSHIP CELLS ({len(flagship_cells_missing)}): "
            + "; ".join(flagship_cells_missing[:5])
            + ("..." if len(flagship_cells_missing) > 5 else ""),
            fatal=True)
    if flagship_cells_underpowered:
        _warn(
            f"UNDERPOWERED FLAGSHIP CELLS ({len(flagship_cells_underpowered)}): "
            + "; ".join(flagship_cells_underpowered[:5])
            + ("..." if len(flagship_cells_underpowered) > 5 else ""),
            fatal=False)

    # ── 5. angshuf cells exist — advisory (fig17 right panel degrades) ─────
    angshuf_missing: List[str] = []
    for init_shuf in PAPER_INITS_ANGSHUF:
        found = any(
            c.get("init") == init_shuf and c.get("model") == "direct_isolated"
            for c in analysis.values()
        )
        if not found:
            angshuf_missing.append(init_shuf)
    if angshuf_missing:
        _warn(
            f"MISSING ANGSHUF CELLS (fig17 right panel will be blank): "
            + ", ".join(angshuf_missing),
            fatal=False)

    # ── 6. PM cross-check cells exist (fatal — manuscript claims it) ───────
    pm_cells = [c for c in analysis.values()
                if c.get("model") == "pm_periodic" and not c.get("underpowered")]
    if not pm_cells:
        _warn(
            "NO POWERED PM-PERIODIC CELLS — cross-check claims in text are unsupported",
            fatal=True)

    # ── 7. No NaN winner_gap_mean in direct_isolated primary target (fatal) ─
    poisoned = [
        k for k, c in analysis.items()
        if c.get("model") == "direct_isolated"
        and c.get(f"winner_gap_mean_{PRIMARY_TARGET}") is not None
        and (isinstance(c[f"winner_gap_mean_{PRIMARY_TARGET}"], float)
             and math.isnan(c[f"winner_gap_mean_{PRIMARY_TARGET}"]))
    ]
    if poisoned:
        _warn(
            f"NaN WINNER_GAP_MEAN in direct_isolated cells ({len(poisoned)}): "
            + "; ".join(poisoned[:3]) + ("..." if len(poisoned) > 3 else ""),
            fatal=True)

    # ── 8. PM force accuracy spot-check: PM vs direct-sum at N=256 ───────
    # Advisory only — a reviewer can ask "how accurate is PM?" and this
    # answers it with a concrete RMS relative force error.  Values ≥ 1.0
    # indicate a normalization or weight-leak bug and are flagged as fatal.
    try:
        pm_err = _pm_force_rms_error(n=256, eps=0.05)
        if not math.isfinite(pm_err):
            _warn("PM FORCE DIAGNOSTIC: could not evaluate (nan/inf result)", fatal=False)
        elif pm_err >= 1.0:
            _warn(
                f"PM FORCE ACCURACY FATAL: RMS relative force error = {pm_err:.3f} "
                f"(≥ 1.0 — indicates PM normalization or CIC weight-leak bug)",
                fatal=True)
        elif pm_err >= 0.50:
            _warn(
                f"PM FORCE ACCURACY HIGH: RMS relative error = {pm_err:.3f} "
                f"(> 0.50 — PM is coarse; acceptable for cross-check role but "
                f"verify pm_grid is adequate)", fatal=False)
        else:
            # Passes — record as informational (not a warning per se, just logged)
            all_w.append(
                f"PM force accuracy OK: RMS relative error = {pm_err:.3f} "
                f"(N=256, eps=0.05, pm_grid=32)")
    except Exception as pm_exc:
        _warn(f"PM FORCE DIAGNOSTIC EXCEPTION: {pm_exc}", fatal=False)

    return all_w, fatal_w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--workers",      type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--replicates",   type=int, default=PAPER_REPS)
    parser.add_argument("--n-boot",       type=int, default=1000)
    parser.add_argument("--min-ok-hard",  type=int, default=100)
    parser.add_argument("--conv-repeats", type=int, default=10)
    parser.add_argument("--no-run",        action="store_true")
    parser.add_argument("--resume",        action="store_true",
                        help="Load completed rows from paper_battery.csv or "
                             "paper_battery_checkpoint.csv and only run missing configs.")
    parser.add_argument("--skip-showcase", action="store_true",
                        help="Skip showcase sim generation (faster dev iterations).")
    parser.add_argument("--skip-figures",  action="store_true",
                        help="Skip all figure generation.")
    parser.add_argument("--skip-tables",   action="store_true",
                        help="Skip all LaTeX table writing.")
    parser.add_argument("--strict-validation", action="store_true",
                        help="Abort with sys.exit(1) if any fatal validation check fails.")
    parser.add_argument("--use-numba",    dest="use_numba",
                        action="store_true", default=True)
    parser.add_argument("--no-numba",     dest="use_numba", action="store_false")
    args = parser.parse_args()

    if args.use_numba and not _HAS_NUMBA:
        print("numba not found — falling back to NumPy")
        args.use_numba = False

    ensure_dirs()

    # ── Run manifest ──────────────────────────────────────────────────────────
    manifest = {
        "timestamp":        datetime.datetime.utcnow().isoformat() + "Z",
        "python_version":   sys.version,
        "platform":         platform.platform(),
        "numpy_version":    np.__version__,
        "matplotlib_version": __import__("matplotlib").__version__,
        "numba_used":       args.use_numba,
        "PAPER_N":          PAPER_N,
        "PAPER_EPS":        PAPER_EPS,
        "PAPER_MODELS":     PAPER_MODELS,
        "PAPER_INITS":          PAPER_INITS,
        "PAPER_INITS_ANGSHUF":  PAPER_INITS_ANGSHUF,
        "PAPER_STEPS":      PAPER_STEPS,
        "DIRECT_N_MAX":     DIRECT_N_MAX,
        "replicates":       args.replicates,
        "n_boot":           args.n_boot,
        "min_ok_hard":      args.min_ok_hard,
        "conv_repeats":     args.conv_repeats,
        "SHOWCASE_N":       SHOWCASE_N,
        "SHOWCASE_SEED":    SHOWCASE_SEED,
        "no_run":           args.no_run,
    }
    manifest_path = os.path.join(DATA_DIR, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Run manifest written to {manifest_path}")

    battery_csv = os.path.join(DATA_DIR, "paper_battery.csv")
    checkpoint  = os.path.join(DATA_DIR, "paper_battery_checkpoint.csv")
    conv_path   = os.path.join(DATA_DIR, "convergence.json")

    # ── Battery ───────────────────────────────────────────────────────────────
    if args.no_run:
        # Merge battery_csv + checkpoint (deduplicated) so that rows written only
        # to the checkpoint after a killed resume run are not silently dropped.
        merged_rows: List[Dict[str, Any]] = []
        merged_keys: set = set()
        found_any = False
        for src in (battery_csv, checkpoint):
            if os.path.exists(src):
                found_any = True
                for row in load_csv_rows(src):
                    k = _paper_done_key(row)
                    if k not in merged_keys:
                        merged_keys.add(k)
                        merged_rows.append(row)
        if not found_any:
            raise FileNotFoundError(
                f"--no-run: neither {battery_csv} nor {checkpoint} found.")
        print(f"--no-run: loaded {len(merged_rows)} rows "
              f"(from battery_csv + checkpoint, deduplicated)")
        # Filter to the current config grid so stale rows from old experiments
        # don't silently pollute analysis when --replicates or grid params changed.
        current_keys = {
            (cfg.n, cfg.eps, cfg.k_fine, cfg.model, cfg.init, cfg.seed)
            for cfg in build_configs(reps=args.replicates)
        }
        rows_all = merged_rows
        rows = [r for r in merged_rows if _paper_done_key(r) in current_keys]
        n_stale = len(rows_all) - len(rows)
        if n_stale:
            print(f"  (filtered out {n_stale} stale rows not in current config grid)")
    elif args.resume:
        # Load completed rows from BOTH battery_csv and checkpoint, deduplicating.
        # No break between sources — if a resume run was killed mid-way, battery_csv
        # holds the prior existing rows and checkpoint holds the newer partial rows;
        # we need both to reconstruct the full done set.
        existing_rows: List[Dict[str, Any]] = []
        done_keys: set = set()
        for src in (battery_csv, checkpoint):
            if os.path.exists(src):
                for row in load_csv_rows(src):
                    k = _paper_done_key(row)
                    if k not in done_keys:
                        done_keys.add(k)
                        existing_rows.append(row)
        print(f"Resume: {len(existing_rows)} existing rows loaded.")
        # Write ALL existing rows to battery_csv immediately (unfiltered) so that
        # a subsequent kill + resume can always reload from battery_csv and no
        # completed work is ever lost — even rows from prior grid configurations.
        if existing_rows:
            write_csv_rows(battery_csv, existing_rows)
        configs = build_configs(reps=args.replicates)
        current_keys = {
            (cfg.n, cfg.eps, cfg.k_fine, cfg.model, cfg.init, cfg.seed)
            for cfg in configs
        }
        # Rows that belong to the current config grid — used for analysis and for
        # determining which configs still need to run.  Stale rows (from a prior run
        # with different --replicates / grid params) are excluded from analysis but
        # kept on disk (written above) so they are never permanently destroyed.
        existing_rows_current = [
            r for r in existing_rows if _paper_done_key(r) in current_keys
        ]
        n_stale = len(existing_rows) - len(existing_rows_current)
        if n_stale:
            print(f"  (note: {n_stale} rows from a prior grid config are preserved "
                  f"on disk but excluded from this run's analysis)")
        # done_keys restricted to current grid — stale rows must not block re-running
        # configs that are genuinely absent from the current experiment.
        done_keys_current = {_paper_done_key(r) for r in existing_rows_current}
        pending = [
            cfg for cfg in configs
            if (cfg.n, cfg.eps, cfg.k_fine, cfg.model, cfg.init, cfg.seed) not in done_keys_current
        ]
        n_direct = sum(1 for c in pending if c.model == "direct_isolated")
        n_pm     = sum(1 for c in pending if c.model == "pm_periodic")
        print(f"  pending: {len(pending)} runs  ({len(configs) - len(pending)} already done)\n"
              f"  ({n_direct} direct-isolated, {n_pm} PM-periodic)")
        if pending:
            new_rows = run_battery(args.workers, pending, args.use_numba, checkpoint)
        else:
            new_rows = []
        # rows used for analysis = current-grid existing + new only.
        # battery_csv on disk gets ALL existing + new so nothing is ever lost.
        rows = existing_rows_current + new_rows
        write_csv_rows(battery_csv, existing_rows + new_rows)
        if os.path.exists(checkpoint):
            os.remove(checkpoint)
    else:
        if os.path.exists(checkpoint):
            os.remove(checkpoint)
        configs = build_configs(reps=args.replicates)
        n_direct = sum(1 for c in configs if c.model == "direct_isolated")
        n_pm     = sum(1 for c in configs if c.model == "pm_periodic")
        print(f"Running battery: {len(configs)} runs  "
              f"({n_direct} direct-isolated, {n_pm} PM-periodic)\n"
              f"  direct_isolated capped at N<={DIRECT_N_MAX}")
        rows = run_battery(args.workers, configs, args.use_numba, checkpoint)
        write_csv_rows(battery_csv, rows)

    ok_count = sum(1 for r in rows if r.get("status") == "ok")
    print(f"ok runs: {ok_count}/{len(rows)}")
    if ok_count == 0:
        raise RuntimeError("No successful runs — refusing to generate blank figures.")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print(f"Running analysis ({args.n_boot} bootstrap resamples, "
          f"min_ok_hard={args.min_ok_hard})...")
    analysis = analyse(rows, n_boot=args.n_boot, min_ok_hard=args.min_ok_hard)
    if not analysis:
        raise RuntimeError("Analysis dict is empty.")

    with open(os.path.join(DATA_DIR, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=_json_safe)

    # ── Convergence ───────────────────────────────────────────────────────────
    print(f"Running VelDisp convergence analysis (n_repeats={args.conv_repeats})...")
    conv = convergence_analysis(rows, n_boot=min(args.n_boot, 200),
                                n_repeats=args.conv_repeats)
    with open(conv_path, "w") as f:
        json.dump(conv, f, indent=2, default=_json_safe)

    # ── Showcase sims ─────────────────────────────────────────────────────────
    showcase_specs = [
        ("bimodal3d",     0.05),
        ("hernquist3d",   0.02),
        ("hernquist3d",   0.10),
        ("plummer3d",     0.05),
        ("cold_clumpy3d", 0.05),
    ]
    showcase_pos0:  Dict[str, np.ndarray]            = {}
    showcase_snaps: Dict[str, Dict[Any, np.ndarray]] = {}
    if args.skip_showcase:
        print("Skipping showcase sims (--skip-showcase).")
    else:
        for init, eps in showcase_specs:
            key   = f"{init}_{eps:.2f}"
            snaps = _run_showcase_sim(init, SHOWCASE_SEED, SHOWCASE_N,
                                      eps, PAPER_STEPS)
            showcase_snaps[key] = snaps
            if 0 in snaps:
                if init not in showcase_pos0 or abs(eps - 0.05) < 1e-12:
                    showcase_pos0[init] = snaps[0]

    # ── Tables ────────────────────────────────────────────────────────────────
    if args.skip_tables:
        print("Skipping table generation (--skip-tables).")
    else:
        write_macros(analysis, rows,
                     os.path.join(DATA_DIR,  "paper_macros.tex"),
                     n_replicates=args.replicates, n_boot=args.n_boot)
        write_verdict_summary(analysis,
                              os.path.join(TABLE_DIR, "verdict_summary.tex"))
        write_sensitivity_table(analysis,
                                os.path.join(TABLE_DIR, "verdict_sensitivity.csv"))
        write_winner_gap_table(analysis,
                               os.path.join(TABLE_DIR, "winner_gap_table.tex"))
        write_n_scaling(analysis,
                        os.path.join(TABLE_DIR, "n_scaling.tex"))
        write_cond_fine(analysis,
                        os.path.join(TABLE_DIR, "cond_fine.tex"))
        write_family_stability(analysis,
                               os.path.join(TABLE_DIR, "family_stability.tex"))
        write_exclusion_summary(
            analysis,
            os.path.join(TABLE_DIR, "exclusion_summary.tex"),
            os.path.join(DATA_DIR,  "exclusion_summary.json"),
        )
        write_diagnostics(rows,
                          os.path.join(TABLE_DIR, "diagnostics.tex"))

    # ── Figures ───────────────────────────────────────────────────────────────
    if args.skip_figures:
        print("Skipping figure generation (--skip-figures).")
    else:
        print("Generating figures...")
        fig01_ic_gallery(showcase_pos0)
        fig02_snapshots(showcase_snaps)
        fig03_verdict_map(analysis)
        fig04_bimodal_anchor(analysis, rows)
        fig05_cond_fine(analysis)
        fig06_eps_transition(analysis)
        fig07_n_scaling(analysis)
        fig08_model_comparison(analysis)
        fig09_diagnostics(rows)
        fig10_summary_matrix(analysis)
        fig11_bimodal_mechanism(showcase_pos0, analysis, rows)
        fig12_veldisp_mechanism(showcase_snaps, analysis=analysis)
        fig13_eps_boundary(analysis)
        fig14_winner_gap_ci_map(analysis)
        fig15_class_gap(analysis)
        fig16_convergence(conv_path)
        fig17_radial_and_null(analysis)

    # ── Output validation ─────────────────────────────────────────────────────
    print("Validating outputs...")
    val_warnings, val_fatal = validate_outputs(
        analysis, rows,
        skip_figures=args.skip_figures,
        skip_tables=args.skip_tables,
    )
    if val_warnings:
        print(f"\nVALIDATION WARNINGS ({len(val_warnings)}, "
              f"{len(val_fatal)} fatal):")
        for w in val_warnings:
            tag = "[FATAL]" if w in val_fatal else "[warn] "
            print(f"  {tag} {w}")
        print()
    else:
        print("  All validation checks passed.")

    # ── Update manifest with realized run metadata ─────────────────────────────
    manifest["n_total_runs"]          = len(rows)
    manifest["n_ok_runs"]             = ok_count
    manifest["battery_csv"]           = battery_csv
    manifest["analysis_json"]         = os.path.join(DATA_DIR, "analysis.json")
    manifest["convergence_json"]      = conv_path
    manifest["completed_at"]          = datetime.datetime.utcnow().isoformat() + "Z"
    manifest["validation_warnings"]   = val_warnings
    manifest["validation_fatal"]      = val_fatal
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if val_fatal and args.strict_validation:
        print(f"ABORTING: --strict-validation is set and {len(val_fatal)} "
              "fatal validation check(s) failed. See warnings above.")
        sys.exit(1)

    if val_warnings:
        print(f"Done — with {len(val_warnings)} validation warning(s). "
              "Check manifest for details.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
