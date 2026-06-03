#!/usr/bin/env python3
"""
coarse_grain_features.py — scale-resolved feature family φ_ℓ for the coarse-graining pilot
==========================================================================================

Given a t₀ particle snapshot (positions, velocities) from the existing N-body engine,
this module builds:

  • φ_ℓ  — a fixed-dimension feature vector describing the system *coarse-grained at
           scale ℓ*.  The t₀ count field is CIC-deposited on a G³ grid (reusing the
           repo's `_cic_deposit3`), Gaussian-smoothed with kernel width ℓ, and reduced to
           a ~15-D vector: smoothed radial mass profile + density-field moments +
           inertia-tensor shape.  ℓ enters *only* through the smoothing width, so the same
           snapshot yields features at every scale with no re-simulation.

  • baselines — the paper's existing single-scale observables (fine positional, fine
           kinematic, fixed-scale coarse, persistence) plus bulk controls.  These reuse the
           `obs_*` functions from `nbody_stress.py` verbatim so the comparison is apples-to-
           apples with the published battery.

Nothing here runs a simulation; it operates purely on cached snapshots.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis, skew

from nbody_3d import (
    _cic_deposit3,
    half_mass_radius,
    kinetic_energy,
    potential_energy_direct,
    virial_ratio,
)
from nbody_stress import (
    StressConfig,
    get_simconfig,
    obs_coarse_conc,
    obs_coarse_rshell_var,
    obs_coarse_var,
    obs_fine_close_pairs,
    obs_fine_knn_all,
    obs_fine_local_vel_disp,
    obs_fine_pk_small,
    obs_fof_groups,
)

Array = np.ndarray

# ── φ_ℓ grid geometry (precomputed once per grid spec) ────────────────────────────

_GEOM_CACHE: Dict[tuple, dict] = {}

N_RADIAL_DEFAULT = 8
R_MAX_DEFAULT    = 0.60   # radial extent of the φ_ℓ description (box half-side is 1.0)
R_LO_DEFAULT     = 0.03   # inner log-bin edge (resolves the cusp core)


def _grid_geometry(g: int, box_size: float,
                   n_radial: int = N_RADIAL_DEFAULT,
                   r_max: float = R_MAX_DEFAULT,
                   r_lo: float = R_LO_DEFAULT) -> dict:
    """Precompute cell-centre geometry for the φ_ℓ reductions.

    Features are measured about the *fixed* box centre (the IC centre), not the
    per-replicate COM.  This keeps a single shared geometry (big speed-up) and turns the
    COM offset into a separate bulk control rather than an implicit re-centring.
    """
    key = (g, round(box_size, 6), n_radial, round(r_max, 6), round(r_lo, 6))
    geom = _GEOM_CACHE.get(key)
    if geom is not None:
        return geom

    cell = box_size / g
    centre = box_size / 2.0
    ax = (np.arange(g) + 0.5) * cell - centre          # cell-centre coord on one axis
    cx, cy, cz = np.meshgrid(ax, ax, ax, indexing="ij")
    coords = np.column_stack([cx.ravel(), cy.ravel(), cz.ravel()])   # (g³, 3)
    r2 = np.sum(coords * coords, axis=1)
    r = np.sqrt(r2)

    sphere = r <= r_max                                # restrict reductions to the IC region
    coords_s = coords[sphere]
    r2_s = r2[sphere]

    edges = np.logspace(np.log10(r_lo), np.log10(r_max), n_radial + 1)
    shell = np.clip(np.digitize(r[sphere], edges) - 1, 0, n_radial - 1)
    shell_counts = np.bincount(shell, minlength=n_radial).astype(float)
    shell_counts[shell_counts == 0] = np.nan           # empty shells → NaN profile entry

    geom = {
        "g": g, "cell": cell, "n_radial": n_radial,
        "sphere": sphere, "coords_s": coords_s, "r2_s": r2_s,
        "shell": shell, "shell_counts": shell_counts,
    }
    _GEOM_CACHE[key] = geom
    return geom


def phi_feature_names(n_radial: int = N_RADIAL_DEFAULT) -> List[str]:
    return (
        [f"phi_rad{i}" for i in range(n_radial)]
        + ["phi_std", "phi_skew", "phi_kurt"]
        + ["phi_eig0", "phi_eig1", "phi_eig2", "phi_rms"]
    )


def _phi_from_smoothed(rho_l: Array, geom: dict) -> Array:
    """Reduce a smoothed count field ρ_ℓ to the fixed φ_ℓ vector."""
    n_radial = geom["n_radial"]
    n_feat = n_radial + 3 + 4
    rho_s = rho_l.ravel()[geom["sphere"]]
    tot = float(rho_s.sum())
    if not math.isfinite(tot) or tot <= 0.0:
        return np.full(n_feat, np.nan)

    # smoothed radial mass profile (mean ρ_ℓ per log shell)
    shell_sum = np.bincount(geom["shell"], weights=rho_s, minlength=n_radial)
    profile = shell_sum / geom["shell_counts"]

    # density-field moments over the IC region (clumpiness at scale ℓ)
    f_std = float(np.std(rho_s))
    f_skew = float(skew(rho_s)) if f_std > 1e-12 else 0.0
    f_kurt = float(kurtosis(rho_s)) if f_std > 1e-12 else 0.0

    # inertia tensor / shape of the smoothed mass about the box centre
    coords_s = geom["coords_s"]
    trace_term = float(np.dot(rho_s, geom["r2_s"]))
    m_ab = np.einsum("n,na,nb->ab", rho_s, coords_s, coords_s)
    inertia = trace_term * np.eye(3) - m_ab
    eig = np.sort(np.linalg.eigvalsh(inertia))[::-1]
    eig_sum = float(eig.sum())
    eig_frac = (eig / eig_sum) if abs(eig_sum) > 1e-30 else np.full(3, np.nan)
    rms = math.sqrt(max(trace_term / tot, 0.0))

    return np.concatenate([
        profile,
        [f_std, f_skew, f_kurt],
        eig_frac,
        [rms],
    ]).astype(np.float64)


def deposit_count_field(pos: Array, g: int, box_size: float) -> Array:
    """CIC count field on a g³ grid (isolated: out-of-box particles discarded)."""
    return _cic_deposit3(pos, mass_pp=1.0, L=box_size, g=g, periodic=False)


def phi_all_scales(pos: Array, g: int, box_size: float,
                   scales: List[float],
                   n_radial: int = N_RADIAL_DEFAULT,
                   r_max: float = R_MAX_DEFAULT,
                   r_lo: float = R_LO_DEFAULT) -> Array:
    """φ_ℓ for every scale, from a single deposit. Returns (n_scales, n_feat)."""
    geom = _grid_geometry(g, box_size, n_radial, r_max, r_lo)
    field = deposit_count_field(pos, g, box_size)
    cell = geom["cell"]
    out = np.empty((len(scales), n_radial + 7), dtype=np.float64)
    for s, ell in enumerate(scales):
        sigma = ell / cell
        # mode='constant' (zeros outside): isolated open-space semantics, matches the
        # discard-not-wrap rule the repo's grid observables use.
        rho_l = gaussian_filter(field, sigma=sigma, mode="constant", cval=0.0)
        out[s] = _phi_from_smoothed(rho_l, geom)
    return out


# ── Baseline feature groups (reuse the paper's observables verbatim) ──────────────

# (column, ordered names) for each baseline group.
BASELINE_GROUPS: Dict[str, List[str]] = {
    "fine_pos":     ["knn_all", "pk_small", "close_pairs", "fof"],
    "fine_kin":     ["vel_disp"],
    "fine_all":     ["knn_all", "pk_small", "close_pairs", "fof", "vel_disp"],
    "coarse_fixed": ["cg4", "cg8", "cg16", "conc", "rshell_var"],
    "persistence":  ["cg8_0"],
    "bulk":         ["virial0", "ke0", "hmr0", "n_in_box", "rms_size", "com_offset"],
}


def baseline_features(pos: Array, vel: Array, cfg: StressConfig) -> Dict[str, float]:
    """Compute every scalar baseline observable from a t₀ snapshot (ℓ-independent).

    All distance/grid observables reuse the exact `obs_*` implementations from the
    published battery, evaluated for an isolated (open-boundary) system.
    """
    box = cfg.box_size
    mass = 1.0 / cfg.n
    sc = get_simconfig(cfg)
    periodic = False

    # fine positional + kinematic
    knn = obs_fine_knn_all(pos, cfg.k_fine, cfg.eps, periodic, box)
    pk = obs_fine_pk_small(pos, periodic, box, cfg.pm_grid, mass)
    close = obs_fine_close_pairs(pos, cfg.eps, periodic, box)
    fof = obs_fof_groups(pos, periodic, box, cfg.fof_b)
    vdisp = obs_fine_local_vel_disp(pos, vel, cfg.k_fine, periodic, box)

    # fixed-scale coarse (Cartesian + radial families)
    cg4 = obs_coarse_var(pos, cfg, 4, periodic)
    cg8 = obs_coarse_var(pos, cfg, 8, periodic)
    cg16 = obs_coarse_var(pos, cfg, 16, periodic)
    conc = obs_coarse_conc(pos, periodic, box)
    rshell = obs_coarse_rshell_var(pos, periodic, box)

    # bulk controls
    ke0 = kinetic_energy(vel, mass)
    pe0 = potential_energy_direct(pos, mass, sc)
    virial0 = virial_ratio(ke0, pe0)
    hmr0 = half_mass_radius(pos, periodic)
    inside = np.all((pos >= 0.0) & (pos < box), axis=1)
    n_in_box = float(np.sum(inside))
    com = np.mean(pos, axis=0)
    rms_size = float(np.sqrt(np.mean(np.sum((pos - com) ** 2, axis=1))))
    com_offset = float(np.linalg.norm(com - box / 2.0))

    return {
        "knn_all": knn, "pk_small": pk, "close_pairs": close, "fof": fof,
        "vel_disp": vdisp,
        "cg4": cg4, "cg8": cg8, "cg16": cg16, "conc": conc, "rshell_var": rshell,
        "cg8_0": cg8,
        "virial0": virial0, "ke0": ke0, "hmr0": hmr0,
        "n_in_box": n_in_box, "rms_size": rms_size, "com_offset": com_offset,
    }


def baseline_matrix(scalars: List[Dict[str, float]], group: str) -> Tuple[Array, List[str]]:
    """Stack a baseline group's scalars into an (n_rep, n_col) matrix."""
    cols = BASELINE_GROUPS[group]
    mat = np.array([[float(s.get(c, np.nan)) for c in cols] for s in scalars],
                   dtype=np.float64)
    return mat, list(cols)
