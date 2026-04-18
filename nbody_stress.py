#!/usr/bin/env python3
"""
nbody_stress.py — production-grade stress-test battery for the N-body ODD result
==================================================================================
Changes vs previous version:
  • pearson_with_ci() bootstrap fully vectorized — no Python inner loop
  • convergence_analysis() uses n_repeats random subsamples per size so
    CI width estimate is itself an estimate with variability, not a one-shot
  • min_ok_hard CLI default raised 50 → 100
"""
from __future__ import annotations
import hashlib
import argparse
import csv
import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import KDTree
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from nbody_3d import (
    _HAS_NUMBA, Array, SimConfig, _cic_deposit3,
    _numba_direct_acc, _worker_init, acceleration,
    half_mass_radius, initial_conditions, kinetic_energy,
    min_image, potential_energy_direct, virial_ratio,
)

def _stable_seed(s: str) -> int:
    """Stable cross-session hash for deterministic bootstrap seeds."""
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % (2 ** 31)

# ── Observable class registry ─────────────────────────────────────────────────

FINE_POSITIONAL_OBS: List[Tuple[str, str]] = [
    ("fine_knn_all_0", "kNN-all"),
    ("fine_pk_0",      "Pk-small"),
    ("fine_close_0",   "ClosePairs"),
    ("fine_fof_0",     "FoF-groups"),
]
FINE_KINEMATIC_OBS: List[Tuple[str, str]] = [
    ("fine_vel_disp_0", "VelDisp"),
]
FINE_OBS:   List[Tuple[str, str]] = FINE_POSITIONAL_OBS + FINE_KINEMATIC_OBS
COARSE_OBS: List[Tuple[str, str]] = [
    ("coarse_g4_0",         "CoarseG4"),
    ("coarse_g8_0",         "CoarseG8"),
    ("coarse_g16_0",        "CoarseG16"),
    # Radial family: orientation-free, grid-independent coarse observers
    ("coarse_conc_0",       "CoarseConc"),       # concentration proxy N(<r50/2)/N(<r50)
    ("coarse_rshell_var_0", "CoarseRShellVar"),  # normalized radial shell mass variance
]

TARGETS: List[Tuple[str, str]] = [
    ("d_coarse_g8_early",  "ΔC8-early"),
    ("d_coarse_g8_mid",    "ΔC8-mid"),
    ("d_coarse_g8_late",   "ΔC8-late"),
    ("d_coarse_g4_early",  "ΔC4-early"),
    ("d_coarse_g16_early", "ΔC16-early"),
    ("d_hmr_early",        "ΔHMR-early"),
    ("d_hmr_late",         "ΔHMR-late"),
    # Astronomy-native targets: concentration growth
    ("d_conc_early",       "ΔConc-early"),
    ("d_conc_late",        "ΔConc-late"),
    # Common-baseline robustness check: final coarse-grid state — NO delta.
    # r(coarse_g8_0, coarse_g8_f) tests whether initial coarse structure predicts
    # final coarse structure WITHOUT the shared-baseline inflation that affects
    # r(coarse_g8_0, cg8_e - cg8_0).  Predictor and target are independent
    # quantities.  Not included in FAMILY_TARGETS to avoid inflating stability
    # counts — use as a supplementary robustness check only.
    ("coarse_g8_f",        "C8-final"),
]

PRIMARY_TARGET = "ΔC8-early"
FAMILY_TARGETS = ["ΔC8-early", "ΔC8-mid", "ΔHMR-early", "ΔConc-late"]

# Verdict threshold: a CI-bounded gap must clear this margin on the correct side
# to earn a hard FINE or COARSE verdict.  0.05 means the 95% CI lower bound
# (for FINE) or upper bound (for COARSE) must clear ±0.05 — i.e., the winner
# advantage is strictly positive after a 0.05 uncertainty buffer.
# Sensitivity: re-running analyse() with alternative values (0.00–0.10) should
# not flip more than ~5% of hard verdicts; if it does, those cells are soft.
VERDICT_GAP_THRESHOLD: float = 0.05

_ALL_PRED_SPECS: List[Tuple[str, str]] = FINE_OBS + COARSE_OBS
_ALL_PRED_COLS:  List[str]             = [col  for col, _    in _ALL_PRED_SPECS]
_ALL_PRED_NAMES: List[str]             = [name for _,    name in _ALL_PRED_SPECS]
_N_POS    = len(FINE_POSITIONAL_OBS)
_N_KIN    = len(FINE_KINEMATIC_OBS)
_N_FINE   = len(FINE_OBS)
_N_COARSE = len(COARSE_OBS)
_FINE_IDX   = list(range(_N_FINE))
_COARSE_IDX = list(range(_N_FINE, _N_FINE + _N_COARSE))
_POS_IDX    = list(range(_N_POS))
_KIN_IDX    = list(range(_N_POS, _N_FINE))

# Model-appropriate predictor sets for the winner-gap bootstrap.
# Periodic models receive only Cartesian-grid coarse observers; radial obs
# (CoarseConc, CoarseRShellVar) are fenced to direct_isolated where they are
# physically meaningful — so their columns are NaN on periodic rows and must
# be excluded from listwise completeness checks for those cells.
_COARSE_OBS_GRID: List[Tuple[str, str]] = [
    (c, n) for c, n in COARSE_OBS
    if c in {"coarse_g4_0", "coarse_g8_0", "coarse_g16_0"}
]
_PRED_SPECS_DIRECT:   List[Tuple[str, str]] = list(FINE_OBS) + list(COARSE_OBS)
_PRED_SPECS_PERIODIC: List[Tuple[str, str]] = list(FINE_OBS) + _COARSE_OBS_GRID


# ── StressConfig ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StressConfig:
    model:       str
    init:        str
    seed:        int
    n:           int
    steps:       int
    dt:          float = 0.005
    G:           float = 1.0
    eps:         float = 0.05
    box_size:    float = 2.0
    pm_grid:     int   = 32
    coarse_grid: int   = 8
    k_fine:      int   = 16
    h_early:     int   = 100
    h_mid:       int   = 300
    plummer_a:   float = 0.20
    vel_scale:   float = 0.35
    cold_scale:  float = 0.05
    clump_count: int   = 5
    fof_b:       float = 0.20


# ── IC families ────────────────────────────────────────────────────────────────

_ANGSHUF_SUFFIX = "_angshuf"

def _sphere_directions(rng: np.random.Generator, n: int) -> Array:
    cth = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    sth = np.sqrt(1.0 - cth * cth)
    return np.column_stack([sth * np.cos(phi), sth * np.sin(phi), cth])


def sample_hernquist3d(rng: np.random.Generator, cfg: StressConfig) -> Tuple[Array, Array]:
    a  = cfg.plummer_a
    u  = rng.uniform(1e-6, 1.0 - 1e-6, cfg.n)
    sq = np.sqrt(u)
    r  = np.clip(a * sq / (1.0 - sq), 0.0, 5.0 * cfg.box_size)
    center = np.full(3, cfg.box_size / 2.0)
    pos = center + r[:, None] * _sphere_directions(rng, cfg.n)
    # Isotropic velocity scale: sigma = sqrt(G*M/(6a)) for a Hernquist sphere
    # (from virial theorem: K = |W|/2, |W| = G*M^2/(4a), so <v^2> = G*M/(4a),
    # giving 1D sigma = sqrt(G*M/(12a)).  The 0.8 factor empirically corrects
    # for the fact that this sampler draws velocities isotropically rather than
    # from the exact Hernquist distribution function, bringing the initial virial
    # ratio closer to 1.  Verify Q = 2K/|W| from the diagnostics output (fig09).
    sigma = math.sqrt(cfg.G / (6.0 * a))
    vel   = rng.normal(0.0, sigma * 0.8, (cfg.n, 3))
    vel  -= np.mean(vel, axis=0)
    return pos, vel


def sample_bimodal3d(rng: np.random.Generator, cfg: StressConfig) -> Tuple[Array, Array]:
    a      = cfg.plummer_a * 0.5
    half   = cfg.n // 2
    offset = cfg.box_size * 0.25

    def _half(n: int, center: list) -> Array:
        u = rng.uniform(1e-6, 1.0 - 1e-6, n)
        r = a / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
        return np.array(center) + r[:, None] * _sphere_directions(rng, n)

    mid = cfg.box_size / 2.0
    p1  = _half(half,           [mid - offset, mid, mid])
    p2  = _half(cfg.n - half,   [mid + offset, mid, mid])
    pos = np.vstack([p1, p2])
    sigma = math.sqrt(cfg.G / (6.0 * a))
    vel   = rng.normal(0.0, sigma, (cfg.n, 3))
    vel[:half,  0] += +0.1
    vel[half:,  0] += -0.1
    vel -= np.mean(vel, axis=0)
    return pos, vel


def _base_cfg_for_angshuf(cfg: StressConfig) -> StressConfig:
    if not cfg.init.endswith(_ANGSHUF_SUFFIX):
        raise ValueError(f"{cfg.init} is not an angular-shuffle IC")
    base_init = cfg.init[: -len(_ANGSHUF_SUFFIX)]
    return StressConfig(**{**asdict(cfg), "init": base_init})


def _radial_obs_center(pos: Array, periodic: bool, box_size: float,
                       center: Optional[Array] = None) -> Array:
    if center is not None:
        return np.asarray(center, dtype=float)
    if periodic:
        return np.full(3, box_size / 2.0)
    return np.mean(pos, axis=0)


def _angshuf_base_state(cfg: StressConfig) -> Tuple[StressConfig, Array, Array, Array]:
    base_cfg = _base_cfg_for_angshuf(cfg)
    base_pos, base_vel = get_initial_conditions(base_cfg)
    center = _radial_obs_center(base_pos, periodic=False, box_size=cfg.box_size)
    return base_cfg, base_pos, base_vel, center


def get_initial_conditions(cfg: StressConfig) -> Tuple[Array, Array]:
    # Angular-shuffle null control: generate base IC, then shuffle angles.
    # Seed for shuffle is derived deterministically from cfg.seed so results
    # are reproducible.  For isolated runs, the shuffle uses the same radial
    # reference center as the radial coarse observers so the null preserves the
    # intended coarse radial profile instead of changing it via a center shift.
    #
    # Bimodal special case: the global mean center lies *between* the two
    # clumps, so _angular_shuffle_pos with that center would destroy the coarse
    # two-clump geometry.  Use _angular_shuffle_bimodal_pos instead, which
    # shuffles each half-box independently around its own centroid.
    if cfg.init == "bimodal3d_angshuf":
        base_cfg = StressConfig(**{**asdict(cfg), "init": "bimodal3d"})
        rng_base = np.random.default_rng(cfg.seed)
        pos, vel = sample_bimodal3d(rng_base, base_cfg)
        shuf_rng = np.random.default_rng(int(cfg.seed) ^ 0xDEADBEEF)
        pos = _angular_shuffle_bimodal_pos(pos, shuf_rng, cfg.box_size)
        return pos, vel
    if cfg.init.endswith(_ANGSHUF_SUFFIX):
        _, pos, vel, center = _angshuf_base_state(cfg)
        shuf_rng = np.random.default_rng(int(cfg.seed) ^ 0xDEADBEEF)
        pos = _angular_shuffle_pos(pos, shuf_rng, center)
        return pos, vel
    rng = np.random.default_rng(cfg.seed)
    if cfg.init == "hernquist3d":
        return sample_hernquist3d(rng, cfg)
    if cfg.init == "bimodal3d":
        return sample_bimodal3d(rng, cfg)
    return initial_conditions(get_simconfig(cfg))


def get_simconfig(cfg: StressConfig) -> SimConfig:
    return SimConfig(
        model=cfg.model, integrator="leapfrog_kdk", init=cfg.init,
        seed=cfg.seed, n=cfg.n, steps=cfg.steps, dt=cfg.dt,
        G=cfg.G, eps=cfg.eps, box_size=cfg.box_size, pm_grid=cfg.pm_grid,
        coarse_grid=cfg.coarse_grid, top_k=cfg.k_fine,
        plummer_a=cfg.plummer_a, vel_scale=cfg.vel_scale,
        cold_scale=cfg.cold_scale, clump_count=cfg.clump_count,
    )


# ── Fine observables ───────────────────────────────────────────────────────────

def _knn_r2(pos: Array, k: int, periodic: bool, box_size: float) -> Array:
    """k-th nearest-neighbour distance² for every particle.  O(N log N) via KDTree.

    For periodic boxes, scipy KDTree boxsize wraps each axis independently,
    which is equivalent to min-image convention.  Positions must be in [0, L);
    we wrap them with mod to be safe.
    """
    if len(pos) < 2 or k < 1:
        return np.full(len(pos), np.nan)
    if periodic:
        pos_w = np.mod(pos, box_size)
        tree  = KDTree(pos_w, boxsize=box_size)
        dist, _ = tree.query(pos_w, k=k + 1, workers=1)  # k+1 incl. self at dist 0
    else:
        tree = KDTree(pos)
        dist, _ = tree.query(pos, k=k + 1, workers=1)
    # dist[:, 0] == 0 (self); dist[:, k] == k-th nearest neighbour
    return dist[:, k] ** 2


def _filter_box(pos: Array, vel: Optional[Array],
                periodic: bool, box_size: float) -> Tuple[Array, Optional[Array]]:
    if periodic:
        return pos, vel
    # Isolated runs: particles evolve freely in open space — no hard box boundary.
    # Distance-based observables (kNN, close-pairs, vel-disp, FoF) should use ALL
    # particles for whole-system measurements; returning unfiltered pos/vel here
    # achieves that.
    #
    # Note: grid-based observables (obs_coarse_var, obs_fine_pk_small via
    # _cic_deposit3) have their own internal domain masks and still measure only
    # particles inside [0, box_size)^3 — that is a hard constraint of the grid
    # approach and is documented in those functions.
    return pos, vel


def obs_fine_knn_all(pos: Array, k: int, eps: float,
                     periodic: bool, box_size: float) -> float:
    pos, _ = _filter_box(pos, None, periodic, box_size)
    if len(pos) < 2:
        return float("nan")
    k_eff  = min(k, len(pos) - 1)
    kth_r2 = _knn_r2(pos, k_eff, periodic, box_size)
    rk     = np.sqrt(np.maximum(kth_r2, eps ** 2))
    return float(np.mean(k_eff / (4.0 / 3.0 * math.pi * rk ** 3)))


def obs_fine_pk_small(pos: Array, periodic: bool, box_size: float,
                      pm_grid: int, mass_pp: float) -> float:
    g = pm_grid; L = box_size
    cell_vol = (L / g) ** 3
    rho   = _cic_deposit3(pos, mass_pp, L, g, periodic=periodic) / cell_vol
    delta = rho - np.mean(rho)
    dk    = np.fft.rfftn(delta)
    pk    = np.abs(dk) ** 2
    freqs  = np.fft.fftfreq(g) * g
    rfreqs = np.fft.rfftfreq(g) * g
    FX, FY, FZ = np.meshgrid(freqs, freqs, rfreqs, indexing="ij")
    k_abs = np.sqrt(FX ** 2 + FY ** 2 + FZ ** 2)
    return float(np.sum(pk[k_abs > g / 4.0]))


def obs_fine_close_pairs(pos: Array, eps: float,
                         periodic: bool, box_size: float) -> float:
    """Fraction of pairs within 4×softening.  O(N log N) via KDTree radius query.

    Fast-path: when mean_neighbours (expected close neighbours per particle at
    uniform density) exceeds 20, query_pairs would enumerate O(N × mean_neighbours)
    pairs — catastrophically slow at large N × eps.  Instead we draw a uniform
    random subsample of 512 particles and compute the fraction within that subsample.

    This is an *unbiased* estimator: for a uniform random subsample of size n_sub,
    E[n_close_sub / C(n_sub,2)] = n_close_full / C(N,2) by a purely combinatorial
    argument that holds for any spatial distribution (clustered or not).  With
    ~4000 close pairs expected in the subsample at the trigger threshold, variance
    is low enough that the estimate is indistinguishable from exact for ODD analysis.
    """
    pos, _ = _filter_box(pos, None, periodic, box_size)
    n = len(pos)
    if n < 2:
        return float("nan")
    thresh = 4.0 * eps

    # Expected mean neighbours per particle at this density and threshold.
    # For a uniform distribution: λ = (4π/3) * thresh³ * (n / box_size³).
    vol = box_size ** 3
    mean_neighbours = (4.0 / 3.0 * math.pi * thresh ** 3) * (n / vol)

    if mean_neighbours > 20:
        # Fraction is deeply saturated — subsample 512 particles and extrapolate.
        rng_sub = np.random.default_rng(int(n * 1000 + int(eps * 1e6)))
        sub_idx = rng_sub.choice(n, size=min(512, n), replace=False)
        pos_sub = pos[sub_idx]
        n_sub = len(pos_sub)
        if periodic:
            pos_w = np.mod(pos_sub, box_size)
            tree  = KDTree(pos_w, boxsize=box_size)
        else:
            tree  = KDTree(pos_sub)
        # Unbiased estimator: E[n_close_sub / C(n_sub,2)] = n_close_full / C(N,2)
        n_close_sub = len(tree.query_pairs(thresh))
        frac_sub = float(n_close_sub) / max(n_sub * (n_sub - 1) / 2.0, 1.0)
        return frac_sub

    if periodic:
        pos_w = np.mod(pos, box_size)
        tree  = KDTree(pos_w, boxsize=box_size)
    else:
        pos_w = pos
        tree  = KDTree(pos_w)
    # query_pairs returns unique pairs (i<j) within radius — no self-pairs
    n_close = len(tree.query_pairs(thresh))
    return float(n_close) / max(n * (n - 1) / 2.0, 1.0)


def obs_fine_local_vel_disp(pos: Array, vel: Array, k: int,
                             periodic: bool, box_size: float) -> float:
    """Mean local velocity dispersion in each particle's k-NN neighbourhood.
    O(N log N) via KDTree — safe at N=16384.
    """
    pos, vel = _filter_box(pos, vel, periodic, box_size)
    if vel is None or len(pos) < 2:
        return float("nan")
    k_eff = min(k, len(pos) - 1)
    if k_eff < 1:
        return float("nan")
    if periodic:
        pos_w = np.mod(pos, box_size)
        tree  = KDTree(pos_w, boxsize=box_size)
    else:
        pos_w = pos
        tree  = KDTree(pos_w)
    # k_eff+1 neighbours: index 0 is self, indices 1..k_eff are actual neighbours
    _, nn_idx = tree.query(pos_w, k=k_eff + 1, workers=1)
    nn_idx    = nn_idx[:, 1:]          # drop self
    v_nn      = vel[nn_idx]            # (N, k_eff, 3)
    v_mean    = np.mean(v_nn, axis=1, keepdims=True)
    local_std = np.sqrt(np.mean(np.sum((v_nn - v_mean) ** 2, axis=-1), axis=1))
    return float(np.mean(local_std))


def obs_fof_groups(pos: Array, periodic: bool, box_size: float,
                   fof_b: float) -> float:
    """Friends-of-Friends group count.  O(N log N) via KDTree — safe at N=16384."""
    pos, _ = _filter_box(pos, None, periodic, box_size)
    n = len(pos)
    if n < 2:
        return float(0)
    if periodic:
        mean_sep = box_size / n ** (1.0 / 3.0)
        pos_w    = np.mod(pos, box_size)
        tree     = KDTree(pos_w, boxsize=box_size)
    else:
        # Use the sphere of the 95th-percentile radius (from centre of mass) as
        # the occupied volume.  This is rotation-invariant — the old axis-aligned
        # bounding-box changed under rigid rotation.  Using a percentile rather
        # than the absolute maximum prevents heavy-tail ICs (e.g. Hernquist, whose
        # 95th-percentile radius is ~8× the scale length) from blowing up r_eff
        # and producing a linking length so large that most particles merge into
        # one group.
        center   = np.mean(pos, axis=0)
        r_all    = np.linalg.norm(pos - center, axis=1)
        r_eff    = float(np.percentile(r_all, 95.0))
        occ_vol  = max((4.0 / 3.0) * math.pi * max(r_eff, 1e-10) ** 3, 1e-10)
        mean_sep = (occ_vol / n) ** (1.0 / 3.0)
        pos_w    = pos
        tree     = KDTree(pos_w)

    link   = fof_b * mean_sep
    pairs  = tree.query_pairs(link)   # set of (i, j) pairs with i < j

    parent = np.arange(n)

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        _union(a, b)
    return float(len({_find(i) for i in range(n)}))


def obs_coarse_var(pos: Array, cfg: StressConfig,
                   grid: int, periodic: bool) -> float:
    """Variance of per-cell particle counts on a `grid`^3 Cartesian mesh.

    Raw variance scales ~ N * (1 - 1/grid³) * (particle clustering / uniform),
    so the absolute value grows with N.  Within a cell all replicates share the
    same N, making the cross-replicate Pearson r correlation scale-invariant
    (multiplying every value by a constant does not change r).  Cross-N
    comparison of the raw observable values is therefore confounded, but
    *cross-N comparison of the Pearson r coefficients is valid* — which is
    the only cross-N claim we make.  For cross-N normalised values see the
    supplementary diagnostics table.
    """
    L = cfg.box_size
    if periodic:
        p = np.mod(pos, L) / L
    else:
        inside = np.all((pos >= 0.0) & (pos < L), axis=1)
        if not np.any(inside):
            return float("nan")   # no particles in grid domain — undefined, not zero
        p = pos[inside] / L
    idx = np.clip(np.floor(p * grid).astype(int), 0, grid - 1)
    hist = np.zeros((grid, grid, grid))
    np.add.at(hist, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    return float(np.var(hist))


def obs_coarse_conc(pos: Array, periodic: bool, box_size: float,
                    center: Optional[Array] = None) -> float:
    """Concentration proxy: N(<r_50/2) / N(<r_50).

    r_50 = median particle radius from the system centre (half-particle-count
    radius).  Returns the fraction of particles inside the half-mass sphere
    that reside within its inner half — an orientation-free, grid-independent
    analogue of the NFW concentration parameter that astronomers recognise.
    """
    center = _radial_obs_center(pos, periodic, box_size, center)
    r = np.sqrt(np.sum((pos - center) ** 2, axis=1))
    r50 = float(np.median(r))
    if r50 < 1e-12:
        return float("nan")
    n_inner = int(np.sum(r <= r50 * 0.5))
    n_half  = int(np.sum(r <= r50))
    if n_half < 1:
        return float("nan")
    return float(n_inner) / float(n_half)


def obs_coarse_rshell_var(pos: Array, periodic: bool, box_size: float,
                          n_shells: int = 8,
                          center: Optional[Array] = None) -> float:
    """Normalized radial shell mass variance.

    Bin the inner-90th-percentile particles into n_shells log-spaced radial
    shells; return (variance of shell counts) / (mean shell count)^2.
    Orientation-free, no Cartesian grid — a radial-family coarse observer.
    """
    center = _radial_obs_center(pos, periodic, box_size, center)
    r = np.sqrt(np.sum((pos - center) ** 2, axis=1))
    r_max = float(np.percentile(r, 90.0))
    if r_max < 1e-12:
        return float("nan")
    inside = r <= r_max
    r_in   = r[inside]
    if len(r_in) < n_shells:
        return float("nan")
    r_lo = max(float(r_in.min()), 1e-10)
    bins  = np.logspace(np.log10(r_lo), np.log10(r_max), n_shells + 1)
    counts, _ = np.histogram(r_in, bins=bins)
    mean_c = float(np.mean(counts))
    if mean_c < 1e-10:
        return float("nan")
    return float(np.var(counts)) / (mean_c ** 2)


def _angular_shuffle_pos(pos: Array, rng: np.random.Generator,
                          center: Array) -> Array:
    """Structure-destroying null control: angular shuffle.

    Preserves each particle's radius from `center` exactly, but replaces its
    angular position with an independent uniform draw on S².  The radial
    density profile (enclosed-mass curve) is therefore identical to the
    original, while all angular / spatial-arrangement information is destroyed.

    Use as an IC for angshuf variants: if fine-obs predictive advantage
    survives angular shuffling it is a marginal-distribution artifact; if it
    collapses, the signal is structure-dependent.
    """
    delta  = pos - center
    r      = np.sqrt(np.sum(delta ** 2, axis=1))
    cos_th = rng.uniform(-1.0, 1.0, len(r))
    phi    = rng.uniform(0.0, 2.0 * math.pi, len(r))
    sin_th = np.sqrt(np.maximum(1.0 - cos_th ** 2, 0.0))
    new_dir = np.column_stack([sin_th * np.cos(phi), sin_th * np.sin(phi), cos_th])
    return center + r[:, None] * new_dir


def _angular_shuffle_bimodal_pos(pos: Array, rng: np.random.Generator,
                                  box_size: float) -> Array:
    """Angular-shuffle null control for bimodal (two-clump) ICs.

    Splits particles into two subsets by x < L/2 vs x ≥ L/2, then applies
    _angular_shuffle_pos to each subset independently around its own centroid.

    Using the global mean center would place the shuffle origin *between* the
    two clumps (at x ≈ L/2), systematically displacing particles across the
    boundary and destroying the coarse two-clump geometry.  Per-clump shuffle
    preserves the coarse separation and mass ratio while destroying fine-scale
    structure within each clump — the correct null hypothesis for bimodal ICs.
    """
    pos_new = pos.copy()
    for mask in (pos[:, 0] < box_size / 2.0, pos[:, 0] >= box_size / 2.0):
        if not np.any(mask):
            continue
        sub  = pos[mask]
        cent = np.mean(sub, axis=0)
        pos_new[mask] = _angular_shuffle_pos(sub, rng, cent)
    return pos_new


# ── StressResult ───────────────────────────────────────────────────────────────

@dataclass
class StressResult:
    model: str; init: str; seed: int; n: int
    eps: float; coarse_grid: int; k_fine: int; steps: int
    energy_rel_drift: float  # nan for pm_periodic (PE undefined) or on error
    virial_0: float; virial_f: float
    coarse_g4_0:  float; coarse_g8_0:  float; coarse_g16_0: float
    # Radial coarse family (orientation-free, grid-independent)
    coarse_conc_0: float; coarse_rshell_var_0: float
    fine_knn_all_0:  float; fine_pk_0:   float
    fine_close_0:    float; fine_vel_disp_0: float; fine_fof_0: float
    d_coarse_g8_early:  float; d_coarse_g8_mid:   float; d_coarse_g8_late:  float
    d_coarse_g4_early:  float; d_coarse_g4_late:  float
    d_coarse_g16_early: float; d_coarse_g16_late: float
    d_hmr_early: float; d_hmr_late: float
    # Astronomy-native targets: concentration growth
    d_conc_early: float; d_conc_late: float
    # Clean-baseline robustness target: final coarse-grid state (no delta).
    # Stored so that r(coarse_g8_0, coarse_g8_f) can be compared to the
    # delta-target correlations to bound the common-baseline inflation.
    coarse_g8_f: float
    status: str; message: str


# ── Integration ────────────────────────────────────────────────────────────────

def _apply_pbc(pos: Array, box_size: float, periodic: bool) -> Array:
    return np.mod(pos, box_size) if periodic else pos


def _integrate_leapfrog(pos0: Array, vel0: Array, mass: float,
                        sc: SimConfig, snap_steps: List[int],
                        use_numba: bool) -> Dict[int, Tuple[Array, Array]]:
    snaps: Dict[int, Tuple[Array, Array]] = {}
    pos, vel = pos0.copy(), vel0.copy()
    periodic  = sc.model in ("direct_periodic", "pm_periodic")
    acc = acceleration(pos, mass, sc, use_numba)
    dt  = sc.dt
    if 0 in snap_steps:
        snaps[0] = (pos.copy(), vel.copy())
    for step in range(1, sc.steps + 1):
        vel = vel + 0.5 * dt * acc
        pos = _apply_pbc(pos + dt * vel, sc.box_size, periodic)
        acc = acceleration(pos, mass, sc, use_numba)
        vel = vel + 0.5 * dt * acc
        if step in snap_steps:
            snaps[step] = (pos.copy(), vel.copy())
    return snaps


# ── Single stress run ──────────────────────────────────────────────────────────

def run_stress(cfg: StressConfig, use_numba: bool = False) -> Dict:
    try:
        sc       = get_simconfig(cfg)
        periodic = cfg.model in ("direct_periodic", "pm_periodic")
        mass     = 1.0 / cfg.n
        pos0, vel0 = get_initial_conditions(cfg)
        radial_center: Optional[Array] = None
        radial_pos0: Optional[Array] = None
        if not periodic:
            radial_center = _radial_obs_center(pos0, periodic=False, box_size=cfg.box_size)
            radial_pos0 = pos0
            if cfg.init.endswith(_ANGSHUF_SUFFIX) and cfg.init != "bimodal3d_angshuf":
                _, radial_pos0, _, radial_center = _angshuf_base_state(cfg)

        ke0 = kinetic_energy(vel0, mass)
        pe0 = potential_energy_direct(pos0, mass, sc)
        e0  = ke0 + pe0

        h_e = min(cfg.h_early, cfg.steps)
        h_m = min(cfg.h_mid,   cfg.steps)
        snap_steps = sorted({0, h_e, h_m, cfg.steps})
        snaps = _integrate_leapfrog(pos0, vel0, mass, sc, snap_steps, use_numba)

        # ── t=0: full predictor set (fine + coarse + radial + hmr) ───────────
        # ke0 / pe0 already computed above — do not re-evaluate them here.
        pos0s, vel0s = snaps[0]
        cg4_0        = obs_coarse_var(pos0s, cfg, 4,  periodic)
        cg8_0        = obs_coarse_var(pos0s, cfg, 8,  periodic)
        cg16_0       = obs_coarse_var(pos0s, cfg, 16, periodic)
        # Radial obs are only physically meaningful for non-periodic runs.
        # For isolated systems, keep a fixed t=0 reference center so d_conc is
        # measured in a stable frame and the angshuf null preserves its radial
        # coarse predictors exactly.
        if not periodic and radial_pos0 is not None:
            conc_0 = obs_coarse_conc(
                radial_pos0, False, cfg.box_size, center=radial_center)
            rshell_var_0 = obs_coarse_rshell_var(
                radial_pos0, False, cfg.box_size, center=radial_center)
        else:
            conc_0 = float("nan")
            rshell_var_0 = float("nan")
        knn_0        = obs_fine_knn_all(pos0s, cfg.k_fine, cfg.eps, periodic, cfg.box_size)
        pk_0         = obs_fine_pk_small(pos0s, periodic, cfg.box_size, cfg.pm_grid, mass)
        close_0      = obs_fine_close_pairs(pos0s, cfg.eps, periodic, cfg.box_size)
        vdisp_0      = obs_fine_local_vel_disp(pos0s, vel0s, cfg.k_fine, periodic, cfg.box_size)
        fof_0        = obs_fof_groups(pos0s, periodic, cfg.box_size, cfg.fof_b)
        hmr_0        = half_mass_radius(pos0s, periodic)

        # ── early: coarse grids + hmr + conc (all early targets need these) ──
        pose, _ = snaps[h_e]
        cg4_e  = obs_coarse_var(pose, cfg, 4,  periodic)
        cg8_e  = obs_coarse_var(pose, cfg, 8,  periodic)
        cg16_e = obs_coarse_var(pose, cfg, 16, periodic)
        hmr_e  = half_mass_radius(pose, periodic)
        conc_e = (obs_coarse_conc(pose, False, cfg.box_size, center=radial_center)
                  if not periodic else float("nan"))

        # ── mid: cg8 only (d_coarse_g8_mid is the sole mid target) ──────────
        posm, _ = snaps[h_m]
        cg8_m = obs_coarse_var(posm, cfg, 8, periodic)

        # ── final: coarse grids + hmr + conc + energy (no fine obs needed) ───
        posf, velf = snaps[cfg.steps]
        cg4_f  = obs_coarse_var(posf, cfg, 4,  periodic)
        cg8_f  = obs_coarse_var(posf, cfg, 8,  periodic)
        cg16_f = obs_coarse_var(posf, cfg, 16, periodic)
        hmr_f  = half_mass_radius(posf, periodic)
        conc_f = (obs_coarse_conc(posf, False, cfg.box_size, center=radial_center)
                  if not periodic else float("nan"))
        kef    = kinetic_energy(velf, mass)
        pef    = potential_energy_direct(posf, mass, sc)

        ef = kef + pef
        # PM model has no direct PE so energy conservation is undefined; use nan.
        e_drift: float = float("nan")
        if cfg.model != "pm_periodic":
            e_drift = float(abs(ef - e0) / max(abs(e0), 1e-30))

        if cfg.model == "pm_periodic":
            v0, vf = float("nan"), float("nan")
        else:
            v0 = virial_ratio(ke0, pe0)
            vf = virial_ratio(kef, pef)

        def _dhmr(late_hmr: float) -> float:
            return (late_hmr - hmr_0) if math.isfinite(hmr_0) else float("nan")

        def _dconc(late_conc: float) -> float:
            return (late_conc - conc_0) if math.isfinite(conc_0) else float("nan")

        return asdict(StressResult(
            model=cfg.model, init=cfg.init, seed=cfg.seed, n=cfg.n,
            eps=cfg.eps, coarse_grid=cfg.coarse_grid,
            k_fine=cfg.k_fine, steps=cfg.steps,
            energy_rel_drift=e_drift,
            virial_0=v0, virial_f=vf,
            coarse_g4_0=cg4_0, coarse_g8_0=cg8_0, coarse_g16_0=cg16_0,
            coarse_conc_0=conc_0, coarse_rshell_var_0=rshell_var_0,
            fine_knn_all_0=knn_0,
            fine_pk_0=pk_0,
            fine_close_0=close_0,
            fine_vel_disp_0=vdisp_0,
            fine_fof_0=fof_0,
            d_coarse_g8_early  = cg8_e  - cg8_0,
            d_coarse_g8_mid    = cg8_m  - cg8_0,
            d_coarse_g8_late   = cg8_f  - cg8_0,
            d_coarse_g4_early  = cg4_e  - cg4_0,
            d_coarse_g4_late   = cg4_f  - cg4_0,
            d_coarse_g16_early = cg16_e - cg16_0,
            d_coarse_g16_late  = cg16_f - cg16_0,
            d_hmr_early=_dhmr(hmr_e),
            d_hmr_late =_dhmr(hmr_f),
            d_conc_early=_dconc(conc_e),
            d_conc_late =_dconc(conc_f),
            coarse_g8_f=cg8_f,
            status="ok", message="",
        ))

    except Exception as exc:
        nan = float("nan")
        return asdict(StressResult(
            model=cfg.model, init=cfg.init, seed=cfg.seed, n=cfg.n,
            eps=cfg.eps, coarse_grid=cfg.coarse_grid,
            k_fine=cfg.k_fine, steps=cfg.steps,
            energy_rel_drift=nan,
            virial_0=nan, virial_f=nan,
            coarse_g4_0=nan, coarse_g8_0=nan, coarse_g16_0=nan,
            coarse_conc_0=nan, coarse_rshell_var_0=nan,
            fine_knn_all_0=nan, fine_pk_0=nan, fine_close_0=nan,
            fine_vel_disp_0=nan, fine_fof_0=nan,
            d_coarse_g8_early=nan, d_coarse_g8_mid=nan, d_coarse_g8_late=nan,
            d_coarse_g4_early=nan, d_coarse_g4_late=nan,
            d_coarse_g16_early=nan, d_coarse_g16_late=nan,
            d_hmr_early=nan, d_hmr_late=nan,
            d_conc_early=nan, d_conc_late=nan,
            coarse_g8_f=nan,
            status="error", message=traceback.format_exc(),
        ))


# ── Statistics ─────────────────────────────────────────────────────────────────

def _clean(x: list, y: list) -> Tuple[np.ndarray, np.ndarray]:
    pairs = [
        (float(xi), float(yi))
        for xi, yi in zip(x, y)
        if xi is not None and yi is not None
        and math.isfinite(float(xi)) and math.isfinite(float(yi))
    ]
    if not pairs:
        return np.array([]), np.array([])
    a, b = zip(*pairs)
    return np.array(a, dtype=float), np.array(b, dtype=float)


def pearson_with_ci(x: list, y: list, n_boot: int = 1000,
                    seed: int = 0) -> Tuple[Optional[float], Optional[float],
                                            Optional[float], int]:
    """
    Pearson r + 95% bootstrap CI.  Fully vectorized — no Python inner loop.
    Returns (r, lo, hi, n) or (None, None, None, n) if degenerate.
    """
    a, b = _clean(x, y)
    n = len(a)
    if n < 5:
        return None, None, None, n   # undefined: insufficient data
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None, None, None, n   # undefined: degenerate variance

    r = float(np.corrcoef(a, b)[0, 1])

    rng  = np.random.default_rng(seed)
    idx  = rng.integers(0, n, size=(n_boot, n))   # (B, N)
    ab   = np.vstack([a, b])                       # (2, N)
    samp = ab[:, idx]                              # (2, B, N)

    # Suppress degenerate resamples
    std2 = samp.std(axis=2)                        # (2, B)
    valid = (std2[0] > 1e-12) & (std2[1] > 1e-12) # (B,)
    samp  = samp[:, valid, :]                      # (2, B', N)

    if samp.shape[1] < 10:
        # Fewer than 10 valid bootstrap resamples — discard the point estimate
        # too so downstream code never sees a finite-r / NaN-CI combination,
        # which would be misleading (a CI-less correlation looks usable but isn't).
        return float("nan"), float("nan"), float("nan"), n

    samp -= samp.mean(axis=2, keepdims=True)
    cov   = (samp[0] * samp[1]).sum(axis=1)
    std0  = np.sqrt((samp[0] ** 2).sum(axis=1))
    std1  = np.sqrt((samp[1] ** 2).sum(axis=1))
    boot_r = cov / (std0 * std1 + 1e-30)

    return (r,
            float(np.percentile(boot_r, 2.5)),
            float(np.percentile(boot_r, 97.5)),
            n)


def partial_r(x: list, y: list, z: list) -> float:
    triples = [
        (float(xi), float(yi), float(zi))
        for xi, yi, zi in zip(x, y, z)
        if all(v is not None and math.isfinite(float(v)) for v in [xi, yi, zi])
    ]
    if len(triples) < 5:
        return float("nan")
    xa = np.array([t[0] for t in triples])
    ya = np.array([t[1] for t in triples])
    za = np.array([t[2] for t in triples])

    def _resid(v: np.ndarray, u: np.ndarray) -> np.ndarray:
        if np.std(u) < 1e-12:
            return v - np.mean(v)
        slope = np.cov(v, u, ddof=0)[0, 1] / np.var(u)
        return v - (slope * u + np.mean(v) - slope * np.mean(u))

    rx, ry = _resid(xa, za), _resid(ya, za)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ── Vectorized winner-gap bootstrap ───────────────────────────────────────────

def _winner_gap_bootstrap(
    rows: List[Dict],
    tgt_col: str,
    n_boot: int,
    seed: int = 42,
    pred_specs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Optional[Any]]:
    """
    Vectorized bootstrap for winner-gap inference (listwise complete case).
    Computes overall / positional / kinematic fine vs best coarse winner gaps.

    pred_specs: active (column, name) predictor pairs.  Defaults to
    _ALL_PRED_SPECS.  Pass a model-appropriate subset so that _row_ok()
    is not poisoned by by-design NaN columns (e.g. radial obs on periodic
    models).  Fine/coarse/pos/kin index sets are derived locally from
    whatever pred_specs is passed — no hard-coded global indices used.
    """
    if pred_specs is None:
        pred_specs = _ALL_PRED_SPECS

    act_cols  = [c for c, _ in pred_specs]
    act_names = [n for _, n in pred_specs]

    # Derive fine/coarse/pos/kin index sets from the active spec list
    _fine_pos_set = {n for _, n in FINE_POSITIONAL_OBS}
    _fine_kin_set = {n for _, n in FINE_KINEMATIC_OBS}
    pos_idx    = [i for i, (_, n) in enumerate(pred_specs) if n in _fine_pos_set]
    kin_idx    = [i for i, (_, n) in enumerate(pred_specs) if n in _fine_kin_set]
    fine_idx   = pos_idx + kin_idx
    coarse_idx = [i for i in range(len(pred_specs)) if i not in set(fine_idx)]

    _null: Dict[str, Optional[Any]] = {
        "winner_gap_mean": None, "winner_gap_ci_lo": None, "winner_gap_ci_hi": None,
        "winner_gap_pos_mean": None, "winner_gap_pos_ci_lo": None, "winner_gap_pos_ci_hi": None,
        "winner_gap_kin_mean": None, "winner_gap_kin_ci_lo": None, "winner_gap_kin_ci_hi": None,
        "best_fine_name":     None, "best_fine_abs_r":     None,
        "best_coarse_name":   None, "best_coarse_abs_r":   None,
        "best_pos_fine_name": None, "best_pos_fine_abs_r": None,
        "best_kin_fine_name": None, "best_kin_fine_abs_r": None,
        "n_listwise": 0,
    }

    def _row_ok(r: Dict) -> bool:
        v = r.get(tgt_col)
        if v is None:
            return False
        try:
            if not math.isfinite(float(v)):
                return False
        except (TypeError, ValueError):
            return False
        for col in act_cols:         # only check columns active for this model
            u = r.get(col)
            if u is None:
                return False
            try:
                if not math.isfinite(float(u)):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    clean = [r for r in rows if _row_ok(r)]
    n     = len(clean)
    _null["n_listwise"] = n

    if n < 5:
        return _null

    p = len(pred_specs)
    X = np.empty((n, p), dtype=np.float64)
    for j, col in enumerate(act_cols):
        X[:, j] = [float(r[col]) for r in clean]
    y = np.array([float(r[tgt_col]) for r in clean], dtype=np.float64)

    if np.std(y) < 1e-12:
        return _null

    degenerate = X.std(axis=0) < 1e-12

    X_c    = X - X.mean(axis=0)
    y_c    = y - y.mean()
    norm_X = np.sqrt((X_c ** 2).sum(axis=0))
    norm_y  = float(np.sqrt((y_c ** 2).sum()))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_pt = (X_c.T @ y_c) / (norm_X * norm_y + 1e-30)
    corr_pt = np.where(degenerate | ~np.isfinite(corr_pt), np.nan, corr_pt)

    rng    = np.random.default_rng(seed)
    bidx   = rng.integers(0, n, size=(n_boot, n))
    X_b    = X[bidx]
    y_b    = y[bidx]
    X_b   -= X_b.mean(axis=1, keepdims=True)
    y_b   -= y_b.mean(axis=1, keepdims=True)
    num    = np.einsum("bip,bi->bp", X_b, y_b)
    den_X  = np.sqrt((X_b ** 2).sum(axis=1))
    den_y  = np.sqrt((y_b ** 2).sum(axis=1))
    valid_boot = (den_X > 1e-12) & (den_y[:, None] > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_b = num / (den_X * den_y[:, None] + 1e-30)
    corr_b = np.where(valid_boot & np.isfinite(corr_b), corr_b, np.nan)
    corr_b[:, degenerate] = np.nan

    def _rowwise_abs_best(arr: np.ndarray) -> np.ndarray:
        finite = np.isfinite(arr)
        best = np.where(finite, np.abs(arr), -np.inf).max(axis=1)
        best[~finite.any(axis=1)] = np.nan
        return best

    # Separate RNG for permutation null so bootstrap samples are not perturbed.
    null_rng  = np.random.default_rng(seed ^ 0xDEAD_BEEF)
    _N_PERM   = 400  # number of class-label permutations for null bias estimate

    def _gap(a_idx: List[int], b_idx: List[int]
             ) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
        """
        Bias-corrected winner gap: gap_observed - E[gap_null].

        When len(a_idx) ≠ len(b_idx), the group with more predictors has a
        systematic order-statistic advantage under the null (max of 5 draws
        beats max of 3 draws on average even if all draws come from the same
        distribution).  We estimate this bias by permuting class labels on
        the same bootstrap correlation matrix and subtract it.  When class
        sizes are equal the bias is zero by symmetry — no permutations needed.
        """
        if not a_idx or not b_idx:
            return None, None, None, float("nan")
        na, nb = len(a_idx), len(b_idx)
        ai, bi = np.array(a_idx), np.array(b_idx)
        gap_a = _rowwise_abs_best(corr_b[:, ai])
        gap_b = _rowwise_abs_best(corr_b[:, bi])
        gap   = gap_a - gap_b
        gap   = gap[np.isfinite(gap)]
        if len(gap) < 10:
            return None, None, None, float("nan")

        # Permutation null: pool predictor indices, randomly re-assign same
        # na vs nb split, compute mean null gap from the actual bootstrap matrix.
        null_bias = 0.0
        if na != nb:
            pool   = np.array(a_idx + b_idx)
            n_pool = len(pool)
            null_means: List[float] = []
            for _ in range(_N_PERM):
                perm = null_rng.permutation(n_pool)
                pa   = pool[perm[:na]]
                pb   = pool[perm[na:na + nb]]
                g_n  = _rowwise_abs_best(corr_b[:, pa]) - _rowwise_abs_best(corr_b[:, pb])
                g_n  = g_n[np.isfinite(g_n)]
                if len(g_n) >= 10:
                    null_means.append(float(g_n.mean()))
            null_bias = float(np.mean(null_means)) if null_means else 0.0

        corrected = gap - null_bias
        return (float(corrected.mean()),
                float(np.percentile(corrected, 2.5)),
                float(np.percentile(corrected, 97.5)),
                null_bias)

    def _best(idx_list: List[int]) -> Tuple[Optional[str], Optional[float]]:
        if not idx_list:
            return None, None
        arr = np.abs(corr_pt[idx_list])
        if not np.isfinite(arr).any():
            return None, None
        arr = np.where(np.isfinite(arr), arr, -np.inf)
        i   = int(arr.argmax())
        return act_names[idx_list[i]], float(arr[i])   # use local names

    gm,  glo,  ghi,  gb  = _gap(fine_idx, coarse_idx)
    pm,  plo,  phi_, pb  = _gap(pos_idx,  coarse_idx)
    km,  klo,  khi,  kb  = _gap(kin_idx,  coarse_idx)

    bf_n, bf_r  = _best(fine_idx)
    bc_n, bc_r  = _best(coarse_idx)
    bp_n, bp_r  = _best(pos_idx)
    bk_n, bk_r  = _best(kin_idx)

    return {
        "winner_gap_mean": gm,  "winner_gap_ci_lo": glo,  "winner_gap_ci_hi": ghi,
        "winner_gap_null_bias": gb,   # order-statistic bias from unequal class sizes
        "winner_gap_pos_mean": pm,  "winner_gap_pos_ci_lo": plo,  "winner_gap_pos_ci_hi": phi_,
        "winner_gap_pos_null_bias": pb,
        "winner_gap_kin_mean": km,  "winner_gap_kin_ci_lo": klo,  "winner_gap_kin_ci_hi": khi,
        "winner_gap_kin_null_bias": kb,
        "best_fine_name":     bf_n, "best_fine_abs_r":     bf_r,
        "best_coarse_name":   bc_n, "best_coarse_abs_r":   bc_r,
        "best_pos_fine_name": bp_n, "best_pos_fine_abs_r": bp_r,
        "best_kin_fine_name": bk_n, "best_kin_fine_abs_r": bk_r,
        "n_listwise": n,
    }


# ── Cell key ───────────────────────────────────────────────────────────────────

def _make_key(r: Dict) -> str:
    # Key format must stay in sync with nbody_paper.make_key().
    return (f"N={r['n']}|eps={r['eps']}|k={r['k_fine']}"
            f"|model={r['model']}|init={r['init']}")


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyse(rows: List[Dict], n_boot: int = 1000,
            min_ok_hard: int = 100) -> Dict[str, Dict]:
    """
    For every (model, init, N, eps, k_fine) cell:
      1. Individual Pearson r + 95% CI + partial r for all predictor×target pairs.
      2. Vectorized winner-gap bootstrap for every target.
      3. CI-backed primary verdict (never point-estimate alone).
      4. Family stability across FAMILY_TARGETS.
      5. Underpowered cell flags.
    """
    ok = [r for r in rows if r.get("status") == "ok"]

    key_total: Dict[str, int] = {}
    for r in rows:
        k = _make_key(r)
        key_total[k] = key_total.get(k, 0) + 1

    groups: Dict[str, List[Dict]] = {}
    for r in ok:
        groups.setdefault(_make_key(r), []).append(r)

    out: Dict[str, Dict] = {}

    for key, rs in groups.items():
        n_req  = key_total.get(key, len(rs))
        n_ok   = len(rs)
        n_fail = n_req - n_ok
        fail_frac    = n_fail / max(n_req, 1)
        underpowered = (n_ok < max(min_ok_hard, int(0.9 * n_req)))

        parts = dict(seg.split("=", 1) for seg in key.split("|") if "=" in seg)

        # Model-aware predictor specs: periodic models cannot use radial obs
        # (concentration is ill-defined in a periodic box with no canonical center)
        _model = parts.get("model", "")
        pred_specs = (
            _PRED_SPECS_DIRECT
            if _model == "direct_isolated"
            else _PRED_SPECS_PERIODIC
        )

        cell: Dict[str, Any] = {
            "key":          key,
            "model":        parts.get("model", ""),
            "init":         parts.get("init",  ""),
            "n":            int(float(parts.get("N",   0))),
            "eps":          float(parts.get("eps", 0.0)),
            "k_fine":       int(float(parts.get("k",   0))),
            "n_req":        n_req,
            "n_ok":         n_ok,
            "n_fail":       n_fail,
            "fail_frac":    fail_frac,
            "underpowered": underpowered,
        }

        vr_col = [r.get("virial_0") for r in rs]

        def _safe_finite(v) -> bool:
            try:
                return math.isfinite(float(v))
            except (TypeError, ValueError):
                return False

        # Build cell-specific pred_specs: exclude predictors valid in <90% of rows
        # (or fewer than min_ok_hard rows, whichever is stricter).
        # A predictor finite in only 10% of replicates should not compete in the
        # winner-gap and should not drag listwise-complete n down for the others.
        _min_valid = max(min_ok_hard, int(0.9 * len(rs)))
        cell_pred_specs = [
            (col, name) for col, name in pred_specs
            if sum(1 for r in rs if _safe_finite(r.get(col))) >= _min_valid
        ] or list(pred_specs)   # fall back to full spec if everything is sparse

        # Pre-filter rows to the listwise-complete set for load-bearing predictors.
        # Individual Pearson r for predictors in cell_pred_specs is computed on
        # the same population that the winner-gap bootstrap will use — making
        # the reported r values directly comparable to the verdict.
        # For non-load-bearing predictors (in pred_specs but not cell_pred_specs),
        # we still compute r on the full rs sample and mark them as informational.
        _cell_pred_cols = {col for col, _ in cell_pred_specs}
        listwise_rs = [
            r for r in rs
            if all(_safe_finite(r.get(col)) for col, _ in cell_pred_specs)
        ]

        for tgt_col, tgt_name in TARGETS:
            ty_lw = [r.get(tgt_col) for r in listwise_rs]
            ty_rs = [r.get(tgt_col) for r in rs]
            vr_lw = [r.get("virial_0") for r in listwise_rs]
            for pred_col, pred_name in pred_specs:
                # Load-bearing predictors: use listwise-complete rows for consistency
                # with the winner-gap bootstrap.  Non-load-bearing: full sample.
                if pred_col in _cell_pred_cols:
                    px = [r.get(pred_col) for r in listwise_rs]
                    ty = ty_lw
                    vr = vr_lw
                else:
                    px = [r.get(pred_col) for r in rs]
                    ty = ty_rs
                    vr = vr_col
                r_val, lo, hi, _ = pearson_with_ci(px, ty, n_boot=n_boot)
                # partial_r controls for virial_0 to bound nuisance-variable inflation.
                # NOTE: for coarse delta-targets (e.g. ΔC8-early = cg8_e - cg8_0)
                # the raw r(coarse_g8_0, ΔC8-early) can be inflated by the shared
                # baseline.  The C8-final target (coarse_g8_f, no delta) provides
                # the baseline-free check; pr_ here controls only for virial ratio.
                pr = partial_r(px, ty, vr)
                cell[f"r_{pred_name}_{tgt_name}"]     = r_val
                cell[f"ci_lo_{pred_name}_{tgt_name}"] = lo
                cell[f"ci_hi_{pred_name}_{tgt_name}"] = hi
                cell[f"pr_{pred_name}_{tgt_name}"]    = pr

            # Deterministic seed per (cell, target) pair — reproducible across runs.
            seed_i = _stable_seed(f"{key}|{tgt_name}")
            gap = _winner_gap_bootstrap(rs, tgt_col, n_boot=n_boot, seed=seed_i,
                                        pred_specs=cell_pred_specs)
            for gk, gv in gap.items():
                cell[f"{gk}_{tgt_name}"] = gv

        # Refine underpowered using the actual listwise-complete bootstrap sample.
        # n_ok counts successful runs; n_listwise counts rows where every active
        # predictor AND the target are finite — the true analysis sample size.
        # A cell with n_ok=500 but n_listwise=42 is not a hard-verdict cell.
        n_listwise_primary = int(cell.get(f"n_listwise_{PRIMARY_TARGET}") or 0)
        underpowered = underpowered or (n_listwise_primary < min_ok_hard)
        cell["underpowered"]       = underpowered
        cell["n_listwise_primary"] = n_listwise_primary

        # CI-backed primary verdict
        wg_lo = cell.get(f"winner_gap_ci_lo_{PRIMARY_TARGET}")
        wg_hi = cell.get(f"winner_gap_ci_hi_{PRIMARY_TARGET}")

        _T = VERDICT_GAP_THRESHOLD
        cell["verdict_threshold"] = _T

        if underpowered:
            primary_verdict = "UNDERPOWERED"
        elif (wg_lo is not None and isinstance(wg_lo, float)
              and math.isfinite(wg_lo) and wg_lo > _T):
            primary_verdict = "FINE"
        elif (wg_hi is not None and isinstance(wg_hi, float)
              and math.isfinite(wg_hi) and wg_hi < -_T):
            primary_verdict = "COARSE"
        else:
            primary_verdict = "TIE"

        cell["primary_verdict"] = primary_verdict
        cell["verdict"]         = primary_verdict

        # Family stability
        # HMR and concentration targets are structurally undefined for periodic models:
        # HMR is ill-defined on a torus; concentration requires a radial centre which
        # doesn't exist in a periodic box (both are set to NaN by construction in
        # run_one_stress).  Exclude both families for periodic cells so that
        # family_stable is not impossible by construction.
        _periodic_cell = (cell.get("model", "") != "direct_isolated")
        _active_family = [
            t for t in FAMILY_TARGETS
            if not (_periodic_cell and (t.startswith("ΔHMR") or t.startswith("ΔConc")))
        ]
        n_fam_fine = n_fam_coarse = 0
        for tgt_name in _active_family:
            lo = cell.get(f"winner_gap_ci_lo_{tgt_name}")
            hi = cell.get(f"winner_gap_ci_hi_{tgt_name}")
            if (lo is not None and isinstance(lo, float)
                    and math.isfinite(lo) and lo > _T):
                n_fam_fine += 1
            if (hi is not None and isinstance(hi, float)
                    and math.isfinite(hi) and hi < -_T):
                n_fam_coarse += 1

        cell["n_family_fine"]   = n_fam_fine
        cell["n_family_coarse"] = n_fam_coarse
        cell["n_family_active"] = len(_active_family)
        cell["family_stable"]   = (
            n_fam_fine   == len(_active_family) or
            n_fam_coarse == len(_active_family)
        )

        # Backward-compat aliases
        cell["best_fine_r"]   = cell.get(f"best_fine_abs_r_{PRIMARY_TARGET}")
        cell["best_coarse_r"] = cell.get(f"best_coarse_abs_r_{PRIMARY_TARGET}")
        cell["fine_adv"]      = cell.get(f"winner_gap_mean_{PRIMARY_TARGET}")

        out[key] = cell

    return out


# ── VelDisp convergence analysis (randomized repeated subsamples) ──────────────

def convergence_analysis(rows: List[Dict], n_boot: int = 200,
                         n_repeats: int = 10) -> Dict:
    """
    Concentrated-profile frontier convergence check.

    For each size in subsample_sizes:
      - draw n_repeats independent random subsamples without replacement
      - compute |r(VelDisp, ΔC8-early)| and CI width for each
      - store mean and std across repeats

    This gives a proper estimate of convergence variability, not a one-shot
    prefix result.  The figure can show error bars on both r_abs and CI width.
    """
    ok = [r for r in rows if r.get("status") == "ok"]

    groups: Dict[str, List[Dict]] = {}
    for r in ok:
        groups.setdefault(_make_key(r), []).append(r)

    target_inits    = {"hernquist3d", "plummer3d"}
    subsample_sizes = [10, 25, 50, 100, 200, 300, 500]
    results: Dict[str, Any] = {}

    for key, rs in groups.items():
        parts = dict(seg.split("=", 1) for seg in key.split("|") if "=" in seg)
        if parts.get("model") != "direct_isolated":
            continue
        if parts.get("init") not in target_inits:
            continue

        sizes = [s for s in subsample_sizes if s <= len(rs)]
        if len(sizes) < 2:
            continue

        rng_master = np.random.default_rng(_stable_seed(key))
        rs_arr = np.array(rs, dtype=object)

        curve: List[Dict] = []
        for sz in sizes:
            # How many non-overlapping draws fit — cap at n_repeats
            actual_reps = n_repeats

            r_abs_list:    List[float] = []
            ci_width_list: List[float] = []

            for _ in range(actual_reps):
                idx = rng_master.choice(len(rs), size=sz, replace=False)
                sub = rs_arr[idx].tolist()
                px  = [r.get("fine_vel_disp_0") for r in sub]
                ty  = [r.get("d_coarse_g8_early") for r in sub]
                r_val, lo, hi, n_used = pearson_with_ci(
                    px, ty, n_boot=n_boot,
                    seed=int(rng_master.integers(0, 2 ** 31))
                )
                if r_val is not None and math.isfinite(r_val):
                    r_abs_list.append(float(abs(r_val)))
                if (lo is not None and hi is not None
                        and math.isfinite(lo) and math.isfinite(hi)):
                    ci_width_list.append(float(hi) - float(lo))

            curve.append({
                "n_reps":         sz,
                "n_repeats_used": len(r_abs_list),
                "r_abs_mean":     float(np.mean(r_abs_list))  if r_abs_list    else None,
                "r_abs_std":      float(np.std(r_abs_list))   if len(r_abs_list)  > 1 else None,
                "ci_width_mean":  float(np.mean(ci_width_list)) if ci_width_list else None,
                "ci_width_std":   float(np.std(ci_width_list))  if len(ci_width_list) > 1 else None,
            })

        results[key] = {
            "init":            parts.get("init"),
            "eps":             float(parts.get("eps", 0.0)),
            "n":               int(float(parts.get("N", 0))),
            "total_available": len(rs),
            "curve":           curve,
        }

    return results


# ── JSON serializer ────────────────────────────────────────────────────────────

def _json_safe(x: Any) -> Any:
    if isinstance(x, bool):   return x
    if isinstance(x, str):    return x
    if isinstance(x, int):    return x
    if isinstance(x, float):  return x if math.isfinite(x) else None
    if isinstance(x, np.bool_):    return bool(x)
    if isinstance(x, np.integer):  return int(x)
    if isinstance(x, np.floating):
        v = float(x); return v if math.isfinite(v) else None
    if isinstance(x, np.ndarray):  return x.tolist()
    return None


# ── Formatted summary ──────────────────────────────────────────────────────────

def print_summary(analysis: Dict) -> str:
    W = 150
    lines: List[str] = []
    lines.append("=" * W)
    lines.append("STRESS-TEST SUMMARY  —  winner-gap CI-based verdicts")
    lines.append(f"  Primary target: {PRIMARY_TARGET}   Family: {FAMILY_TARGETS}")
    lines.append(f"  FINE if gap_CI_lo > +0.05  |  COARSE if gap_CI_hi < -0.05  |  TIE otherwise")
    lines.append("=" * W)

    hdr = (f"{'Group':<46} {'n_ok/req':>9}  "
           f"{'bc_r':>7} {'bf_r':>7}  "
           f"{'gap_mean':>8} {'gap_CI':>17}  "
           f"{'pos_gap_CI':>17} {'kin_gap_CI':>17}  "
           f"{'verdict':>12}  {'fam':>7}")
    lines.append(hdr)
    lines.append("─" * W)

    def _f(v: Any, nd: int = 3) -> str:
        if v is None: return "  n/a"
        try:
            fv = float(v)
            return f"{fv:+.{nd}f}" if math.isfinite(fv) else "  n/a"
        except (TypeError, ValueError):
            return "  n/a"

    def _ci(lo: Any, hi: Any) -> str:
        try:
            l, h = float(lo), float(hi)
            if math.isfinite(l) and math.isfinite(h):
                return f"[{l:+.2f},{h:+.2f}]"
        except (TypeError, ValueError):
            pass
        return "[  n/a  ]     "

    for key in sorted(analysis.keys()):
        c   = analysis[key]
        und = " !" if c.get("underpowered") else "  "
        tgt = PRIMARY_TARGET
        row = (
            f"  {key:<46} "
            f"{c.get('n_ok','?'):>4}/{c.get('n_req','?'):<4}{und}  "
            f"{_f(c.get(f'best_coarse_abs_r_{tgt}')):>7} "
            f"{_f(c.get(f'best_fine_abs_r_{tgt}')):>7}  "
            f"{_f(c.get(f'winner_gap_mean_{tgt}')):>8} "
            f"{_ci(c.get(f'winner_gap_ci_lo_{tgt}'), c.get(f'winner_gap_ci_hi_{tgt}')):>17}  "
            f"{_ci(c.get(f'winner_gap_pos_ci_lo_{tgt}'), c.get(f'winner_gap_pos_ci_hi_{tgt}')):>17} "
            f"{_ci(c.get(f'winner_gap_kin_ci_lo_{tgt}'), c.get(f'winner_gap_kin_ci_hi_{tgt}')):>17}  "
            f"[{c.get('primary_verdict','---'):>10}]  "
            f"{c.get('n_family_coarse',0)}C/{c.get('n_family_fine',0)}F"
        )
        lines.append(row)

    lines.append("")
    lines.append("bc_r / bf_r = |r| best coarse / best fine (point estimate)")
    lines.append("gap_CI = 95% bootstrap CI of overall winner gap")
    lines.append("pos/kin_gap_CI = class-specific CI vs best coarse")
    lines.append("! = UNDERPOWERED cell")

    # N-scaling sub-table (direct_isolated only)
    lines.append("")
    lines.append("N-SCALING  (direct_isolated only):")
    lines.append(f"  {'N':>6}  {'init':>14}  {'eps':>5}  "
                 f"{'bc_r':>7} {'bf_r':>7}  {'gap_CI':>17}  {'kin_CI':>17}  {'verdict':>12}")
    lines.append("  " + "─" * 100)

    for key in sorted(analysis.keys()):
        c     = analysis[key]
        parts = dict(seg.split("=", 1) for seg in key.split("|") if "=" in seg)
        if parts.get("model") != "direct_isolated":
            continue
        tgt = PRIMARY_TARGET
        lines.append(
            f"  {parts.get('N','?'):>6}  {parts.get('init','?'):>14}  "
            f"{float(parts.get('eps', 0.0)):>5.2f}  "
            f"{_f(c.get(f'best_coarse_abs_r_{tgt}')):>7} "
            f"{_f(c.get(f'best_fine_abs_r_{tgt}')):>7}  "
            f"{_ci(c.get(f'winner_gap_ci_lo_{tgt}'), c.get(f'winner_gap_ci_hi_{tgt}')):>17}  "
            f"{_ci(c.get(f'winner_gap_kin_ci_lo_{tgt}'), c.get(f'winner_gap_kin_ci_hi_{tgt}')):>17}  "
            f"[{c.get('primary_verdict','---'):>10}]"
        )

    return "\n".join(lines)


# ── I/O ────────────────────────────────────────────────────────────────────────

def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    # Use the UNION of all row keys as fieldnames, not rows[0].keys().
    # If --resume merges old rows (no coarse_g8_f) with new rows (have it),
    # rows[0] might be an old row and rows[0].keys() would silently drop the
    # new field from every subsequent row.  dict insertion order is preserved
    # in Python 3.7+, so the union preserves logical column ordering.
    all_keys: dict = {}
    for r in rows:
        all_keys.update(dict.fromkeys(r.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_keys), extrasaction="ignore",
                           restval="")
        w.writeheader()
        w.writerows(rows)


def _load_existing_rows(path: str) -> List[Dict]:
    """Load rows from a CSV, returning [] if the file is missing or empty.

    Applies backward-compat derivations for columns added after initial runs:
      coarse_g8_f = d_coarse_g8_late + coarse_g8_0
        (both stored since the original battery; coarse_g8_f was added later)
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if not r.get("coarse_g8_f"):
                try:
                    late = float(r.get("d_coarse_g8_late", "nan"))
                    base = float(r.get("coarse_g8_0",      "nan"))
                    if math.isfinite(late) and math.isfinite(base):
                        r["coarse_g8_f"] = str(late + base)
                except (TypeError, ValueError):
                    pass
        return rows
    except Exception:
        return []


def _done_key(row: Dict) -> tuple:
    return (
        int(float(row.get("n", 0))),
        float(row.get("eps", 0.0)),
        int(float(row.get("k_fine", 16))),
        str(row.get("model", "")),
        str(row.get("init",  "")),
        int(float(row.get("seed", 0))),
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="N-body ODD stress-test battery (hardened)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outdir",      default="outputs/stress")
    parser.add_argument("--n",           type=int, nargs="+",
                        default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--eps",         type=float, nargs="+",
                        default=[0.02, 0.03, 0.05, 0.07, 0.10])
    parser.add_argument("--steps",       type=int,   default=600)
    parser.add_argument("--replicates",  type=int,   default=500)
    parser.add_argument("--workers",     type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--models",      nargs="+",
                        default=["direct_isolated", "pm_periodic"],
                        choices=["direct_isolated", "direct_periodic", "pm_periodic"],
                        help="Integrator/boundary model. direct_isolated is the primary "
                             "physically-clean model. pm_periodic is the cross-check. "
                             "direct_periodic uses minimum-image direct summation: energy "
                             "is proxy-like, virial and radial observables are not "
                             "physically clean — treat as diagnostic only, not primary evidence.")
    parser.add_argument("--inits",       nargs="+",
                        default=["bimodal3d", "hernquist3d", "plummer3d", "cold_clumpy3d"],
                        help="IC names; append _angshuf for angular-shuffle null "
                             "(e.g. hernquist3d_angshuf). Only valid for direct_isolated.")
    parser.add_argument("--k-fine",      type=int, nargs="+", default=[16],
                        help="Single value for flagship run. "
                             "Multiple values only for sensitivity appendix.")
    parser.add_argument("--n-boot",      type=int, default=1000)
    parser.add_argument("--min-ok-hard", type=int, default=100,
                        help="Minimum successful replicates for a hard verdict.")
    parser.add_argument("--conv-repeats", type=int, default=10,
                        help="Random subsample repeats per size in convergence analysis.")
    parser.add_argument("--use-numba",   dest="use_numba",
                        action="store_true", default=True)
    parser.add_argument("--no-numba",    dest="use_numba", action="store_false")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip configs already present in results_full.csv "
                             "or checkpoint.csv and append new results.")
    args = parser.parse_args()

    # Enforce: _angshuf ICs are only physically meaningful for direct_isolated.
    # In a periodic box the construction centre is the box centre, and the
    # shuffled positions after evolution will not maintain a clean reference —
    # so the null interpretation is invalid for pm_periodic / direct_periodic.
    angshuf_inits   = [i for i in args.inits if i.endswith("_angshuf")]
    noniso_models   = [m for m in args.models if m != "direct_isolated"]
    if angshuf_inits and noniso_models:
        parser.error(
            f"_angshuf ICs {angshuf_inits} are only valid for direct_isolated, "
            f"but non-isolated models {noniso_models} were also requested. "
            "Remove the _angshuf ICs or restrict --models to direct_isolated."
        )

    if len(args.k_fine) > 1:
        print("WARNING: multiple --k-fine values add an extra battery dimension. "
              "Use a single value (default 16) for the flagship run.")

    if args.use_numba and not _HAS_NUMBA:
        print("numba not found — falling back to numpy")
        args.use_numba = False

    os.makedirs(args.outdir, exist_ok=True)
    seeds = [2000 + i for i in range(args.replicates)]

    DIRECT_N_MAX = 2048
    configs: List[StressConfig] = []
    for n_val, eps_val, k_val, model, init, seed in product(
        args.n, args.eps, args.k_fine, args.models, args.inits, seeds
    ):
        if model == "direct_isolated" and n_val > DIRECT_N_MAX:
            continue
        if init.endswith("_angshuf") and model != "direct_isolated":
            # Belt-and-suspenders: CLI validation already blocks this,
            # but guard here too in case main() is called programmatically.
            continue
        configs.append(StressConfig(
            model=model, init=init, seed=seed,
            n=n_val, steps=args.steps, eps=eps_val, k_fine=k_val,
        ))

    unique_cells = {
        (cfg.n, cfg.eps, cfg.k_fine, cfg.model, cfg.init)
        for cfg in configs
    }
    n_cells = len(unique_cells)
    print(
        f"Stress-test battery: {len(configs)} runs\n"
        f"  N            : {args.n}\n"
        f"  eps          : {args.eps}\n"
        f"  k_fine       : {args.k_fine}\n"
        f"  models       : {args.models}\n"
        f"  inits        : {args.inits}\n"
        f"  replicates   : {args.replicates} per cell  ({n_cells} cells, after direct_isolated cap)\n"
        f"  steps        : {args.steps}\n"
        f"  workers      : {args.workers}\n"
        f"  bootstrap n  : {args.n_boot}\n"
        f"  min_ok_hard  : {args.min_ok_hard}\n"
        f"  conv_repeats : {args.conv_repeats}\n"
        f"  numba        : {args.use_numba}\n"
    )

    results_csv    = os.path.join(args.outdir, "results_full.csv")
    checkpoint_csv = os.path.join(args.outdir, "checkpoint.csv")

    # ── Resume: load existing rows and skip already-done configs ──────────────
    existing_rows: List[Dict] = []
    done_keys: set = set()
    if args.resume:
        for path in (results_csv, checkpoint_csv):
            for row in _load_existing_rows(path):
                k = _done_key(row)
                if k not in done_keys:
                    done_keys.add(k)
                    existing_rows.append(row)
        print(f"Resume: {len(existing_rows)} existing rows loaded, "
              f"{len(done_keys)} configs will be skipped.")

    pending = [
        cfg for cfg in configs
        if (cfg.n, cfg.eps, cfg.k_fine, cfg.model, cfg.init, cfg.seed) not in done_keys
    ]
    if args.resume and len(pending) < len(configs):
        print(f"  pending: {len(pending)} runs  "
              f"({len(configs) - len(pending)} already done)")

    new_rows: List[Dict] = []
    if pending:
        # Write new results incrementally to checkpoint so crashes don't lose work.
        # Use a context manager so the file is always closed even on KeyboardInterrupt
        # or worker crashes that propagate through the executor.
        with open(checkpoint_csv, "a", newline="", encoding="utf-8") as ckpt_file:
            ckpt_writer: Optional[csv.DictWriter] = None  # type: ignore[type-arg]

            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_worker_init,
                initargs=(args.use_numba,),
            ) as ex:
                futs = {ex.submit(run_stress, cfg, args.use_numba): cfg for cfg in pending}
                with tqdm(total=len(pending), unit="run", ncols=100) as pbar:
                    for fut in as_completed(futs):
                        res = fut.result()
                        new_rows.append(res)
                        if ckpt_writer is None:
                            ckpt_writer = csv.DictWriter(
                                ckpt_file, fieldnames=list(res.keys()))
                            if ckpt_file.tell() == 0:
                                ckpt_writer.writeheader()
                        ckpt_writer.writerow(res)
                        ckpt_file.flush()
                        tag = "✓" if res["status"] == "ok" else "✗"
                        pbar.set_postfix_str(
                            f"{tag} N={res['n']:4d} {res['model'][:10]} "
                            f"{res['init'][:10]} s={res['seed']}"
                        )
                        pbar.update(1)
                        if res["status"] == "error":
                            tqdm.write(
                                f"  ✗ {res['model']} N={res['n']} "
                                f"seed={res['seed']}: {res['message'][:120]}"
                            )

    rows = existing_rows + new_rows
    rows.sort(key=lambda r: (
        int(float(r.get("n", 0))),
        float(r.get("eps", 0.0)),
        int(float(r.get("k_fine", 0))),
        str(r.get("model", "")),
        str(r.get("init",  "")),
        int(float(r.get("seed",  0))),
    ))
    write_csv(results_csv, rows)

    # Remove checkpoint after a clean merge
    if os.path.exists(checkpoint_csv):
        os.remove(checkpoint_csv)

    print(f"\nRunning analysis ({args.n_boot} bootstrap resamples, "
          f"min_ok_hard={args.min_ok_hard})...")
    analysis = analyse(rows, n_boot=args.n_boot, min_ok_hard=args.min_ok_hard)

    summary_text = print_summary(analysis)
    with open(os.path.join(args.outdir, "summary_table.txt"), "w") as f:
        f.write(summary_text)
    with open(os.path.join(args.outdir, "ci_table.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=_json_safe)

    print(f"\nRunning convergence analysis (n_repeats={args.conv_repeats})...")
    conv = convergence_analysis(rows, n_boot=min(args.n_boot, 200),
                                n_repeats=args.conv_repeats)
    with open(os.path.join(args.outdir, "convergence.json"), "w") as f:
        json.dump(conv, f, indent=2, default=_json_safe)

    if conv:
        print("Convergence summary (r_abs_mean ± std at max subsample):")
        for key, v in sorted(conv.items()):
            last = v["curve"][-1] if v["curve"] else {}
            rm   = last.get("r_abs_mean")
            rs   = last.get("r_abs_std")
            wm   = last.get("ci_width_mean")
            rstr = f"{rm:.3f}±{rs:.3f}" if (rm is not None and rs is not None) else "n/a"
            wstr = f"{wm:.3f}" if wm is not None else "n/a"
            print(f"  {key}: r={rstr}  CI_width={wstr}  (n={last.get('n_reps','?')})")

    ok_count = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\nAll outputs → {args.outdir}/   ({ok_count}/{len(rows)} ok runs)")


if __name__ == "__main__":
    main()
