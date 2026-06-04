#!/usr/bin/env python3
"""
phase_space_coarse_features.py — coarse phase-space features ψ_ℓ + relaxation observables
=========================================================================================

New branch (see nbody_phase_space_coarse_graining_plan.md): coarse-graining moved to phase
space.  Given a t-snapshot (pos, vel) of a self-gravitating system, this module provides:

  • relaxation observables (scalars used to build TARGETS and the bulk control):
      Q = 2K/|U| (virial ratio), E = K+U, coarse phase-space entropy S(r,v_r) on a FIXED
      grid (the mixing measure), global radial dispersion σ_r, tangential σ_t, anisotropy β.

  • ψ_ℓ — coarse phase-space FEATURES at radial resolution ℓ: radial profiles of the
      kinematic moments (σ_r, σ_t, mean v_r, β) Gaussian-smoothed along r at scale ℓ and
      sampled at fixed radii, plus the smoothed (r,v_r) occupancy entropy/spread at scale ℓ.
      ℓ enters only through the smoothing width, so one snapshot yields every scale for free.

  • bulk-control features: Q₀, E, a coarse radial mass profile, and C₈(t₀).

Velocities are decomposed about the COM into radial v_r = v·r̂ and tangential v_t.
Reuses the repo's exact potential / coarse-grid functions so K, U, Q, C₈ match the battery.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import entropy as _shannon_entropy

from nbody_3d import kinetic_energy, potential_energy_direct
from nbody_stress import StressConfig, get_simconfig, obs_coarse_var

Array = np.ndarray

# ── Phase-space entropy grid (COMOVING: r/r50, v_r/σ) ────────────────────────────
# Scaling by the half-mass radius r50(t) and radial dispersion σ(t) at each time makes the
# entropy a measure of the phase-space distribution *shape* (folding / mixing), insensitive
# to overall expansion/contraction and to escapers leaving a fixed box.  A fixed-box entropy
# would instead track mass loss — a bulk effect the controls already capture.
PS_RMAX = 5.0     # (r/r50, v_r/σ) entropy grid: scaled radius
PS_VMAX = 4.0     # scaled |radial velocity|
PS_NR = 24
PS_NV = 24

PROF_RMAX = 0.70  # feature radial profiles
PROF_NFINE = 40
SAMPLE_RADII = (0.10, 0.20, 0.35, 0.50)

OCC_NR = 32       # feature (r, v_r) occupancy fine grid (smoothed at ℓ)
OCC_NV = 32
OCC_VMAX = 3.5

RAD_PROFILE_EDGES = (0.10, 0.20, 0.35, 0.50)   # bulk coarse radial mass profile


def psi_feature_names() -> List[str]:
    names: List[str] = []
    for prof in ("sigr", "sigt", "vr", "beta"):
        for rr in SAMPLE_RADII:
            names.append(f"psi_{prof}_r{rr:g}")
    names += ["psi_occ_entropy", "psi_occ_std"]
    return names


def bulk_feature_names() -> List[str]:
    return ["Q0", "E0", "C8_0"] + [f"Menc_r{rr:g}" for rr in RAD_PROFILE_EDGES]


# ── velocity decomposition ───────────────────────────────────────────────────────

def decompose(pos: Array, vel: Array, center: Array) -> Tuple[Array, Array, Array]:
    """Return (r, v_r signed, v_t speed) about `center`."""
    d = pos - center
    r = np.sqrt(np.sum(d * d, axis=1))
    rsafe = np.where(r > 1e-12, r, 1e-12)
    rhat = d / rsafe[:, None]
    v_r = np.sum(vel * rhat, axis=1)
    v2 = np.sum(vel * vel, axis=1)
    v_t = np.sqrt(np.maximum(v2 - v_r * v_r, 0.0))
    return r, v_r, v_t


# ── relaxation observables (targets + bulk scalars) ──────────────────────────────

def _phase_entropy(r: Array, v_r: Array) -> float:
    """Shannon entropy of coarse COMOVING (r/r50, v_r/σ) occupancy — a mixing measure.

    Scaling by the half-mass radius and radial dispersion isolates phase-space *shape*
    change (folding / coarse-grained mixing) from overall expansion: a self-similar rescaling
    leaves S unchanged, while phase-space folding raises the coarse-grained entropy.
    """
    r50 = float(np.median(r))
    sig = float(np.std(v_r))
    if r50 < 1e-9 or sig < 1e-9:
        return float("nan")
    rc = np.clip(r / r50, 0.0, PS_RMAX - 1e-9)
    vc = np.clip(v_r / sig, -PS_VMAX + 1e-9, PS_VMAX - 1e-9)
    H, _, _ = np.histogram2d(rc, vc, bins=(PS_NR, PS_NV),
                             range=[[0.0, PS_RMAX], [-PS_VMAX, PS_VMAX]])
    p = H.ravel()
    p = p[p > 0]
    if p.sum() <= 0:
        return float("nan")
    return float(_shannon_entropy(p))          # natural log; normalisation-invariant


def relaxation_observables(pos: Array, vel: Array, cfg: StressConfig) -> Dict[str, float]:
    """Scalars describing relaxation state: Q, E, phase-space entropy, global σ_r/σ_t/β."""
    mass = 1.0 / cfg.n
    sc = get_simconfig(cfg)
    center = np.mean(pos, axis=0)
    r, v_r, v_t = decompose(pos, vel, center)

    ke = kinetic_energy(vel, mass)
    pe = potential_energy_direct(pos, mass, sc)        # exact for direct_isolated
    Q = 2.0 * ke / abs(pe) if (math.isfinite(pe) and abs(pe) > 1e-30) else float("nan")
    E = ke + pe

    inside = r <= PS_RMAX
    vr_in, vt_in = v_r[inside], v_t[inside]
    sigr = float(np.std(vr_in)) if vr_in.size > 2 else float("nan")
    sigt = float(np.sqrt(np.mean(vt_in ** 2))) if vt_in.size > 2 else float("nan")
    beta = (1.0 - sigt ** 2 / (2.0 * sigr ** 2)) if (sigr and sigr > 1e-9) else float("nan")

    return {"Q": Q, "E": E, "S_mix": _phase_entropy(r, v_r),
            "sigr": sigr, "sigt": sigt, "beta": beta, "ke": ke, "pe": pe}


# ── ψ_ℓ coarse phase-space features ──────────────────────────────────────────────

def _fine_radial_profiles(r: Array, v_r: Array, v_t: Array) -> Dict[str, Array]:
    """Per fine-radial-bin kinematic moments (σ_r, σ_t, mean v_r, β)."""
    bw = PROF_RMAX / PROF_NFINE
    idx = np.clip((r / bw).astype(int), 0, PROF_NFINE - 1)
    cnt = np.bincount(idx, minlength=PROF_NFINE).astype(float)
    s_vr = np.bincount(idx, weights=v_r, minlength=PROF_NFINE)
    s_vr2 = np.bincount(idx, weights=v_r * v_r, minlength=PROF_NFINE)
    s_vt2 = np.bincount(idx, weights=v_t * v_t, minlength=PROF_NFINE)
    safe = np.where(cnt > 0, cnt, np.nan)
    mean_vr = s_vr / safe
    var_vr = np.maximum(s_vr2 / safe - mean_vr ** 2, 0.0)
    sigr = np.sqrt(var_vr)
    sigt = np.sqrt(np.maximum(s_vt2 / safe, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = 1.0 - sigt ** 2 / (2.0 * sigr ** 2)
    # empty/degenerate bins → 0 so smoothing can blend across them
    out = {}
    for name, arr in (("sigr", sigr), ("sigt", sigt), ("vr", mean_vr), ("beta", beta)):
        out[name] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def psi_all_scales(pos: Array, vel: Array, scales: List[float]) -> Array:
    """ψ_ℓ for every radial-resolution scale.  Returns (n_scales, n_feat)."""
    center = np.mean(pos, axis=0)
    r, v_r, v_t = decompose(pos, vel, center)
    prof = _fine_radial_profiles(r, v_r, v_t)
    bw = PROF_RMAX / PROF_NFINE
    sample_idx = [min(int(round(rr / bw)), PROF_NFINE - 1) for rr in SAMPLE_RADII]

    # fine (r, v_r) occupancy for the smoothed-occupancy features
    rc = np.clip(r, 0.0, PROF_RMAX - 1e-9)
    vc = np.clip(v_r, -OCC_VMAX + 1e-9, OCC_VMAX - 1e-9)
    occ0, _, _ = np.histogram2d(rc, vc, bins=(OCC_NR, OCC_NV),
                                range=[[0.0, PROF_RMAX], [-OCC_VMAX, OCC_VMAX]])
    occ_bw_r = PROF_RMAX / OCC_NR

    n_feat = len(SAMPLE_RADII) * 4 + 2
    out = np.empty((len(scales), n_feat), dtype=np.float64)
    for s, ell in enumerate(scales):
        sigma = ell / bw
        feats: List[float] = []
        for name in ("sigr", "sigt", "vr", "beta"):
            sm = gaussian_filter1d(prof[name], sigma=max(sigma, 1e-3), mode="nearest")
            feats.extend(sm[i] for i in sample_idx)
        # smoothed occupancy at this radial resolution → entropy + spread
        occ_sm = gaussian_filter(occ0, sigma=(ell / occ_bw_r, ell / occ_bw_r),
                                 mode="constant")
        p = occ_sm.ravel(); p = p[p > 0]
        ent = float(_shannon_entropy(p)) if p.sum() > 0 else 0.0
        feats.append(ent)
        feats.append(float(np.std(occ_sm)))
        out[s] = feats
    return out


# ── bulk control features ────────────────────────────────────────────────────────

def bulk_features(pos: Array, vel: Array, cfg: StressConfig,
                  relax0: Dict[str, float]) -> Dict[str, float]:
    """{Q0, E0, C8(t0), coarse radial enclosed-mass profile}."""
    center = np.mean(pos, axis=0)
    r = np.sqrt(np.sum((pos - center) ** 2, axis=1))
    n = len(r)
    out = {"Q0": relax0["Q"], "E0": relax0["E"],
           "C8_0": obs_coarse_var(pos, cfg, 8, False)}
    for rr in RAD_PROFILE_EDGES:
        out[f"Menc_r{rr:g}"] = float(np.sum(r <= rr)) / max(n, 1)
    return out
