#!/usr/bin/env python3
"""
nbody_anisotropic_ic.py — anisotropic equilibrium initial conditions (Stage-0 of the anisotropy battery)
=========================================================================================================

Self-contained DF-based samplers (numpy only; units G=M=1, scale a). NOT the repo's approximate-Gaussian
samplers — these draw velocities from the actual distribution function so beta(r) is correct and the
system is a genuine equilibrium (validated in Stage 0).

Families:
  - isotropic        : Eddington inversion  f(E)
  - radial (OM)       : Osipkov-Merritt      f(Q), Q = E - L^2/(2 r_a^2),   beta(r)=r^2/(r^2+r_a^2)
  - tangential        : constant beta=-0.5   f(E,L)=L * f_E(E),             beta(r)=-1/2

Positions: exact inverse-CDF. Velocities: rejection sampling on the per-radius DF.
Profiles: 'hernquist' (cusp) and 'plummer' (core). Potentials are UNSOFTENED here (the IC); Stage 0
separately checks equilibrium under the SOFTENED production potential.
"""
from __future__ import annotations

import math
import numpy as np

# ---- profile primitives (G=M=1) -------------------------------------------------
def rho(profile, r, a):
    r = np.asarray(r, float)
    if profile == "hernquist":
        return a / (2 * np.pi * r * (r + a) ** 3)
    if profile == "plummer":
        return 3.0 / (4 * np.pi * a ** 3) * (1.0 + (r / a) ** 2) ** -2.5
    raise ValueError(profile)

def Mlt(profile, r, a):
    if profile == "hernquist":
        return r ** 2 / (r + a) ** 2
    if profile == "plummer":
        return r ** 3 / (r ** 2 + a ** 2) ** 1.5

def psi(profile, r, a):                       # relative potential, >0, ->0 at infinity
    if profile == "hernquist":
        return 1.0 / (r + a)
    if profile == "plummer":
        return 1.0 / np.sqrt(r ** 2 + a ** 2)

def r_of_psi(profile, p, a):
    if profile == "hernquist":
        return 1.0 / p - a
    if profile == "plummer":
        return np.sqrt(np.maximum(1.0 / p ** 2 - a ** 2, 0.0))

def W_analytic(profile, a):                   # total potential energy (G=M=1)
    if profile == "hernquist":
        return -1.0 / (6.0 * a)
    if profile == "plummer":
        return -3.0 * np.pi / (32.0 * a)

def _sphere_dirs(rng, n):
    v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1)[:, None]

def sample_radii(profile, n, a, rng):
    u = rng.uniform(1e-7, 1 - 1e-7, n)
    if profile == "hernquist":
        s = np.sqrt(u); r = a * s / (1.0 - s)
    elif profile == "plummer":
        r = a / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    # truncate to 50a (= 5*box in the production setup; matches the committed Hernquist sampler).
    # Removes far outliers that destabilize centering, keeps ~96% (Hernquist)/~100% (Plummer) of mass,
    # and keeps Psi(r) well within the DF grid.
    return np.clip(r, 1e-4 * a, 50.0 * a)


# ---- DF construction (numerical inversion on a Psi grid) ------------------------
class _DF:
    """Callable f(Q) (OM/iso) or f_E(E) (constant-beta), built by numerical inversion."""
    def __init__(self, profile, a, kind, param):
        self.profile, self.a, self.kind, self.param = profile, a, kind, param
        # dense r-grid -> Psi (ascending); evaluate the relevant 'density vs Psi'
        r = np.logspace(math.log10(1e-5 * a), math.log10(1e4 * a), 8000)
        p = psi(profile, r, a)                       # decreasing in r
        order = np.argsort(p); p = p[order]; r = r[order]
        if kind == "om":
            ra = param
            dens = (1.0 + (r / ra) ** 2) * rho(profile, r, a)   # OM augmented density
        elif kind == "beta":                          # constant beta=-0.5 -> invert rho/r
            dens = rho(profile, r, a) / r
        else:
            dens = rho(profile, r, a)                 # isotropic
        # unique Psi
        p, ui = np.unique(p, return_index=True); dens = dens[ui]
        self.pgrid = p
        if kind == "beta":
            # f_E(Psi) ∝ d^2(rho/r)/dPsi^2  (Cuddeford, beta=-1/2). Numerical 2nd derivative.
            d1 = np.gradient(dens, p)
            d2 = np.gradient(d1, p)
            self.ftab = np.clip(d2, 0.0, None)        # non-negativity floor
        else:
            # Eddington / OM:  f(Q) = 1/(sqrt8 pi^2) [ INT_0^Q d2dens/dPsi^2 / sqrt(Q-Psi) dPsi
            #                                            + (1/sqrt(Q)) (ddens/dPsi)|_0 ]
            d1 = np.gradient(dens, p)
            d2 = np.gradient(d1, p)
            self._d2 = d2; self._d1_0 = d1[0]; self._p0 = p[0]
            Q = p.copy(); ft = np.empty_like(Q)
            tt, wt = np.polynomial.legendre.leggauss(96)   # t in (-1,1)->(0,1)
            t = 0.5 * (tt + 1.0); w = 0.5 * wt
            for i, Qi in enumerate(Q):
                Psi = Qi * (1.0 - t ** 2)                   # substitution kills sqrt singularity
                d2i = np.interp(Psi, p, d2)
                integ = np.sum(w * d2i) * 2.0 * math.sqrt(Qi)   # INT d2/sqrt(Q-Psi) dPsi
                bound = (self._d1_0 / math.sqrt(Qi)) if Qi > 0 else 0.0
                ft[i] = (integ + bound) / (math.sqrt(8.0) * np.pi ** 2)
            self.ftab = np.clip(ft, 0.0, None)
        self.fmax = float(np.max(self.ftab))

    def f(self, Q):                                   # interpolate; 0 outside grid
        Q = np.asarray(Q, float)
        out = np.interp(Q, self.pgrid, self.ftab, left=0.0, right=0.0)
        out[Q <= 0] = 0.0
        return out


def sample(profile, n, a, aniso, rng, r_a=None):
    """aniso in {'isotropic','radial','tangential'}; returns (pos[n,3], vel[n,3]) centred at origin."""
    r = sample_radii(profile, n, a, rng)
    pos = r[:, None] * _sphere_dirs(rng, n)
    Psi = psi(profile, r, a)
    vesc = np.sqrt(2.0 * Psi)
    if aniso == "isotropic":
        df = _DF(profile, a, "iso", None); kind = "iso"
    elif aniso == "radial":
        df = _DF(profile, a, "om", r_a); kind = "om"
    elif aniso == "tangential":
        df = _DF(profile, a, "beta", -0.5); kind = "beta"
    else:
        raise ValueError(aniso)
    vr = np.empty(n); vt = np.empty(n)
    # weight w(v_r,v_t):  om: f(Q)*v_t ; iso: f(E)*v_t ; beta=-1/2: f_E(E)*v_t^2
    def _w(Pr, rr, vrr, vtt):
        v2 = vrr ** 2 + vtt ** 2
        ins = v2 <= 2.0 * Pr
        if kind == "om":
            Q = Pr - 0.5 * v2 - (vtt ** 2) * (rr ** 2) / (2.0 * r_a ** 2); ww = df.f(Q) * vtt
        elif kind == "beta":
            ww = df.f(Pr - 0.5 * v2) * vtt ** 2
        else:
            ww = df.f(Pr - 0.5 * v2) * vtt
        return np.where(ins, ww, 0.0)
    # v_t proposal cap: for OM the valid region (Q>=0) confines v_t <= vesc/sqrt(1+r^2/r_a^2)
    # (radial bias suppresses tangential motion at large r); iso/beta use the full vesc.
    if kind == "om":
        vt_cap = vesc / np.sqrt(1.0 + (r / r_a) ** 2)
    else:
        vt_cap = vesc.copy()
    # precompute per-particle envelope ONCE on a fine (v_r,v_t) grid (fixed per particle)
    ng = 64; gg = (np.arange(ng) + 0.5) / ng
    VR = (2 * gg[None, :] - 1) * vesc[:, None]; VT = gg[None, :] * vt_cap[:, None]
    WW = _w(Psi[:, None], r[:, None], VR, VT)
    env = np.max(WW, axis=1) * 1.6 + 1e-300
    # rejection with the precomputed envelope (loop is O(remaining) per iter; capped)
    todo = np.arange(n); it = 0
    while todo.size and it < 4000:
        it += 1; m = todo.size
        vr_p = rng.uniform(-1, 1, m) * vesc[todo]; vt_p = rng.uniform(0, 1, m) * vt_cap[todo]
        w = _w(Psi[todo], r[todo], vr_p, vt_p)
        over = w > env[todo]                       # envelope-bound violations (should be ~0)
        acc = (rng.uniform(0, 1, m) * env[todo] < w) | over
        idx = todo[acc]; vr[idx] = vr_p[acc]; vt[idx] = vt_p[acc]; todo = todo[~acc]
    if todo.size:
        raise RuntimeError(f"rejection did not converge for {todo.size} particles ({kind})")
    # assemble Cartesian velocity: v_r along r_hat, v_t in random perpendicular direction
    rhat = pos / np.maximum(r[:, None], 1e-30)
    # build a perpendicular unit vector per particle
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    alt = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    use_alt = np.abs(rhat[:, 0]) > 0.9
    ref[use_alt] = alt[use_alt]
    e1 = np.cross(rhat, ref); e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(rhat, e1)
    phi = rng.uniform(0, 2 * np.pi, n)
    that = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
    vel = vr[:, None] * rhat + vt[:, None] * that
    vel -= np.mean(vel, axis=0)
    return pos, vel
