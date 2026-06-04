# Anisotropy Mechanism — Test D (causal time-course)

Hernquist, ε=0.05, N=1024, θ=20°, 100 matched pairs. Times (steps): [0, 5, 10, 20, 40, 80, 160, 320, 600, 1000].

## Verdict — 🟢 MECHANISM RESOLVED

> **MECHANISM RESOLVED [conserved |L|-depletion (upstream, permanent) → β (transient signature) + orbital concentration → C₈] — |L| effect is PERMANENT (per-particle |L_i| conserved in the spherical potential: -0.34→-0.34); β is transient (decays) (+0.65→+0.10) decaying toward the natural attractor; L→β generation: the β-null L-matched arm grows β +0.07→+0.34 (early) from pure |L|-depletion → |L| is UPSTREAM of β; timing: M(<0.05) first-sig t=5, M(<0.1) t=5, C₈ t=10 (concentration BEFORE C₈); sham clean: True; antisymmetry late: True. The durable causal variable is the orbital angular-momentum distribution (|L_i| conserved); β is its decaying signature; concentration mediates C₈. A non-rotational handle could vary |L| without the β-gradient confound, but the conserved-|L|/transient-β ordering is already clear. No AWS.**

## First significant time (radialize − sham): M(<0.05) t=5, M(<0.1) t=5, C₈ t=10 → concentration-before-C₈ = True

## Radialize effect (− sham) vs time

| quantity | t=0 | t=5 | t=10 | t=20 | t=40 | t=80 | t=160 | t=320 | t=600 | t=1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| beta | +0.652 | +0.628 | +0.593 | +0.540 | +0.488 | +0.426 | +0.329 | +0.222 | +0.148 | +0.101 |
| Lspec | -0.344 | -0.344 | -0.343 | -0.342 | -0.341 | -0.339 | -0.337 | -0.337 | -0.336 | -0.336 |
| M05 | +0.000 | +0.004 | +0.010 | +0.016 | +0.009 | +0.008 | +0.005 | +0.004 | +0.003 | +0.002 |
| M10 | +0.000 | +0.003 | +0.012 | +0.030 | +0.030 | +0.021 | +0.016 | +0.018 | +0.016 | +0.011 |
| M20 | +0.000 | +0.001 | +0.002 | +0.011 | +0.035 | +0.024 | +0.028 | +0.022 | +0.028 | +0.034 |
| C8 | +0.000 | +0.073 | +0.242 | +1.297 | +4.021 | +3.805 | +2.353 | +1.367 | +2.321 | +2.622 |
| sigr | +0.239 | +0.231 | +0.219 | +0.204 | +0.186 | +0.161 | +0.124 | +0.086 | +0.062 | +0.045 |
| S | -0.109 | -0.099 | -0.081 | -0.052 | -0.033 | -0.008 | +0.017 | +0.015 | +0.029 | +0.051 |

## β(t) effect: radialize vs tangentialize vs L-matched

| arm | t=0 | t=5 | t=10 | t=20 | t=40 | t=80 | t=160 | t=320 | t=600 | t=1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| rad | +0.652 | +0.628 | +0.593 | +0.540 | +0.488 | +0.426 | +0.329 | +0.222 | +0.148 | +0.101 |
| tan | -1.975 | -1.695 | -1.397 | -1.073 | -0.826 | -0.640 | -0.378 | -0.206 | -0.114 | -0.064 |
| lmatch | +0.070 | +0.127 | +0.195 | +0.279 | +0.343 | +0.291 | +0.241 | +0.150 | +0.077 | +0.044 |

- β pattern (radialize): **transient (decays)**; **|L| permanent: True** (conserved per-particle); **L→β generation: True** (L-matched β +0.07→+0.34 early).
- sham clean: True; antisymmetry holds late: True; conservation ΔKE/KE=8.2e-05, ΔQ/Q=8.2e-05.
- natural relaxation (orig): β -0.00→+0.61, M(<0.1) 0.078→0.018.

