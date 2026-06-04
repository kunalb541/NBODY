# Phase-Space Coarse-Graining Pilot — Report

**Families:** hernquist3d, plummer3d (cusps)  **ε:** [0.02, 0.05, 0.1]  **N:** 1024  **reps:** 500  **horizons:** [100, 600]  **regressor:** RidgeCV  
**Headline target:** `dS_late` = Δ coarse phase-space (r,v_r) entropy (mixing).

## Verdict — 🔴 STOP / FREEZE

> **STOP / FREEZE — after the hardened control (bulk + initial target value), ψ_ℓ adds no robust signal for the mixing headline (survives 1/6 cells). The strong raw Δβ/Δσ_r signals were baseline-sharing (β₀/σ_r₀ predicting their own change — the same trap as C₈). Best partial survivor: dSigr_early 3/6. Freeze N-body as a boundary result. No AWS.**

## Hardened control — the real bar

Raw 'ψ beyond bulk' is contaminated by baseline-sharing: ψ_ℓ contains the initial β(r)/σ_r(r) profiles, and Δβ=β(t₁)−β(t₀) shares −β(t₀) (the C₈ trap). The honest control adds the **initial target quantity** (β₀/σ_r₀/S₀) to bulk.

| target | raw ψ>bulk | **hardened ψ>(bulk+self₀)** |
|---|---|---|
| dS_late | 2/6 | **1/6** |
| dS_early | 2/6 | **2/6** |
| dQ_late | 1/6 | **—** |
| dQ_early | 3/6 | **—** |
| dSigr_late | 4/6 | **0/6** |
| dBeta_late | 4/6 | **0/6** |

## The reported questions

1. **Does ψ_ℓ beat bulk controls?** RAW yes for several targets, but that is baseline-sharing. Under the **hardened** control the mixing headline `dS_late` survives only **1/6** cells, and the strong raw Δβ/Δσ_r collapse to 0/6.
2. **Does ψ_ℓ beat spatial φ_ℓ?** YES for the headline — phase-space features carry far more relaxation info than spatial ones (this echoes the paper's known kinematic/VelDisp advantage, not a new scale law).
3. **Which target works (hardened)?** none at the ≥majority bar.
4. **ℓ*(ε) (secondary, must NOT revive ℓ*∼ε):** {'hernquist3d': -0.8660254037844387, 'plummer3d': 0.0} — inconsistent; no force-resolution scale law.
5. **AWS?** No. Ran 3000 reps in 1078s on 9 local workers.

## Per-cell, per-target (held-out R² and ψ increments)

| family | ε | target | R²(bulk) | R²(φ*) | R²(ψ*) | ψ−bulk [CI] | ψ−φ [CI] | ℓ*_ψ |
|---|---|---|---|---|---|---|---|---|
| hernquist3d | 0.02 | Δentropy(late) | +0.147 | -0.012 | +0.127 | -0.018 [-0.099,+0.062] | +0.127 [+0.002,+0.242] | 0.3 |
| hernquist3d | 0.02 | Δentropy(early) | +0.013 | +0.004 | -0.126 | -0.528 [-1.698,+0.051] | -0.089 [-0.283,+0.052] | 0.2 |
| hernquist3d | 0.02 | ΔQ(late) | +0.980 | +0.666 | +0.637 | -0.004 [-0.009,-0.001] | +0.147 [+0.035,+0.247] | 0.3 |
| hernquist3d | 0.02 | ΔQ(early) | +0.954 | +0.699 | -1.259 | -0.013 [-0.046,+0.003] | +0.181 [+0.111,+0.270] | 0.3 |
| hernquist3d | 0.02 | Δσr(late) | +0.298 | +0.236 | +0.456 | +0.267 [+0.162,+0.364] | +0.264 [+0.073,+0.407] | 0.3 |
| hernquist3d | 0.02 | Δβ(late) | +0.223 | +0.172 | +0.398 | +0.132 [-0.208,+0.353] | +0.287 [+0.120,+0.418] | 0.04 |
| hernquist3d | 0.05 | Δentropy(late) | +0.348 | +0.138 | +0.196 | -0.056 [-0.126,+0.012] | +0.172 [+0.006,+0.325] | 0.2 |
| hernquist3d | 0.05 | Δentropy(early) | +0.182 | +0.061 | -0.100 | -0.052 [-0.140,+0.028] | +0.034 [-0.055,+0.117] | 0.025 |
| hernquist3d | 0.05 | ΔQ(late) | +0.975 | +0.646 | +0.708 | -0.001 [-0.003,+0.002] | +0.203 [+0.133,+0.317] | 0.3 |
| hernquist3d | 0.05 | ΔQ(early) | +0.980 | +0.719 | -3.066 | -0.012 [-0.023,-0.002] | -1.954 [-6.861,+0.194] | 0.3 |
| hernquist3d | 0.05 | Δσr(late) | +0.163 | +0.101 | +0.330 | -2.258 [-8.141,+0.257] | -1.109 [-4.524,+0.378] | 0.2 |
| hernquist3d | 0.05 | Δβ(late) | +0.180 | +0.138 | +0.255 | +0.052 [-0.731,+0.433] | +0.201 [+0.055,+0.318] | 0.2 |
| hernquist3d | 0.1 | Δentropy(late) | +0.332 | +0.213 | +0.343 | +0.054 [+0.001,+0.112] | +0.240 [+0.116,+0.354] | 0.2 |
| hernquist3d | 0.1 | Δentropy(early) | +0.249 | +0.009 | +0.207 | +0.005 [-0.065,+0.079] | +0.222 [+0.105,+0.320] | 0.09 |
| hernquist3d | 0.1 | ΔQ(late) | +0.963 | +0.612 | +0.675 | +0.008 [+0.003,+0.016] | +0.258 [+0.175,+0.364] | 0.3 |
| hernquist3d | 0.1 | ΔQ(early) | +0.978 | +0.607 | +0.735 | +0.006 [+0.001,+0.011] | +0.260 [+0.174,+0.368] | 0.3 |
| hernquist3d | 0.1 | Δσr(late) | +0.159 | +0.076 | +0.300 | +0.294 [+0.196,+0.394] | +0.366 [+0.228,+0.485] | 0.04 |
| hernquist3d | 0.1 | Δβ(late) | +0.087 | +0.061 | +0.437 | +0.378 [+0.282,+0.477] | +0.414 [+0.319,+0.510] | 0.06 |
| plummer3d | 0.02 | Δentropy(late) | +0.630 | +0.339 | +0.643 | +0.033 [+0.005,+0.069] | +0.339 [+0.241,+0.452] | 0.3 |
| plummer3d | 0.02 | Δentropy(early) | +0.478 | +0.174 | +0.508 | +0.051 [-0.008,+0.113] | +0.361 [+0.252,+0.467] | 0.14 |
| plummer3d | 0.02 | ΔQ(late) | +0.632 | +0.222 | +0.529 | +0.007 [-0.065,+0.071] | +0.333 [+0.178,+0.502] | 0.3 |
| plummer3d | 0.02 | ΔQ(early) | +0.832 | +0.553 | +0.717 | +0.056 [+0.033,+0.086] | +0.286 [+0.216,+0.383] | 0.3 |
| plummer3d | 0.02 | Δσr(late) | +0.504 | +0.139 | +0.608 | +0.202 [+0.124,+0.284] | +0.528 [+0.437,+0.612] | 0.14 |
| plummer3d | 0.02 | Δβ(late) | +0.207 | +0.065 | +0.568 | +0.414 [+0.282,+0.544] | +0.530 [+0.402,+0.648] | 0.2 |
| plummer3d | 0.05 | Δentropy(late) | +0.646 | +0.291 | +0.533 | +0.004 [-0.039,+0.044] | +0.273 [+0.173,+0.379] | 0.14 |
| plummer3d | 0.05 | Δentropy(early) | +0.295 | +0.130 | +0.387 | +0.093 [+0.014,+0.178] | +0.270 [+0.123,+0.411] | 0.09 |
| plummer3d | 0.05 | ΔQ(late) | +0.886 | +0.441 | +0.729 | +0.014 [-0.002,+0.030] | +0.346 [+0.275,+0.430] | 0.2 |
| plummer3d | 0.05 | ΔQ(early) | +0.939 | +0.560 | +0.751 | +0.015 [+0.007,+0.026] | +0.289 [+0.199,+0.405] | 0.3 |
| plummer3d | 0.05 | Δσr(late) | +0.290 | +0.095 | +0.576 | -0.041 [-0.917,+0.370] | +0.295 [-0.394,+0.619] | 0.04 |
| plummer3d | 0.05 | Δβ(late) | +0.226 | +0.148 | +0.601 | +0.392 [+0.255,+0.563] | +0.477 [+0.359,+0.599] | 0.2 |
| plummer3d | 0.1 | Δentropy(late) | +0.672 | +0.152 | +0.559 | +0.013 [-0.015,+0.046] | +0.411 [+0.237,+0.631] | 0.3 |
| plummer3d | 0.1 | Δentropy(early) | +0.309 | +0.112 | +0.340 | +0.084 [+0.021,+0.149] | +0.283 [+0.139,+0.411] | 0.09 |
| plummer3d | 0.1 | ΔQ(late) | +0.972 | +0.421 | +0.786 | -0.010 [-0.022,-0.001] | +0.439 [+0.348,+0.547] | 0.2 |
| plummer3d | 0.1 | ΔQ(early) | +0.996 | +0.434 | +0.768 | +0.000 [-0.000,+0.001] | +0.403 [+0.299,+0.535] | 0.3 |
| plummer3d | 0.1 | Δσr(late) | +0.306 | +0.012 | +0.595 | +0.291 [+0.181,+0.404] | +0.592 [+0.494,+0.701] | 0.06 |
| plummer3d | 0.1 | Δβ(late) | +0.163 | +0.035 | +0.668 | +0.530 [+0.382,+0.702] | +0.655 [+0.545,+0.769] | 0.14 |

