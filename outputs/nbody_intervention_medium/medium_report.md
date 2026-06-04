# N-body Causal Anisotropy Handle — Medium Confirmation

Matched-pair (orig/int/sham), N=1024, 100 pairs/cell, horizon t₁=600. Handle: speed-preserving radial anisotropy rotation. **Goal: try to break the handle.**

## Verdict — 🟢 CONFIRMED

> **CONFIRMED — the causal anisotropy handle scales with θ (Spearman θ↔imposed=+1.00, θ↔effect=+1.00), the effect CI excludes 0 in 7/7 cells, sham stays null, KE/Q preserved, and gravitational persistence (23%) exceeds the free-streaming baseline (8%). Controlled anisotropy perturbations causally modulate relaxation outcomes. No AWS.**

## The four stress tests

1. **θ-scaling:** Spearman(θ, imposed Δβ₀) = +1.00; Spearman(θ, effect) = +1.00. effect by θ: { 5°:+0.035, 10°:+0.080, 20°:+0.151, 30°:+0.212 }.
2. **ε-robustness:** causal β effect by ε: { 0.02:+0.160[+0.152,+0.169], 0.05:+0.151[+0.145,+0.158], 0.1:+0.118[+0.112,+0.124] }.
3. **Sham null:** True.  **Conservation OK:** True.  effect CI>0 in 7/7 cells.
4. **Trivial-memory baseline:** gravitational persistence (median) 23% vs **free-streaming** 8% → non-trivial (gravity sustains the imposed anisotropy beyond ballistic carryover).

## Per-cell

| family | ε | θ | imposed Δβ₀ | causal β [CI] | persist grav/free | sham Δβ₀ | ΔKE/KE | causal σr | causal Q | causal S | causal C₈ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hernquist3d | 0.05 | 5° | +0.232 | +0.035 [+0.030,+0.040] | 15%/5% | +0.000 | 5.4e-06 | +0.013 | -0.001 | +0.014 | +0.277 |
| hernquist3d | 0.05 | 10° | +0.410 | +0.080 [+0.075,+0.085] | 20%/5% | +0.000 | 2.2e-05 | +0.030 | -0.002 | +0.024 | +0.880 |
| hernquist3d | 0.05 | 20° | +0.654 | +0.151 [+0.145,+0.158] | 23%/8% | +0.001 | 8.2e-05 | +0.061 | -0.004 | +0.029 | +2.321 |
| hernquist3d | 0.05 | 30° | +0.802 | +0.212 [+0.204,+0.220] | 26%/11% | +0.002 | 1.8e-04 | +0.092 | -0.005 | +0.027 | +4.326 |
| hernquist3d | 0.02 | 20° | +0.654 | +0.160 [+0.152,+0.169] | 25%/8% | +0.001 | 8.2e-05 | +0.061 | -0.000 | +0.027 | +2.213 |
| hernquist3d | 0.1 | 20° | +0.654 | +0.118 [+0.112,+0.124] | 18%/8% | +0.001 | 8.2e-05 | +0.055 | -0.019 | +0.032 | +2.091 |
| plummer3d | 0.05 | 20° | +0.649 | +0.160 [+0.152,+0.169] | 25%/3% | -0.000 | 8.3e-05 | +0.055 | +0.008 | -0.007 | +3.446 |

*Context: unperturbed β relaxes 0→~0.46 (radial-anisotropy attractor); the handle is a modulation on top.*

