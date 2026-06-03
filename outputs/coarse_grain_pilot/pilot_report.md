# Coarse-Graining Scale Pilot — Report

**Model:** `direct_isolated` (only model where ε affects the force)  
**Families:** hernquist3d, plummer3d, bimodal3d  
**ε:** [0.02, 0.05, 0.1]  **N:** 1024 (fixed across ε)  **steps:** 100 (ΔC8-early)  **reps/cell:** 500  
**Smoothing scales ℓ:** [0.025, 0.04, 0.06, 0.09, 0.14, 0.2, 0.3]  **grid:** 96³  **regressor:** RidgeCV  **split:** 70%/30% fixed

## Verdict — 🔴 RED — retire the coarse-graining *law*; the prior result stands only as a methods / stylized observable-class study

> **STOP — ℓ*(ε) is flat/unstructured; retire the force-resolution-scale claim.**

## Kill tests

1. **R²(ℓ) structured (not flat)?** NO (median peak prominence = 0.13× CI width; >1 ⇒ structured).
2. **ℓ* increases with ε (cusps)?** NO  
   - hernquist3d: Spearman(ε, ℓ*) = +0.00, P(ℓ* non-decreasing) = 0.09
   - plummer3d: Spearman(ε, ℓ*) = -0.87, P(ℓ* non-decreasing) = 0.21
3. **Coarse optimum beats fine features?** NO (Δ_coarse 95% CI excludes 0 in ≥1 cusp cell).
4. **Survives bulk controls?** NO (φ_ℓ* adds held-out R² beyond bulk + persistence cg8₀ controls, CI > 0, in ≥1 cusp cell — i.e. genuine future-prediction power, not baseline-sharing).
5. **Runtime / AWS?** sim stage 147.0s for 4500 sims (32.7 ms/sim wall on 9 workers); features 291.0s. **AWS needed: False.**

## ℓ*(ε) by family

| family | ε | ℓ* | ℓ* 95% CI | R²(ℓ*) | R²(fine) | Δ_coarse [CI] | φ beyond ctrl [CI] | winner |
|---|---|---|---|---|---|---|---|---|
| hernquist3d | 0.02 | 0.09 | [0.06, 0.14] | 0.211 | 0.392 | -0.181 [-0.294, -0.067] | +0.010 [-0.029, +0.049] | fine |
| hernquist3d | 0.05 | 0.14 | [0.025, 0.2] | 0.189 | 0.364 | -0.175 [-0.276, -0.068] | +0.000 [-0.031, +0.031] | fine |
| hernquist3d | 0.1 | 0.09 | [0.025, 0.14] | 0.431 | 0.495 | -0.065 [-0.157, +0.020] | -0.004 [-0.022, +0.014] | tie |
| plummer3d | 0.02 | 0.14 | [0.025, 0.14] | 0.191 | 0.342 | -0.150 [-0.305, +0.002] | +0.013 [-0.014, +0.039] | tie |
| plummer3d | 0.05 | 0.14 | [0.04, 0.14] | 0.278 | 0.411 | -0.133 [-0.273, +0.004] | -0.014 [-0.036, +0.008] | tie |
| plummer3d | 0.1 | 0.06 | [0.025, 0.14] | 0.537 | 0.666 | -0.129 [-0.235, -0.042] | -0.004 [-0.011, +0.003] | fine |
| bimodal3d | 0.02 | 0.2 | [0.14, 0.3] | 0.545 | 0.264 | +0.281 [+0.167, +0.393] | +0.000 [-0.001, +0.002] | coarse-large |
| bimodal3d | 0.05 | 0.3 | [0.09, 0.3] | 0.574 | 0.556 | +0.019 [-0.033, +0.075] | -0.000 [-0.001, +0.000] | tie |
| bimodal3d | 0.1 | 0.3 | [0.14, 0.3] | 0.643 | 0.680 | -0.037 [-0.104, +0.028] | -0.000 [-0.000, +0.000] | tie |

## Target robustness — absolute C₈(t₁) (no delta)

| family | ℓ*(ε) [abs target] | Spearman(ε,ℓ*) | φ beyond ctrl > 0 anywhere? |
|---|---|---|---|
| hernquist3d | [0.14, 0.09, 0.09] | -0.87 | no |
| plummer3d | [0.14, 0.09, 0.025] | -1.00 | no |
| bimodal3d | [0.025, 0.2, 0.3] | +1.00 | yes |

## The five questions

**1. Does ℓ* track ε?**  **NO.**  R²(ℓ) curves are flat (median peak prominence 0.13× the bootstrap CI width; a real peak needs >1). For the cusp families where softening should matter most, Spearman(ε, ℓ*) = hernquist=+0.00, plummer=-0.87 — i.e. no increase (the hypothesis predicts a clear rise). P(ℓ* strictly increasing across ε) ≈ 0 in every family.

**2. Does Δ_coarse survive the hardened control?**  **NO.**  After controlling for bulk + cg8₀, the scale-resolved φ_ℓ* adds ≈0 held-out R² in every cell (all 95% CIs include 0). The apparent φ_ℓ R² is fully explained by bulk quantities + the trivial −cg8₀ baseline-sharing. And Δ_coarse vs fine_all is *negative* for the cusps (fine features win) — coarse does not beat fine.

**3. Which family shows the cleanest effect?**  None show a force-resolution *scale* effect. **bimodal** has the cleanest *predictability* (R² up to ~0.6, coarse_fixed up to ~0.99) but its ℓ* sits at the largest scales and is set by clump geometry, not ε — the paper's known coarse dominance, re-derived. The **cusps (hernquist, plummer)**, where ε should matter most, give the cleanest **null**: flat R²(ℓ) and ℓ* that does not rise with ε.

**4. Is the signal target-specific or generic?**  **Neither survives — the weak apparent trend is target-specific and wrong-signed.**  Swapping ΔC8 → absolute C₈(t₁): cusp Spearman(ε, ℓ*) = hernquist=-0.87, plummer=-1.00 (vs delta hernquist=+0.00, plummer=-0.87) — the sign changes with the target and is ≤0 for cusps under both. φ beyond control stays ≈0 for the absolute target too. There is no target-generic, control-surviving scale law.

**5. Is AWS justified?**  **NO.**  The pilot is red, so by the decision rule there is nothing to scale up. Independently, runtime is trivial: 4500 sims in 147s + features 291s on 9 local workers; the full grid extrapolates to a single local day — below the ~2–3 day AWS threshold.

## Baseline R² reference (per cell)

See `results.csv` (kind=baseline) for fine_pos / fine_kin / fine_all / coarse_fixed / persistence / bulk R² in every cell, and `summary.json` → `target_robustness` for the full absolute-target sweep.
