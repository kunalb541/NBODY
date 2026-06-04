# Anisotropy Mechanism — Test A (antisymmetry) + mediation

Hernquist, ε=0.05, N=1024, θ=20°, 100 matched quadruples (orig/radialize/tangentialize/sham), t₁=600.

## Verdict — 🟢 ANTISYM + CONCENTRATION-MEDIATED

> **ANTISYMMETRIC + CONCENTRATION-MEDIATED — radialize raises C₈ (+2.32) and central mass M(<0.1) (+0.0162); tangentialize lowers both (-1.16, -0.0084) — clean SIGN antisymmetry. The clustering tracks central CONCENTRATION at the mean level, via orbital pericenter reduction (lower specific L → deeper plunge), NOT bulk infall velocity (inner ⟨v_r⟩ effect null). MAGNITUDE is asymmetric/nonlinear: radialize is ~6× more effective per unit Δβ₀ (concentration is one-directional). β and |L_i| both flip (entangled) — the β-vs-L split needs the L-matched control (test B). [per-pair ΔC₈↔ΔMcen r=-0.14: weak, scale mismatch.] Proceed to test B.**

## Causal effects (intervention − sham, t₁), paired 95% CI

| quantity | radialize | tangentialize | antisymmetric? |
|---|---|---|---|
| beta | +0.1478 [+0.1428, +0.1524] | -0.1142 [-0.1194, -0.1090] | yes |
| Lspec | -0.3360 [-0.3407, -0.3310] | +0.2100 [+0.2063, +0.2139] | yes |
| C8 | +2.3209 [+1.4888, +3.2247] | -1.1617 [-1.8372, -0.4727] | yes |
| Mcen | +0.0162 [+0.0130, +0.0195] | -0.0084 [-0.0104, -0.0064] | yes |
| vr_inner | -0.0067 [-0.0157, +0.0018] | +0.0041 [-0.0032, +0.0115] | — |

## Imposed at t₀ (handle check)

- radialize: Δβ₀ = +0.6519 [+0.6453, +0.6586], Δ⟨|L_i|⟩ = -0.3437 [-0.3477, -0.3396]
- tangentialize: Δβ₀ = -1.9751 [-2.0021, -1.9488], Δ⟨|L_i|⟩ = +0.2130 [+0.2101, +0.2159]
- sham Δβ₀ = -0.0002 [-0.0054, +0.0046] (≈0)

## Mediation & controls

- ΔC₈ ↔ Δ central-mass M(<0.1) correlation (radialize−sham): r = -0.14
- conservation: ΔKE/KE = 8.2e-05, ΔQ/Q = 8.2e-05; sham null = True
- natural relaxation (orig t₀→t₁): {'beta': [-0.004, 0.536], 'Lspec': [1.275, 1.276], 'C8': [34.825, 23.639], 'Mcen': [0.078, 0.026], 'vr_inner': [0.003, 0.011]}

## β vs L (entangled — needs test B)

radialize raises β and lowers ⟨|L_i|⟩; tangentialize does the opposite. C₈ follows β↑ / |L_i|↓ together, so this pilot cannot separate them — the L-matched (β-null) control (test B) is required.

