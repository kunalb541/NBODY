# N-body Anisotropy Handle — Non-Cusp Contrast (uniform box)

Contrast family: **uniform3d** (homogeneous, single-component, non-concentrated). Same handle/sham/controls. θ=20°, ε∈[0.02, 0.05, 0.1], N=1024, 100 pairs/cell.

Cusp reference (Hernquist+Plummer): β effect 0.11-0.19, persistence grav 17-29% vs free 3-8%.

## Verdict — 🟡 CONCENTRATED-SPECIFIC

> **CONCENTRATED-MECHANISM-SPECIFIC — β technically responds in the uniform box (3/3), but via a DIFFERENT mechanism: the box is inert for β (natural relaxation orig β stays ~0, attractor |Δ|=0.01 vs ~0.5 in cusps), so the imposed anisotropy persists PASSIVELY (75%, no relaxation to erase it), and the clustering response REVERSES sign (C₈=-0.83 vs +2…+3 in cusps: dispersal, not infall). The active-relaxation causal handle remains a CONCENTRATED-FAMILY effect. The contrast sharpens, not broadens, the scope.**

## Diagnostics

- β responds: 3/3 ε cells · σ_r: 3/3 · C₈: 3/3
- sham null: True · conservation OK: True · sign OK: True
- persistence gravity 75% vs free-streaming 20% → non-trivial (dynamical)
- natural β relaxation (orig 0→t₁) by ε: {'0.02': [-0.0, -0.01], '0.05': [-0.0, -0.01], '0.1': [-0.0, -0.02]}

## Per-cell

| ε | imposed Δβ₀ | causal β [CI] | persist grav/free | sham Δβ₀ | ΔKE/KE | causal σr | causal Q | causal S | causal C₈ | orig β 0→1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | +0.650 | +0.474 [+0.464,+0.485] | 73%/20% | -0.002 | 9.0e-05 | +0.059 | -0.079 | +0.006 | -0.804 | -0.00→-0.01 |
| 0.05 | +0.650 | +0.485 [+0.476,+0.494] | 75%/20% | -0.002 | 9.0e-05 | +0.060 | -0.080 | -0.001 | -0.833 | -0.00→-0.01 |
| 0.1 | +0.650 | +0.508 [+0.497,+0.518] | 78%/20% | -0.002 | 9.0e-05 | +0.062 | -0.081 | -0.010 | -0.938 | -0.00→-0.02 |

