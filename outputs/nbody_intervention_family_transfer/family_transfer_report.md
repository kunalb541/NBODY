# N-body Anisotropy Handle — Family Transfer (Hernquist → Plummer)

Same handle / sham / controls as the confirmation. {Hernquist, Plummer} × ε∈{0.02,0.05,0.10}, θ=20°, N=1024, 100 matched pairs/cell.

## Verdict — 🟢 ROBUST CUSP-FAMILY

> **ROBUST CUSP-FAMILY HANDLE — velocity anisotropy is a causal handle on future relaxation in BOTH Hernquist and Plummer (β CI>0 in 3/3 / 3/3 ε cells, sham null, KE/Q preserved, gravity persistence 25% > free-streaming 3%). The handle transfers across cusp families. No AWS.**

## By family

- **hernquist3d** — β CI>0 in 3/3 ε cells; sign_ok=True; sham_null=True; conservation_ok=True; persistence grav/free = 23%/8% (non-trivial); C₈ responds in 3/3 cells; **passed=True**.
- **plummer3d** — β CI>0 in 3/3 ε cells; sign_ok=True; sham_null=True; conservation_ok=True; persistence grav/free = 25%/3% (non-trivial); C₈ responds in 3/3 cells; **passed=True**.

**C₈ response family-specific:** False.

## Per-cell

| family | ε | imposed Δβ₀ | causal β [CI] | persist grav/free | sham Δβ₀ | ΔKE/KE | causal σr | causal Q | causal S | causal C₈ | orig β 0→1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hernquist3d | 0.02 | +0.654 | +0.160 [+0.152,+0.169] | 25%/8% | +0.001 | 8.2e-05 | +0.061 | -0.000 | +0.027 | +2.213 | -0.01→+0.39 |
| hernquist3d | 0.05 | +0.654 | +0.151 [+0.145,+0.158] | 23%/8% | +0.001 | 8.2e-05 | +0.061 | -0.004 | +0.029 | +2.321 | -0.01→+0.46 |
| hernquist3d | 0.1 | +0.654 | +0.118 [+0.112,+0.124] | 18%/8% | +0.001 | 8.2e-05 | +0.055 | -0.019 | +0.032 | +2.091 | -0.01→+0.56 |
| plummer3d | 0.02 | +0.649 | +0.186 [+0.176,+0.196] | 29%/3% | -0.000 | 8.3e-05 | +0.062 | +0.007 | -0.006 | +2.933 | +0.00→+0.35 |
| plummer3d | 0.05 | +0.649 | +0.160 [+0.152,+0.169] | 25%/3% | -0.000 | 8.3e-05 | +0.055 | +0.008 | -0.007 | +3.446 | +0.00→+0.43 |
| plummer3d | 0.1 | +0.649 | +0.108 [+0.100,+0.115] | 17%/3% | -0.000 | 8.3e-05 | +0.044 | +0.003 | +0.023 | +3.028 | +0.00→+0.58 |

