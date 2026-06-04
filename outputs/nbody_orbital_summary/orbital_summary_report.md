# Concentration-relevant orbital summary

Hernquist, ε=0.05, N=1024, 100 pairs. Shell-local radializations (inner/mid/outer thirds) + full radialize/tangentialize/sham; integrate to 100 steps.

## Verdict — 🟢 ORBITAL SUMMARY FOUND

> **INNER LOW-PERICENTER FRACTION IS THE SUMMARY — across interventions, ΔM(<0.1) is best tracked by ΔL_inner (r=-0.96), while global ⟨|L|⟩ does NOT track it (r=-0.55). Pattern by ΔM(<0.1): [('rad', 0.0315), ('inner3', 0.0291), ('mid3', 0.0149), ('outer3', 0.0036), ('tan', -0.0119)] — inner-third concentrates, outer-third does NOT, tangentialize reverses. Global mean ⟨|L|⟩ is RETIRED as the causal summary; the causal variable is the low-pericenter / inner-low-|L| orbit population. No AWS.**

## ΔM(<r_c) response and Δsummary by intervention (− sham)

| arm | ΔM(<0.05) | **ΔM(<0.1)** | ΔC₈ | Δ⟨L⟩_glob | Δ⟨L⟩_inner | Δ⟨L⟩_outer | Δf(peri<.05) | Δf(peri<.1) |
|---|---|---|---|---|---|---|---|---|
| rad | +0.01579 | **+0.03148** | +4.83 | -0.344 | -0.039 | -0.866 | +0.1241 | +0.1324 |
| inner3 | +0.01573 | **+0.02905** | +0.84 | -0.014 | -0.039 | -0.002 | +0.0678 | +0.0548 |
| mid3 | +0.00495 | **+0.01492** | +4.22 | -0.044 | +0.000 | -0.002 | +0.0488 | +0.0632 |
| outer3 | +0.00193 | **+0.00362** | +0.84 | -0.289 | +0.000 | -0.867 | +0.0067 | +0.0153 |
| tan | -0.00116 | **-0.01188** | +0.01 | +0.212 | +0.025 | +0.534 | -0.0436 | -0.0639 |

## Cross-arm correlation of each Δsummary with ΔM(<0.1)

| summary | corr with ΔM(<0.1) | corr with ΔM(<0.05) |
|---|---|---|
| L_global | -0.55 | -0.44 |
| L_inner | -0.96 | -0.98 |
| L_mid | -0.69 | -0.50 |
| L_outer | -0.47 | -0.37 |
| fperi_005 | +0.96 | +0.91 |
| fperi_01 | +0.92 | +0.82 |
| fperi_02 | +0.79 | +0.63 |

**Best predictor: L_inner** (r=-0.96). Global ⟨|L|⟩ corr = -0.55 → global mean retired = True.
conservation ΔKE/KE=8.2e-05, ΔQ/Q=8.2e-05.

