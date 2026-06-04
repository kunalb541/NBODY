# Low-pericenter dose-response

Hernquist, ε=0.05, N=1024, 100 pairs. Graded inner radializations + mid/full/tangentialize/sham. Integrate to 100 steps.

## Verdict — 🟢 CONFIRMED

> **CONFIRMED (non-strict monotone) — strong dose-response (corr {'0.05': 0.91, '0.1': 0.86, '0.2': 0.98}), M leads C₈; the low-pericenter mechanism holds though arm ordering is not perfectly monotone.**

## Dose-response (across arms, including sham at origin)

| r_c | corr(ΔM, Δf_peri) | slope |
|---|---|---|
| 0.05 | +0.91 | +0.18 |
| 0.1 | +0.86 | +0.31 |
| 0.2 | +0.98 | +0.36 |

M leads C₈: M sig t=5, C₈ t=20; monotone=False; conservation ΔKE/KE=4.9e-05.

## Per-arm (sorted by Δf_peri<0.1)

| arm | Δf(<0.05) | Δf(<0.1) | Δf(<0.2) | ΔM(<0.05) | **ΔM(<0.1)** | ΔM(<0.2) | ΔC₈ |
|---|---|---|---|---|---|---|---|
| tan | -0.0436 | -0.0639 | -0.0638 | -0.00662 | **-0.02080** | -0.02750 | -2.28 |
| inner_w | +0.0316 | +0.0269 | +0.0105 | +0.00626 | **+0.01413** | +0.00589 | -0.44 |
| inner_m | +0.0678 | +0.0548 | +0.0182 | +0.01567 | **+0.02903** | +0.01111 | -0.81 |
| mid | +0.0488 | +0.0632 | +0.0721 | +0.00339 | **+0.01307** | +0.02643 | +3.95 |
| inner_s | +0.1261 | +0.0895 | +0.0231 | +0.03140 | **+0.05081** | +0.01465 | -1.26 |
| full | +0.1241 | +0.1324 | +0.1138 | +0.01559 | **+0.02977** | +0.03652 | +4.71 |
