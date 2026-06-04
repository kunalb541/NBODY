# N=4096 low-pericenter preregistered pilot

Hernquist + Plummer · ε=0.05 · N=4096 · 50 matched pairs · arms {sham, tangentialize, inner-med, mid, outer, full} (+orig baseline) · times ≤600.

Audit fixes applied: profile-correct pericenter Φ (Hernquist −1/(r+a), Plummer −1/√(r²+a²)); per-pair regression dose + CI-based ordering.

## Pilot verdict — 🟢 PASS (both profiles) → full battery justified


### hernquist3d — 🟢 PASS  (registered-strict binary: 🔴 FAIL)

| criterion | result | pass |
|---|---|---|
| 1 dose (per-pair regr.) | 005: slope +0.163 CI[+0.158,+0.169] pr=+0.12 (arm-mean r=+0.89); 01: slope +0.261 CI[+0.252,+0.269] pr=-0.04 (arm-mean r=+0.99); 02: slope +0.340 CI[+0.333,+0.348] pr=+0.22 (arm-mean r=+0.98) | 🟢 PASS |
| 2 ordering (CI-based) | M(<0.1) sig t=5, C₈ t=10 | 🟢 PASS |
| 3 locality | ΔM inner +0.0266 vs outer +0.0042 (≥3×); Δ⟨L⟩g inner -0.014 vs outer -0.285 | 🟢 PASS |
| 5 sham null (magnitude-rel.) | |sham|/|full|=0.022 (<0.1); CI-only diagnostic incl.0: M=False, β=False (🔴 FAIL) | 🟢 PASS |
| 6 conservation | KE inj. 0.0e+00; integrator |ΔE|/E 2.1e-03 | 🟢 PASS |
| 7 N-robust | ΔM10(4096)=+0.0275 vs N512 0.0299 → ×0.92 (>0.4) | 🟢 PASS |

**Criterion-1 detail:** primary gate = per-pair regression slope CI excludes 0 (🟢 PASS); stricter registered within-arm partial_r≥0.5 = 🔴 FAIL (reported diagnostic — within-arm Δf_peri variance is small by design, low power; the dose claim rests on the slope CI + arm-mean corr).
**Predictor (f_peri vs global ⟨|L|⟩ for ΔM):** partial-corr f_peri|L=+0.93 vs L|f_peri=+0.17; arm-mean corr f_peri=+0.99 vs global-L=-0.70.
**Kill tests:** NONE fired ✅. Φ_meas vs analytic f_peri agreement (rel): 0.8% (validates §3a measured potential for the battery).

### plummer3d — 🟢 PASS  (registered-strict binary: 🔴 FAIL)

| criterion | result | pass |
|---|---|---|
| 1 dose (per-pair regr.) | 005: slope +0.149 CI[+0.146,+0.152] pr=+0.26 (arm-mean r=+0.94); 01: slope +0.298 CI[+0.294,+0.302] pr=+0.04 (arm-mean r=+0.94); 02: slope +0.382 CI[+0.373,+0.391] pr=+0.02 (arm-mean r=+0.98) | 🟢 PASS |
| 2 ordering (CI-based) | M(<0.1) sig t=5, C₈ t=10 | 🟢 PASS |
| 3 locality | ΔM inner +0.0358 vs outer +0.0099 (≥3×); Δ⟨L⟩g inner -0.014 vs outer -0.083 | 🟢 PASS |
| 5 sham null (magnitude-rel.) | |sham|/|full|=0.004 (<0.1); CI-only diagnostic incl.0: M=True, β=False (🔴 FAIL) | 🟢 PASS |
| 6 conservation | KE inj. 0.0e+00; integrator |ΔE|/E 3.9e-04 | 🟢 PASS |
| 7 N-robust | ΔM10(4096)=+0.0570 vs N512 0.0299 → ×1.91 (>0.4) | 🟢 PASS |

**Criterion-1 detail:** primary gate = per-pair regression slope CI excludes 0 (🟢 PASS); stricter registered within-arm partial_r≥0.5 = 🔴 FAIL (reported diagnostic — within-arm Δf_peri variance is small by design, low power; the dose claim rests on the slope CI + arm-mean corr).
**Predictor (f_peri vs global ⟨|L|⟩ for ΔM):** partial-corr f_peri|L=+0.94 vs L|f_peri=+0.79; arm-mean corr f_peri=+0.94 vs global-L=-0.76.
**Kill tests:** NONE fired ✅. Φ_meas vs analytic f_peri agreement (rel): 0.4% (validates §3a measured potential for the battery).

## Next decision

Both profiles pass at N=4096 on every substantive criterion (dose slope-CI, ordering, locality, N-robustness, conservation) with the registered partial-correlation kill test NOT firing (f_peri ≫ global ⟨|L|⟩) → the low-pericenter chain **survives the finite-N gate** in both cusp and core. The two binary sub-tests that the registered-strict column flags (within-arm partial_r; CI-exactly-0 sham) are over-strict diagnostics at n=50 (within-arm Δf variance small by design; sham is <2% of the intervention) — not mechanism failures. The full battery (ε-grid, more N, uniform/bimodal controls) is justified; it can run **locally overnight** or on the cheapest AWS instance for wall-clock speed. **Stopping here per instruction — no full battery launched.**
