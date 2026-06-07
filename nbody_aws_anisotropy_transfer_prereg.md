# AWS Preregistration — Anisotropy-Transfer Battery (N-body)

**Status:** Preregistration. **Do NOT launch until approved.** Criteria fixed *before* running; no
post-hoc adjustment. Governs by the kill-test constitution used for the low-pericenter battery.
`paper.tex` untouched. **Date:** 2026-06-07.

## 0. Motivation — the transfer question

The low-pericenter handle is earned for **isotropic** Hernquist/Plummer concentrated systems
(battery-positive, N-robust to 4096, controls negative; see `outputs/nbody_aws_battery/`). The
mechanism claims central response is driven by **low-pericenter accessibility**, *not* by global
angular momentum or generic anisotropy. The hardest fair test of that claim:

> **Does Δf_peri(<r_c) remain the causal handle on ΔM(<r_c) when the baseline orbital anisotropy is
> deliberately varied (isotropic / radial / tangential)?**

This directly attacks the literature-facing worry (radial-orbit-instability / anisotropy summaries):
radial anisotropy is precisely where β and the low-pericenter population are most entangled, so if the
pericenter dose still beats anisotropy summaries there, the mechanism is genuinely the variable.

## 1. Core claim (single, falsifiable)

> Across isotropic, radially-anisotropic, and tangentially-anisotropic Hernquist and Plummer
> equilibria, the intervention-induced **low-pericenter dose Δf_peri(<0.1)** remains the best
> predictor of the inner-mass response **ΔM(<0.1)** — beating global ⟨|L|⟩, β-based anisotropy
> summaries, radial kinetic fraction, baseline central mass, and outer-shell response, *after
> controlling for baseline structure.*

## 2. Initial conditions — the six anisotropy families (the critical new piece)

Six families: {hernquist3d, plummer3d} × {isotropic, radial, tangential}.

**Construction (preregistered, fixed):**
- **Isotropic:** existing Eddington-inversion samplers (as used in the low-pericenter battery).
- **Radial:** **Osipkov–Merritt** distribution function with anisotropy radius **r_a = a = 0.20**
  (β = 0 at centre rising outward; analytic OM DF for Hernquist per Hernquist 1990, numerical
  Eddington–OM inversion for Plummer). Strong-radial variant r_a = a/2 deferred to the optional grid.
- **Tangential:** a **constant-β = −0.5** distribution (Cuddeford-type construction) sampled at the
  same density profile, giving a global tangential bias.

**Mandatory pre-flight EQUILIBRIUM GATE (per family, N, ε):** integrate the *orig* (un-intervened)
arm to the full horizon and require the baseline to be stationary —
`median |ΔM(<0.1)|/M(<0.1) < 0.10` **and** `|Δ⟨β⟩| < 0.05` over the horizon. A family that fails the
gate is **not** a clean equilibrium (the "anisotropy" would be transient relaxation, confounding the
test); it is re-derived or excluded, and the gate result is reported. *This gate is the difference
between testing anisotropy transfer and testing relaxation.* Matched-pair (arm − sham) further
subtracts any residual common-mode breathing within a family.

## 3. Battery grid

| axis | values |
|---|---|
| family | 6 (above) |
| N | 2048, 4096 |
| ε | **0.05 primary**; ε=0.02, 0.10 at N=2048 **optional** (only if compute is cheap) |
| arms | reuse: orig · sham · tangentialize · inner-{weak,med,strong} · mid · outer · full · vt-rescale |
| times | preregistered {0,5,10,20,50,100,300,600,1000} |
| pairs | 100 (N=2048), 50 (N=4096); matched seeds 2000+ |

**Primary cells:** 6 families × 2 N × 1 ε = **12 cells.** (Optional ε-sweep adds 6×2 = 12 cells at
N=2048.) Reuses the existing isotropic Hernquist/Plummer battery cells as the anchor — only the four
anisotropic families are new science.

## 4. Primary target, predictor, and competitors

- **Target:** ΔM(<0.1) (peak paired effect, arm − sham).
- **Primary causal predictor:** Δf_peri(<0.1) at t₀ (arm − sham).
- **Competing predictors** (each as Δ, arm − sham, plus baseline structure as controls):
  global ⟨|L|⟩; β-based anisotropy (Δβ and outer-shell β); **radial kinetic fraction**
  σ_r²/(σ_r²+σ_t²); **baseline M(<0.1)**; **outer-shell response** ΔM(outer third).

## 5. Analysis — multi-predictor competition (the heart)

Pool per-pair points (arm × pair) across the six families. Compute **partial correlations** of ΔM
with each candidate, each controlling for the others **and for baseline structure** (baseline M(<0.1),
baseline β):
- **Primary:** partial-corr(Δf_peri, ΔM | all competitors + baseline) is **positive with a bootstrap
  95% CI excluding 0**, and is the **largest** among candidates.
- Report a full partial-correlation table per family and pooled, plus the per-family dose-slope
  decomposition (as in the low-pericenter battery), and the equilibrium-gate diagnostics.

## 6. Pass criteria (ALL)

1. **Handle survives anisotropy:** Δf_peri remains the top predictor of ΔM (largest partial-corr, CI
   excludes 0) **pooled and within each of the radial and tangential families** — not only isotropic.
2. **Ordering & sham:** M before C₈ (CI-based); sham magnitude-relative null (|sham|/|intervention| <
   0.1) on intensive channels.
3. **Conservation:** KE injection ~0; integrator |ΔE|/E < 10⁻².
4. **Equilibrium gate passed** for every reported family (§2).

## 7. Kill tests (any TRUE ⇒ mechanism incomplete / claim retired)

| kill test | threshold | consequence |
|---|---|---|
| anisotropy summary beats f_peri | partial-corr(β-summary or radial-KE-fraction, ΔM \| f_peri + baseline) > partial-corr(Δf_peri, ΔM \| that + baseline) | **mechanism incomplete** — anisotropy carries the signal |
| handle vanishes in anisotropic families | Δf_peri partial-corr CI includes 0 in the radial **or** tangential family | **no transfer**; scope stays isotropic |
| sham explains response | |sham−orig| ≥ 0.5 × intervention, or KE/Q drift > 10⁻² | reject intervention |
| outer-shell/random control reproduces inner response | ΔM(outer) > 0.6 × ΔM(full), or random-axis control matches | reject locality |

## 8. Runtime estimate (measured per-pair from the low-pericenter battery)

Per-pair wall (2-vCPU free-tier, measured): N=2048 ≈ 60 s, N=4096 ≈ 223 s.

| grid | cells | pairs | free-tier 2-vCPU | paid 8-vCPU (spot) |
|---|---|---|---|---|
| primary (ε=0.05) | 12 | 600×N2048 + 300×N4096 | **≈ 29 h (~1.2 d)**, $0 (credits) | **≈ 5 h, ~$1–2** |
| + optional ε-sweep | +12 | +1200×N2048 | +~20 h (free) | +~3 h, +~$1 |

Reuses the hardened, checkpointed runner — same self-terminate + per-cell S3 sync + resume. Free-tier
is feasible but ~1.2 d; a paid 8-vCPU upgrade does the primary grid in ~5 h for ~$1–2 (well within the
$200 credits / $10 budget). **Your call which.**

## 9. Output / S3 structure (unchanged schema + anisotropy fields)

`s3://nbody-battery-<acct>/outputs/nbody_aws_anisotropy/` (rebuild bucket+role like before), one
dir per cell `cell_<family>_N<N>_eps<eps>/` with `rows.jsonl` (+ per-row baseline β, radial-KE
fraction), `cell_summary.json` (partial-corr table, dose-slope, equilibrium-gate), root
`manifest.json` + `verdict.json` + `equilibrium_gate.json`.

## 10. Stop-rule (fail-fast, after the first family)

Run the **hardest family first: hernquist3d radial** (N=2048 then 4096). If its **equilibrium gate
fails**, or **Δf_peri is not the top predictor** of ΔM (anisotropy summary wins, or its CI includes
0), **STOP and report** — do not run the remaining families. Radial anisotropy is the maximal
β/low-pericenter confound; if the handle wins there it wins everywhere.

## 11. Exact command (gated on approval — NOT run now)

```
# 0. (one-time) rebuild S3 bucket + IAM role (as before); implement anisotropic samplers + extended
#    predictor analysis in a generalized runner (nbody_aws_anisotropy_battery.py).
# 1. pre-flight equilibrium gate (cheap, local or 1 cloud cell):
python nbody_aws_anisotropy_battery.py --equilibrium-gate-only
# 2. stop-rule family first:
python nbody_aws_anisotropy_battery.py --families hernquist_radial --pairs 100
# 3. full primary grid on pass:
python nbody_aws_anisotropy_battery.py --pairs 100 --resume
```

## 12. New code required (gated on approval)

- **Anisotropic IC samplers:** Osipkov–Merritt (radial) + constant-β Cuddeford (tangential) for
  Hernquist & Plummer, with the §2 equilibrium gate. *(The one genuinely new physics-adjacent code;
  must be unit-validated against known β(r) profiles before any science run.)*
- **Extended predictor analysis:** add radial-KE fraction, baseline-M / baseline-β controls, and the
  multivariate partial-correlation table to `analyse_cell`.
- No change to the verified intervention/integration/pericenter core.

## 13. Alternative future batteries (described, NOT implemented)

**A. Clumpy / triaxial / rotating transfer.** Plummer/Hernquist with bound subclumps; bound merging
bimodal; rotating flattened; triaxial ellipsoid. Tests whether the handle survives realistic
asymmetry. Messier than anisotropy (no clean spherical pericenter / single centre) — needs a
per-structure centre and a revised pericenter definition. Do **after** anisotropy transfer.

**B. Central-sink / loss-cone-inspired toy.** Add a central absorbing radius; measure how
interventions change the **flux into the sink**. Connects to loss-cone language ("low-pericenter
intervention controls sink-feeding rate") — but becomes a *different paper* requiring careful
preregistration, proper relaxation/loss-cone-timescale framing, and the loss-cone literature. Scope
honestly as **toy sink-feeding, not galactic-nucleus loss-cone theory.** Do **last**, only after
anisotropy (and ideally clumpy) transfer.

---
**Decision requested:** approve (a) the anisotropy-transfer prereg as written, (b) free-tier
(~1.2 d, $0) vs paid 8-vCPU (~5 h, ~$1–2), or (c) revise families/criteria. No code is implemented
and no resources are provisioned until then.
