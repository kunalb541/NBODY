# N-body Causal-Intervention Pilot — Plan

**Status:** Plan + smallest pilot. No AWS. No huge grid. `paper.tex` untouched.
**Date:** 2026-06-04
**Context:** the two *correlational* coarse-scale hypotheses (ℓ*∼ε for ΔC₈; phase-space mixing
scale) are retired — both failed because **target-self baselines + bulk controls** explained the
apparent signal. The N-body branch stays **open**; we change the *class* of question.

## The new question — causal, not correlational

> If we **perturb** a fine / kinematic / coarse handle at t₀, does it **causally change** a future
> relaxation/clustering target?

This moves N-body from "predictability" to "causal handle." The design choice that kills the
baseline-sharing trap is a **matched pair**: same seed, same IC, the *only* difference is the
intervention. Effects are measured within-pair, so there is no initial-value to share.

## Handles (perturbations at t₀)

1. **Fine kinematic — compensated velocity-anisotropy rotation (FIRST PILOT).**
   Per particle, rotate the velocity **toward the radial direction** by a fixed angle θ, *in the
   plane spanned by* r̂ *and the current tangential direction*, **preserving the particle's speed**.
   - Decompose: `v_r = v·r̂`, `t̂ = (v − v_r r̂)/|v − v_r r̂|`, `v_t = v·t̂`, `φ = atan2(v_t, v_r)`.
   - New angle `φ' = max(φ − θ, 0)`; `v' = |v|(cos φ' · r̂ + sin φ' · t̂)`.
   - Purely-radial particles (`v_t ≈ 0`) are left unchanged.
   - Then subtract the mean velocity to restore total momentum ≈ 0.
2. Coarse geometry — shell-wise radial rescaling / concentration change (later).
3. Bimodal clump-separation perturbation (alternative first pilot).

**Sham (magnitude-matched control).** Rotate each particle's velocity by the **same angle θ** but
about a **uniformly random 3-D axis** (Rodrigues' formula), preserving speed. Same per-particle
speed and same `|Δv| = 2|v|sin(θ/2)`, but **no systematic anisotropy** — isolates the
*anisotropy-direction* of the intervention from a generic equal-magnitude velocity kick.

## Conserved / changed quantities (by construction)

| quantity | intervention | sham |
|---|---|---|
| per-particle speed → **KE, total E, virial Q** | **exactly preserved** | **exactly preserved** |
| positions → **PE, C₈(t₀), radial profile** | unchanged | unchanged |
| **anisotropy β(t₀)** | **raised** (radialised) | ≈ unchanged |
| angular momentum L | lowered (less tangential) | ≈ unchanged |
| total momentum | re-zeroed (tiny correction) | re-zeroed |

Because E and Q are preserved *by construction*, any effect on a future target **cannot** be a
bulk-energy/virial effect — kill-test #1 is satisfied structurally (and verified numerically).

## Targets (future, at horizon t₁)

future anisotropy **β(t₁)** (headline), future radial dispersion σ_r(t₁), future Q(t₁), future
ΔC₈, future coarse phase-space entropy. **First pilot target: β(t₁).**

## Matched-pair effects

For each seed *i* (same IC), run **three** simulations to t₁ — `orig`, `int`, `sham` — and record:
- imposed handle: `Δβ₀ = β_int(t₀) − β_orig(t₀)` (should be ≫ 0; sham ≈ 0) — confirms the handle.
- generic perturbation effect: `β_sham(t₁) − β_orig(t₁)`.
- intervention effect: `β_int(t₁) − β_orig(t₁)`.
- **causal handle (headline):** `E_i = β_int(t₁) − β_sham(t₁)` — paired across seeds, 95% CI by
  paired bootstrap. Also the **persistence ratio** `[β_int(t₁) − β_sham(t₁)] / Δβ₀`.

## First pilot (smallest)

- Family **Hernquist** (cusp relaxation); N = **1024**; horizon **t₁ = 600**; **100 matched pairs**
  (seeds 2000…2099); intervention angle **θ = 20°**; one handle, one target (β).
- Three runs per seed (orig/int/sham) = **300 integrations**.

## Runtime estimate

300 integrations to 600 steps at N=1024 ≈ 0.95 s each → ≈ **285 CPU-s ≈ <1 min on 9 workers**.
Trivial; entirely local.

## Kill tests

1. **Bulk-energy disguise** — if the effect needs ΔQ₀/ΔE, it's bulk. Here E, Q are preserved by
   construction (verified per run); a surviving effect is therefore *not* bulk.
2. **Sham equivalence** — if the sham produces the same future β (`E_i` CI includes 0), there is no
   causal anisotropy handle (relaxation erases it).
3. **Unphysical size** — if only large θ works, report θ-dependence; a handle that needs
   near-total radialisation is not useful.
4. **Below noise** — if `|E_i|` is smaller than the cross-pair scatter / paired-bootstrap CI, stop.

## What counts as ALIVE

The paired causal effect `E_i = β_int(t₁) − β_sham(t₁)` has a 95% CI **excluding 0**, with the
**same sign as the imposed Δβ₀** (the radialisation persists relative to sham), at a physical θ,
with E/Q verified uncontaminated. → *initial velocity anisotropy is a causal handle on future
anisotropy, independent of energy/virial.* Otherwise **null** (relaxation erases the handle) — a
clean, useful negative that we record and move on from.

## Deliverables

`nbody_intervention_pilot.py`; `outputs/nbody_intervention_pilot/{results.csv, summary.json,
figures, pilot_report.md}`. Report: paired effect size, CI, E/Q contamination, sham comparison,
alive/null verdict. Design is unambiguous → proceed directly to the smallest pilot.
