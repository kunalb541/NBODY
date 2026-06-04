# N-body Phase-Space Coarse-Graining Pilot — Science Plan

**Status:** Pilot complete. No AWS. `paper.tex` untouched.
**Date:** 2026-06-04
**Predecessor:** the spatial-clustering branch (`ℓ*∼ε` for ΔC₈) was tested and **retired**
([`nbody_coarse_graining_branch_notes.md`](nbody_coarse_graining_branch_notes.md)).

> **Pilot verdict: 🔴 STOP / FREEZE.** Coarse phase-space ψ_ℓ predicts relaxation targets far
> better than spatial φ_ℓ (6/6 cells) — but that is the *known kinematic advantage*, not a new
> scale law. The raw "ψ beyond bulk" looked strong for Δβ/Δσ_r (4/6), but **collapsed to 0/6 under
> the hardened control** (bulk + the initial target quantity β₀/σ_r₀) — pure baseline-sharing, the
> same C₈ trap. The clean mixing headline (Δ comoving phase-space entropy) survives the hardened
> control in only **1/6** cells. Only residue: `dSigr_early` 3/6 (weak, secondary). **N-body is
> frozen as a boundary result.** Full results: [`outputs/phase_space_coarse_pilot/pilot_report.md`](outputs/phase_space_coarse_pilot/pilot_report.md).

> **This is a new branch, not a C₈ rescue.** It moves coarse-graining to where it is physically
> native in collisionless dynamics — **phase space** — and changes the target from spatial
> clustering (C₈) to **collisionless relaxation / phase mixing**. The retired law is not reopened;
> `ℓ*∼ε` appears here only as a *secondary* diagnostic, never the headline.

---

## 1. Why phase space, and why now

The retired branch smoothed *configuration-space* clustering features and asked whether the optimal
spatial smoothing scale tracks the force softening ε for the ΔC₈ target. It failed cleanly: across
early/mid/late ΔC₈ and absolute C₈, coarse spatial features added **no** held-out signal beyond
bulk + C₈(t₀), and ℓ* did not track ε in cusp families.

The literature points to a different, physically-native home for coarse-graining:

- **Barnes (2012)** — gravitational softening is itself a *smoothing of the mass distribution*, so
  coarse-graining is physically meaningful, but it is not automatically predictive for a spatial
  clustering target (consistent with our null).
- **Halle, Colombi, Peirani (2017/2019)** — collisionless self-gravitating systems develop fine
  phase-space structure during violent relaxation, while **coarse-grained phase-space quantities
  remain meaningful** even when fine spiral structure is scrambled. → coarse-graining is native in
  *phase space*, not configuration space.
- **Giachetti & Casetti (2019)** — coarse-grained collisionless (Vlasov / long-range) dynamics
  studied explicitly as a function of the **coarse-graining scale**, including self-gravitating
  models. → a scale-resolved phase-space description is literature-standard.
- **Lucie-Smith et al. (2018)** — ML structure-formation work uses **predictive performance under
  different input features** to infer which physical information drives outcomes. → a
  predictability-based diagnostic is literature-compatible when framed as feature/information
  sufficiency, not as a universal law.

**Central hypothesis.** In softened self-gravitating *collisionless relaxation*, the predictively
useful description scale lives in **coarse phase-space variables**. Concretely: a coarse
phase-space feature family `ψ_ℓ` (radial-shell kinematics + coarse (r, v_r) occupancy at resolution
ℓ) predicts **future relaxation/mixing** better than fine descriptions or purely spatial coarse
features `φ_ℓ`, **and adds held-out signal beyond bulk controls** (Q₀, E, radial profile, C₈(t₀)).

**Headline metric:** `ΔR²(ψ_ℓ)` beyond bulk controls, and beyond spatial `φ_ℓ`.
**Secondary only:** `ℓ*(ε)`. A positive `ℓ*(ε)` trend would be interpreted as phase-space
description scale, **never** as a revival of the spatial `ℓ*∼ε` law.

---

## 2. Phase-0 audit — what exists (done)

| Asset | Status | Consequence |
|---|---|---|
| t₀ **positions** (500×1024×3) per cell | **Cached** (`cache/snapshots/*.npz`) | spatial `φ_ℓ` free |
| t₀ **velocities** (500×1024×3) per cell | **Cached** (mean \|v\|≈2.06, non-trivial) | **`ψ_ℓ` phase-space features free — no re-sim** |
| seeds, ΔC₈-early, cg8₀ | Cached | reproducible; C₈(t₀) bulk control free |
| **late-time (pos, vel) at t₁** | **NOT cached** | relaxation **targets** require one re-integration pass |
| Q, β, σ_r, phase-space entropy (any time) | **Never computed** | new feature/target code needed |
| Engine: forces, leapfrog, ICs (hernquist/plummer) | In `nbody_3d.py` / `nbody_stress.py` | reuse verbatim |
| RidgeCV R² + bootstrap + `_scale_fit` + split | In `coarse_grain_pilot.py` | reuse verbatim |

**Bottom line:** features (`ψ_ℓ`, `φ_ℓ`, baselines) are computable **from cache with no
simulation**; only the relaxation **targets** need a re-integration of the cached t₀ snapshots to
the horizon(s) — the same cheap operation the late-horizon closure already performed.

---

## 3. What must be added

1. **`phase_space_features.py`** — the coarse phase-space family `ψ_ℓ` + the relaxation observables
   used for targets. Pure functions on a (pos, vel) snapshot. Reuses COM/centering from the repo.
2. **`phase_space_pilot.py`** — orchestrator with cached stages:
   - **targets:** re-integrate cached t₀ snapshots to {early=100, late=600}; at each horizon compute
     the relaxation observables (Q, σ_r-profile, β, phase-space entropy). Cache target scalars.
   - **features:** `ψ_ℓ` (all scales) + spatial `φ_ℓ` (reuse `coarse_grain_features`) + baselines,
     from cached t₀. Cached separately. Changing ℓ never re-integrates.
   - **analyze:** RidgeCV held-out R² per (target, predictor set, ℓ); `ΔR²` beyond bulk & beyond `φ`.

### 3.1 Phase-space coarse features `ψ_ℓ` (radial resolution ℓ)
From t₀ (pos, vel), centred on the COM, with `r`, radial velocity `v_r = v·r̂`, tangential
`v_t = |v − v_r r̂|`. ℓ is the **radial coarse-graining scale** (shell width). For each ℓ:
- **radial-shell kinematic moments** (shells of width ∼ℓ): `σ_r(r)`, `σ_t(r)`, mean `v_r(r)`
  (net infall/expansion) — the collisionless-relaxation-relevant moments;
- **anisotropy profile** `β(r) = 1 − σ_t²/(2σ_r²)` per shell;
- **coarse (r, v_r) occupancy**: 2-D histogram (r-bins width ℓ × v_r-bins scaled to the global
  dispersion), reduced to low-order moments + **occupancy (Shannon) entropy** — the coarse
  phase-space density and its mixing state;
- **optional** specific angular-momentum `|L|(r)` per shell.
ℓ enters only through the radial bin width, so one snapshot yields `ψ_ℓ` at every scale with no
re-integration. (Variant for later: a coupled 2-D (r, v_r) scale rather than radial-only.)

### 3.2 Comparison predictor sets (the actual science is the *contrast*)
- `fine_pos` — spatial fine (kNN, Pₖ, close-pairs, FoF) [reuse].
- `fine_kin` — fine kinematic (local velocity dispersion, + richer velocity stats) [reuse + extend].
- `spatial_coarse φ_ℓ` — the retired branch's smoothed-density features [reuse] — included so we can
  show phase space adds something **spatial coarse-graining cannot**.
- `phase_coarse ψ_ℓ` — **new**.
- `bulk_control` — **Q₀ = 2K/|U|, total energy E, coarse radial density profile, C₈(t₀)** — the
  trivial predictors. This is the decisive control: relaxation/virialization is partly fixed by the
  initial virial ratio and energy, so `ψ_ℓ` must beat these to be a real result.

### 3.3 Targets (future relaxation / mixing), at horizon t₁
- **Primary: `ΔQ = Q(t₁) − Q(t₀)`**, `Q = 2K/|U|` — virialization, a canonical robust scalar.
- Secondary: **Δ phase-space entropy** (coarse f(r,v_r) — mixing); **Δσ_r profile** (central/mean
  radial dispersion — relaxation); **Δβ(r)** (anisotropy); **mixing index** from binned (r, v_r)
  (coarse-grained DF flattening / occupancy spread).
Horizons: **early (100)** and **late (600)**. Targets need K and U at t₁ (PE is the existing
O(N²) `potential_energy_direct`, exact for `direct_isolated`).

---

## 4. Exact pilot matrix (tiny, local)

| Axis | Value | Rationale |
|---|---|---|
| Model | `direct_isolated` | exact PE → exact Q; the physically-clean model |
| Families | `hernquist3d`, `plummer3d` (**cusps only**) | where relaxation/softening physics is sharpest; bimodal is a merger, not a relaxation test |
| Softening ε | {0.02, 0.05, 0.10} | reuse; ℓ* vs ε is secondary |
| N | 1024 (fixed) | reuse cached t₀; isolate ε from sampling scale |
| Replicates | 300–500 (seeds 2000…) | reuse cached t₀ snapshots exactly |
| Horizons | early (100) + late (600) | relaxation is a late-time phenomenon → late matters here |
| ℓ grid | radial resolutions spanning core→envelope (e.g. shells from ~0.03 to ~0.6) | secondary ℓ* diagnostic |
| Regressor | RidgeCV, fixed split | reuse pilot machinery; RF/HGB only after Ridge |

**Cells:** 2 families × 3 ε = 6 cells. Targets re-integrated once per (cell, rep).

---

## 5. Metrics & kill-tests

**Per (target, ε):** held-out R² for each predictor set; **`ΔR²(ψ_ℓ*)` beyond `bulk_control`**
(primary metric, with bootstrap CI); **`ΔR²(ψ_ℓ*)` beyond `spatial φ_ℓ*`**; ℓ*(ε) (secondary).

**Kill-tests (stringent — designed to prevent a disguised rescue):**
1. **No info beyond bulk** → if `ψ_ℓ` adds no `ΔR²` beyond {Q₀, E, radial profile, C₈(t₀)} (CI
   includes 0) for every target → **STOP**.
2. **Spatial matches phase-space** → if `ΔR²(ψ beyond φ)` CI includes 0 → no genuinely
   *phase-space* result (it was spatial all along).
3. **Single-target** → if only one relaxation target shows the effect → label **target-specific**,
   not a general phase-space coarse-graining result.
4. **ℓ* secondary** → an `ℓ*(ε)` trend is reported but **never** used to claim revival of `ℓ*∼ε`.

**Decision rule:** STOP and **freeze N-body as a boundary result** if kill-test 1 or 2 fires across
targets. If `ψ_ℓ` adds robust `ΔR²` beyond bulk **and** beyond spatial `φ_ℓ` on ≥1 canonical target
(ΔQ or phase-space entropy), the branch is **alive** → medium local confirmation (more reps /
horizons / targets), still **no AWS** unless a later, much larger grid becomes the bottleneck.

---

## 6. Estimated runtime (local, no AWS)

| Stage | Estimate | Note |
|---|---|---|
| Targets: re-integrate cusps to early(100) | ~1 min | 6 cells × ~500 reps × 33 ms/sim wall (measured) |
| Targets: re-integrate cusps to late(600) | **~6–8 min** | ~0.95 s/sim; the dominant cost (measured via the closure) |
| Features `ψ_ℓ` + `φ_ℓ` + baselines (from cached t₀) | ~2–4 min | phase-space binning is cheaper than 96³ smoothing |
| Analysis (RidgeCV + bootstrap) | < 1 min | reuse pilot machinery |
| **Total pilot** | **~10–15 min** | cusp-only; entirely local |

**Simulations are the only re-run cost, and they are minutes.** AWS is not in scope for the pilot,
and a later full grid (4 families × 5 ε × N up to 2048 × multiple horizons) still extrapolates to a
single local day with feature caching — below the ~2–3 day AWS threshold.

---

## 7. Is this genuinely new, or a C₈ rescue? (honest assessment)

**Genuinely new — three independent ways it differs from the retired branch:**
1. **Different feature space** — `ψ_ℓ` is built from **velocities / phase space** (σ_r, β, (r,v_r)
   occupancy, phase-space entropy); the retired `φ_ℓ` used positions only.
2. **Different target** — **relaxation / mixing / virialization** (ΔQ, Δβ, Δσ_r, Δentropy), not
   spatial clustering ΔC₈.
3. **Different physics** — violent relaxation and phase mixing, the native home of coarse-grained
   distribution functions, vs configuration-space clumping.

**But the honest risk (stated up front):** the *headline* targets — especially ΔQ — may be largely
determined by **bulk** quantities (Q₀, E). Virialization is strongly set by the initial virial
ratio and energy, so it is entirely possible that `ψ_ℓ` adds little beyond the bulk control and this
branch **also closes**. That is exactly what kill-test 1 is for. The phase-space entropy / mixing
targets are the ones most likely to carry genuine scale-resolved phase-space information beyond
bulk — they are the real test of the hypothesis. We will not over-claim: if bulk + fine-kinematic
features already saturate predictability, the result is "phase-space coarse-graining adds nothing
predictive here," and N-body freezes as a boundary result.

**This is the one remaining N-body branch worth a tiny pilot.** If it fails, freeze. If it succeeds,
the N-body story connects to collisionless dynamics, violent relaxation, and coarse-grained DFs —
real astronomy, not internal observer-scale language.

---

## 8. Deliverables (this step) and the stop point

- **This file** (`nbody_phase_space_coarse_graining_plan.md`).
- **Stop and report** (below) — **no pilot run until approved**:
  1. cached snapshots/velocities exist? **Yes** (t₀ pos+vel, all cells); late-time **not** cached.
  2. estimated runtime? **~10–15 min local**, cusp-only.
  3. exact features/targets? **§3.1 / §3.3** above.
  4. genuinely new or rescue? **Genuinely new** (§7), with the bulk-control risk stated honestly.
