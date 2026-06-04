# N-body Coarse-Graining Branch — Decision Record

**Date:** 2026-06-03  **Status:** kill-test complete. No AWS. No paper rewrite. `paper.tex` untouched.

## Headline

> **Coarse-graining scale is not automatically the force-resolution scale.**

The pilot did not show "coarse-graining failed." It showed something sharper and more useful:
in this arena the *predictive description scale is set by the system's physical organization
(profile / clump geometry), not by the gravitational softening ε.* That is a real boundary for
the ODD program: observer scale is **not** simply matched to the numerical force scale; it
depends on the target and on how the system is organized.

## The claim that dies (precise scope)

> *Force resolution selects the predictive coarse-graining scale* — i.e. `ℓ*(ε) ∼ Cε`, or any
> landmark "coarse-graining-law" framing based on the early ΔC₈ / C₈(t₁) clustering target in
> this pilot matrix.

This specific claim is **retired**. Nothing broader is claimed dead.

## Earned / retained

- **Coarse summaries can strongly predict future clustering in structured systems**, especially
  bimodal / clump-like initial conditions (held-out R² up to ~0.64; fixed-scale coarse up to
  ~0.97–0.99).
- **Fine-vs-coarse predictive privilege remains target- and system-dependent.**
- **The original observable-class result is not killed by this pilot.** It stands as a
  methods / stylized observable-class comparison.

## Retired

- **`ℓ* ∼ ε`** / "force resolution selects the optimal predictive coarse-graining scale" for the
  tested early clustering target.
- **Any landmark coarse-graining-*law* framing based on ΔC₈.**

## Boundary learned

- In this arena, **predictive scale is tied more to profile / clump geometry than to force
  softening ε.** The cusp families (hernquist, plummer) — where ε should matter most — give the
  cleanest *null*; ℓ* sits near the profile scale (~a/2 ≈ 0.09–0.14), stationary or *decreasing*
  in ε. The only family with a rising ℓ* is bimodal, whose ℓ* is pinned at the clump-geometry
  scale and is ε-independent in any force-resolution sense.
- **The hardened control `bulk + C₈(t₀)` is decisive.** The apparent scale signal was mostly
  baseline / persistence structure: after the control, the scale-resolved features add ≈0
  held-out R² in **every** cell (all 95% CIs include 0). Random/trivial baselines match the
  coarse features.

## Evidence (early target ΔC₈, the paper's primary; full numbers in `outputs/coarse_grain_pilot/`)

| Test | Result |
|---|---|
| R²(ℓ) has a clear peak? | **No** — flat; median peak prominence 0.13× the bootstrap CI width |
| ℓ* increases with ε (cusps)? | **No** — Spearman(ε,ℓ*) = +0.00 (hernquist), −0.87 (plummer); P(strictly increasing) ≈ 0 |
| Coarse optimum beats fine? | **No** — Δ_coarse is negative for cusps (fine wins) |
| Survives bulk + C₈(t₀) control? | **No** — φ_ℓ* increment ≈ 0 in every cell, all CIs include 0 |
| Target-generic? | **No** — swapping ΔC₈ → absolute C₈(t₁) flips the cusp trend (−0.87 / −1.00) |
| AWS justified? | **No** — pilot is red; runtime trivial (4500 sims in ~150 s local) |

**Pilot configuration:** `direct_isolated` (the only model where ε affects the force) ×
{hernquist, plummer, bimodal} × ε∈{0.02,0.05,0.10} × N=1024 (fixed across ε) × 500 reps ×
7 smoothing scales; RidgeCV held-out R², fixed 70/30 split.

## Late-horizon closure (pre-registered rule)

Run only to seal the loophole cleanly — **not** as a rescue attempt. Rule:
- **If the late horizon also fails** (no ε-increasing, control-surviving ℓ* in cusps) → **branch
  closed**; the law is retired across early, mid, and late ΔC₈ targets.
- **If the late horizon shows something** → it becomes a **new target-specific hypothesis**, not a
  revival of the general force-resolution law.

### Result — 🔴 BRANCH CLOSED

**Validated.** Self-test passed (the `early_delta` re-run reproduces the primary ℓ* exactly:
hernquist [0.09,0.14,0.09], plummer [0.14,0.14,0.06]); φ_ℓ↔target alignment was 500/500
matched and 500/500 unique-keyed in every cusp cell (no row mispairing — the first attempt's
bug, caught and fixed).

**Cusp ℓ*(ε) and Spearman(ε, ℓ*) across horizons/targets** (the families where softening
should matter most):

| target | hernquist ℓ*(ε) | ρ | plummer ℓ*(ε) | ρ | φ beyond bulk+C₈(t₀) |
|---|---|---|---|---|---|
| early ΔC8 | [0.09, 0.14, 0.09] | +0.00 | [0.14, 0.14, 0.06] | −0.87 | none |
| mid ΔC8   | [0.09, 0.14, 0.06] | −0.50 | [0.09, 0.14, 0.09] | +0.00 | 1 cell* |
| late ΔC8  | [0.14, 0.09, 0.20] | +0.50 | [0.09, 0.14, 0.06] | −0.50 | none |
| late absC8| [0.30, 0.09, 0.14] | −0.50 | [0.14, 0.14, 0.30] | +0.87 | none |

The Spearman signs **scatter symmetrically around zero** (−0.87 … +0.87) with no consistent
direction — the signature of no real trend (a genuine ℓ*∼ε law would show consistently strong
positive ρ). The hardened control survives in exactly **1 of 24** cusp cells (hernquist, mid,
ε=0.10: +0.067 [+0.006, +0.123]) — ~4%, the expected 95%-CI false-positive rate, isolated
inside an otherwise-null, non-monotonic pattern. *Not a coherent signal.*

**The loophole is closed exactly where it had to be checked:** early / mid / late ΔC8 + absolute
C8, in the cusp families, under the bulk+C₈(t₀) control, on aligned matched samples. No AWS, no
larger smoothing grid, and no horizon rescue.

> **Branch status (recorded):** In the N-body branch, a force-resolution-matched coarse-graining
> law was tested and retired. Across early/mid/late ΔC8 and absolute C8, the optimal smoothing
> scale does not track softening in cusp families, and coarse features add no stable held-out
> signal beyond bulk + C₈(t₀). The surviving lesson is negative but useful: predictive scale in
> this setup is set by **system geometry / target structure, not by the numerical force-softening
> scale**.

**On bimodal (geometry anchor — not part of the cusp closure rule).** Bimodal *does* show a
consistent Spearman(ε, ℓ*) = **+0.87** across early/mid/late ΔC8. This must not be misread as a
revival of ℓ*∼ε, for two decisive reasons:
1. ℓ* is pinned at the **largest** smoothing scales (0.2–0.3 ≈ the clump-separation / clump-size
   geometry), **not** at ∼ε — the trend is "bigger systems prefer bigger ℓ," set by geometry.
2. `φ beyond bulk + C₈(t₀)` = **none** in every bimodal cell — even this trend adds no stable
   held-out signal once the trivial baselines are controlled.

So bimodal supports only the weaker, target/system claim — **"geometry scale matters in clumpy
systems"** — and **does not revive the force-resolution law ℓ*∼ε.**

## Disposition

- **No AWS.** **No rewrite.** This was a useful kill-test: a clean, credible negative that
  sharpens (rather than destroys) the ODD program's read of observer scale.

---

## Follow-up branch: phase-space coarse-graining — also retired (N-body FROZEN)

Plan: [`nbody_phase_space_coarse_graining_plan.md`](nbody_phase_space_coarse_graining_plan.md).
Result: [`outputs/phase_space_coarse_pilot/pilot_report.md`](outputs/phase_space_coarse_pilot/pilot_report.md).

The next physically-native place for coarse-graining (collisionless relaxation / phase mixing)
was tested: coarse phase-space features ψ_ℓ (radial-shell σ_r, σ_t, v_r, β + (r,v_r) occupancy
entropy at resolution ℓ) predicting future **mixing** (Δ comoving phase-space entropy, headline)
and ΔQ / Δσ_r / Δβ (secondary), in hernquist + plummer × ε{0.02,0.05,0.10} × N=1024 × 500 reps.

**Outcome — 🔴 STOP / FREEZE:**
- ψ_ℓ beats spatial φ_ℓ **6/6** on the headline — but that is the paper's *known kinematic /
  VelDisp advantage* (velocities carry more relaxation info than positions), not a new scale law.
- Raw "ψ_ℓ beyond bulk {Q₀,E,radial,C₈}" looked strong for Δβ / Δσ_r (4/6 cells) — **but it was
  baseline-sharing.** ψ_ℓ contains the initial β(r)/σ_r(r), and Δβ = β(t₁)−β(t₀) shares −β(t₀).
  Adding the **initial target quantity** to the control (β₀/σ_r₀/S₀) collapsed Δβ_late and
  Δσ_r_late from 4/6 → **0/6**. Same trap as C₈, caught by the same hardening.
- The clean mixing **headline** (Δ comoving phase-space entropy) survives the hardened control in
  only **1/6** cells (chance). Best partial survivor `dSigr_early` 3/6 — weak, secondary,
  early-horizon, below the majority bar.

**Boundary lesson (now general):** in this stylised setup, predictive structure is carried by
**bulk energy / virial / profile quantities and the initial value of the target**, *not* by a
coarse-graining scale — in **either** configuration space (ΔC₈) **or** phase space (mixing /
relaxation). Once baseline-sharing is properly controlled, no scale-resolved coarse-graining
description adds robust held-out predictive power.

**N-body program status: OPEN.** Two *specific correlational coarse-scale hypotheses* (spatial
clustering scale ℓ*∼ε; phase-space mixing scale) were tested and retired with hardened controls —
**but the N-body branch is not finished.** Both failed the same way (target-self baseline + bulk
controls explain the apparent signal), which tells us the *class* of question to stop repeating,
not to stop exploring. No AWS, no rewrite. The original observable-class paper stands as a
methods / stylised study; the coarse-graining *law* framing is not supported.

**Next direction (open):** an **interventional / causal-handle** test — if we perturb a fine /
kinematic / coarse handle at t₀, does it *causally* change a future relaxation/clustering target?
A matched-pair design (same seed, same IC, only the intervention differs) sidesteps the
baseline-sharing trap that killed the correlational tests. See
[`nbody_intervention_pilot_plan.md`](nbody_intervention_pilot_plan.md).
