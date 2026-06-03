# N-body Coarse-Graining Scale Pilot — Science Plan

**Status:** Pilot complete. Local only. No AWS. `paper.tex` untouched.
**Date:** 2026-06-03

> **Pilot verdict: 🔴 RED — STOP.** Across hernquist / plummer / bimodal × ε ∈ {0.02, 0.05, 0.10},
> `R²(ℓ)` is flat (peak prominence 0.13× the bootstrap CI width), `ℓ*` does **not** increase with ε
> in the cusp families (Spearman 0.00 and −0.87), and once controlled for **bulk + C₈(t₀)** the
> scale-resolved features add ≈0 held-out R² in every cell (all CIs include 0) — for both ΔC8 and the
> absolute C₈(t₁) target. The force-resolution coarse-graining *law* is not supported; the prior
> result stands as a methods / stylized observable-class study. AWS is **not** justified.
> Full results: [`outputs/coarse_grain_pilot/pilot_report.md`](outputs/coarse_grain_pilot/pilot_report.md).

## 1. The hypothesis being tested

In softened self-gravitating N-body dynamics, the predictively optimal *description scale*
tracks the *force-resolution scale*. Concretely, with a smoothed feature family `φ_ℓ` built
from the t₀ particle distribution at coarse-graining scale `ℓ`, and a fixed future-structure
target,

```
ℓ*(ε) = argmax_ℓ R²(ℓ; ε)
```

should **increase monotonically with the gravitational softening ε**, ideally `ℓ* ≈ C·ε`.

This is a *new, scale-resolved* analysis. The existing paper compares fixed observable
*classes* (coarse vs fine, single scalars). It never sweeps a continuous smoothing scale,
never fits a multi-feature regressor, and never locates an optimal `ℓ`. The pilot asks the
cheapest possible question: **is `ℓ*(ε)` structured and does it move with ε?** If flat/noisy,
retire the claim. If alive, decide whether a full battery (and AWS) is justified.

---

## 2. What already exists (audit)

### 2.1 Simulation generators
| File | Role | Reused by pilot |
|---|---|---|
| `nbody_3d.py` | Core physics: force models, integrators, IC samplers, observables. Standalone utility. | **Yes** — forces, leapfrog, deposit, potential/KE |
| `nbody_stress.py` | Production battery runner + IC families (hernquist, bimodal) + the paper's observables + stats. Imports physics from `nbody_3d`. | **Yes** — `get_initial_conditions`, `_integrate_leapfrog`, all `obs_*` |
| `nbody_paper.py` | Figure/table/macro generation; orchestrates the 140k-run battery. | No (figures only) |
| `test_regression.py` | 36 regression tests. | Reference only |

**Force models:** `direct_isolated` (softened O(N²) direct summation, open boundary, Plummer
softening ε — Numba JIT), `direct_periodic`, `pm_periodic` (FFT Poisson, **ε-invariant by
construction**). Integrators: `leapfrog_kdk` (symplectic, used by paper), `rk4`.

**IC families:** `bimodal3d` (two-clump merger), `hernquist3d` (cusp), `plummer3d` (cusp),
`cold_clumpy3d` (multi-clump), `uniform3d`, plus `*_angshuf` angular-shuffle nulls.
Scale radius `plummer_a = 0.20`; box `L = 2.0`; ICs centred at `(L/2, L/2, L/2) = (1,1,1)`.

### 2.2 Saved simulation outputs
- **No particle snapshots are saved anywhere.** The battery computes scalar observables
  on-the-fly and discards particle data. `outputs/data/paper_battery.csv` (~53 MB) and
  `analysis.json` (~8 MB) are **gitignored and absent** from the working tree.
- Present in `outputs/`: 17 figure PDFs, 8 LaTeX tables, `convergence.json`,
  `exclusion_summary.json`, `paper_macros.tex`, `run_manifest.json`.
- **Implication:** the pilot must run its own simulations and cache its own t₀ snapshots.

### 2.3 Current feature extraction (single-scalar observables at t₀)
Coarse: `coarse_g4/g8/g16` (count variance on 4³/8³/16³ grids → cell sizes 0.5 / 0.25 / 0.125),
`coarse_conc` (concentration N(<r₅₀/2)/N(<r₅₀)), `coarse_rshell_var`.
Fine positional: `fine_knn_all` (k=16), `fine_pk_small`, `fine_close_pairs`, `fine_fof`.
Fine kinematic: `fine_vel_disp` (k=16). **No smoothing-scale sweep. No regression. No ML.**

### 2.4 Current targets
`TARGETS` in `nbody_stress.py`. **Primary = `ΔC8-early` = `coarse_g8(t_early) − coarse_g8(t₀)`**,
the change in 8³ density variance by the early snapshot (`H_EARLY = 100` steps). Also
ΔC8-mid/late, ΔC4/C16-early, ΔHMR, ΔConc, C8-final. The paper's headline coarse-dominance
result (|r| up to 0.984, bimodal) is on `ΔC8-early`. **The pilot adopts `ΔC8-early`.**

### 2.5 Train/test split logic
**None.** The paper uses cross-replicate **Pearson r + bootstrap CI** (single feature → target)
and a winner-gap bootstrap — not held-out predictive evaluation. The pilot **introduces** a
fixed-seed train/test split (70/30), identical across all `ℓ` and baselines within a cell, and
reports held-out **R²** (the quantity the hypothesis is stated in).

### 2.6 Softening, particle numbers, runtime
- **Softening grid:** ε ∈ {0.02, 0.03, 0.05, 0.07, 0.10}. With a=0.20 ⇒ ε/a ∈ {0.1,…,0.5}.
- **Particle numbers:** `direct_isolated` N ∈ {256, 512, 1024, 2048} (capped at 2048, O(N²));
  `pm_periodic` N ∈ {4096, 8192, 16384}.
- **Steps:** paper `PAPER_STEPS = 600`, `H_EARLY = 100`, `H_MID = 300`, dt = 0.005.
- **Measured per-sim runtime** (this machine, 10 cores, Numba, `direct_isolated`, 100 steps):
  N=256 → 10 ms, N=512 → 39 ms, **N=1024 → 158 ms**, scaling ∝ N² as expected.
  The `ΔC8-early` target only needs integration to step 100 (the step-100 state is identical
  whether or not integration continues to 600), giving a 6× saving over a full run.

---

## 3. What must be added

1. **`coarse_grain_features.py`** — the scale-resolved feature family `φ_ℓ` (Gaussian
   coarse-graining of the t₀ field at scale `ℓ`) plus the fixed baselines, all derived from a
   cached t₀ snapshot. Reuses the repo's `obs_*` functions for baselines.
2. **`coarse_grain_pilot.py`** — orchestrator with three cached stages:
   - **sim:** run `direct_isolated` to step 100, cache t₀ (pos, vel) + `ΔC8-early` per replicate.
   - **features:** deposit → smooth at each `ℓ` → `φ_ℓ`; compute baselines. Cached separately.
     **Re-running with a different `ℓ` grid never re-runs simulations.**
   - **analyze:** Ridge R²(ℓ; ε) on the fixed split, `ℓ*(ε)`, Δ_coarse, kill tests, figures, report.
3. **Train/test split, RidgeCV regression, R² + bootstrap CIs, `ℓ*` bootstrap** — none exist today.
4. **Config JSON** for reproducibility; tqdm progress bars; snapshot + feature caches.

### 3.1 Feature family `φ_ℓ` (positional coarse-graining at scale ℓ)
From t₀ particles: CIC-deposit onto a **G=96³** count field (cell = 0.0208; reuses
`_cic_deposit3`), Gaussian-smooth with kernel width `ℓ` (`scipy.ndimage.gaussian_filter`),
then extract a fixed ~15-D vector from the smoothed field `ρ_ℓ`, centred on the box centre:
- **radial mass profile** — `ρ_ℓ` averaged in 8 log-spaced radial shells (concentration at scale ℓ);
- **density-field moments** — std / skewness / kurtosis of `ρ_ℓ` (clumpiness at scale ℓ);
- **inertia tensor / shape** — 3 normalised eigenvalues + RMS size (low-order multipole/shape).
Velocities are cached but unused by the core positional `φ_ℓ`, so a **phase-space `φ_ℓ`**
extension needs no new simulations.

### 3.2 Baselines (fixed, ℓ-independent; each fit on the same split)
- `fine_pos` = [knn_all, pk_small, close_pairs, fof]; `fine_kin` = [vel_disp]; `fine_all` = both.
- `coarse_fixed` = [cg4, cg8, cg16, conc, rshell_var] (the paper's fixed-scale coarse class).
- `persistence` = [cg8₀] (does the current coarse value predict its own future change).
- `bulk` = [virial₀, KE₀, half-mass radius, N-in-box, RMS size, COM offset] — the **bulk control**
  for kill-test #4.

---

## 4. Exact pilot matrix

| Axis | Value | Rationale |
|---|---|---|
| Model | `direct_isolated` only | The **only** model where ε affects the force (PM is ε-invariant). |
| Families | `hernquist3d`, `plummer3d` (primary cusps) + `bimodal3d` (anchor/contrast) | Cusps are where the paper found ε matters; bimodal is a geometry-set negative control (ℓ* expected ε-independent). |
| Softening ε | {0.02, 0.05, 0.10} (low / med / high) | Spread across the existing grid for maximum ℓ* leverage. |
| Particle count N | **1024**, fixed across ε | Holding N fixed makes the mean interparticle spacing constant, so any ℓ* movement isolates ε from the sampling scale (∝ N^−1/3). |
| Steps | 100 (`H_EARLY`) | Exactly reproduces the paper's `ΔC8-early`; 6× cheaper than a full run. |
| Target | `ΔC8-early` | The paper's designated primary, strongest future-structure target. |
| Replicates / cell | **500** (seeds 2000…2499, paper convention) | 350 train / 150 test; bootstrap CIs on R² and ℓ*. |
| Smoothing scales ℓ (native) | {0.025, 0.04, 0.06, 0.09, 0.14, 0.20, 0.30} | ℓ/a ∈ {0.13, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5}; brackets ε (0.02–0.10) and extends to the paper's C8 cell (0.25). Floor set by the 96³ grid cell (0.021); the user's ℓ/a grid, adapted to repo units + grid resolution. |
| Regressor | RidgeCV (StandardScaler + LOO-CV α) | Simple first, per the plan; RF/HGB optional flags after Ridge. |

**Cells:** 3 families × 3 ε = **9 cells**; **4500 simulations** total.

---

## 5. Estimated runtime

| Quantity | Estimate |
|---|---|
| Per sim (N=1024, 100 steps, 1 core) | 158 ms (measured) |
| Simulation compute, 4500 sims | ≈ 711 CPU-s ≈ **12 min single-core**, **~1.5–2 min on 9 workers** |
| Feature extraction (4500 sims × 7 ℓ, 96³ smoothing) | **~4–6 min** |
| Ridge analysis + bootstrap + figures | **< 1 min** |
| **Total wall (pilot)** | **≈ 6–10 min** |

**Full-grid extrapolation** (for the AWS decision, *not run now*): all 4 IC families × 5 ε ×
direct N {256,512,1024,2048} × 500 reps × (early+mid+late) is dominated by N=2048
(~0.63 s/sim). Order ~10⁵–10⁶ sims but still ≈ tens of CPU-hours → **single-day local job on
9 workers with feature caching**. This sits *below* the "~2–3 local days" AWS threshold, so the
expectation is **AWS is not required even for the full scale battery** — the pilot will confirm
or revise this with a measured number.

---

## 6. Metrics & decision (Phases 2–3)

**Compute per (ε, ℓ):** held-out `R²(ℓ; ε)` (+bootstrap CI); per family `ℓ*(ε) = argmax_ℓ R²`
(+bootstrap distribution); `Δ_coarse = R²(ℓ*) − R²_fine`.

**Outputs:** `outputs/coarse_grain_pilot/results.csv`, `summary.json`, `config.json`,
heatmap `R²(ℓ, ε)`, line `ℓ* vs ε`, winner map (fine / coarse-small / coarse-large / tie),
runtime report, `pilot_report.md`.

**Kill tests:** (1) is `R²(ℓ)` structured or flat/noisy? (2) does `ℓ*` increase with ε
[Spearman + bootstrap P(increasing)]? (3) does the coarse optimum beat fine features
[Δ_coarse CI > 0]? (4) is it killed by bulk controls [does `φ_ℓ*` add R² beyond `bulk`]?
(5) is local runtime small enough, or does the full grid need AWS?

**Decision rule:**
- No structured/monotonic `ℓ*(ε)` → **stop**, retire the force-resolution-scale claim.
- Real but noisy → run a **medium local confirmation** (more reps / families / ε / a later horizon).
- AWS **only if** the pilot survives **and** simulations are the bottleneck **and** the full grid
  would exceed ~2–3 local days **and** feature caching is already implemented.

---

## 7. Implementation requirements (met by design)
tqdm progress bars; snapshot cache (`cache/snapshots/*.npz`) and feature cache
(`cache/features/*.npz`) kept separate; **changing `ℓ` recomputes features only, never
simulations**; `config.json` saved; models kept simple (Ridge first); `paper.tex` not touched.
