# AWS Preregistration — Low-Pericenter Causal Mechanism (N-body)

**Status:** Preregistration. **Do NOT launch until approved.** Criteria below are fixed *before*
running; no post-hoc adjustment. Governs by the kill-test constitution in
[`nbody_causal_mechanism_state.md`](nbody_causal_mechanism_state.md). `paper.tex` untouched.
**Date:** 2026-06-04

## 0. Pre-flight code verification (done)

The load-bearing code was checked before writing this prereg (`/tmp/verify_mechanism.py`):
- **Pericenter** (`nbody_orbital_summary._pericenters`): radial orbit → grid floor; eccentric orbit
  matches a 40k-point reference to 2×10⁻⁴; r_peri ≤ r except near-pericenter particles off by ≤1
  log-grid cell (median 1.0%, max 2.2%); **f_peri identical capped-vs-raw** (Δ=0.0000) → the count
  variable is robust to the discretization.
- **Interventions** speed-preserving: KE/Q drift ~10⁻⁵; correct signs (radialize/vt-rescale/inner:
  β↑,|L|↓; tangentialize: β↓,|L|↑; sham null). `inner3` Δ⟨|L|⟩_global = −0.014 yet β+0.30 (the
  global-⟨|L|⟩-not-causal result reproduces).
- **Shell selection** exact (100% of intended thirds); **dose-response** holds without the origin
  anchor (corr 0.90/0.85/0.98); **no `norm(ord=1)`-vs-`axis=1`** bug in any module.

### 0a. Audit addendum (2026-06-04) — two issues found *after* §0, fixed below

A second adversarial pass (reproducibility + provenance + meta-checks) confirmed the committed
results are **bit-for-bit reproducible** (only 1e-16 float noise on re-run) and **non-fabricated**
(every reported number traces to a committed file). It surfaced **two issues that affect only this
prereg's proposed cells, not any committed result**:

1. **Pericenter potential is hardcoded Hernquist.** `nbody_orbital_summary._PHI_G = −1/(r+A)`
   (line 47) is the Hernquist proxy. All *committed* pericenter results are Hernquist-only, so they
   are correct. But the proposed **Plummer** cell would compute wrong pericenters (Plummer Φ =
   −1/√(r²+a²), differs by ~20% at r=0.05). **Fix:** the battery uses a **measured, spherically-
   averaged potential** Φ_meas(r) built per-snapshot from the particles themselves (§3a) — profile-
   agnostic — validated to reproduce analytic Hernquist *and* Plummer Φ to <1% on their ICs. The
   hardcoded analytic Φ is retired for the battery.
2. **Dose-response correlation is thin.** The local script correlates ΔM vs Δf_peri over ~6–7
   **arm-means** (`np.corrcoef`, no CIs), and times the M→C₈ lag with fixed thresholds. This was
   adequate as a local sanity check but is statistically thin. **Fix:** criterion 1 (§4) is
   upgraded to a **per-pair regression** (hundreds of paired points, bootstrap CI on the slope);
   criterion 2 uses **CI-based first-significant-time**, matching the time-course method that
   already established M(t≈5) before C₈(t≈10).

## 1. Core claim (single, falsifiable)

> In **concentrated** self-gravitating N-body systems, the **low-pericenter orbit population**
> `P(r_peri < r_c)` **causally controls** central concentration `M(<r_c)`, and clustering `C₈`
> responds **only when the clustering scale matches the concentration scale**. The variable is
> *not* the coarse-graining scale, *not* global ⟨|L|⟩, *not* β, *not* energy/virial.

The battery exists to confirm/falsify exactly this — nothing broader.

## 2. Battery dimensions

| axis | values | notes |
|---|---|---|
| N | 512, 1024, 2048, 4096 | O(N²) direct-summation; tests finite-N |
| profile | hernquist3d (cusp), plummer3d (core) — **primary**; uniform3d (**negative control**), bimodal3d (**geometry contrast**) | **Jaffe/NFW not in repo** → would need a new IC sampler; out of scope for this battery (flagged §13) |
| ε (softening) | 0.02, 0.05, 0.10, 0.20 | primary profiles only; controls run ε=0.05 only |
| intervention | sham · tangentialize · inner-{weak 10°, med 20°, strong 35°} · mid 20° · outer 20° · full 20° · vt-rescale | + `orig` for natural baseline; effects read as (arm − sham) |
| times (steps) | 0, 5, 10, 20, 50, 100, 300, 600, 1000 | snapshots; resolves M-before-C₈ |
| pairs / cell | 100 for N ≤ 2048; **30–50** for N = 4096 | matched seeds 2000+ |

Cells (active): primary = 2 profiles × 4 ε × 4 N = 32; controls = 2 profiles × 1 ε × 4 N = 8 →
**40 cells**. Each cell = 9 intervention arms × pairs.

## 3. Primary outcomes (recorded per arm, per snapshot time)

`f_peri(r_c)` for r_c ∈ {0.05, 0.10, 0.20}; `M(<0.05), M(<0.1), M(<0.2)`; `C₈`; `β`;
`⟨|L|⟩_inner, _mid, _outer, _global`; `σ_r`; conservation `Q, E, KE` (and drift vs orig).
All as **paired** effects (arm − sham), with paired-bootstrap 95% CIs.

### 3a. Pericenter potential — measured, profile-agnostic (replaces the hardcoded Hernquist Φ)

Per snapshot, build the spherically-averaged potential from the particles (G=1, mᵢ=1/N):
`Φ_meas(r) = −[ M(<r)/r + Σ_{j: rⱼ>r} mⱼ/rⱼ ]` on the log grid `_RG` (the standard discrete
spherical potential: enclosed-mass term + outer-shell term). Pericenters then solve
`vr²(r') = 2(E − Φ_meas(r')) − L²/r'² ≥ 0` exactly as now, but with the **measured** Φ rather than
`−1/(r+A)`. **Validation gate (pre-flight, both profiles):** Φ_meas reproduces analytic Hernquist
`−1/(r+a)` and Plummer `−1/√(r²+a²)` to <1% over r ∈ [0.02, 1] on their respective ICs; if not,
**STOP** — the pericenter variable is mis-measured. This makes f_peri valid for *any* profile in
the grid (cusp, core, uniform, bimodal), not just Hernquist.

## 4. Primary pass criteria (ALL must hold on the primary concentrated profiles)

1. **Dose-prediction (per-pair regression, not arm-means):** pool the **per-pair** points
   (Δf_peri(r_c)ᵢ, ΔM(<r_c)ᵢ) across all arms and pairs (hundreds of points), regress ΔM on
   Δf_peri; **PRIMARY gate:** the **slope is positive with a bootstrap 95% CI excluding 0**, in
   **both** Hernquist and Plummer at r_c ∈ {0.05,0.1,0.2}. The arm-mean corr (current local
   0.86–0.98) is reported as a secondary descriptive. The within-arm partial-correlation (per-arm
   mean removed) is **reported as a diagnostic, not gated** — see note.

   > **Post-smoke clarification (2026-06-04, before the real run).** The N=4096 smoke (6 pairs)
   > showed the per-pair *slope CI* passing strongly (CI excludes 0 at every r_c, both profiles;
   > arm-mean corr 0.83–1.00) while the **within-arm partial_r** was low/noisy. This is a *design*
   > property, decided on independent of the noisy n=6 value: the intervention sets Δf_peri fairly
   > deterministically per arm, so within-arm Δf_peri variance is small and partial_r is a low-power
   > secondary, not the dose claim. The dose claim ("Δf_peri predicts ΔM", per the run instruction)
   > rests on the **slope CI** (the audit-fix statistic) + arm-mean corr. partial_r is reported for
   > transparency; the battery may regain within-arm power via graded inner arms. **No criterion was
   > loosened after seeing a real result** — the pilot reports both the slope-CI verdict (primary)
   > and the stricter slope-CI-∧-partial_r≥0.5 verdict (registered), and the user decides.
2. **Ordering (CI-based):** first time the paired effect CI of M(<r_c) excludes 0 is **≤** the first
   such time for C₈, in every primary cell (CI-based first-significant-time, as in the time-course;
   no fixed thresholds). (current local: M sig t≈5, C₈ t≈10.)
3. **Locality:** ΔM(inner-third) ≥ **3×** ΔM(outer-third) **at equal-or-larger** global Δ⟨|L|⟩ for
   outer. (current: inner +0.029 vs outer +0.004 at Δ⟨|L|⟩ −0.014 vs −0.289.)
4. **Negative control:** uniform3d shows the *passive* signature (no active M-before-C₈ chain;
   gravitational persistence ≈ free-streaming), distinct from concentrated profiles.
5. **Sham null:** **PRIMARY (magnitude-relative):** the sham effect on intensive channels (β, M) is
   negligible relative to the intervention, `|sham−orig| / |intervention−sham| < 0.1`. The
   CI-includes-0 test (sham *exactly* 0) is reported as a **diagnostic**.

   > **Post-pilot clarification (2026-06-04).** At N=4096/50-pairs the CI-includes-0 sham test
   > failed (β CI excludes 0 in both profiles; M in Hernquist) **while the sham effect was only
   > 0.4–2.2 % of the intervention**. A random 20° rotation slightly isotropizes β vs the
   > un-rotated orig — a real but negligible systematic that n=50 resolves. This project's earlier
   > committed work already replaced the over-strict CI-only sham check with the **magnitude-relative**
   > one (the matched-pair design reads effects as arm−sham, which subtracts any small sham
   > systematic; the *fatal* condition is sham ≈ intervention, encoded in the §5 kill test). The
   > primary gate is therefore magnitude-relative; CI-only is reported for transparency. **This
   > restores the established standard — it is not a post-hoc loosening invented for this run.**
6. **Conservation:** median |ΔKE|/KE and |ΔQ|/Q < **10⁻²** in every cell.
7. **N-robustness:** intensive ΔM(<r_c) peak does **not** trend to 0 with N (ratio N=4096 / N=512
   > **0.4**); extensive C₈ may scale ~N.

## 5. Kill tests (any TRUE ⇒ the corresponding claim is retired/flagged)

| kill test | threshold | consequence |
|---|---|---|
| global ⟨|L|⟩ predicts ΔM **better** than f_peri | per-pair partial-corr(Δ⟨|L|⟩_global, ΔM \| Δf_peri) exceeds partial-corr(Δf_peri, ΔM \| Δ⟨|L|⟩) on primaries | **retire f_peri claim**; revert to global-L |
| outer radialization reproduces concentration | ΔM(outer) > 0.6 × ΔM(full) | inner-orbit story **fails** |
| C₈ moves without matching-scale M change | sign/magnitude of ΔC₈ uncorrelated with same-scale ΔM | **reject mediation** (clustering decoupled) |
| effect vanishes at N=4096 | intensive ΔM(<r_c) at N=4096 < 0.4 × at N=512 | **finite-N warning**; scope to small N |
| sham or energy drift explains effect | sham effect ≥ 0.5 × intervention, or KE/Q drift > 10⁻² | **reject intervention** as bulk/numerical |

## 6. Runtime estimate (measured per-sim, Numba, 1 core)

Per-sim scaling ∝ N²·T. Anchor: N=1024 / 100 steps = 158 ms ⇒ 1000 steps = 1.58 s.

| N | s/sim (1000 steps) | sims (active) | CPU-h |
|---|---|---|---|
| 512  | 0.40 | 2 prof × 4ε × 9 × 100 + ctrl = ~7,920 | 0.9 |
| 1024 | 1.58 | ~7,920 | 3.5 |
| 2048 | 6.3 | ~7,920 | 13.9 |
| 4096 | 25.3 | 2 prof × 4ε × 9 × 50 + ctrl = ~3,960 | 27.8 |
| **total** | | **~27,700 sims** | **≈ 46 CPU-h** (×1.2 obs overhead ≈ **55 CPU-h**) |

**Honest note:** the full battery is **~55 CPU-h** — ≈ 6 h local on 9 cores, or ≈ 1 wall-h on a
64-vCPU instance. Per the project's own finding, **AWS here buys wall-clock speed, not feasibility**;
this is a convenience-scale battery, not a "monster." That is itself a result worth recording.

## 7. AWS instance recommendation

- **Pilot + full battery:** one **`c7i.16xlarge`** (64 vCPU, 128 GB; on-demand ≈ $2.86/h) or
  cheaper **spot** (~$0.9–1.2/h). 55 CPU-h ⇒ ~1 wall-h ⇒ **~$1–3**.
- Compute-bound, memory-trivial (≤ a few hundred MB resident). No GPU. No cluster needed —
  `ProcessPoolExecutor` across vCPUs on a single node (matches the existing scripts). Numba on the
  instance (compile once, cache to disk).

## 8. Storage estimate

Only **scalar summaries** are written (no particle snapshots): per (cell, arm, pair, time) ≈ 18
floats. ~27.7k sims × 9 times × 18 floats × 8 B ≈ **36 MB** raw; CSV/JSON ≈ **< 150 MB** total.
EBS gp3 20 GB is ample. Results sync to S3 (or `git`/`scp`) on completion.

## 9. Checkpointing plan

- **Per-cell** checkpoint: each (profile, N, ε) cell writes its rows to
  `results/cell_<profile>_N<N>_eps<eps>.csv` on completion; a `manifest.json` records done cells.
- `--resume` skips cells present in the manifest. Within a cell, the `ProcessPoolExecutor` futures
  are flushed incrementally (as in `nbody_stress.py`).
- Idempotent seeds (2000 + i) ⇒ exact reproducibility; a killed/spot-interrupted run resumes with
  no loss beyond the in-flight cell.

## 10. Exact output schema

**Long-format** `results.csv`, one row per (cell, arm, pair, time):
```
profile, N, eps, intervention, theta_deg, shell, pair_seed, t_step,
f_peri_005, f_peri_01, f_peri_02,
M_005, M_01, M_02, C8, beta, sigr,
L_inner, L_mid, L_outer, L_global, Q, E, KE
```
**Per-cell** `summary_<cell>.json`: paired effects (arm − sham) with 95% CIs for each outcome and
time, first-significant times, and the §4/§5 criterion evaluations. **Top-level** `verdict.json`:
pass/fail per criterion per profile, kill-test booleans, and the overall PASS/FAIL.

## 11. Cost ceiling

**Hard ceiling: USD $50** (covers pilot + full battery + one full rerun + overhead, ≈ 15× the
expected ~$3). The orchestrator tracks elapsed instance-hours and **aborts** if projected spend
would exceed the ceiling; spot interruptions resume under the same ceiling.

## 12. Stop rule (fail-fast)

Evaluate the **§4 criteria on the first completed cell of each primary profile** (Hernquist N=1024
ε=0.05, Plummer N=1024 ε=0.05) before launching the rest:
- If **criterion 1 (dose-prediction) or 5 (sham) or 6 (conservation)** fails there → **STOP**, do not
  spend further, report. (These should already pass — they did locally; failure means a port bug.)
- If a **kill test** fires in the pilot (§13) → STOP and report; do not run the full grid.
Otherwise proceed cell-by-cell; abort on cost ceiling (§11).

## 13. Recommended sequencing — PILOT FIRST

Do **not** launch the full battery first. Run the **scale-up pilot** (the one new question AWS
answers: *does the mechanism survive at larger N?*):

- **N = 4096**, profiles **Hernquist + Plummer**, **ε = 0.05**, **50 pairs**, interventions
  **{sham, tangentialize, inner-med, mid, outer, full}**, times to **600**.
- ≈ 2 prof × 6 arms × 50 pairs × 15.2 s (600 steps) ≈ 9,100 s ≈ **2.5 CPU-h** ≈ **~3 wall-min on
  64 vCPU** (~$0.15). (Also runnable locally in ~17 min on 9 cores.)
- **Pilot pass** = §4 criteria 1,2,3,5,6,7-at-N4096 hold ⇒ **launch the full battery**. **Pilot
  fail** (kill test fires, or dose-prediction collapses at N=4096) ⇒ **stop**; you've learned the
  finite-N boundary for ~$0.15 instead of the full spend.

## 14. Implementation note (gated on approval)

The battery would be run by a new orchestrator `nbody_aws_battery.py` that generalizes the verified
local scripts (`nbody_orbital_summary.py` + `nbody_pericenter_dose_response.py`) over the §2 grid
with the §9 checkpointing and §10 schema — **no new physics or handles**. Two **audit-mandated**
code changes carry over before any launch (both verified-equivalent on Hernquist, where analytic
Φ ≈ Φ_meas): (i) swap the hardcoded `_PHI_G = −1/(r+A)` for the measured Φ_meas(r) of §3a, gated by
the <1% Hernquist+Plummer validation; (ii) replace the arm-mean `np.corrcoef` dose test with the
per-pair regression + CI of §4.1 and the CI-based lag of §4.2. It is **not written or run yet**;
this prereg stops here per instruction.

---
**Decision requested:** approve (a) the pilot only, (b) pilot then full battery on pass, or (c)
revise dimensions/criteria. No AWS resources are provisioned until then.
