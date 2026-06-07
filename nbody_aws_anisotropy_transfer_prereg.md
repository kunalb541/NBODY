# AWS Preregistration — Anisotropy-Transfer Battery (N-body), v2

**Status:** Preregistration. **Do NOT launch until approved.** Criteria fixed *before* running; no
post-hoc adjustment. `paper.tex` untouched. **Date:** 2026-06-07.
**v2 changes** (grounded by a literature-verification workflow, run 2026-06-07): radial IC moved off
the radial-orbit-instability boundary (`r_a = 1.5a`, not `a`); **Stage 0** validation gate added with
a shape/ROI check; multivariate + cluster-bootstrap analysis specified; ξ=2T_r/T_t competitor added;
confound guards mandated. References in §13.

## 0. Motivation — the transfer question

The low-pericenter handle is earned for **isotropic** Hernquist/Plummer (battery-positive, N-robust to
4096, controls negative; `outputs/nbody_aws_battery/`, independently re-audited to machine precision).
Hardest fair test of the claim that central response is driven by **low-pericenter accessibility**, not
anisotropy:

> **Does Δf_peri(<0.1) remain the best predictor of ΔM(<0.1) when baseline orbital anisotropy is
> deliberately varied (isotropic / radial / tangential), after controlling for the anisotropy
> summaries and baseline structure?**

## 1. Core claim (single, falsifiable)

> Across isotropic, radially-anisotropic, and tangentially-anisotropic Hernquist and Plummer
> equilibria, the intervention-induced low-pericenter dose Δf_peri(<0.1) remains the top predictor of
> ΔM(<0.1) — beating global ⟨|L|⟩, β(r), the Polyachenko–Shukhman ratio ξ=2T_r/T_t, radial kinetic
> fraction, baseline central mass, and outer-shell response — **within the radial and tangential
> families**, not only isotropic.

## 2. Initial conditions — six families (the experiment's MAIN RISK)

{hernquist3d, plummer3d} × {isotropic, radial, tangential}. Construction (preregistered, fixed;
feasibility literature-verified):

- **Isotropic:** existing Eddington-inversion samplers (as in the committed battery).
- **Radial (Osipkov–Merritt):** β(r)=r²/(r²+r_a²). **r_a = 1.5a = 0.30 (primary).** Analytic OM DF for
  Hernquist (Hernquist 1990; Baes & Dejonghe 2002); Plummer via Eddington–OM (analytic, Breen, Varri
  & Heggie 2017). *DF non-negativity floors* (r_a,min ≈ 0.20a Hernquist; 0.75a Plummer) are easily
  satisfied. **The binding constraint is the radial-orbit instability:** r_a,crit ≈ 1.0a for Hernquist
  (Meza & Zamorano 1997; ξ_crit=2.31±0.27); **r_a=a is ON the boundary and is rejected.** r_a=1.5a is
  comfortably ROI-stable. (Honest note: at r_a=1.5a the radial bias lives mostly in the envelope — β is
  mild inside <0.1 — so the *inner* anisotropy contrast comes chiefly from the tangential families;
  inner β(r)/f_peri are measured and reported per family, §5.)
- **Tangential (constant-β):** β(r) = **−0.5 constant** at all radii. Feasible for both: Hernquist
  needs β≤½ (An & Evans 2006), Plummer (core) needs β≤0 (An & Evans 2005; β=−0.5 allowed, β>0
  impossible); no lower bound (Baes & van Hese; Cuddeford 1991). Provides the strong inner-anisotropy
  contrast.
- **Softening caveat (preregistered):** ICs are sampled from the *unsoftened* DF but evolved in the
  *softened* (ε=0.05, ε/a=0.25) potential. We therefore (i) exclude r < 2ε = 0.10 from β/density
  validation and from inner-anisotropy claims, and (ii) Stage 0 verifies the sampled IC is a quiet
  equilibrium *in the softened potential it will actually be integrated in* (the equilibrium-hold test).

## 2a. STAGE 0 — anisotropic IC validation gate (mandatory; the load-bearing addition)

Run **before any science battery**, **at N=2048, ε=0.05 first** (cheapest place to expose
construction/ROI errors), for each new family {Hernquist-radial, Hernquist-tangential, Plummer-radial,
Plummer-tangential}. Eight gates:

1. **Density fidelity.** Binned M(<r) and ρ(r) match the intended Hernquist/Plummer to Poisson error:
   every fit bin |z|<3, KS p>0.05 vs the analytic CDF.
2. **β(r) sign/profile.** Binned β(r)=1−σ_t²/2σ_r² over resolved shells r∈[2ε,5a]=[0.10,1.0] traces the
   target (radial: r²/(r²+r_a²); tangential: −0.5): ≥80% of resolved shells within tolerance.
3. **Virial at t₀.** 0.90 ≤ 2T/|W| ≤ 1.10 with the *softened* W.
4. **Baseline M(<0.1) stationary.** Integrate the un-intervened orig arm to the full 1000-step horizon;
   median_t |ΔM(<0.1;t)|/M < 0.10 (matches the committed battery's gate).
5. **Baseline β(r) stationary.** |Δ⟨β⟩_global(t)| < 0.05 over the horizon (no anisotropy washout).
6. **Shape / ROI gate (NEW — catches bar formation).** On the un-intervened run, track inertia-tensor
   axis ratios c/a, b/a and the inner m=2 quadrupole; require **c/a, b/a stay > 0.9** and Lagrangian
   radii (10/50/90%) drift < 0.10. *(The drift-only gates 4–5 alone do NOT catch an ROI bar — this is
   why r_a=a was rejected.)*
7. **Anchor reproduction.** The generalized runner in *isotropic* mode reproduces the committed-battery
   anchors (Hernquist base f_peri(<0.1)≈0.34, peak ΔM10≈0.027; Plummer ≈0.36, ≈0.056) to <2% at
   matched seeds — proves the refactor didn't perturb the verified core.
8. **Tangential f_peri power.** Constant-β=−0.5 thins the inner plunging tail; confirm f_peri(<0.1)
   per-pair Poisson noise is small enough to resolve the dose at N=2048/4096; else raise N for those.

**Hard-gate logic:** any family failing 1–6 is **excluded** from the science battery until fixed; if
**Hernquist-radial fails Stage 0, stop the whole transfer battery** (its construction is the riskiest).
Gate 7 failing ⇒ a refactor bug ⇒ stop and fix before anything.

## 3. Battery grid

6 families × N∈{2048, 4096} × ε=0.05 (primary) = **12 cells**; ε∈{0.02,0.10} at N=2048 optional. Arms:
**orig, sham, tangentialize, inner-{weak,med,strong}, mid, outer, full, vt-rescale, and the L-matched
β-null arm — all MANDATED in every family** (not "reuse") so within-family f_peri/β decorrelation is
testable (§7 confound). Times preregistered {0,5,10,20,50,100,300,600,1000}. Pairs 100 (N=2048), 50
(N=4096), seeds 2000+.

## 4. Target, predictor, competitors

- **Target:** ΔM(<0.1) (peak paired effect, arm − sham).
- **Primary predictor:** Δf_peri(<0.1) at t₀.
- **Competitors (Δ = arm−sham):** global ⟨|L|⟩; β(r) (Δβ, inner & outer); **ξ = 2T_r/T_t
  (Polyachenko–Shukhman; the true ROI/anisotropy order parameter)**; radial kinetic fraction
  σ_r²/(σ_r²+σ_t²); outer-shell response ΔM(outer third). **Baseline controls:** baseline M(<0.1),
  baseline β, **family fixed effects**.

## 5. Analysis — multivariate competition (upgraded per referee)

- **Multivariate partial correlation (not the 1-control `_pcorr`).** Residualize ΔM and Δf_peri on the
  **full covariate matrix** Z = [Δ⟨|L|⟩, Δβ, Δξ, Δradial-KE-frac, Δouter, baseline M, baseline β,
  family dummies] by multiple regression, then correlate residuals. Primary statistic: partial-corr
  (Δf_peri, ΔM | Z) > each competitor's partial-corr (controlling for Δf_peri + rest).
- **Cluster bootstrap over pairs** (resample matched pairs, keeping all arms together) for CIs — the 8
  arm-points per pair are not independent (shared sham, near-constant within-arm dose). No naive
  per-point CIs.
- **Within-family AND pooled.** The within-family result (radial; tangential) is load-bearing; the
  pooled fit carries family dummies.
- **Report** inner β(r) and f_peri per family at t₀ (document the actual inner-anisotropy contrast),
  plus the per-family dose×slope decomposition and the Stage-0 diagnostics.

## 6. Pass criteria (ALL)

1. Δf_peri is the **top** predictor of ΔM (largest multivariate partial-corr, cluster-bootstrap CI
   excludes 0) **within the radial family AND the tangential family** (not only isotropic), in both
   profiles.
2. Ordering M-before-C₈ (CI-based); sham magnitude-relative null (|sham|/|intervention|<0.1).
3. KE injection ≈0; integrator |ΔE|/E < 10⁻².
4. **Stage 0 passed** for every reported family.

## 7. Kill tests (any TRUE ⇒ mechanism incomplete / scope retracted)

| kill test | threshold | consequence |
|---|---|---|
| anisotropy summary beats f_peri | multivariate partial-corr(β or ξ or radial-KE, ΔM \| rest) > partial-corr(Δf_peri, ΔM \| rest) | **mechanism incomplete** — anisotropy is load-bearing under varied equilibria |
| no transfer | Δf_peri partial-corr CI includes 0 in the radial **or** tangential family | scope stays isotropic |
| confound unbroken | within a family, Δf_peri and Δβ/Δξ cannot be decorrelated across the mandated arm set (collinearity > 0.95) | test under-identified → report as inconclusive, not pass |
| sham / outer reproduces | |sham−orig| ≥ 0.5×intervention; or ΔM(outer)>0.6×ΔM(full) | reject |

## 8. Runtime & compute (decision: paid 8-vCPU)

Per-pair wall (measured): N=2048 ≈ 60 s, N=4096 ≈ 223 s (2-vCPU). Primary grid (12 cells, ~900 pairs)
≈ **29 h on free-tier 2-vCPU** vs **≈ 5 h on a paid c7i.2xlarge (8 vCPU), ~$1–2 spot.** **Decision (per
approval): paid 8-vCPU** — credits available; prioritize short wall-clock + clean monitoring. Stage 0
itself is cheap (4 families × 1 cell + short equilibrium-hold runs, < 1 h). Reuses the hardened
checkpointed runner (self-terminate + per-cell S3 sync + resume + 16h backstop).

## 9. Output / S3

`s3://nbody-battery-<acct>/outputs/nbody_aws_anisotropy/` (rebuild bucket+role as before).
`stage0_validation.json` (8 gates per family), per-cell `rows.jsonl` (+ baseline β, ξ, radial-KE
fraction, inner β/f_peri), `cell_summary.json` (multivariate partial-corr table, cluster-boot CIs,
dose×slope), root `manifest.json` + `verdict.json`.

## 10. Stop-rule (fail-fast)

(a) **Stage 0 first** — if Hernquist-radial fails Stage 0, **stop**. (b) Then the **hardest science
family first: hernquist3d radial** (N=2048→4096). If Δf_peri is not the top predictor there → **stop**,
**but first a power pre-check**: confirm the achieved Δf_peri dose range and the f_peri↔β collinearity
are sufficient to identify the effect (hernquist-radial has the *smallest* radialize dose — a null
there must be a real null, not low power). If under-powered, run a tangential family before deciding.

## 11. Exact command (gated — NOT run now)

```
# (one-time) rebuild S3 bucket+IAM; implement anisotropic samplers + Stage-0 gates + multivariate/
#   cluster-bootstrap analysis in a generalized runner (nbody_aws_anisotropy_battery.py).
python nbody_aws_anisotropy_battery.py --stage0-only                 # validate 4 families @ N2048/eps0.05
python nbody_aws_anisotropy_battery.py --families hernquist_radial --pairs 100   # stop-rule family
python nbody_aws_anisotropy_battery.py --pairs 100 --resume          # full primary grid on pass
```

## 12. New code required (gated on approval; unit-validated before any science run)

- **Anisotropic IC samplers:** OM radial (r_a=1.5a) + constant-β=−0.5 tangential, Hernquist & Plummer.
  *Validate against the analytic β(r) and σ_r(r) (Baes & Dejonghe 2002 Eq. 61) before use.*
- **Stage 0 gates**, incl. the inertia-tensor / m=2 shape gate.
- **Analysis upgrade:** multivariate-residualization partial correlation + cluster bootstrap over
  pairs + family dummies; ξ=2T_r/T_t and radial-KE-fraction computation; inner-β/f_peri reporting.
- No change to the verified intervention / integration / pericenter core (re-audited to machine
  precision).

## 13. References (anisotropic-IC feasibility; verify exact vol/page on ADS)

Hernquist (1990) ApJ — model + isotropic & OM DF. Baes & Dejonghe (2002) — OM Hernquist DF, σ_r, λ_max.
Buyle et al. (2007) — Hernquist OM non-negativity/stability. **Meza & Zamorano (1997)** — Hernquist ROI
threshold r_a,crit≈1.0a. Polyachenko & Shukhman (1981) — ROI criterion ξ=2T_r/T_t. Breen, Varri &
Heggie (2017) — Plummer OM non-negativity (r_a>0.75a). An & Evans (2005, 2006) — constant-β existence /
central cusp-anisotropy theorem β₀≤γ/2. Cuddeford (1991); Baes & van Hese — generalized anisotropic DFs.

## 14. Alternative future batteries (described, NOT implemented)

**A. Clumpy / triaxial / rotating transfer** — needs per-structure centre + revised pericenter
definition; do after anisotropy. **B. Central-sink / loss-cone toy** — a *different paper* (relaxation/
loss-cone timescales, sink boundary); scope as toy sink-feeding, not galactic-nucleus theory; do last.

---
**Decision requested:** approve v2, then (per prior approval) implement + Stage 0 on paid 8-vCPU. No
code implemented and no resources provisioned until then.
