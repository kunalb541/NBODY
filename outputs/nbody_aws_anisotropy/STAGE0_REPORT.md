# Stage 0 report — anisotropic IC validation (anisotropy-transfer battery)

**Run:** N=2048, eps=0.05, r_a=1.5a (radial), beta=-0.5 (tangential), 5 seeds/family, 1000-step
horizon. Local. No science battery run (gated on Stage 0). `paper.tex` untouched.

## Verdict: BLOCKED — do NOT launch the science battery.

The samplers are **correct** (static gates pass), but the ICs are **not stationary** at the
M(<0.1)=2eps aperture under the softened production potential (dynamic gates fail) — uniformly across
**isotropic and anisotropic** families. Per the preregistered decision rule (*Hernquist-radial fails
Stage 0 -> stop the whole battery*), the anisotropy science grid is **blocked pending an IC fix.**
Stage 0 did exactly its job: it prevented running anisotropy-transfer science on ICs that would
breathe ~30% regardless of anisotropy.

## Static gates (1,2,3,7) — PASS

Unit-validated at N=20000 (Eddington machinery verified vs the analytic Hernquist DF to ratio 1.000):

| family | density KS D | 2T/|W| (unsoftened) | max|beta-target| |
|---|---|---|---|
| hernquist isotropic | 0.007 | 1.003 | ~0 (iso) |
| hernquist radial    | 0.007 | 1.014 | 0.026 (beta(r)=r^2/(r^2+r_a^2)) |
| hernquist tangential| 0.007 | 1.009 | 0.062 (beta=-0.5) |
| plummer isotropic   | 0.007 | 0.998 | ~0 |
| plummer radial      | 0.007 | 1.001 | 0.025 |
| plummer tangential  | 0.007 | 1.005 | 0.073 |

Density, beta(r), and (unsoftened) virial are all correct. **The DF samplers work.**

## Dynamic gates (4 M-drift, 5 beta-drift, 6 shape/ROI) — FAIL (4,5); PASS (6 shape)

| family | M(<0.1) drift | beta drift | breathing | c/a min | b/a min | gates |
|---|---|---|---|---|---|---|
| hernquist isotropic | 0.318 | 0.113 | 0.396 | 0.914 | 0.952 | 4✗ 5✗ 6✗ |
| hernquist radial    | 0.327 | 0.068 | 0.427 | 0.908 | 0.948 | 4✗ 5✗ 6✗ |
| hernquist tangential| 0.413 | 0.227 | 0.601 | 0.924 | 0.948 | 4✗ 5✗ 6✗ |
| plummer isotropic   | 0.301 | 0.107 | 0.217 | 0.931 | 0.957 | 4✗ 5✗ 6✗ |
| plummer radial      | 0.284 | 0.059 | 0.202 | 0.928 | 0.966 | 4✗ 5✗ 6✗ |
| plummer tangential  | 0.344 | 0.239 | 0.243 | 0.935 | 0.966 | 4✗ 5✗ 6✗ |

Thresholds: M-drift<0.10, beta-drift<0.05, breathing<0.15, axis ratios>0.85.

## Diagnosis

- **Not the ROI / not anisotropy.** The shape gate PASSES (axis ratios ~0.91-0.97, no bars) — the
  v2 fix r_a=1.5a successfully avoided the radial-orbit instability. And **isotropic families fail
  identically** to anisotropic ones, so the breathing is **not** an anisotropy artifact.
- **Not global expansion.** The softened virial 2T/|W_soft| is only ~1.06-1.07 (mild super-virial).
- **It is the softening-scale aperture.** M(<0.1) sits at 2eps=0.10; the ICs are sampled from the
  *unsoftened* DF but evolved in the *softened* potential (eps/a=0.25), so the inner structure
  (<~2eps) re-adjusts -> ~30% drift in M(<0.1) and the inner Lagrangian radius. The bulk (50/90%
  radii) is far more stable; the failure is concentrated at the softening scale.

## Important: the committed battery is unaffected

The committed low-pericenter battery used these same (isotropic) profiles at the same aperture and
its causal claims are read as **(arm - sham)** within matched pairs, which **subtracts this
common-mode breathing**. The breathing only fails the *strict baseline-stationarity* Stage-0 gate;
it does not invalidate matched-pair causal differences. (Independently re-audited to machine precision.)

## Fix options (user decision — design change to the prereg)

1. **Softened-potential DF.** Build the DF from the *softened* potential Psi_soft(r) so the inner
   structure matches what is integrated. Cleanest; most work.
2. **Pre-relaxation.** Integrate each IC briefly, let the inner settle, then re-validate (Stage 0
   again). Simple; check it settles rather than runs away.
3. **Move the aperture off the softening scale.** Use M(<0.2) (=4eps) or M(<0.3) as the primary
   target, or reduce eps (e.g. 0.02), so the measured aperture is in the stationary region.
4. **Reframe the Stage-0 gate.** Require (arm - sham) *difference* stability (what the science
   actually measures) rather than baseline absolute stationarity — consistent with how the committed
   battery's causal claims are made.

## Battery status: BLOCKED (not cleared, not partial). Awaiting a chosen IC-stationarity fix.
