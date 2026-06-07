# Stage 0B report — aperture M(<0.2)=4eps + short pre-relaxation

**Fix applied (approved):** primary aperture M(<0.1)=2eps -> M(<0.2)=4eps; add 300-step pre-relaxation
under the softened production potential before measurement. N=2048, eps=0.05, r_a=1.5a, beta=-0.5,
5 seeds, 1000-step measurement horizon. Local. No science battery run. paper.tex untouched.

## Verdict: PARTIALLY UNBLOCKED — radial + isotropic families PASS (incl. the stop-rule family); tangential marginal.

The aperture+pre-relax fix RESOLVES the equilibrium-hold problem. The stop-rule family
**hernquist_radial PASSES**, so the battery is **not** blocked by the stop rule. The two TANGENTIAL
families fail on ONE gate only (beta-profile fidelity after pre-relax); their M-drift, breathing,
beta-drift, and shape all pass.

## Results (both profiles)

| family | M(<0.1) drift no-PR | M(<0.2) drift no-PR | M(<0.2) drift PR | breathing | beta drift | beta-prof maxdev | c/a,b/a min | PASS |
|---|---|---|---|---|---|---|---|---|
| hernquist isotropic | 0.318 | 0.159 | 0.015 | 0.049 | 0.041 | 0.029 | 0.91 | YES |
| hernquist radial    | 0.327 | 0.154 | 0.007 | 0.062 | 0.010 | 0.041 | 0.90 | YES |
| hernquist tangential| 0.413 | 0.202 | 0.010 | 0.038 | 0.029 | 0.165 | 0.92 | no (beta-prof) |
| plummer isotropic   | 0.301 | 0.165 | 0.007 | 0.067 | 0.029 | 0.103 | 0.93 | YES |
| plummer radial      | -     | -     | 0.006 | 0.076 | 0.019 | 0.057 | 0.91 | YES |
| plummer tangential  | -     | -     | 0.008 | 0.060 | 0.028 | 0.159 | 0.92 | no (beta-prof) |

Gate thresholds: M(<0.2) drift<0.10, beta drift<0.05, breathing<0.15, axis ratios>0.85,
beta-profile maxdev<0.12.

## What the fix bought (attribution)

- **Aperture (M(<0.1) -> M(<0.2)):** drift ~0.30 -> ~0.16. The 2eps aperture was half the problem.
- **Pre-relaxation (300 steps):** ~0.16 -> ~0.01. Settles the collisionless inner phase-mixing
  transient; breathing 0.4-0.6 -> 0.04-0.08. The anisotropy survives (2-body relaxation at N=2048 is
  ~10^4 steps away): radial beta(r) maxdev 0.04-0.06, well within tolerance.
- **Shape/ROI:** still clean (axis ratios ~0.90-0.93, no bars; r_a=1.5a stable).

## The remaining issue: tangential beta-profile fidelity

Both tangential (constant beta=-0.5) families pass M-stationarity/breathing/shape but the INNER
tangential bias relaxes slightly during the 300-step pre-relaxation: beta-profile maxdev 0.16-0.17 vs
the 0.12 gate (the inner shell relaxes from beta=-0.5 toward ~-0.34; sign and bulk are preserved).
This is a fidelity nuance, not a stationarity failure.

Options for the tangential families (user decision):
1. **Shorter pre-relaxation** for tangential (e.g. 100 steps) — less beta washout; re-check M-drift.
2. **Accept beta~-0.4** as a valid (still clearly tangential, distinct-from-iso) baseline; relax the
   beta-profile tolerance to ~0.18 with justification.
3. **Softened-potential / constant-beta DF** for tangential only (cleaner, more work).
4. **Run radial+isotropic transfer first** (fully validated) and treat tangential as a flagged
   secondary.

## Battery status: UNBLOCKED for the radial transfer test (the key max-confound case) + isotropic baseline.
Tangential pending a small fidelity fix. Per protocol: requesting approval before the science battery.
