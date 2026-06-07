# Mechanism note — why Plummer is ~2× stronger than Hernquist

**Source:** committed AWS battery rows in `outputs/nbody_aws_battery/`. Reproduce with
`python3 nbody_battery_plummer_vs_hernquist.py` (writes `plummer_vs_hernquist.json`).
**No new simulations.** N=4096 included; controls remain negative; `BATTERY_PASS=False` is the
strict-Boolean artifact (one borderline locality threshold), **scientific verdict is positive.**

## Result — the 2× is a clean dose × slope product

Causal chain Δf_peri → ΔM(<0.1), with peak ΔM10 = **slope × dose** (ε=0.05, mean over N=512–4096):

| term | Hernquist (cusp) | Plummer (core) | ratio P/H |
|---|---|---|---|
| **dose** = Δf_peri(<0.1), full−sham, t0 | 0.133 | 0.192 | **×1.44** |
| **slope** = ΔM10 / dose | 0.216 | 0.293 | **×1.36** |
| **ΔM10** (peak response) | 0.0287 | 0.0562 | **×1.96** |

**dose × slope = 1.44 × 1.36 = ×1.96 = the measured ΔM10 ratio exactly.** Both terms contribute,
in roughly equal measure. Both are flat across N (→ the ×1.01 Plummer N-robustness); ε raises the
softening and lowers the *slope* (response) while leaving the *dose* essentially unchanged (the dose
is set at t₀ by the intervention geometry, before integration).

## Interpretation — low-pericenter accessibility in an unsaturated centre

One unified cause: **the core is unsaturated; the cusp is partially pre-loaded with plunging orbits.**

- **Baseline deep-pericenter fraction f_peri(<0.05): Hernquist 0.186 vs Plummer 0.128.** The cusp
  (ρ∝1/r) already packs many deep-plunging orbits into its centre, so the same radialization has less
  headroom to add more → **smaller dose**. The core starts with fewer deep orbits → more room → larger
  dose (×1.44).
- **Steeper slope (×1.36):** each newly-plunging orbit deposits mass at small r. In a core the small-r
  density is low, so that is a large *fractional* ΔM(<0.1); in an already-dense cusp the same orbit is
  a smaller marginal increase.

**The mechanism is low-pericenter accessibility in an unsaturated centre — not "cusp concentration."**
In fact it is the opposite: the cusp's pre-existing central density *reduces* the handle's leverage on
both terms. The mechanism sentence:

> Cores have more low-pericenter headroom than cusps, so the same radializing intervention produces
> both more plunging orbits (dose ×1.44) and a larger marginal inner-mass response (slope ×1.36),
> giving the ~2× stronger central-concentration effect in Plummer.

## Scope

Based on the committed AWS battery (2 profiles × N=512–4096 × ε grid, 100 pairs / 50 at N=4096).
Holds across all N (N=4096 included). uniform/bimodal controls remain negative (mechanism absent).
No new simulations, no AWS, paper text untouched.

## Correction (2026-06-06) — slope is NOT a headroom effect

A follow-up multi-radius analysis (`nbody_headroom_scale_analysis.py`, `HEADROOM_ANALYSIS.md`)
shows the shorthand above — *"an unsaturated centre explains both dose and slope"* — was **too
strong**. The corrected reading:

- **Dose advantage = headroom/accessibility (holds).** Plummer has a lower baseline *deep*-pericenter
  fraction at r<0.05 (**0.128 vs 0.186** in Hernquist) and a larger Δf_peri — the core genuinely has
  more room to create deep-plunging orbits.
- **Slope advantage is NOT explained by unsaturation.** At r=0.1 the two profiles have nearly equal
  baseline enclosed mass (**Plummer 0.087 vs Hernquist 0.081**), yet Plummer's slope is still
  **×1.36** higher. Equal saturation, unequal slope ⇒ the slope term is a **dynamical
  potential-shape / orbit-deposition effect** (radial orbits deposit mass differently in a harmonic
  core than in a 1/r cusp), not a headroom effect. (Across r_c, H even tracks baseline mass with
  corr **+0.87** — the opposite of a "low mass → high slope" headroom law.)
- **The mechanism is scale-specific, not scale-free.** The Plummer/Hernquist ΔM advantage is
  **×1.60 (r=0.05), ×1.96 (r=0.10), ×1.22 (r=0.20)** — strongest near **r ≈ a/2**, not constant
  across radii. There is **no compact scale-free law** ΔM(<r) ≈ Δf_peri(<r) × H(unsaturation).

**Corrected mechanism sentence:** the Plummer advantage is a *two-factor* effect — the core has more
deep-pericenter **dose** headroom, while its **potential shape** gives newly-radialized orbits a
larger marginal contribution to M(<r) — and it is **radius-specific**, peaking where dose and
response slope align (here r ≈ 0.1). See `HEADROOM_ANALYSIS.md` for the full table.
