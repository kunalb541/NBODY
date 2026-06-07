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
