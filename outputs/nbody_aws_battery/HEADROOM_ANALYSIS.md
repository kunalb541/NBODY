# Headroom-law test across central radii — is the handle scale-free?

**Source:** committed AWS battery rows under `outputs/nbody_aws_battery/`. Reproduce with
`python3 nbody_headroom_scale_analysis.py` (writes `headroom_scale.json`, `figures/fig5_headroom_scale.png`).
**No new simulations.** ε=0.05, mean over N=512–4096. `dM` = peak-over-time ΔM(<r) (full−sham);
`dose` = Δf_peri(<r) at t₀ (full−sham); `H` = dM/dose; `baseM`, `basef` = baseline (orig, t₀)
enclosed mass and pericenter fraction within r.

## Question

Can the low-pericenter handle be written as a compact, scale-free **headroom law**

    ΔM(<r) ≈ Δf_peri(<r) × H(r),   with H(r) set by baseline unsaturation?

## Corrected decomposition table

**Hernquist (cusp)**

| r_c | dose | ΔM | H | baseM | basef |
|---|---|---|---|---|---|
| 0.05 | 0.125 | 0.0154 | 0.123 | 0.019 | 0.186 |
| 0.10 | 0.133 | 0.0287 | 0.216 | 0.081 | 0.334 |
| 0.20 | 0.114 | 0.0365 | 0.318 | 0.230 | 0.512 |

**Plummer (core)**

| r_c | dose | ΔM | H | baseM | basef |
|---|---|---|---|---|---|
| 0.05 | 0.172 | 0.0247 | 0.144 | 0.014 | 0.128 |
| 0.10 | 0.192 | 0.0562 | 0.293 | 0.087 | 0.365 |
| 0.20 | 0.129 | 0.0447 | 0.346 | 0.350 | 0.677 |

**Plummer / Hernquist ratios by r_c:** ΔM ×1.60 / ×1.96 / ×1.22 at r = 0.05 / 0.10 / 0.20;
dose ×1.37 / ×1.44 / ×1.13; slope ×1.17 / ×1.36 / ×1.09.

## Findings

1. **Not scale-free.** The Plummer/Hernquist ΔM advantage is **×1.60 (r=0.05), ×1.96 (r=0.10),
   ×1.22 (r=0.20)** — it peaks near **r ≈ a/2** (a=0.20) and washes out by r=a. No single law over
   all radii.

2. **H(r) tracks geometry, not unsaturation.** H rises monotonically with r_c and correlates
   **+0.87 with baseline mass** across the 6 (profile, r_c) points — a *bigger* sphere simply catches
   more of each plunging orbit's mass. This is the *opposite* of "low baseline mass → high H."

3. **The slope advantage is NOT unsaturation.** At r=0.10 the two profiles have nearly equal
   baseline enclosed mass (Plummer **0.087** vs Hernquist **0.081**), yet Plummer's slope is still
   **×1.36** higher. Equal saturation, unequal slope ⇒ the slope term is a *dynamical*
   potential-shape / orbit-deposition effect (radial orbits deposit mass differently in a harmonic
   core than in a 1/r cusp), not headroom.

4. **The dose advantage IS consistent with headroom** — but only for *deep* pericenters: at r<0.05
   Plummer's baseline plunging fraction is lower (basef **0.128** vs **0.186**) and its dose is
   larger, i.e. the core has more room to create deep-pericenter orbits.

## Conclusion (corrected law)

**No compact scale-free law ΔM(<r) ≈ Δf_peri(<r) × H(unsaturation).** Instead:

> ΔM(<r) decomposes into a **pericenter-dose** term and a **radius/profile-dependent response
> slope**. The dose is governed partly by low-pericenter *headroom* (accessibility of deep
> pericenters, clearest at r<0.05). The slope is governed by *potential shape / orbit-deposition
> geometry*, not unsaturation. The handle is strongest at the target radius where dose and response
> slope align — here **r ≈ 0.1 = a/2**.

This supersedes the earlier shorthand "an unsaturated centre explains both dose and slope" (see the
correction stanza in `MECHANISM_NOTE.md`).
