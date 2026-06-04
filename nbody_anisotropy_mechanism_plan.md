# N-body Anisotropy Handle — Mechanism Plan

**Status:** Science-first mechanism interrogation. No paper, no journals, no AWS. `paper.tex` untouched.
**Date:** 2026-06-04

## What is confirmed (the phenomenon, not an endpoint)

Speed-preserving **radialization** of velocities at t₀ causally modulates future **β, σ_r, entropy,
and C₈** in concentrated Hernquist/Plummer systems — sham-null, KE/Q clean, gravity persistence
(~23%) > free-streaming, and `uniform3d` showing a *different* passive mechanism (concentration
required). This is a **live causal phenomenon**. The question now is **why**.

## The mechanism question

Radialization does several things at once. We must find which is the causal variable:

```
radialize  ──►  β ↑   (velocity-dispersion anisotropy)
            ──►  |L_i| ↓  (per-particle specific angular momentum: radial orbits carry less L)
            ──►  smaller pericenters ──► central concentration ↑ ──► C₈ ↑
```

β and |L_i| are **geometrically entangled** for a speed-preserving rotation (you cannot make an
orbit more radial without lowering its specific angular momentum). So the core confounds are:

- **H1 — Anisotropy (β):** the dispersion anisotropy itself is the causal variable.
- **H2 — Angular-momentum depletion (L):** lower specific L is the real driver; β is incidental.
- **H3 — Infall mediation (for C₈):** the C₈ response is *mediated* by radial infall → central
  concentration growth, not a direct effect.

## Exact intervention formulas (speed-preserving → KE, E, Q conserved by construction)

Decompose each velocity about the COM: `v_r = v·r̂`, `t̂ = (v − v_r r̂)/|·|`, `φ = atan2(|v_t|, v_r) ∈ [0,π]`.

- **Radialize (β↑, |L_i|↓):** `φ' = φ−θ` for `φ≤π/2`, `φ' = φ+θ` for `φ>π/2` (toward nearest radial
  axis, clamp to [0,π]); `v' = |v|(cos φ' r̂ + sin φ' t̂)`. *(the confirmed handle.)*
- **Tangentialize (β↓, |L_i|↑):** move toward the tangential plane: `φ' = min(φ+θ, π/2)` for
  `φ≤π/2`, `φ' = max(φ−θ, π/2)` for `φ>π/2`; same reconstruction. Purely-radial particles (|v_t|≈0)
  get a random unit `t̂ ⊥ r̂`.
- **Sham (β≈0, magnitude-matched):** rotate `v` by angle θ about a uniformly random axis (Rodrigues).
- **L-matched control (test B):** target the same Δ⟨|L_i|⟩ as radialization but with **no net Δβ**,
  by radializing the inner half and tangentializing the outer half (or vice-versa) in proportions
  tuned so ⟨σ_t²/σ_r²⟩ is preserved while ⟨|L_i|⟩ drops by the matched amount. Speed-preserving.
- **Shell-local (test E):** apply radialization **only** to particles in an inner / middle / outer
  radial shell (by r-percentile), leaving the rest at orig; same sham per shell.

All interventions re-zero total momentum afterward (tiny correction; KE/Q drift recorded).

## The five mechanism tests

| Test | Question | What it runs |
|---|---|---|
| **A** | Is the causal direction antisymmetric? | radialize / tangentialize / sham, compare ΔC₈, Δβ, Δ⟨\|L_i\|⟩ |
| **B** | β or L? | L-matched (β-null) control vs radialization — does the effect survive at matched ΔL? |
| **C** | Is C₈ infall-mediated? | measure radial infall rate, central mass M(<r_c), shell crossing alongside ΔC₈ |
| **D** | Transient or stable memory? | track the effect at t₁ ∈ {100, 300, 600, 1000} |
| **E** | Where does control live? | inner / middle / outer shell-local radialization → causal map |

## What would falsify each mechanism (kill-tests)

| Mechanism | Falsified if … |
|---|---|
| **H1 (β)** | the L-matched (β-null) control reproduces the full effect → it was L, not β |
| **H2 (L)** | radialize/tangentialize C₈ effects track Δβ better than Δ⟨\|L_i\|⟩, AND the L-matched control fails to reproduce the effect |
| **H3 (infall→C₈)** | C₈ changes **without** a matching change in central mass / infall → C₈ is via another channel (e.g. shell phase structure) |
| **Causality (test A)** | tangentialization does **not** give opposite-sign C₈ → interpretation is nonlinear/contaminated |
| **Persistence** | the effect vanishes by t₁=1000 → transient memory, not a durable handle |
| **Locality** | shell-local effects are spatially uniform → no localized causal control |

## First pilot (smallest decisive) — Test A + mediation diagnostics

- **Hernquist, ε=0.05, N=1024, θ=20°, 100 matched quadruples** (orig / radialize / tangentialize / sham).
- **Horizon t₁=600.**
- **Measured** at t₀ and t₁ per run: β, mean specific angular momentum ⟨|L_i|⟩, C₈, central-mass
  fraction M(<0.1), inner radial velocity ⟨v_r⟩(r<0.3) (infall proxy).
- **Reported:** paired causal effects (X − sham) for radialize and tangentialize on each quantity;
  **antisymmetry** C₈(radialize) vs C₈(tangentialize); **mediation** — does ΔC₈ co-move with
  Δ central mass across the two interventions; conservation (ΔKE/KE, ΔQ/Q); sham null.

### Decision
The first-pilot design is **unambiguous** (radialize/tangentialize/sham is the clean causality
check). Per the directive, run **only** this minimal test now; tests B–E follow only if it is
promising. Runtime: 100×4 = 400 integrations to step 600 ≈ <2 min local.

### What the first pilot can already decide
- **Antisymmetry (Q3):** radialize C₈↑ and tangentialize C₈↓ with opposite sign → causality clean;
  else → interpretation incomplete (stop and rethink).
- **Infall (Q2, partial):** if ΔC₈ co-moves with Δ central mass → infall-mediation supported; if C₈
  moves without central-mass change → H3 in doubt.
- **β vs L (Q1, partial):** records whether ΔC₈ tracks Δβ or Δ⟨|L_i|⟩ across the two interventions;
  a clean separation needs test B (L-matched control), deferred.
