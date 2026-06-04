# Anisotropy Mechanism — Test B (β-null, L-matched control)

Hernquist, ε=0.05, N=1024, 100 matched quadruples (orig / radialize / L-matched-β-null / sham), t₁=600.

## Verdict — 🟠 CONFOUNDED — β/|L| not cleanly separable

> **CONFOUNDED CONSTRUCTION (β/|L| not cleanly separable) — the L-matched arm hit the GLOBAL scalar targets (Δβ₀=+0.070≈0, Δ|L|=-0.29≈ref -0.34), but depleting the radius-weighted ⟨|L|⟩ at fixed β REQUIRES a radial anisotropy gradient (radialize outer / tangentialize inner), and that gradient itself reshapes scale-dependent clustering. Result is scale-split: core concentration M(<0.1) rises (+0.0044, +27% of radialize, SAME sign → L-depletion DOES drive concentration) while mid-scale C₈ REVERSES (-1.08, opposite sign, from inner tangentialization). So C₈ cannot cleanly separate β vs L here. CLEAN findings: (1) L-depletion at β-null causally raises central mass → angular-momentum depletion contributes to concentration; (2) the L-matched arm sustains β(t₁)=+0.077 from an imposed +0.07 → L-depletion is partly upstream of anisotropy. A confound-free β/L split is likely unreachable under speed-preserving rotations (β and |L| geometrically coupled, as anticipated). Next: a non-rotational handle, or a single-scale concentration target instead of C₈.**

## Construction check (t₀ — must hit Δβ₀≈0 with matched Δ|L|)

- tuned params: split=50%, θ_out=20°, θ_in=25°
- radialize: Δβ₀=+0.652, Δ⟨|L|⟩=-0.344
- **L-matched: Δβ₀=+0.070** (β-null=True), **Δ⟨|L|⟩=-0.295** (L-matched=True)
- construction OK: **True**; sham Δβ₀=-0.0002 [-0.0055, +0.0049]; conservation ΔKE/KE=9.1e-05, ΔQ/Q=9.1e-05

## Causal effects at t₁ (intervention − sham)

| quantity | radialize | L-matched β-null | reproduced |
|---|---|---|---|
| beta | +0.1478 [+0.1431, +0.1524] | +0.0770 [+0.0722, +0.0817] | 52% |
| Lspec | -0.3360 [-0.3407, -0.3309] | -0.3005 [-0.3053, -0.2955] | 89% |
| C8 | +2.3209 [+1.5051, +3.2582] | -1.0837 [-1.8096, -0.2835] | -47% |
| Mcen | +0.0162 [+0.0129, +0.0195] | +0.0044 [+0.0019, +0.0068] | 27% |
| sigr | +0.0616 [+0.0596, +0.0635] | +0.0361 [+0.0341, +0.0381] | 59% |
| S | +0.0285 [+0.0227, +0.0346] | +0.0025 [-0.0037, +0.0090] | 9% |

C₈ reproduced: -47%; central-mass reproduced: 27%.

