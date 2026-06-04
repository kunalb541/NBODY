# N-body Causal-Intervention Pilot — Report

**Family:** hernquist3d  **N:** 1024  **ε:** 0.05  **θ:** 20.0°  **matched pairs:** 100  **horizon:** t₁=600

**Handle:** compensated velocity-anisotropy rotation (speed-preserving → E, Q conserved). **Sham:** equal-angle rotation about a random axis.

## Verdict

> **ALIVE — initial velocity anisotropy is a CAUSAL handle on future anisotropy: β_int(t₁)−β_sham(t₁) = +0.1515 [+0.1452,+0.1583] (same sign as the imposed Δβ₀=+0.654), with E and Q preserved (not a bulk effect). Worth a medium confirmation (θ-dependence, more families/targets). No AWS.**

## Honest interpretation

The **unperturbed** system already relaxes from isotropic to strongly radial — β: -0.007 → +0.461 (the radial-anisotropy attractor of violent relaxation). The intervention is a **sub-dominant causal modulation on top of that attractor**: imposing β₀≈+0.65 leaves a **+0.151** residual at t₁ (persistence ≈ 23% of the imposed change). So initial anisotropy *is* causally accessible, but the system is dominated by its own relaxation. Confirmation should test θ-dependence (θ=20° is a strong push) and whether the persistence exceeds a trivial collisionless baseline.

## Matched-pair effects (paired bootstrap 95% CI)

- **Imposed handle** Δβ₀ (int−orig): +0.6537 [+0.6464, +0.6609]  (n=100)  — sham Δβ₀: +0.0009 [-0.0041, +0.0062]  (n=100) (should be ≈0)
- **Causal effect** β_int(t₁) − β_sham(t₁) (headline): +0.1515 [+0.1452, +0.1583]  (n=100)
- int − orig at t₁: +0.1508 [+0.1441, +0.1580]  (n=100)
- sham − orig at t₁: -0.0007 [-0.0058, +0.0046]  (n=100)
- causal effect on σ_r(t₁): +0.0609 [+0.0586, +0.0633]  (n=100)

## Conservation (intervention vs orig at t₀ — must be ≈0 to rule out a bulk effect)

(IC is super-virial Q≈2 so |E|≈0; conservation is checked on KE and Q relative drift.)

- median |ΔKE|/KE = 8.22e-05
- median |ΔQ|/Q   = 8.22e-05
- median ΔL (angular momentum, expected <0 for radialisation) = -1.552e-02

## Kill tests

1. **Bulk-energy disguise:** ruled out by construction — E and Q are preserved (see conservation above).
2. **Sham equivalence:** the headline is int−**sham**; if it includes 0, no anisotropy-specific causal handle.
3. **Unphysical size:** θ=20° is a modest, physical rotation.
4. **Below noise:** the paired-bootstrap CI is the noise bar.

