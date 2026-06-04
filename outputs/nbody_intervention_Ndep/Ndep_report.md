# N-dependence of the |L| causal chain (Hernquist, ε=0.05, θ=20°)

N ∈ [512, 1024, 2048] (pairs: [100, 100, 50]). Times: [0, 5, 10, 20, 50, 100, 300, 600].

## Verdict — 🟢 PARTICLE-NUMBER ROBUST

> **PARTICLE-NUMBER ROBUST — across N=[512, 1024, 2048]: ordering (M before/with C₈) holds at all N, |L| permanent at all N, β transient at all N, C₈ sign consistent (+), intensive sham clean, KE/Q clean; the INTENSIVE M(<r_c) effect does NOT vanish (peak 0.030→0.027, ratio 0.91) while the extensive C₈ grows ~with N. The |L|→concentration→C₈ chain is a real dynamical mechanism in the tested range, not a finite-N realization artifact. (C₈ sham is noisy at N=[512] — expected: C₈ is a count variance, small and noisy at low N; the intensive β/M(<r_c) shams are clean at all N.) No AWS.**

## Per-N summary

| N | pairs | M(<0.05) t | M(<0.1) t | C₈ t | conc<C₈ | |L| perm | β trans | peak M(<0.1) | peak C₈ | C₈ sign | sham✓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 100 | 5 | 5 | 5 | True | True | True | 0.030 | 1.29 | +1 | False |
| 1024 | 100 | 5 | 5 | 10 | True | True | True | 0.030 | 4.71 | +1 | True |
| 2048 | 50 | 5 | 5 | 5 | True | True | True | 0.027 | 16.87 | +1 | True |

- **intensive M(<0.1) peak by N:** {512: 0.03, 1024: 0.03, 2048: 0.027} (ratio hiN/loN = 0.91; vanishes=False).
- extensive C₈ peak by N: {512: 1.3, 1024: 4.7, 2048: 16.9} (grows ~with N, as expected for a count variance).
- ordering holds at all N: True; |L| permanent all N: True; β transient all N: True; C₈ sign consistent: True.

