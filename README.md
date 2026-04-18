## Predictive Structure in Three-Dimensional Self-Gravitating N-body Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**140,000 three-dimensional N-body simulations testing which initial-condition observables — coarse density contrast, fine positional structure, or phase-space kinematics — best predict future gravitational clustering across dark matter halo profiles, galaxy mergers, and globular cluster models.**

Code and analysis pipeline for the paper:

> **Predictive Structure in Three-Dimensional Self-Gravitating N-body Systems:
> Initial-Condition Families, Observable Classes, and the Role of Gravitational Softening**
>
> Kunal Bhatia (2026)
> [ORCID: 0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

## Scientific Summary

This paper asks: **which scale of initial structure in a self-gravitating particle system best predicts future gravitational clustering?**

We test three observable classes:
- **Coarse positional**: grid-scale density variance (analogous to large-scale overdensity in cosmological simulations)
- **Fine positional**: nearest-neighbour density, Fourier power, close-pair fraction, FoF group count
- **Fine kinematic**: local velocity dispersion (the only observable accessing phase-space information)

across four initial-condition families (plus three angular-shuffle null controls = 7 total):
- **Bimodal** (two-cluster merger): models galaxy-galaxy interactions and dark matter halo mergers
- **Hernquist** cusp: models elliptical galaxies and the stellar component of dark matter haloes
- **Plummer** sphere: models globular clusters and dwarf spheroidal galaxies
- **Cold-clumpy** (multi-clump): models young stellar groups and early dark matter substructure assembly

### Key Results

1. **Bimodal mergers**: Coarse density variance dominates in every tested cell (r = 0.984 at ε = 0.05, ranging from 0.961 to 0.994 across ε; winner-gap CIs entirely below zero in all cells). The inter-cluster density contrast at the Jeans scale governs the merger dynamics.

2. **Concentrated cusps**: Local velocity dispersion outperforms coarse predictors at low softening (Hernquist at ε ≤ 0.05, Plummer at ε = 0.02; gap CIs exclude zero), indicating that phase-space substructure below the softening scale carries predictive information when the force resolution partially resolves the cusp.

3. **Softening transition**: The fine kinematic advantage erodes with increasing softening and vanishes by ε = 0.10, where the softening length exceeds the cusp scale radius. The zero-crossing occurs between ε/a ≈ 0.25 and 0.50.

4. **No positional fine advantage**: No tested positional fine-scale observable outperforms the coarse predictor in any parameter cell, with one exception: cold-clumpy at ε = 0.10, where ClosePairs acts as a proxy for inter-clump connectivity rather than true fine-scale structure.

5. **Practical implication**: For simulations of cusp-dominated systems, force resolution must satisfy ε/a ≲ 0.25 for initial velocity substructure to carry predictive information. Above this threshold, only the coarse density field matters.

## Battery Design

- **140,000 total simulations** across 280 parameter cells
- Direct-isolated: N ∈ {256, 512, 1024, 2048}; PM-periodic: N ∈ {4096, 8192, 16384}
- ε ∈ {0.02, 0.03, 0.05, 0.07, 0.10}
- 500 independent realisations per cell
- 1000 percentile bootstrap resamples with null-bias correction for CIs
- Two force models: direct-summation (isolated, O(N²)) and particle-mesh (periodic, O(N log N))
- Three angular-shuffle null controls (bimodal, Hernquist, Plummer)
- PM forces are exactly ε-invariant; only ClosePairs varies across ε for PM runs

## Repository Structure

```
.
├── nbody_3d.py           # Force models, integrators, observables (standalone utility)
├── nbody_stress.py       # Battery runner, statistical analysis, bootstrap CI engine
├── nbody_paper.py        # Figure/table generation, analysis pipeline, paper macros
├── test_regression.py    # 34 regression tests
├── build.sh              # Reproducible build: run battery + compile paper
├── paper.tex             # LaTeX manuscript
├── paper.pdf             # Compiled paper
├── .gitignore            # Excludes large data files, caches, build artifacts
├── LICENSE               # MIT License
├── outputs/
│   ├── data/             # Generated macros, convergence data, run manifest
│   ├── figures/          # 17 publication-ready PDF figures
│   └── tables/           # 8 LaTeX table files
└── README.md
```

**Note**: The full battery CSV (~53 MB) and analysis JSON (~8 MB) are gitignored.
They can be regenerated from scratch (see below).

## Quick Start

### Requirements

```bash
pip install numpy scipy numba tqdm matplotlib
```

Python 3.10+ required. Numba is optional but provides ~10x speedup for direct-summation force calculations via Newton's 3rd law optimisation.

### Run regression tests

```bash
python -m pytest test_regression.py -v
```

All 34 tests should pass.

### Regenerate figures and tables (from existing data)

```bash
python nbody_paper.py --no-run --replicates 500
```

This loads the pre-computed battery CSV and regenerates all 17 figures, 8 tables, and paper macros. Takes ~15–30 minutes (bootstrap analysis).

### Run the full battery from scratch

```bash
python nbody_paper.py --workers 30 --replicates 500
```

**Warning**: This runs 140,000 N-body simulations and takes approximately 28 hours on 32 vCPUs. Use `--resume` to safely restart interrupted runs without losing progress.

### Build the paper

```bash
./build.sh
```

Runs the analysis pipeline and compiles the LaTeX manuscript via `latexmk`. Requires a LaTeX distribution.

## Force Models

| Model | Method | Boundary | Cost | Softening |
|-------|--------|----------|------|-----------|
| `direct_isolated` | Pairwise O(N²) with Newton's 3rd law | Open | N(N−1)/2 pairs | Plummer ε |
| `pm_periodic` | FFT Poisson solver on 32³ grid | Periodic | O(N log N) | Grid resolution (~L/32) |

Direct-isolated is capped at N ≤ 2048. PM-periodic runs the full N range up to 16384. The two models represent distinct physics and are compared as separate experiments.

## Observable Classes

| Class | Observable | Description |
|-------|-----------|-------------|
| Coarse (grid) | CoarseG4/G8/G16 | Density variance on 4³, 8³, 16³ grids |
| Coarse (radial) | CoarseConc | Concentration proxy: N(<r₅₀/2) / N(<r₅₀) |
| Coarse (radial) | CoarseRShellVar | Normalised radial shell mass variance |
| Fine positional | kNN-all | Mean k-nearest-neighbour density (k=16) |
| Fine positional | Pk-small | Small-scale Fourier power (|k| > g/4) |
| Fine positional | ClosePairs | Fraction of pairs within 4ε |
| Fine structural | FoF | Friends-of-friends group count (b=0.20 d̄) |
| Fine kinematic | VelDisp | Mean local velocity dispersion (k=16 neighbours) |

Grid-based observables restrict to particles inside [0, L)³. Distance-based observables (kNN, ClosePairs, VelDisp, FoF) use all particles.

## Citation

If you use this code or data, please cite:

```
Bhatia, K. (2026). Predictive Structure in Three-Dimensional Self-Gravitating
N-body Systems: Initial-Condition Families, Observable Classes, and the Role
of Gravitational Softening. Preprint.
```

## License

MIT License. See [LICENSE](LICENSE) for details.
