 Predictive Structure in Three-Dimensional Self-Gravitating N-body Systems

Code and analysis pipeline for the paper:

> **Predictive Structure in Three-Dimensional Self-Gravitating N-body Systems:
> Initial-Condition Families, Observable Classes, and the Role of Gravitational Softening**
>
> Kunal Bhatia (2026)

## Scientific Summary

This paper asks: **which scale of initial structure in a self-gravitating particle system best predicts future gravitational clustering?**

We test three observable classes:
- **Coarse positional**: grid-scale density variance (analogous to large-scale overdensity in cosmological simulations)
- **Fine positional**: nearest-neighbour density, Fourier power, close-pair fraction, FoF group count
- **Fine kinematic**: local velocity dispersion (the only observable accessing phase-space information)

across four initial-condition families:
- **Bimodal** (two-cluster merger): models galaxy-galaxy interactions and dark matter halo mergers
- **Hernquist** cusp: models elliptical galaxies and the stellar component of dark matter haloes
- **Plummer** sphere: models globular clusters and dwarf spheroidal galaxies
- **Cold-clumpy** (multi-clump): models young stellar groups and early dark matter substructure assembly

### Key Results

1. **Bimodal mergers**: Coarse density variance dominates in every tested cell (r = +0.984, gap CI entirely below zero). The inter-cluster density contrast at the Jeans scale governs the merger rate.

2. **Concentrated cusps**: Local velocity dispersion outperforms coarse predictors at low softening (Hernquist at epsilon <= 0.05, Plummer at epsilon = 0.02), indicating that phase-space substructure below the softening scale carries predictive information when the force resolution partially resolves the cusp.

3. **Softening transition**: The fine kinematic advantage erodes with increasing softening and vanishes by epsilon = 0.10, where the softening length exceeds the cusp scale radius. The transition occurs between epsilon/a ~ 0.25 and 0.50.

4. **No positional fine advantage**: No tested positional fine-scale observable outperforms the coarse predictor in any parameter cell.

5. **Practical implication**: For simulations of cusp-dominated systems, force resolution must satisfy epsilon/a < 0.25 for initial velocity substructure to carry predictive information.

## Battery Design

- **140,000 total simulations** across 280 parameter cells
- N in {256, 512, 1024, 2048, 4096, 8192, 16384}
- epsilon in {0.02, 0.03, 0.05, 0.07, 0.10}
- 500 independent realisations per cell
- 1000 bootstrap resamples for confidence intervals
- Two force models: direct-summation (isolated) and particle-mesh (periodic)
- Three angular-shuffle null controls

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
├── outputs/
│   ├── data/             # Generated macros, convergence data, run manifest
│   ├── figures/          # 17 publication-ready PDF figures
│   └── tables/           # 8 LaTeX table files
└── README.md
```

## Quick Start

### Requirements

```bash
pip install numpy scipy numba tqdm matplotlib
```

Python 3.10+ required. Numba is optional but provides ~10x speedup for direct-summation force calculations.

### Run regression tests

```bash
python -m pytest test_regression.py -v
```

All 34 tests should pass.

### Regenerate figures and tables (from existing data)

```bash
python nbody_paper.py --no-run --replicates 500
```

This loads the pre-computed battery CSV and regenerates all 17 figures, 8 tables, and paper macros. Takes ~15-30 minutes (bootstrap analysis).

### Run the full battery from scratch

```bash
python nbody_paper.py --workers 30 --replicates 500
```

**Warning**: This runs 140,000 N-body simulations and takes approximately 28 hours on 32 vCPUs. Use `--resume` to safely restart interrupted runs.

### Build the paper

```bash
./build.sh
```

Runs the analysis pipeline and compiles the LaTeX manuscript. Requires `latexmk` and a LaTeX distribution.

## Force Models

| Model | Method | Boundary | Cost | Softening |
|-------|--------|----------|------|-----------|
| `direct_isolated` | Pairwise O(N^2) with Newton's 3rd law | Open | N(N-1)/2 pairs | Plummer epsilon |
| `pm_periodic` | FFT Poisson solver on 32^3 grid | Periodic | O(N log N) | Grid resolution |

Direct-isolated is capped at N <= 2048. PM-periodic runs the full N range up to 16384.

## Observable Classes

| Class | Observable | Description |
|-------|-----------|-------------|
| Coarse | CoarseG4/G8/G16 | Density variance on 4^3, 8^3, 16^3 grids |
| Coarse (radial) | CoarseConc, CoarseRShellVar | Concentration proxy, radial shell variance |
| Fine positional | kNN-all | Mean k-nearest-neighbour density |
| Fine positional | Pk-small | Small-scale Fourier power |
| Fine positional | ClosePairs | Fraction of pairs within 4*epsilon |
| Fine structural | FoF | Friends-of-friends group count |
| Fine kinematic | VelDisp | Mean local velocity dispersion |

## Citation

If you use this code or data, please cite:

```
Bhatia, K. (2026). Predictive Structure in Three-Dimensional Self-Gravitating
N-body Systems: Initial-Condition Families, Observable Classes, and the Role
of Gravitational Softening. Preprint.
```

## License

This code is released for academic use. See the paper for full methodology and results.
