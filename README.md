### Observable-Class Predictability in Stylised Three-Dimensional Self-Gravitating N-body Families

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**140,000 three-dimensional N-body simulations testing which observable class — coarse positional, fine positional, or fine kinematic — best predicts future gravitational clustering across four stylised initial-condition families, as a function of particle count and gravitational softening.**

Code and analysis pipeline for the paper:

> **Observable-Class Predictability in Stylised Three-Dimensional Self-Gravitating N-body Families**
>
> Kunal Bhatia (2026)
> [ORCID: 0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

## Scientific Summary

This paper asks: **in a controlled setting with stylised initial-condition families, which observable class — coarse positional, fine positional, or fine kinematic — best predicts future clustering?**
The systems are intentionally simplified (isolated or periodic boundaries, idealised IC families, fixed scalar target, no cosmological expansion), so the result is a statement about predictive structure in this controlled setting, not a general claim about self-gravitating systems.

We test three observable classes:
- **Coarse positional**: grid-scale density variance and radial concentration proxies
- **Fine positional**: nearest-neighbour density, Fourier power, close-pair fraction, FoF group count
- **Fine kinematic**: local velocity dispersion (the only observable accessing phase-space information)

across four initial-condition families (plus three angular-shuffle null controls = 7 total):
- **Bimodal** (two-cluster merger): stylised merger-like initial condition
- **Hernquist** cusp: concentrated cusp profile (scale radius *a*)
- **Plummer** sphere: softer concentrated profile (scale radius *a*)
- **Cold-clumpy** (multi-clump): pressureless multi-clump initial condition

### Key Results

1. **Bimodal (calibration anchor)**: Coarse density variance dominates in every tested cell (|r| = 0.984 at ε = 0.05, ranging from 0.961 to 0.994 across ε; winner-gap CIs entirely below zero at all N, ε, and both force models).

2. **Concentrated cusps — restricted kinematic advantage**: Local velocity dispersion outperforms the coarse predictor for concentrated cusp profiles at low softening (Hernquist ε ≤ 0.05, Plummer ε = 0.02; gap CIs exclude zero). The effect is modest: VelDisp explains ~18–28% of variance (R² at N = 1024). The advantage erodes with increasing ε and vanishes by ε = 0.10; the transition lies between ε/a ≈ 0.35 and 0.50 for these idealised families.

3. **No positional fine advantage**: No tested positional fine-scale observable outperforms the coarse predictor in any cell, with one exception: cold-clumpy at ε = 0.10, where ClosePairs acts as a proxy for inter-clump connectivity rather than true fine-scale structure.

4. **Scope**: All results are specific to the scalar target ΔC_8^early and the stylised families studied here. The paper is an observable-class comparison under a fixed target choice, not a broader characterisation of predictive structure in gravity. The ε/a transition range is a feature of these idealised profiles and should not be extrapolated to cosmological populations.

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
├── test_regression.py    # 36 regression tests
├── build.sh              # Reproducible build: run battery + compile paper
├── paper.tex             # LaTeX manuscript (anonymised for double-blind review)
├── paper.pdf             # Compiled paper (anonymised for double-blind review)
├── highlights.txt        # Elsevier submission highlights
├── declaration_of_interest.txt  # Declaration of no competing interests
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

Runs the analysis pipeline and compiles the LaTeX manuscript via `latexmk`.
Requires a LaTeX distribution with AASTeX 7.0.1 (`aastex701.cls`).
Install via `tlmgr --usermode install aastex` if not already present.

## Force Models

| Model | Method | Boundary | Cost | Softening |
|-------|--------|----------|------|-----------|
| `direct_isolated` | Pairwise O(N²) with Newton's 3rd law | Open | N(N−1)/2 pairs | Plummer ε |
| `pm_periodic` | FFT Poisson solver on 32³ grid | Periodic | O(N log N) | Grid resolution (~L/32) |

Direct-isolated is capped at N ≤ 2048. PM-periodic runs the full N range up to 16384. PM-periodic serves as a numerical cross-check with a different force architecture and boundary conditions; results are reported separately rather than pooled.

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
Bhatia, K. (2026). Observable-Class Predictability in Stylised Three-Dimensional
Self-Gravitating N-body Families: Coarse Dominance, Kinematic Advantage, and
Softening Dependence. Zenodo. https://doi.org/10.5281/zenodo.19643717
```

## License

MIT License. See [LICENSE](LICENSE) for details.
