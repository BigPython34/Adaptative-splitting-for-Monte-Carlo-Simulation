# Adaptive Splitting for Rare Event Simulation

A comprehensive implementation of Sequential Monte Carlo (SMC) methods with adaptive threshold selection for rare event probability estimation. This project implements and compares three different approaches to rare event simulation, providing a robust framework for comparative analysis.

## Overview

Rare event estimation is a critical problem in many domains (finance, reliability engineering, physics) where standard Monte Carlo methods are inefficient. This project implements state-of-the-art variance reduction techniques based on importance splitting.

### Implemented Methods

1. **Adaptive SMC (Algorithm 2)**: Automatically selects optimal thresholds based on sample quantiles, adapting to the problem geometry
2. **Fixed-Level SMC (Algorithm 1)**: Uses predetermined equally-spaced threshold levels
3. **Naive Monte Carlo**: Baseline direct sampling for comparison

## Project Structure

```
adaptative_splitting/
├── smc/                          # Core SMC library
│   ├── __init__.py
│   ├── config.py                 # Centralized configuration
│   ├── core.py                   # Mathematical core functions (phi, MCMC kernel)
│   ├── smc_algorithms.py         # Pure algorithm implementations
│   └── plotting.py               # Visualization functions
├── run_single_demo.py            # Quick demonstration script
├── run_comparaison_study.py      # Comprehensive comparative analysis
├── run_risk_analysis.py          # Financial risk application
├── run_simple_L_comparaison.py   # Simple threshold comparison
├── figures/                      # Generated visualizations
│   ├── analyse_adaptive_smc/
│   └── comparaison/
└── results/                      # Cached simulation results
```

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy, SciPy, Matplotlib, tqdm

### Setup

```powershell
# Navigate to project directory
cd adaptative_splitting

# Install dependencies (if not already installed)
pip install numpy scipy matplotlib tqdm
```

## Quick Start

### 1. Single Demonstration Run

Run a single comparison of all three methods:

```powershell
python run_single_demo.py
```

**Output**: Console comparison table showing:
- Theoretical probability
- Each method's estimate, relative error, and runtime
- Number of iterations/samples used

**Use case**: Quick verification that algorithms work correctly.

### 2. Comprehensive Comparative Study

Run extensive simulations across multiple threshold levels:

```powershell
python run_comparaison_study.py
```

**Output**:
- `figures/comparaison/compare_methods_graph_all.png`: Relative error vs threshold
- `figures/comparaison/plot_thresholds_on_gaussian.png`: Threshold visualization
- `figures/comparaison/compare_variances_boxplot.png`: Estimator variance comparison
- `results/simulation_results.pkl`: Cached results for reuse

**Use case**: Publication-quality comparative analysis.

### 3. Financial Risk Application

Apply SMC to portfolio risk analysis (Value-at-Risk estimation):

```powershell
python run_risk_analysis.py
```

**Application**: Estimates tail probabilities for portfolio losses using a realistic stock+option portfolio.

## Core Algorithm Details

### Problem Setup

We estimate rare event probabilities of the form:

```
P(φ(X) > L) where X ~ N(0,1) and φ(x) = -x
```

For threshold L = 7, this probability ≈ 1.3e-12, making naive Monte Carlo impractical.

### Adaptive SMC Algorithm

**Key steps**:
1. Initialize N particles from target distribution
2. Compute score φ(x) for each particle
3. Set threshold Lₖ as (1-p₀) quantile of scores
4. Keep only particles with φ(x) ≥ Lₖ (fraction p₀ survives)
5. Resample survivors to N particles
6. Apply MCMC mutations to explore constrained space
7. Repeat until target threshold L reached

**Advantages**:
- Automatic threshold adaptation
- Robust to problem geometry
- Typically more efficient than fixed levels

### MCMC Kernel

The Metropolis-Hastings kernel samples from N(0,1) truncated at x ≤ -L:

- **Proposal**: x' ~ N(x, σ²)
- **Acceptance probability**: min(1, exp(-0.5(x'² - x²)))
- **Constraint**: Only accept if x' ≤ -L

Tuning parameter σ controls acceptance rate (target ~30-70%).

## Configuration

Configuration is centralized in `smc/config.py`:

### Default Parameters

```python
DEFAULT_MCMC_PARAMS = {
    "p0": 0.5,          # Survival probability per level
    "sigma": 1.0,       # MCMC proposal standard deviation
    "n_mcmc": 80,       # MCMC steps per mutation phase
    "L_target": 7,      # Target threshold
    "N": 5000,          # Number of particles
    "max_iter": 100,    # Maximum iterations
}
```

### Key Parameters

- **p0**: Controls number of levels (lower → more levels, lower variance, slower)
- **sigma**: MCMC exploration width (tune for ~30-70% acceptance)
- **n_mcmc**: MCMC equilibration steps (higher → better mixing, slower)
- **N**: Particle count (higher → lower variance, slower)
- **L_target**: Rare event threshold (higher → rarer event)

## API Reference

### Core Functions

#### `phi(x: np.ndarray) -> np.ndarray`

Score function for rare event detection.

```python
from smc.core import phi
scores = phi(samples)  # Returns -x for Gaussian tail problem
```

#### `mcmc_kernel(x, L_current, n_steps, sigma, return_trace=False)`

Metropolis-Hastings kernel for truncated sampling.

**Returns**: `(x_final, acceptance_rate, trace)`

### Algorithm Functions

#### `adaptive_smc_run(N, p0, phi_function, initial_sampler, L_target, n_mcmc, sigma, max_iter)`

Executes adaptive SMC algorithm.

**Returns**: `AdaptiveSMCResult` object with:
- `prob_est`: Estimated probability
- `thresholds`: List of adaptive thresholds
- `acc_rates`: MCMC acceptance rates per level
- `n_iter`: Number of iterations
- `particle_means`, `particle_vars`: Diagnostics

#### `fixed_smc_run(N, thresholds, n_mcmc, sigma)`

Executes fixed-level SMC algorithm.

**Returns**: `FixedSMCResult` object

#### `run_naive_mc(L_target, num_samples, batch_size)`

Naive Monte Carlo estimation.

**Returns**: Estimated probability (float)

### Visualization Functions

All plotting functions in `smc/plotting.py` accept configuration dictionaries:

- `plot_methods_graph_all()`: Relative error comparison
- `plot_thresholds_on_gaussian()`: Threshold overlay on PDF
- `plot_variances_boxplot()`: Estimator variance comparison
- `plot_relative_errors_boxplot()`: Relative error boxplots

## Performance

### Benchmarks (L_target = 7)

On a typical modern CPU with 60-second time budget:

| Method | Relative Error | Speedup vs Naive |
|--------|---------------|------------------|
| Naive MC | ~50-100% | 1x |
| Fixed SMC | ~5-15% | ~10x |
| Adaptive SMC | ~2-8% | ~20x |

**Note**: Adaptive SMC typically achieves 10-100× variance reduction compared to naive MC.

### Parallelization

Comparison studies use multiprocessing for time-budgeted experiments:

```python
# Automatically distributes runs across available cores
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(adaptive_smc_run, ...) for _ in range(n_runs)]
```

## Customization

### Custom Score Functions

To adapt for your rare event problem:

```python
def custom_phi(x):
    """Your score function."""
    return your_transformation(x)

# Use in adaptive_smc_run via phi_function parameter
result = adaptive_smc_run(
    N=5000, 
    p0=0.5, 
    phi_function=custom_phi,  # Your function
    initial_sampler=your_sampler,
    L_target=your_threshold,
    ...
)
```

### Custom Initial Distributions

```python
def custom_sampler(N):
    """Generate N initial particles."""
    return your_distribution.rvs(size=N)

result = adaptive_smc_run(..., initial_sampler=custom_sampler, ...)
```

## Troubleshooting

### Algorithm Fails (Returns None)

**Cause**: No particles survived at some level

**Solutions**:
- Increase N (more particles)
- Increase p0 (keep more survivors per level)
- Increase n_mcmc (better MCMC mixing)
- Tune sigma (check acceptance rates in output)

### High Variance / Poor Accuracy

**Solutions**:
- Increase N (more particles)
- Decrease p0 (more levels, more splitting)
- Increase n_mcmc (reduce MCMC bias)
- Run multiple independent replications

### Slow Performance

**Solutions**:
- Decrease n_mcmc (if acceptance rates are good)
- Increase p0 (fewer levels)
- Use multiprocessing for independent runs
- Consider fixed-level SMC (faster but less accurate)

## Theoretical Background

This implementation is based on the adaptive multilevel splitting literature:

- Cérou, F., & Guyader, A. (2007). "Adaptive multilevel splitting for rare event analysis"
- See `../Morio-article2.pdf` for additional theoretical context

### Key Concepts

- **Importance Splitting**: Gradually focus sampling on rare region
- **Adaptive Thresholds**: Data-driven level selection
- **MCMC Mutations**: Maintain diversity while respecting constraints
- **Unbiased Estimation**: Proper weighting ensures correctness

## Extensions

Potential extensions for advanced users:

1. **Multi-dimensional problems**: Extend φ to vector-valued inputs
2. **Different MCMC kernels**: Hamiltonian Monte Carlo, Langevin dynamics
3. **Alternative resampling schemes**: Stratified, systematic
4. **Parallel tempering**: Multiple MCMC chains per particle
5. **Application to other distributions**: Non-Gaussian targets

## Citation

If you use this code in academic work, please cite:

```bibtex
@software{adaptive_smc_2025,
  author = {BigPython34},
  title = {Adaptive Splitting for Monte Carlo Simulation},
  year = {2025},
  url = {https://github.com/BigPython34/Adaptative-splitting-for-Monte-Carlo-Simulation}
}
```

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

## Related Projects

- See `../my_article/` for stochastic interest rate modeling applications
- Both projects demonstrate advanced Monte Carlo techniques in different domains

## License

See main repository LICENSE file.
