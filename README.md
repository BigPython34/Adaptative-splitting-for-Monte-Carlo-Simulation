# Monte Carlo Simulation Research Projects

This repository contains two independent but complementary research projects focused on advanced Monte Carlo simulation techniques applied to rare event estimation and financial mathematics.

## Repository Structure

```
├── adaptative_splitting/     # Sequential Monte Carlo with adaptive splitting
│   ├── smc/                  # Core SMC library
│   └── run_*.py             # Simulation scripts
├── my_article/              # Stochastic interest rate models
│   ├── finance_lib/         # Financial modeling library
│   └── run_*.py            # Analysis scripts
└── figures/                 # Generated visualizations
```

## Project 1: Adaptive Splitting for Rare Event Simulation

**Location**: `adaptative_splitting/`

A sophisticated implementation of Sequential Monte Carlo (SMC) methods with adaptive threshold selection for rare event probability estimation. This project implements and compares three approaches:

- **Adaptive SMC**: Automatically selects optimal thresholds based on quantiles
- **Fixed-Level SMC**: Uses predetermined threshold levels
- **Naive Monte Carlo**: Baseline comparison method

### Key Features

- Pure algorithm implementations with minimal dependencies
- Comprehensive comparative analysis framework
- Parallel execution support for performance benchmarking
- Extensive visualization toolkit for results analysis
- Configuration-driven architecture for reproducibility

### Applications

- Rare event probability estimation (tail probabilities)
- Financial risk analysis (Value-at-Risk, portfolio losses)
- Reliability engineering
- Statistical physics simulations

### Quick Start

```powershell
cd adaptative_splitting
python run_single_demo.py      # Single demonstration run
python run_comparaison_study.py # Comprehensive comparative study
python run_risk_analysis.py     # Financial risk application
```

See [`adaptative_splitting/README.md`](adaptative_splitting/README.md) for detailed documentation.

## Project 2: Stochastic Interest Rate Models

**Location**: `my_article/`

A professional-grade implementation of stochastic short rate models for fixed income derivatives pricing. The library includes:

- **Cox-Ingersoll-Ross (CIR)** model with analytical bond pricing
- **Hull-White** model with flexible time-dependent parameters
- Monte Carlo pricing engine for exotic derivatives
- Real-world calibration to market yield curves

### Key Features

- Object-oriented architecture with clean abstractions
- Analytical formulas where available (zero-coupon bonds)
- Monte Carlo pricing for path-dependent derivatives
- Advanced calibration with flexible drift specifications
- Comprehensive validation against closed-form solutions
- Professional visualization suite

### Applications

- Zero-coupon bond pricing
- European options on short rates
- Range accrual notes
- Model calibration to market data
- Yield curve analysis
- Interest rate derivatives risk management

### Quick Start

```powershell
cd my_article
python run_model_showcase.py    # Model dynamics visualization
python run_validator_zc.py       # Validate analytical formulas
python run_calibration_study.py  # Calibrate to market data
python run_mc_analytics.py       # Monte Carlo pricing analysis
```

See [`my_article/README.md`](my_article/README.md) for detailed documentation.

## Technical Stack

### Common Dependencies

- **Python** 3.8+
- **NumPy**: Numerical computations
- **SciPy**: Statistical functions and optimization
- **Matplotlib**: Visualization
- **tqdm**: Progress bars (for long simulations)

### Specialized Tools

- **Multiprocessing**: Parallel execution for performance studies
- **Pickle**: Results caching and persistence

## Installation

1. Clone the repository:
```powershell
git clone https://github.com/BigPython34/Adaptative-splitting-for-Monte-Carlo-Simulation.git
cd Adaptative-splitting-for-Monte-Carlo-Simulation
```

2. Install dependencies:
```powershell
pip install numpy scipy matplotlib tqdm
```

3. Run tests/demos:
```powershell
# Test adaptive splitting
cd adaptative_splitting
python run_single_demo.py

# Test interest rate models
cd ..\my_article
python run_model_showcase.py
```

## Project Origins

These projects originated from academic research in computational finance and rare event simulation:

- **Adaptive Splitting**: Based on Sequential Monte Carlo literature for rare event estimation, with applications to financial risk metrics
- **Interest Rate Models**: Implementation of classical short rate models (CIR, Hull-White) with modern Monte Carlo techniques for derivatives pricing

## Results and Visualizations

Both projects generate publication-quality figures in their respective `figures/` directories:

- **Adaptive Splitting**: Method comparison plots, convergence analysis, threshold visualization
- **Interest Rate Models**: Trajectory plots, yield curves, calibration fits, pricing convergence

## Performance Considerations

- **Adaptive SMC**: Uses multiprocessing for time-budgeted comparisons
- **Monte Carlo Pricing**: Efficient vectorized operations for large simulations
- **Caching**: Results are cached to avoid redundant computations

## Code Quality

- Clean separation of concerns (algorithms, configuration, visualization)
- Type hints for improved code clarity
- Comprehensive docstrings
- Modular architecture for extensibility
- English code and comments (cleaned for clarity)

## Contributing

This is a research codebase. If you find issues or have suggestions:

1. Open an issue describing the problem
2. For major changes, discuss in an issue first
3. Follow the existing code style and structure

## License

See LICENSE file for details.

## Authors

Research and implementation by BigPython34

## References

### Adaptive Splitting
- Cérou, F., & Guyader, A. (2007). "Adaptive multilevel splitting for rare event analysis"
- See `Morio-article2.pdf` for additional theoretical background

### Interest Rate Models
- Cox, J. C., Ingersoll Jr, J. E., & Ross, S. A. (1985). "A theory of the term structure of interest rates"
- Hull, J., & White, A. (1990). "Pricing interest-rate-derivative securities"
- See `my_article/CIR_and_pricing.pdf` for detailed formulas

## Acknowledgments

Thanks to the scientific computing and quantitative finance communities for the foundational work these implementations build upon.
