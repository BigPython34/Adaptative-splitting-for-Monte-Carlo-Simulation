# Stochastic Interest Rate Models

A professional-grade Python library for stochastic short rate modeling and interest rate derivatives pricing. This project provides robust implementations of the Cox-Ingersoll-Ross (CIR) and Hull-White models with analytical formulas, Monte Carlo pricing, and market calibration tools.

## Overview

Short rate models are fundamental tools in fixed income quantitative finance. This library provides:

- **Two classical short rate models** with analytical solutions where available
- **Monte Carlo pricing engine** for path-dependent derivatives
- **Advanced calibration framework** with flexible specifications
- **Comprehensive validation suite** against closed-form solutions
- **Professional visualization tools** for analysis and presentation

## Project Structure

```
my_article/
├── finance_lib/                  # Core library
│   ├── __init__.py
│   ├── models.py                 # CIR and Hull-White model implementations
│   ├── pricing_engine.py         # Monte Carlo derivatives pricer
│   ├── calibration.py            # Market calibration tools
│   ├── visualization.py          # Plotting utilities
│   ├── risk_analysis.py          # Risk metrics (VaR, CVaR)
│   └── data_fetcher.py           # Market data retrieval
├── run_model_showcase.py         # Model dynamics visualization
├── run_validator_zc.py            # Validate analytical formulas
├── run_calibration_study.py      # Calibrate to market data
├── run_mc_analytics.py            # Monte Carlo pricing analysis
├── run_advanced_applications.py  # Advanced derivatives pricing
├── figures/                       # Generated visualizations
│   ├── cir_hw_comparison/
│   ├── validation/
│   ├── calibration/
│   └── monte_carlo/
└── CIR_and_pricing.pdf           # Theoretical reference
```

## Installation

### Prerequisites

- Python 3.8+
- NumPy, SciPy, Matplotlib

### Setup

```powershell
# Navigate to project directory
cd my_article

# Install dependencies
pip install numpy scipy matplotlib
```

## Quick Start

### 1. Model Dynamics Visualization

Explore the behavior of CIR and Hull-White models:

```powershell
python run_model_showcase.py
```

**Output**:
- Short rate trajectory plots
- Distribution of rates at maturity
- Yield curve comparison

**Use case**: Understand model characteristics before pricing.

### 2. Validate Analytical Formulas

Verify that Monte Carlo simulations match analytical bond prices:

```powershell
python run_validator_zc.py
```

**Output**: Validation plots comparing analytical vs simulated zero-coupon bond prices

**Use case**: Ensure implementation correctness.

### 3. Calibrate to Market Data

Fit model parameters to observed yield curves:

```powershell
python run_calibration_study.py
```

**Output**:
- Calibrated parameters for both models
- Fit quality visualization
- Model vs market yield curve comparison

**Use case**: Prepare models for real-world pricing.

### 4. Monte Carlo Pricing Analysis

Price derivatives using Monte Carlo and analyze convergence:

```powershell
python run_mc_analytics.py
```

**Output**:
- Option price estimates with confidence intervals
- Convergence analysis plots
- Distribution of discounted payoffs

**Use case**: Price exotic derivatives and assess accuracy.

## Models

### Cox-Ingersoll-Ross (CIR) Model

**Stochastic differential equation**:
```
dr(t) = (b - β·r(t))dt + σ·√r(t)·dW(t)
```

**Features**:
- Ensures non-negative rates (important for realism)
- Mean-reverting dynamics
- Analytical zero-coupon bond formula
- Feller condition: 2b ≥ σ² ensures strict positivity

**Parameters**:
- `b`: Long-term mean level (drift)
- `β` (beta): Mean reversion speed
- `σ` (sigma): Volatility
- `r0`: Initial short rate

**Example**:
```python
from finance_lib.models import CIRModel

cir = CIRModel(b=0.06, beta=0.2, sigma=0.15, r0=0.03)

# Simulate trajectories
paths = cir.simulate_euler(T=5.0, n_steps=250, n_paths=1000)

# Analytical bond price
price = cir.bond_price_analytical(t=0, T=10, r_at_t=0.03)

# Yield curve
maturities = np.linspace(0.5, 10, 20)
yields = cir.yield_curve(t=0, maturities=maturities, r_at_t=0.03)
```

### Hull-White Model

**Stochastic differential equation**:
```
dr(t) = (b(t) - β·r(t))dt + σ·dW(t)
```

**Features**:
- Time-dependent drift b(t) allows perfect fit to term structure
- Gaussian rates (can be negative)
- Analytical zero-coupon bond formula
- Constant volatility (tractable and commonly used)

**Parameters**:
- `b_function`: Time-dependent drift function
- `β` (beta): Mean reversion speed
- `σ` (sigma): Constant volatility
- `r0`: Initial short rate

**Example**:
```python
from finance_lib.models import HullWhiteModel

# Constant drift
hw = HullWhiteModel(beta=0.2, sigma=0.01, r0=0.03, 
                    b_function=lambda t: 0.03)

# Piecewise-constant drift for better calibration
def flexible_drift(t):
    if t <= 2.0: return 0.001
    elif t <= 10.0: return 0.012
    else: return 0.011

hw_flex = HullWhiteModel(beta=0.2, sigma=0.01, r0=0.03,
                         b_function=flexible_drift)

# Same API as CIR
paths = hw.simulate_euler(T=5.0, n_steps=250, n_paths=1000)
price = hw.bond_price_analytical(t=0, T=10, r_at_t=0.03)
```

## Monte Carlo Pricing Engine

### MonteCarloDerivativesPricer

A versatile pricing engine for interest rate derivatives.

**Supported Products**:

#### 1. European Call on Short Rate

**Payoff**: `max(r(T) - K, 0)`

```python
from finance_lib.models import CIRModel
from finance_lib.pricing_engine import MonteCarloDerivativesPricer

model = CIRModel(b=0.06, beta=0.2, sigma=0.15, r0=0.03)
pricer = MonteCarloDerivativesPricer(model)

result = pricer.price_european_call_on_rate(
    T=1.0,           # Maturity
    strike=0.03,     # Strike rate
    n_sims=100000,   # Number of simulations
    n_steps=300      # Time steps per path
)

print(f"Price: {result['price']:.6f}")
print(f"Std Error: {result['std_error']:.6f}")
print(f"95% CI: {result['confidence_interval_95']}")
```

#### 2. Range Accrual Note

**Payoff**: `Notional × (fraction of time r(t) ∈ [r_min, r_max])`

```python
price = pricer.price_range_accrual_note(
    T=2.0,
    r_min=0.02,
    r_max=0.04,
    notional=1000000,
    n_observations=100,
    n_sims=50000
)
```

### Pricing Features

- **Stochastic discounting**: Proper path-dependent discount factors
- **Standard errors**: Automatic computation of Monte Carlo error
- **Confidence intervals**: 95% CI provided by default
- **Efficient vectorization**: NumPy-based for speed

## Calibration

### Market Calibration

Fit model parameters to observed market yield curves using optimization.

**Basic CIR Calibration**:
```python
from finance_lib.calibration import calibrate_cir_model
import numpy as np

# Market data
market_maturities = np.array([1, 2, 5, 10, 20, 30])
market_yields = np.array([0.01, 0.015, 0.025, 0.03, 0.032, 0.033])
r0_proxy = market_yields[0]

# Calibrate
calibrated_cir = calibrate_cir_model(market_maturities, market_yields, r0_proxy)

# Result: CIR model with optimal (b, beta, sigma) parameters
print(f"b = {calibrated_cir.b:.4f}")
print(f"beta = {calibrated_cir.beta:.4f}")
print(f"sigma = {calibrated_cir.sigma:.4f}")
```

**Flexible Hull-White Calibration**:
```python
from finance_lib.calibration import calibrate_hw_model_flexible

# Piecewise-constant drift for better fit
calibrated_hw = calibrate_hw_model_flexible(
    market_maturities, 
    market_yields, 
    r0_proxy
)

# Better fit to term structure than constant drift
```

### Calibration Features

- **L-BFGS-B optimization**: Constrained optimization for parameter bounds
- **Mean squared error objective**: Minimize yield curve fitting error
- **Flexible drift specification**: Piecewise-constant b(t) for Hull-White
- **Weighted fitting**: Optional weighting to emphasize short maturities
- **Robust bounds**: Physical constraints (positive volatilities, etc.)

## Validation

The library includes comprehensive validation:

### Zero-Coupon Bond Validation

Compares analytical formulas to Monte Carlo simulations:

```python
# run_validator_zc.py demonstrates:
# 1. Generate paths to maturity T
# 2. Compute stochastic discount factor for each path
# 3. Average discounts = MC bond price
# 4. Compare to analytical formula

# Typical accuracy: <0.5% relative error with 100k paths
```

### Convergence Analysis

Monitors Monte Carlo estimator convergence:

```python
# run_mc_analytics.py shows:
# - Price vs number of simulations
# - Standard error decay (~ 1/√N)
# - Comparison to analytical benchmarks (Hull-White call)
```

## API Reference

### BaseShortRateModel (Abstract Base Class)

```python
class BaseShortRateModel:
    def __init__(self, r0: float)
    def simulate_euler(self, T, n_steps, n_paths=1) -> np.ndarray
    def bond_price_analytical(self, t, T, r_at_t) -> float
    def yield_curve(self, t, maturities, r_at_t) -> np.ndarray
```

### CIRModel

```python
class CIRModel(BaseShortRateModel):
    def __init__(self, b: float, beta: float, sigma: float, r0: float)
    # Inherits: simulate_euler, bond_price_analytical, yield_curve
```

### HullWhiteModel

```python
class HullWhiteModel(BaseShortRateModel):
    def __init__(self, beta: float, sigma: float, r0: float, 
                 b_function: Callable[[float], float])
    # Inherits: simulate_euler, bond_price_analytical, yield_curve
```

### MonteCarloDerivativesPricer

```python
class MonteCarloDerivativesPricer:
    def __init__(self, model: BaseShortRateModel)
    
    def price_european_call_on_rate(self, T, strike, n_sims, n_steps) -> Dict
    # Returns: {"price": float, "std_error": float, "confidence_interval_95": tuple}
    
    def price_range_accrual_note(self, T, r_min, r_max, notional, 
                                  n_observations, n_sims) -> float
```

### Calibration Functions

```python
def calibrate_cir_model(market_maturities, market_yields, r0_proxy) -> CIRModel

def calibrate_hw_model(market_maturities, market_yields, r0_proxy) -> HullWhiteModel

def calibrate_hw_model_flexible(market_maturities, market_yields, r0_proxy) -> HullWhiteModel
```

## Visualization

### Available Plots

```python
from finance_lib.visualization import (
    plot_short_rate_trajectories,
    plot_final_rate_distribution,
    plot_yield_curves_comparison
)

# Trajectory plot
plot_short_rate_trajectories(model, "CIR", T=5, n_steps=250, n_paths=10)

# Distribution at maturity
plot_final_rate_distribution(model, "CIR", T=1, n_sims=20000)

# Yield curve comparison
models_dict = {"CIR": cir_model, "Hull-White": hw_model}
plot_yield_curves_comparison(models_dict, r_current=0.03, max_maturity=10)
```

All plots are saved to `figures/` with publication-quality settings.

## Performance Tips

### Simulation Performance

```python
# For pricing, use:
n_sims = 100000   # Good balance for most applications
n_steps = 300     # Ensures accuracy for T=1 year options

# For quick tests:
n_sims = 10000
n_steps = 100

# For high precision:
n_sims = 1000000
n_steps = 500
```

### Memory Considerations

Large simulations generate arrays of shape `(n_paths, n_steps+1)`:

- 1M paths × 300 steps × 8 bytes ≈ 2.4 GB
- Consider batching for very large simulations
- Or use generators for streaming computation

## Advanced Applications

### Custom Derivatives

Extend the pricing engine for custom payoffs:

```python
class MyDerivativesPricer(MonteCarloDerivativesPricer):
    def price_custom_derivative(self, T, n_sims, n_steps):
        paths = self.model.simulate_euler(T, n_steps, n_sims)
        
        # Your custom payoff function
        payoffs = self.compute_payoff(paths)
        
        # Stochastic discounting
        dt = T / n_steps
        integrated_rates = np.sum(paths[:, :-1], axis=1) * dt
        discount_factors = np.exp(-integrated_rates)
        
        price = np.mean(payoffs * discount_factors)
        std_error = np.std(payoffs * discount_factors) / np.sqrt(n_sims)
        
        return {"price": price, "std_error": std_error}
```

### Time-Dependent Parameters

For CIR, extend to time-dependent parameters:

```python
class TimeVaryingCIRModel(CIRModel):
    def __init__(self, b_func, beta_func, sigma_func, r0):
        # Override _get_drift_diffusion to use time-dependent params
        pass
```

### Multi-Factor Models

Extend to multi-factor frameworks:

```python
# Two-factor model: r(t) = x(t) + y(t)
# Each factor follows its own SDE
# Correlated Brownian motions via Cholesky decomposition
```

## Theoretical Background

This implementation follows standard quantitative finance references:

- **CIR Model**: Cox, Ingersoll, Ross (1985), "A Theory of the Term Structure of Interest Rates"
- **Hull-White Model**: Hull & White (1990), "Pricing Interest-Rate-Derivative Securities"
- **Analytical Formulas**: See `CIR_and_pricing.pdf` for derivations

### Key Formula: CIR Bond Price

```
P(t,T) = A(t,T) · exp(-B(t,T) · r(t))

where:
B(t,T) = 2(e^(d·τ) - 1) / [(β+d)(e^(d·τ)-1) + 2d]
A(t,T) = [2d·e^((β+d)τ/2) / ((β+d)(e^(d·τ)-1) + 2d)]^(2b/σ²)
d = √(β² + 2σ²)
τ = T - t
```

### Key Formula: Hull-White Bond Price

```
P(t,T) = exp(A(t,T) - B(t,T)·r(t))

where:
B(t,T) = (1 - e^(-β(T-t))) / β
A(t,T) = ∫ₜᵀ [0.5σ²B²(s,T) - b(s)B(s,T)] ds
```

## Troubleshooting

### Negative Rates in CIR

**Symptom**: Warnings about negative rates

**Causes**: 
- Feller condition not met: 2b < σ²
- Too large time steps (Euler discretization error)

**Solutions**:
- Ensure 2b ≥ σ² when setting parameters
- Increase n_steps (smaller dt)
- Model automatically clips negative values to zero

### Calibration Fails

**Symptom**: Poor fit to market curve

**Solutions**:
- Use `calibrate_hw_model_flexible` for better flexibility
- Check that market data is reasonable (no arbitrage)
- Try different initial parameter guesses
- Increase bounds if optimizer hits limits

### High Monte Carlo Error

**Symptom**: Large standard errors in pricing

**Solutions**:
- Increase n_sims (primary solution)
- Use variance reduction techniques (not implemented yet)
- Check if payoff is too discontinuous (use more steps)

## Extensions & Future Work

Potential enhancements:

1. **Additional Models**: Vasicek, G2++, SABR
2. **Variance Reduction**: Control variates, antithetic variables
3. **Calibration**: Include swaption volatilities, caps/floors
4. **American Options**: LSM or tree methods
5. **GPU Acceleration**: CuPy for massive simulations
6. **Interest Rate Swaps**: Complete swap pricer
7. **Greeks**: Delta, gamma, vega via finite differences or pathwise

## Citation

```bibtex
@software{stochastic_rates_2025,
  author = {BigPython34},
  title = {Stochastic Interest Rate Models Library},
  year = {2025},
  url = {https://github.com/BigPython34/Adaptative-splitting-for-Monte-Carlo-Simulation}
}
```

## Related Projects

- See `../adaptative_splitting/` for advanced Monte Carlo variance reduction techniques
- Both projects demonstrate state-of-the-art simulation methods

## Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Contributions welcome!

## License

See main repository LICENSE file.
