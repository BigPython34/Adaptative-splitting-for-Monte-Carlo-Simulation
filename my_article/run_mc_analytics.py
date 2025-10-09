#!/usr/bin/env python3
"""
Script for the analysis of the Monte Carlo pricing engine for a call option.
(High-precision version with a large number of simulations)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.stats import norm
from scipy.integrate import quad
# --- Imports from our finance library ---
from finance_lib.models import HullWhiteModel, CIRModel
from finance_lib.pricing_engine import MonteCarloDerivativesPricer

# --- Analytical Formula Implementation ---
def price_hw_call_analytical(model: HullWhiteModel, T: float, K: float) -> float:
    """
    Calculates the analytical price of a European call option on r(T).
    """
    beta, sigma = model.beta, model.r0
    
    mean_r_T_Q = model.r0 * np.exp(-beta * T)
    
    def integrand_mean(s):
        return np.exp(-beta * (T - s)) * model.b_function(s)
    mean_r_T_Q += quad(integrand_mean, 0, T)[0]
    
    v_T = np.sqrt((model.sigma**2 / (2 * beta)) * (1 - np.exp(-2 * beta * T)))
    
    shift = (model.sigma**2 / beta**2) * (1 - np.exp(-beta * T)) - \
            (model.sigma**2 / (2*beta**2)) * (1 - np.exp(-2*beta*T))
    mean_r_T_forward = mean_r_T_Q - shift
    
    d = (mean_r_T_forward - K) / v_T
    
    call_price = model.bond_price_analytical(0, T, model.r0) * (
        (mean_r_T_forward - K) * norm.cdf(d) + v_T * norm.pdf(d)
    )
    return call_price

# --- Main Analysis Function ---
def main():
    """Main function for Monte Carlo analysis."""
    print("--- Starting Monte Carlo Engine Analysis for European Call Option (High Precision) ---")

    FIGURES_DIR = Path("figures/monte_carlo")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Setup with YOUR Calibrated Models ---
    r0 = 0.024 

    # Calibrated CIR Model
    cir_params = {'b': 0.0132, 'beta': 0.3043, 'sigma': 0.1010, 'r0': r0}
    cir_model = CIRModel(**cir_params)
    cir_pricer = MonteCarloDerivativesPricer(cir_model)

    # Calibrated Flexible Hull-White Model
    hw_beta, hw_sigma = 0.2061, 0.0120
    def hw_b_func_flexible(t):
        if t <= 2.0: return 0.0014
        elif t <= 10.0: return 0.0121
        else: return 0.0107
    hw_model_flex = HullWhiteModel(hw_beta, hw_sigma, r0, hw_b_func_flexible)
    hw_pricer = MonteCarloDerivativesPricer(hw_model_flex)

    # Option parameters
    T, K, n_steps = 1.0, r0, 300

    # --- 2. Price Calculation (Analytical vs. MC) ---
    print("\n--- Pricing Comparison (High Precision) ---")
    # MODIFICATION : Augmentation du nombre de simulations à 1,000,000
    n_sims_large = 1_000_000 
    # ATTENTION : Ceci peut prendre quelques minutes.

    price_hw_analytic = price_hw_call_analytical(hw_model_flex, T, K)
    print(f"Hull-White Analytical Price: {price_hw_analytic:.6f} (Benchmark)")
    
    res_hw_mc = hw_pricer.price_european_call_on_rate(T, K, n_sims_large, n_steps)
    price_hw_mc = res_hw_mc["price"]
    print(f"Hull-White Monte Carlo Price:  {price_hw_mc:.6f} (N={n_sims_large:,})")

    res_cir_mc = cir_pricer.price_european_call_on_rate(T, K, n_sims_large, n_steps)
    price_cir_mc = res_cir_mc["price"]
    print(f"CIR Monte Carlo Price:         {price_cir_mc:.6f} (N={n_sims_large:,})")

    # --- 3. Convergence Analysis (Separate Plots) ---
    print("\n[1/2] Running convergence analysis...")
    # MODIFICATION : Ajout de points de simulation plus élevés
    n_sims_list = [1000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000]
    hw_prices, cir_prices = [], []

    for n_sims in tqdm(n_sims_list, desc="Convergence Simulation"):
        hw_prices.append(hw_pricer.price_european_call_on_rate(T, K, n_sims, n_steps)["price"])
        cir_prices.append(cir_pricer.price_european_call_on_rate(T, K, n_sims, n_steps)["price"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Monte Carlo Price Convergence for ATM Call Option", fontsize=16)

    axes[0].plot(n_sims_list, hw_prices, 'o-', label='MC Price')
    axes[0].axhline(price_hw_analytic, color='red', ls='--', label='Analytical Price')
    axes[0].set_xscale('log'), axes[0].set_title('Hull-White Model (Validation)'), axes[0].legend()
    axes[0].set_xlabel('Number of Simulations (N)'), axes[0].set_ylabel('Estimated Option Price')
    axes[0].grid(True, which='both', linestyle='--')

    axes[1].plot(n_sims_list, cir_prices, 's-', color='darkorange', label='MC Price')
    axes[1].set_xscale('log'), axes[1].set_title('CIR Model (Estimation)'), axes[1].legend()
    axes[1].set_xlabel('Number of Simulations (N)'), axes[1].set_ylabel('Estimated Option Price')
    axes[1].grid(True, which='both', linestyle='--')

    convergence_path = FIGURES_DIR / "monte_carlo_convergence_separate.png"
    plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to: {convergence_path}")

    # --- 4. Discounted Payoff Distribution (Separate Plots) ---
    print("\n[2/2] Analyzing discounted payoff distributions...")
    hw_paths = hw_model_flex.simulate_euler(T, n_steps, n_sims_large)
    cir_paths = cir_model.simulate_euler(T, n_steps, n_sims_large)
    
    hw_payoffs = np.maximum(hw_paths[:, -1] - K, 0)
    hw_df = np.exp(-np.sum(hw_paths[:, :-1] * (T / n_steps), axis=1))
    hw_discounted_payoffs = hw_df * hw_payoffs
    
    cir_payoffs = np.maximum(cir_paths[:, -1] - K, 0)
    cir_df = np.exp(-np.sum(cir_paths[:, :-1] * (T / n_steps), axis=1))
    cir_discounted_payoffs = cir_df * cir_payoffs

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(f"Distribution of Discounted Payoffs (N={n_sims_large:,})", fontsize=16)

    axes[0].hist(hw_discounted_payoffs, bins=100, density=True, range=(0, np.percentile(hw_discounted_payoffs[hw_payoffs > 0], 99.5)))
    axes[0].axvline(price_hw_mc, color='red', ls='--', lw=2, label=f'MC Price = {price_hw_mc:.6f}')
    axes[0].set_title('Hull-White Model'), axes[0].legend()
    axes[0].set_xlabel('Value of Discounted Payoff'), axes[0].set_ylabel('Probability Density')

    axes[1].hist(cir_discounted_payoffs, bins=100, density=True, color='darkorange', range=(0, np.percentile(cir_discounted_payoffs[cir_payoffs > 0], 99.5)))
    axes[1].axvline(price_cir_mc, color='red', ls='--', lw=2, label=f'MC Price = {price_cir_mc:.6f}')
    axes[1].set_title('CIR Model'), axes[1].legend()
    axes[1].set_xlabel('Value of Discounted Payoff')

    distribution_path = FIGURES_DIR / "discounted_payoff_distribution_separate.png"
    plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {distribution_path}")

    print("\n--- Monte Carlo analysis complete. ---")

if __name__ == "__main__":
    main()