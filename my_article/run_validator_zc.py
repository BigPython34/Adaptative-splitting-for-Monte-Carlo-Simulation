#!/usr/bin/env python3
"""
Validation script for zero-coupon (ZC) bond prices.

This script compares the analytical ("true") price of a zero-coupon bond
with the price estimated via Monte Carlo simulation for both CIR and
Hull-White models. It also generates distribution plots to visualize
the convergence of the Monte Carlo estimator's mean towards the theoretical value.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Imports from our finance library ---
from finance_lib.models import CIRModel, HullWhiteModel, BaseShortRateModel

def validate_model_with_plot(model: BaseShortRateModel, model_name: str, T: float, n_sims: int, n_steps_per_year: int):
    """
    Validates a model and generates a plot of the estimator distribution.
    """
    print("-" * 60)
    print(f"--- Validating Model: {model_name} ---")
    
    FIGURES_DIR = Path("figures/validation")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Analytical Price Calculation ---
    analytical_price = model.bond_price_analytical(t=0, T=T, r_at_t=model.r0)
    print(f"Analytical Price (Truth): {analytical_price:.6f}")

    # --- 2. Monte Carlo Price Estimation ---
    total_steps = int(n_steps_per_year * T)
    paths = model.simulate_euler(T=T, n_steps=total_steps, n_paths=n_sims)

    dt = T / total_steps
    integrated_rates = np.sum(paths[:, :-1], axis=1) * dt
    discount_factors = np.exp(-integrated_rates)
    
    mc_price = np.mean(discount_factors)
    std_error = np.std(discount_factors) / np.sqrt(n_sims)
    
    print(f"Monte Carlo Price (Estimate): {mc_price:.6f}")
    
    # --- 3. Generate Distribution Plot ---
    plt.figure(figsize=(12, 7))
    plt.hist(discount_factors, bins=100, density=True, alpha=0.7, label='MC Estimator Distribution')
    
    plt.axvline(mc_price, color='royalblue', linestyle='--', lw=2, 
                label=f'MC Mean (Est. Price) = {mc_price:.6f}')
    
    plt.axvline(analytical_price, color='red', linestyle='-', lw=2.5, 
                label=f'Analytical Price (Truth) = {analytical_price:.6f}')
    
    # --- ENGLISH LABELS FOR THE PLOT ---
    plt.title(f'Validation: Estimator Distribution vs. Analytical Price ({model_name})')
    plt.xlabel("Estimator Value (Stochastic Discount Factor)")
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, linestyle='--')
    
    # Save the figure
    safe_model_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    output_path = FIGURES_DIR / f"validation_{safe_model_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Validation plot saved to: {output_path}")
    # plt.show()
    
    # --- 4. Validation Conclusion ---
    if abs(analytical_price - mc_price) / std_error < 3:
        print(f"✅ SUCCESS: The simulator for {model_name} is validated.")
    else:
        print(f"❌ FAILURE: The simulator for {model_name} appears incorrect or biased.")
    print("-" * 60)

def main():
    """Main function for model validation."""
    print("=" * 60)
    print("=== MC Simulators vs. Analytical Formulas Validation ===")
    print("=" * 60)

    # --- Test Configuration ---
    T = 5.0
    n_sims = 200000
    n_steps_per_year = 300
    r0 = 0.03

    # --- CIR Model Validation ---
    cir_model = CIRModel(b=0.05, beta=0.2, sigma=0.12, r0=r0)
    validate_model_with_plot(cir_model, "CIR", T, n_sims, n_steps_per_year)

    # --- Hull-White Model Validation ---
    hw_model = HullWhiteModel(beta=0.2, sigma=0.01, r0=r0, b_function=lambda t: 0.03)
    validate_model_with_plot(hw_model, "Hull-White (constant b)", T, n_sims, n_steps_per_year)

if __name__ == "__main__":
    main()