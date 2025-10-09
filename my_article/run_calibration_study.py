#!/usr/bin/env python3
"""
Script for the study of interest rate model calibration.

This script:
1. Fetches the real market yield curve from the FRED database.
2. Calibrates three models: CIR, a simple Hull-White (constant drift), 
   and a flexible Hull-White (piecewise-constant drift).
3. Plots a comparison of the models' goodness-of-fit against the market curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Imports from our finance library ---
from finance_lib.data_fetcher import fetch_fred_yield_curve
from finance_lib.calibration import (
    calibrate_cir_model, 
    calibrate_hw_model, 
    calibrate_hw_model_flexible
)

def main():
    """Executes the calibration on the full curve and evaluates the fit."""
    print("--- Starting Calibration Study on the Full Yield Curve ---")
    
    # Define the output directory for figures
    FIGURES_DIR = Path("figures/calibration")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    market_mats, market_yields = fetch_fred_yield_curve()
    if market_mats is None:
        print("Aborting analysis due to data fetching failure.")
        return

    r0_proxy = market_yields[market_mats > 0][0]
    print(f"\nUsing r(0) = {r0_proxy*100:.3f}% (proxy from the shortest maturity)")

    # --- Calibrate all three models ---
    calibrated_cir = calibrate_cir_model(market_mats, market_yields, r0_proxy)
    calibrated_hw_simple = calibrate_hw_model(market_mats, market_yields, r0_proxy)
    calibrated_hw_flexible = calibrate_hw_model_flexible(market_mats, market_yields, r0_proxy)
    
    # --- Calculate the yield curves from the calibrated models ---
    cir_model_yields = calibrated_cir.yield_curve(0, market_mats, r0_proxy)
    hw_simple_yields = calibrated_hw_simple.yield_curve(0, market_mats, r0_proxy)
    hw_flexible_yields = calibrated_hw_flexible.yield_curve(0, market_mats, r0_proxy)

    # --- Visualization ---
    print("\nGenerating model fit comparison plot...")
    plt.figure(figsize=(14, 8))
    
    # Market data is the reference
    plt.plot(market_mats, market_yields * 100, 'o-', 
             label="Observed Market Curve (Target)", 
             color='black', lw=3, markersize=8, zorder=10)
    
    # Simple, rigid models
    plt.plot(market_mats, cir_model_yields * 100, 's--', 
             label="CIR (3-Parameter Fit)", alpha=0.8)
    plt.plot(market_mats, hw_simple_yields * 100, '^--', 
             label="Hull-White (3-Parameter Fit)", alpha=0.8)
             
    # Flexible model
    plt.plot(market_mats, hw_flexible_yields * 100, 'x-', 
             label="Flexible Hull-White (5-Parameter Fit)", 
             lw=2.5, color='crimson', zorder=5)
    
    # Plot formatting
    plt.title("Goodness-of-Fit of Calibrated Models to the Market Yield Curve")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # --- Save the figure ---
    output_path = FIGURES_DIR / "calibration_fit_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    # plt.show() # Optional: uncomment to display the plot interactively
    
    print("\n--- Calibration study complete. ---")

if __name__ == "__main__":
    main()