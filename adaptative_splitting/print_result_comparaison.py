#!/usr/bin/env python3
"""
Script for plotting results from the SMC comparison study.
"""

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm

from smc.config import COMPARISON_CONFIG

def plot_variances_boxplot(var_data: dict, config: dict, output_dir: Path):
    """Affiche un boxplot des estimateurs."""
    sim_cfg = config["simulation"]
    labels_cfg = config["labels"]
    
    L_target = sim_cfg["demo_threshold"]
    theoretical_prob = 1 - norm.cdf(L_target)

    plt.figure(figsize=config["plots"]["boxplot_figsize"])
    plt.boxplot(
        [var_data["naive_estimates"], var_data["adaptive_estimates"], var_data["fixed_estimates"]],
        labels=["Naive MC", "Adaptive SMC", "Fixed-level SMC"]
    )
    plt.axhline(theoretical_prob, color='red', ls='--', label=f'Theoretical P(X > {L_target:.1f})')
    plt.title(f'Estimator Comparison (Time Budget = {sim_cfg["time_budget_seconds"]:.1f}s)')
    plt.ylabel('Estimated Probability')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    output_path = output_dir / "estimator_variances_boxplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Boxplot saved to '{output_path}'")
    plt.close()

def main_plotting():
    """Charge les résultats et génère tous les graphiques."""
    print("\n--- Starting Plotting Script ---")
    cfg = COMPARISON_CONFIG
    paths_cfg = cfg.get("paths", {})
    
    results_dir = Path(paths_cfg.get("results_dir", "results"))
    figures_dir = Path(paths_cfg.get("figures_dir", "figures/comparison"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / paths_cfg.get("results_filename", "comparison_results.pkl")
    
    if not results_file.exists():
        print(f"Error: Results file not found at '{results_file}'.")
        print("Please run 'run_comparison_study.py' first.")
        return
        
    with open(results_file, "rb") as f:
        sim_results = pickle.load(f)
    print(f"Results loaded from '{results_file.name}'.")

    # Générer le boxplot de l'analyse de variance
    if "estimator_variances" in sim_results:
        plot_variances_boxplot(sim_results["estimator_variances"], cfg, figures_dir)
    else:
        print("No variance analysis data found in results file.")

    print("\n--- Plotting complete. ---")

if __name__ == "__main__":
    main_plotting()