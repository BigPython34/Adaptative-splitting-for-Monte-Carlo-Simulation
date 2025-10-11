#!/usr/bin/env python3
"""
Demonstration script for a single execution of SMC algorithms.

This script runs a simulation for each method (adaptive SMC, fixed-level SMC,
and naive Monte Carlo), displays results, and compares them to theoretical values.
"""

import time
import numpy as np
from scipy.stats import norm

from smc.config import DEFAULT_MCMC_PARAMS, MAIN_MESSAGES, COMPARISON_CONFIG
from smc.smc_algorithms import adaptive_smc_run, fixed_smc_run, run_naive_mc

NAIVE_SAMPLES_PER_SECOND = 10_000_000

def main():
    """Main demonstration function."""
    
    params = DEFAULT_MCMC_PARAMS
    L_target = params["L_target"]
    
    print(MAIN_MESSAGES["analysis_header"])
    print(f"Launching comparative demo with L_target = {L_target}...")
    
    print("\n[1/3] Running Adaptive SMC...")
    start_time_adaptive = time.time()
    smc_result_adaptive = adaptive_smc_run(
        N=params["N"],
        p0=params["p0"],
        L_target=L_target,
        n_mcmc=params["n_mcmc"],
        sigma=params["sigma"],
        max_iter=params["max_iter"],
    )
    duration_adaptive = time.time() - start_time_adaptive
    
    print("[2/3] Running Fixed-Level SMC...")
    fixed_params = COMPARISON_CONFIG["simulation"]
    thresholds = np.linspace(
        fixed_params["fixed_threshold_start"],
        L_target,
        fixed_params["fixed_num_levels"]
    )
    
    start_time_fixed = time.time()
    smc_result_fixed = fixed_smc_run(
        N=params["N"],
        thresholds=thresholds,
        n_mcmc=params["n_mcmc"],
        sigma=params["sigma"],
    )
    duration_fixed = time.time() - start_time_fixed

    print("[3/3] Running Naive Monte Carlo (equivalent time budget)...")
    num_naive_samples = int(duration_adaptive * NAIVE_SAMPLES_PER_SECOND)
    
    naive_prob = run_naive_mc(L_target, num_samples=num_naive_samples)
    
    theoretical_prob = norm.cdf(-L_target)
    print("\n" + "="*40)
    print("=== COMPARATIVE DASHBOARD ===")
    print("="*40)
    print(f"Theoretical probability : {theoretical_prob:.4e}\n")

    if smc_result_adaptive:
        prob = smc_result_adaptive.prob_est
        error = abs(prob - theoretical_prob) / theoretical_prob if theoretical_prob > 0 else float('inf')
        print(f"--- Adaptive SMC ---")
        print(f"    Duration    : {duration_adaptive:.2f} seconds")
        print(f"    Estimate    : {prob:.4e}")
        print(f"    Rel. error  : {error:.2%}")
        print(f"    Iterations  : {smc_result_adaptive.n_iter}")
    else:
        print("--- Adaptive SMC : FAILED (no surviving particles) ---")
    if smc_result_fixed:
        prob = smc_result_fixed.prob_est
        error = abs(prob - theoretical_prob) / theoretical_prob if theoretical_prob > 0 else float('inf')
        print(f"\n--- SMC Niveaux Fixes ---")
        print(f"    Durée       : {duration_fixed:.2f} secondes")
        print(f"    Estimation  : {prob:.4e}")
        print(f"    Erreur rel. : {error:.2%}")
        print(f"    Nb niveaux  : {len(smc_result_fixed.thresholds)}")
    else:
        print("\n--- SMC Niveaux Fixes : ÉCHEC (aucune particule n'a survécu) ---")

    # --- Résultats MC Naïf ---
    error = abs(naive_prob - theoretical_prob) / theoretical_prob if theoretical_prob > 0 else float('inf')
    print(f"\n--- Monte-Carlo Naïf ---")
    print(f"    Nb échantillons: {num_naive_samples:,} (sur ~{duration_adaptive:.2f}s)")
    print(f"    Estimation     : {naive_prob:.4e}")
    print(f"    Erreur rel.    : {error:.2%}")
    print("="*40)


if __name__ == "__main__":
    main()