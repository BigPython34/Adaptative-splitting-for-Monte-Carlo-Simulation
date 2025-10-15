#!/usr/bin/env python3
"""
Script pour tester l'efficacité d'une seule exécution SMC par rapport
au Monte Carlo naïf avec un budget de temps équivalent.
"""

import time
import numpy as np
from scipy.stats import norm

# --- Imports depuis notre bibliothèque SMC ---
from smc.smc_algorithms import (
    adaptive_smc_run, create_mcmc_propagation_step,
    fixed_smc_run, run_naive_mc
)
from smc.core import phi as gaussian_phi, mcmc_kernel

# --- Définitions du problème et des stratégies ---
def gaussian_sampler(n): return np.random.randn(n)

# --- Fonction Principale ---
def main():
    """Orchestre le test d'efficacité."""
    print("="*60)
    print("=== Test d'Efficacité : 1 Run SMC vs. MC Naïf Équivalent ===")
    print("="*60)

    # --- 1. Paramètres du Test ---
    L_TARGET = 8.0
    
    # Paramètres partagés pour les simulations SMC
    N_PARTICLES = 50000
    P0 = 0.7
    N_MCMC = 20
    SIGMA_MCMC = 0.5
    NUM_FIXED_LEVELS = 20 # Nombre de niveaux pour le SMC fixe

    # --- 2. Lancement du SMC Adaptatif ---
    print(f"\n--- Exécution du SMC Adaptatif (1 seule fois) ---")
    propagation_strategy = create_mcmc_propagation_step(mcmc_kernel, N_MCMC, SIGMA_MCMC)
    
    start_time_adapt = time.time()
    res_adapt = adaptive_smc_run(
        N=N_PARTICLES, p0=P0, L_target=L_TARGET, phi_function=gaussian_phi,
        initial_sampler=gaussian_sampler, propagation_step=propagation_strategy
    )
    duration_adapt = time.time() - start_time_adapt
    
    if res_adapt:
        print(f"Terminé en {duration_adapt:.2f} secondes.")
    else:
        print("Échec de la simulation.")

    # --- 3. Lancement du SMC Fixe ---
    print(f"\n--- Exécution du SMC à Niveaux Fixes (1 seule fois) ---")
    fixed_thresholds = np.linspace(0, L_TARGET, NUM_FIXED_LEVELS)
    
    start_time_fixed = time.time()
    res_fixed = fixed_smc_run(
        N=N_PARTICLES, thresholds=fixed_thresholds, phi_function=gaussian_phi,
        initial_sampler=gaussian_sampler, mcmc_kernel_func=mcmc_kernel,
        n_mcmc=N_MCMC, sigma=SIGMA_MCMC
    )
    duration_fixed = time.time() - start_time_fixed
    
    if res_fixed:
        print(f"Terminé en {duration_fixed:.2f} secondes.")
    else:
        print("Échec de la simulation.")

    # --- 4. Lancement du Monte Carlo Naïf (avec les budgets de temps mesurés) ---
    SAMPLES_PER_SEC = 5_000_000 
    
    print(f"\n--- Exécution du MC Naïf (budget: {duration_adapt:.2f}s) ---")
    num_samples_adapt_equiv = int(duration_adapt * SAMPLES_PER_SEC)
    prob_naive_adapt = run_naive_mc(
        L_TARGET, gaussian_phi, gaussian_sampler, num_samples=num_samples_adapt_equiv)
    print(f"Terminé. {num_samples_adapt_equiv:,} échantillons générés.")

    print(f"\n--- Exécution du MC Naïf (budget: {duration_fixed:.2f}s) ---")
    num_samples_fixed_equiv = int(duration_fixed * SAMPLES_PER_SEC)
    prob_naive_fixed = run_naive_mc(
        L_TARGET, gaussian_phi, gaussian_sampler, num_samples=num_samples_fixed_equiv)
    print(f"Terminé. {num_samples_fixed_equiv:,} échantillons générés.")
    
    # --- 5. Affichage du Tableau Comparatif ---
    theoretical_prob = 1 - norm.cdf(L_TARGET)
    
    print("\n" + "="*70)
    print(f"=== Tableau Comparatif d'Efficacité (L_target={L_TARGET}) ===")
    print(f"Vérité Théorique : {theoretical_prob:.4e}")
    print("="*70)
    print(f"{'Méthode':<35} | {'Temps (s)':>10} | {'Estimation':>15} | {'Erreur Rel.':>12}")
    print("-" * 70)
    
    def print_result_line(name, duration, estimate):
        if estimate is None or theoretical_prob == 0:
            rel_error = float('inf')
        else:
            rel_error = abs(estimate - theoretical_prob) / theoretical_prob
        
        print(f"{name:<35} | {duration:>10.2f} | {estimate:>15.4e} | {rel_error:>11.2%}")

    if res_adapt:
        print_result_line("1. SMC Adaptatif", duration_adapt, res_adapt.prob_est)
        print_result_line("   -> MC Naïf (budget équivalent)", duration_adapt, prob_naive_adapt)
    
    print("-" * 70)
    
    if res_fixed:
        print_result_line("2. SMC à Niveaux Fixes", duration_fixed, res_fixed.prob_est)
        print_result_line("   -> MC Naïf (budget équivalent)", duration_fixed, prob_naive_fixed)
        
    print("="*70)

if __name__ == "__main__":
    main()