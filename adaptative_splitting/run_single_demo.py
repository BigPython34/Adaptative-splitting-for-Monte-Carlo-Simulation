#!/usr/bin/env python3
"""
Script de démonstration pour une exécution unique des algorithmes SMC.

Ce script lance une simulation pour chaque méthode (SMC adaptatif, SMC à niveaux
fixes, et Monte-Carlo naïf), affiche les résultats et les compare à la
valeur théorique.
"""

import time
import numpy as np
from scipy.stats import norm

# --- Imports depuis notre bibliothèque "smc" ---
from smc.config import DEFAULT_MCMC_PARAMS, MAIN_MESSAGES, COMPARISON_CONFIG
from smc.smc_algorithms import adaptive_smc_run, fixed_smc_run, run_naive_mc

# Constante pour estimer la performance du MC naïf
# (à ajuster en fonction de votre machine si nécessaire)
NAIVE_SAMPLES_PER_SECOND = 10_000_000

def main():
    """Fonction principale de la démonstration."""
    
    # --- Configuration ---
    # On utilise les paramètres par défaut pour la simulation
    params = DEFAULT_MCMC_PARAMS
    L_target = params["L_target"]
    
    print(MAIN_MESSAGES["analysis_header"])
    print(f"Lancement d'une démo comparative avec L_target = {L_target}...")
    
    # ==========================================
    # 1. EXÉCUTION DU SMC ADAPTATIF
    # ==========================================
    print("\n[1/3] Exécution du SMC Adaptatif...")
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
    
    # ==========================================
    # 2. EXÉCUTION DU SMC À NIVEAUX FIXES
    # ==========================================
    print("[2/3] Exécution du SMC à Niveaux Fixes...")
    # On récupère les paramètres pour le SMC fixe depuis la config de comparaison
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

    # ==========================================
    # 3. EXÉCUTION DU MONTE-CARLO NAÏF
    # ==========================================
    print("[3/3] Exécution du Monte-Carlo Naïf (budget temps équivalent)...")
    # On accorde au MC naïf le même budget de calcul que le SMC adaptatif
    num_naive_samples = int(duration_adaptive * NAIVE_SAMPLES_PER_SECOND)
    
    naive_prob = run_naive_mc(L_target, num_samples=num_naive_samples)
    
    # ==========================================
    # 4. AFFICHAGE DES RÉSULTATS
    # ==========================================
    theoretical_prob = norm.cdf(-L_target)
    print("\n" + "="*40)
    print("=== TABLEAU DE BORD COMPARATIF ===")
    print("="*40)
    print(f"Probabilité théorique : {theoretical_prob:.4e}\n")

    # --- Résultats SMC Adaptatif ---
    if smc_result_adaptive:
        prob = smc_result_adaptive.prob_est
        error = abs(prob - theoretical_prob) / theoretical_prob if theoretical_prob > 0 else float('inf')
        print(f"--- SMC Adaptatif ---")
        print(f"    Durée       : {duration_adaptive:.2f} secondes")
        print(f"    Estimation  : {prob:.4e}")
        print(f"    Erreur rel. : {error:.2%}")
        print(f"    Nb itérations: {smc_result_adaptive.n_iter}")
    else:
        print("--- SMC Adaptatif : ÉCHEC (aucune particule n'a survécu) ---")

    # --- Résultats SMC Fixe ---
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