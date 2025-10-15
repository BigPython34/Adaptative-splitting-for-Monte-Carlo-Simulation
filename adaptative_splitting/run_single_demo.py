#!/usr/bin/env python3
"""
Script de démonstration pour le SMC sur un événement rare gaussien.

Ce script utilise le moteur SMC générique pour estimer P(X > 7),
où X suit une loi normale standard N(0,1).
"""

import numpy as np
from scipy.stats import norm

# --- Imports depuis notre bibliothèque SMC ---
from smc.smc_algorithms import adaptive_smc_run, create_mcmc_propagation_step
from smc.core import phi as gaussian_phi, mcmc_kernel

# --- Définition du Problème Spécifique ---

def gaussian_sampler(num_samples: int) -> np.ndarray:
    """Générateur de particules suivant une loi normale N(0,1)."""
    return np.random.randn(num_samples)

# --- Exécution de la Démonstration ---

def main():
    """Fonction principale de la démonstration."""
    print("="*60)
    print("=== Démo SMC pour l'estimation de P(X > 7) avec X ~ N(0,1) ===")
    print("="*60)
    
    N = 50000      # Nombre de particules
    p0 = 0.75      # Taux de survie à chaque étape (75%)
    L_TARGET = 7.0 # Le seuil de l'événement rare

    N_MCMC = 5     # Nombre de pas MCMC pour chaque particule
    SIGMA_MCMC = 0.5 # Écart-type des pas de proposition MCMC

    # --- 2. Construction de la stratégie de propagation ---
    propagation_strategy = create_mcmc_propagation_step(
        mcmc_kernel_func=mcmc_kernel,
        n_mcmc=N_MCMC,
        sigma=SIGMA_MCMC
    )
    
    # --- 3. Lancement du moteur SMC générique ---
    print("Lancement de la simulation SMC adaptative...")
    result = adaptive_smc_run(
        N=N,
        p0=p0,
        L_target=L_TARGET,
        phi_function=gaussian_phi,
        initial_sampler=gaussian_sampler,
        propagation_step=propagation_strategy
    )

    if result is None:
        print("\nLa simulation a échoué (probablement à cause de la mort de toutes les particules).")
        return

    # --- 4. Affichage des résultats ---
    final_probability = result.prob_est
    theoretical_prob = 1 - norm.cdf(L_TARGET)

    print("\n--- Résultats ---")
    print(f"Nombre d'itérations adaptatives : {len(result.thresholds)}")
    print(f"Seuils successifs (L_k) : {[f'{t:.2f}' for t in result.thresholds]}")
    print("-" * 20)
    print(f"Probabilité estimée de P(X > {L_TARGET}) : {final_probability:.4e}")
    print(f"Probabilité théorique :                 {theoretical_prob:.4e}")
    print("-" * 20)
    
    error = abs(final_probability - theoretical_prob) / theoretical_prob
    print(f"Erreur relative : {error:.2%}")
    print("="*60)

if __name__ == "__main__":
    main()