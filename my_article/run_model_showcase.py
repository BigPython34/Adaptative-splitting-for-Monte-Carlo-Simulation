#!/usr/bin/env python3
"""
Script de présentation pour visualiser les dynamiques des modèles CIR et Hull-White.

Ce script utilise des paramètres prédéfinis pour illustrer :
1. Les trajectoires simulées du taux court.
2. La distribution du taux court à une maturité donnée.
3. Les courbes de rendement analytiques.
"""

from finance_lib.models import CIRModel, HullWhiteModel
from finance_lib.visualization import (
    plot_short_rate_trajectories,
    plot_final_rate_distribution,
    plot_yield_curves_comparison,
)

def main():
    """Fonction principale pour la présentation des modèles."""
    print("--- Lancement de la présentation des modèles de taux courts ---")

    # --- Configuration commune ---
    r0 = 0.03
    T_long_term = 5.0
    T_dist = 1.0
    
    # --- Instanciation des modèles ---
    cir_params = {'b': 0.06, 'beta': 0.2, 'sigma': 0.15, 'r0': r0}
    cir_model = CIRModel(**cir_params)

    # Pour Hull-White, on utilise un drift constant pour la simplicité
    hw_params = {'beta': 0.2, 'sigma': 0.01, 'r0': r0}
    hw_model = HullWhiteModel(**hw_params, b_function=lambda t: 0.03)

    models_to_compare = {"CIR Model": cir_model, "Hull-White Model": hw_model}
    
    # --- 1. Visualisation des trajectoires ---
    print("\n[1/3] Génération des graphiques de trajectoires simulées...")
    plot_short_rate_trajectories(
        cir_model, "CIR", T=T_long_term, n_steps=250, n_paths=10
    )
    plot_short_rate_trajectories(
        hw_model, "Hull-White", T=T_long_term, n_steps=250, n_paths=10
    )

    # --- 2. Visualisation de la distribution finale ---
    print("[2/3] Génération des histogrammes de distribution finale...")
    plot_final_rate_distribution(
        cir_model, "CIR", T=T_dist, n_sims=20000
    )
    plot_final_rate_distribution(
        hw_model, "Hull-White", T=T_dist, n_sims=20000
    )

    # --- 3. Comparaison des courbes de rendement ---
    print("[3/3] Génération du graphique comparatif des courbes de rendement...")
    plot_yield_curves_comparison(
        models=models_to_compare, r_current=r0, max_maturity=10.0
    )
    
    print("\n--- Présentation des modèles terminée. ---")

if __name__ == "__main__":
    main()