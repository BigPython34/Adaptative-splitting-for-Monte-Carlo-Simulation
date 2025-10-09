#!/usr/bin/env python3
"""
Script simple pour comparer les méthodes SMC en fonction du seuil L.

Ce script exécute chaque algorithme une seule fois pour différentes valeurs de L
et trace l'évolution de leur erreur relative.
"""

import datetime
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

# --- Imports depuis notre bibliothèque "smc" ---
from smc.config import COMPARISON_CONFIG
from smc.plotting import plot_methods_graph_all
from smc.smc_algorithms import adaptive_smc_run, fixed_smc_run, run_naive_mc

# --- Fonction d'Orchestration ---

def simulate_performance_across_L(config: dict) -> dict:
    """
    Exécute chaque algorithme une fois pour une grille de seuils L.
    """
    grid_cfg = config["grid"]
    sim_cfg = config["simulation"]
    
    # Création de la liste des seuils à tester
    L_values = np.linspace(grid_cfg["l_min"], grid_cfg["l_max"], int(grid_cfg["l_count"]))
    
    results = {
        "L_values": L_values.tolist(),
        "naive_rel_errors": [],
        "smc_rel_errors": [],
        "fixed_rel_errors": [],
    }
    
    # Un grand nombre d'échantillons pour le MC naïf pour avoir une bonne référence
    # (ajustez si nécessaire, mais 10^8 est un bon point de départ)
    NAIVE_SAMPLES_COUNT = 10**8

    # Boucle simple et séquentielle sur les valeurs de L
    for l_val in tqdm(L_values, desc="Simulation pour différentes valeurs de L"):
        theoretical_prob = norm.cdf(-l_val)
        
        # --- 1. SMC Adaptatif ---
        res_adapt = adaptive_smc_run(
            N=sim_cfg["particle_count"],
            p0=sim_cfg["p0"],
            L_target=l_val,
            n_mcmc=sim_cfg["n_mcmc"],
            sigma=sim_cfg["sigma"],
            max_iter=sim_cfg["max_iter"],
        )
        
        # --- 2. SMC Fixe ---
        thresholds = np.linspace(
            sim_cfg["fixed_threshold_start"], l_val, sim_cfg["fixed_num_levels"]
        )
        res_fixed = fixed_smc_run(
            N=sim_cfg["particle_count"],
            thresholds=thresholds,
            n_mcmc=sim_cfg["n_mcmc"],
            sigma=sim_cfg["sigma"],
        )
        
        # --- 3. MC Naïf ---
        prob_naive = run_naive_mc(l_val, num_samples=NAIVE_SAMPLES_COUNT)

        # --- Calcul des erreurs relatives ---
        if theoretical_prob > 0:
            err_naive = abs(prob_naive - theoretical_prob) / theoretical_prob
            err_adapt = abs(res_adapt.prob_est - theoretical_prob) / theoretical_prob if res_adapt else np.nan
            err_fixed = abs(res_fixed.prob_est - theoretical_prob) / theoretical_prob if res_fixed else np.nan
        else:
            err_naive, err_adapt, err_fixed = np.nan, np.nan, np.nan
            
        results["naive_rel_errors"].append(err_naive)
        results["smc_rel_errors"].append(err_adapt)
        results["fixed_rel_errors"].append(err_fixed)
        
    return results

# --- Fonction Principale ---

def main() -> None:
    """Point d'entrée principal pour la comparaison."""
    cfg = COMPARISON_CONFIG
    paths_cfg = cfg["paths"]
    
    start_time = datetime.datetime.now()
    print(f"Début de l'analyse simple : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Utiliser un nom de fichier différent pour ne pas écraser l'étude complexe
    results_file = Path(paths_cfg["results_dir"]) / "simple_L_comparison_results.pkl"

    if results_file.exists():
        choice = input(f"Le fichier de résultats '{results_file.name}' existe. Relancer la simulation ? (o/n) : ")
        redo_simulation = choice.lower() == 'o'
    else:
        redo_simulation = True

    if redo_simulation:
        print(">>> Début du calcul des résultats...")
        sim_results = simulate_performance_across_L(cfg)
        with open(results_file, "wb") as f:
            pickle.dump(sim_results, f)
        print(f">>> Résultats sauvegardés dans '{results_file.name}'")
    else:
        with open(results_file, "rb") as f:
            sim_results = pickle.load(f)
        print(f">>> Résultats chargés depuis '{results_file.name}'")

    # --- Génération du graphique ---
    print(">>> Début du tracé du graphique...")
    plot_methods_graph_all(sim_results, cfg)
    print(">>> Graphique sauvegardé. Analyse terminée.")

if __name__ == "__main__":
    main()