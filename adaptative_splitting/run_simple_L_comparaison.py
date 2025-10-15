#!/usr/bin/env python3
"""
Script simple pour comparer les méthodes SMC en fonction du seuil L.
"""

import datetime
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

# --- Imports depuis la bibliothèque SMC ---
from smc.config import COMPARISON_CONFIG
# from smc.plotting import plot_methods_graph_all # Mettre en commentaire si non utilisé
from smc.smc_algorithms import (
    adaptive_smc_run, create_mcmc_propagation_step,
    fixed_smc_run, run_naive_mc
)
from smc.core import phi as gaussian_phi, mcmc_kernel

# --- Définition du problème gaussien ---
def gaussian_sampler(num_samples: int) -> np.ndarray:
    return np.random.randn(num_samples)

# --- Fonction d'Orchestration ---
def simulate_performance_across_L(config: dict) -> dict:
    grid_cfg = config["grid"]
    sim_cfg = config["simulation"]
    
    L_values = np.linspace(grid_cfg["l_min"], grid_cfg["l_max"], int(grid_cfg["l_count"]))
    results = {"L_values": L_values.tolist(), "naive_rel_errors": [], "smc_rel_errors": [], "fixed_rel_errors": []}
    
    # Construire la stratégie de propagation une seule fois
    propagation_strategy = create_mcmc_propagation_step(
        mcmc_kernel_func=mcmc_kernel,
        n_mcmc=sim_cfg["n_mcmc"],
        sigma=sim_cfg["sigma"]
    )
    
    for l_val in tqdm(L_values, desc="Simulation pour différentes valeurs de L"):
        theoretical_prob = 1 - norm.cdf(l_val)
        
        # --- 1. SMC Adaptatif ---
        res_adapt = adaptive_smc_run(
            N=sim_cfg["particle_count"],
            p0=sim_cfg["p0"],
            L_target=l_val,
            phi_function=gaussian_phi,
            initial_sampler=gaussian_sampler,
            propagation_step=propagation_strategy,
            max_iter=sim_cfg["max_iter"]
        )
        
        # --- 2. SMC Fixe ---
        thresholds = np.linspace(sim_cfg["fixed_threshold_start"], l_val, sim_cfg["fixed_num_levels"])
        res_fixed = fixed_smc_run(
            N=sim_cfg["particle_count"],
            thresholds=thresholds,
            phi_function=gaussian_phi,
            initial_sampler=gaussian_sampler,
            mcmc_kernel_func=mcmc_kernel,
            n_mcmc=sim_cfg["n_mcmc"],
            sigma=sim_cfg["sigma"]
        )
        
        # --- 3. MC Naïf ---
        prob_naive = run_naive_mc(
            L_target=l_val,
            phi_function=gaussian_phi,
            initial_sampler=gaussian_sampler,
            num_samples=int(10**8)
        )

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
def main():
    """Point d'entrée principal pour la comparaison."""

    cfg = COMPARISON_CONFIG
    paths_cfg = cfg["paths"]
    
    start_time = datetime.datetime.now()
    print(f"Début de l'analyse simple : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results_file = Path(paths_cfg.get("results_dir", Path("results"))) / "simple_L_comparison_results.pkl"
    results_file.parent.mkdir(exist_ok=True)

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

    print(">>> Analyse terminée.")

if __name__ == "__main__":
    main()