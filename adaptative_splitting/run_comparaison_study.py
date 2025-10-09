#!/usr/bin/env python3
"""
Script principal pour l'étude comparative des méthodes SMC.

Ce script orchestre les simulations, collecte les données et génère les
graphiques d'analyse en utilisant la bibliothèque `smc`.
"""

import datetime
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

# --- Imports depuis notre bibliothèque "smc" ---
from smc.config import COMPARISON_CONFIG
from smc.plotting import (
    plot_methods_graph_all,
    plot_relative_errors_boxplot,
    plot_thresholds_on_gaussian,
    plot_variances_boxplot,
)
from smc.smc_algorithms import (
    adaptive_smc_run,
    fixed_smc_run,
    run_naive_mc,
)

# --- Fonctions "Wrapper" pour la gestion du budget temps ---
# Ces fonctions vivent ici car elles définissent le "design de l'expérience"

def run_adaptive_smc_on_budget(
    L_target: float, time_budget: float, config: dict
) -> Dict:
    """Répète le SMC adaptatif sur un budget de temps et agrège les résultats."""
    sim_cfg = config["simulation"]
    run_estimates = []
    
    with ProcessPoolExecutor() as executor:
        futures = []
        start_time = time.time()
        # Soumettre des tâches tant que le budget temps n'est pas écoulé
        while time.time() - start_time < time_budget:
            futures.append(
                executor.submit(
                    adaptive_smc_run,
                    sim_cfg["particle_count"],
                    sim_cfg["p0"],
                    L_target,
                    sim_cfg["n_mcmc"],
                    sim_cfg["sigma"],
                    sim_cfg["max_iter"],
                )
            )
        
        # Récupérer les résultats
        for future in tqdm(as_completed(futures), total=len(futures), desc="Adaptive SMC runs"):
            result = future.result()
            if result:
                run_estimates.append(result.prob_est)
                
    return {"avg_estimate": np.mean(run_estimates) if run_estimates else np.nan}


def run_fixed_smc_on_budget(
    L_target: float, time_budget: float, config: dict
) -> Dict:
    """Répète le SMC fixe sur un budget de temps et agrège les résultats."""
    sim_cfg = config["simulation"]
    run_estimates = []
    
    # Le script d'orchestration est responsable de la création des seuils
    thresholds = np.linspace(
        sim_cfg["fixed_threshold_start"], L_target, sim_cfg["fixed_num_levels"]
    )
    
    with ProcessPoolExecutor() as executor:
        futures = []
        start_time = time.time()
        while time.time() - start_time < time_budget:
            futures.append(
                executor.submit(
                    fixed_smc_run,
                    sim_cfg["particle_count"],
                    thresholds,
                    sim_cfg["n_mcmc"],
                    sim_cfg["sigma"],
                )
            )
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fixed SMC runs"):
            result = future.result()
            if result:
                run_estimates.append(result.prob_est)

    return {"avg_estimate": np.mean(run_estimates) if run_estimates else np.nan}

# --- Fonctions d'Orchestration de Simulation ---

def simulate_methods_performance(config: dict) -> Dict:
    """Calcule les erreurs relatives pour une grille de seuils L."""
    grid_cfg = config["grid"]
    sim_cfg = config["simulation"]
    L_values = np.linspace(grid_cfg["l_min"], grid_cfg["l_max"], int(grid_cfg["l_count"]))
    
    results = {"L_values": L_values.tolist(), "naive_rel_errors": [], "smc_rel_errors": [], "fixed_rel_errors": []}
    
    for l_val in tqdm(L_values, desc="Simulating methods across L values"):
        theoretical_prob = norm.cdf(-l_val)
        if theoretical_prob == 0:
            results["naive_rel_errors"].append(np.nan)
            results["smc_rel_errors"].append(np.nan)
            results["fixed_rel_errors"].append(np.nan)
            continue
            
        # Allouer un budget de temps pour chaque méthode à ce niveau L
        time_budget = sim_cfg["time_budget_seconds"]
        
        # Le MC naïf est rapide, on peut l'exécuter directement
        num_naive_samples = int(time_budget * 10_000_000) # Estimation grossière
        naive_prob = run_naive_mc(l_val, num_samples=num_naive_samples)
        
        adaptive_prob = run_adaptive_smc_on_budget(l_val, time_budget, config)["avg_estimate"]
        fixed_prob = run_fixed_smc_on_budget(l_val, time_budget, config)["avg_estimate"]
        
        # Calcul des erreurs relatives
        results["naive_rel_errors"].append(abs(naive_prob - theoretical_prob) / theoretical_prob)
        results["smc_rel_errors"].append(abs(adaptive_prob - theoretical_prob) / theoretical_prob)
        results["fixed_rel_errors"].append(abs(fixed_prob - theoretical_prob) / theoretical_prob)
        
    return results

def simulate_estimator_variances(config: dict) -> Dict:
    """Compare les variances des estimateurs pour un L_target fixe."""
    sim_cfg = config["simulation"]
    L_target = sim_cfg["demo_threshold"]
    n_runs = sim_cfg["variance_runs"]
    time_budget = sim_cfg["time_budget_seconds"]
    
    estimates = {"naive_estimates": [], "adaptive_estimates": [], "fixed_estimates": []}
    
    for _ in tqdm(range(n_runs), desc=f"Simulating variances for L={L_target}"):
        num_naive_samples = int(time_budget * 10_000_000)
        estimates["naive_estimates"].append(run_naive_mc(L_target, num_samples=num_naive_samples))
        estimates["adaptive_estimates"].append(run_adaptive_smc_on_budget(L_target, time_budget, config)["avg_estimate"])
        estimates["fixed_estimates"].append(run_fixed_smc_on_budget(L_target, time_budget, config)["avg_estimate"])
        
    return estimates

# --- Fonction Principale ---

def main() -> None:
    """Point d'entrée principal pour la comparaison des méthodes."""
    cfg = COMPARISON_CONFIG
    paths_cfg = cfg["paths"]
    sim_cfg = cfg["simulation"]
    
    start_time = datetime.datetime.now()
    print(cfg["messages"]["analysis_start"].format(timestamp=start_time.strftime("%Y-%m-%d %H:%M:%S")))
    
    results_file = Path(paths_cfg["results_dir"]) / paths_cfg["results_filename"]

    if results_file.exists():
        choice = input(cfg["prompts"]["existing_results"])
        redo_simulation = choice.lower() == cfg["prompts"]["affirmative"]
    else:
        redo_simulation = True

    if redo_simulation:
        print(cfg["messages"]["results_computation"])
        sim_results = {
            "methods_performance": simulate_methods_performance(cfg),
            "estimator_variances": simulate_estimator_variances(cfg),
        }
        with open(results_file, "wb") as f:
            pickle.dump(sim_results, f)
        print(cfg["messages"]["results_saved"].format(filepath=results_file))
    else:
        with open(results_file, "rb") as f:
            sim_results = pickle.load(f)
        print(cfg["messages"]["results_loaded"].format(filepath=results_file))

    # --- Génération des graphiques ---
    print(cfg["messages"]["plotting_start"])
    
    # 1. Graphe des erreurs relatives vs L
    plot_methods_graph_all(sim_results["methods_performance"], cfg)
    
    # 2. Boxplots des variances pour un L fixe
    plot_variances_boxplot(sim_results["estimator_variances"], cfg)
    plot_relative_errors_boxplot(sim_results["estimator_variances"], cfg)

    # 3. Visualisation des seuils sur la distribution
    # On a besoin de lancer une seule fois les algos pour récupérer leurs seuils
    res_adapt = adaptive_smc_run(
        sim_cfg["particle_count"], sim_cfg["p0"], sim_cfg["demo_threshold"],
        sim_cfg["n_mcmc"], sim_cfg["sigma"], sim_cfg["max_iter"]
    )
    if res_adapt:
        fixed_thresholds = np.linspace(
            sim_cfg["fixed_threshold_start"], sim_cfg["demo_threshold"], sim_cfg["fixed_num_levels"]
        )
        plot_thresholds_on_gaussian(res_adapt.thresholds, fixed_thresholds, cfg)
    else:
        print(cfg["messages"]["adaptive_plot_failure"])
        
    print(cfg["messages"]["plotting_end"])

if __name__ == "__main__":
    main()