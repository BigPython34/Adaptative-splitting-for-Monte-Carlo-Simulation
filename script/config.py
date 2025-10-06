#!/usr/bin/env python3
"""Configuration centralisée pour les scripts SMC et comparaisons."""

from pathlib import Path
from typing import Dict


DIRECTORY_PATHS: Dict[str, Path] = {
    "scripts": Path("script"),
    "analysis_figures": Path("figures/analyse_adaptive_smc"),
    "comparison_figures": Path("figures/comparaison"),
    "results": Path("results"),
}

for directory in (
    DIRECTORY_PATHS["analysis_figures"],
    DIRECTORY_PATHS["comparison_figures"],
    DIRECTORY_PATHS["results"],
):
    directory.mkdir(parents=True, exist_ok=True)

ANALYSIS_FIGURES_DIR = DIRECTORY_PATHS["analysis_figures"]
COMPARISON_FIGURES_DIR = DIRECTORY_PATHS["comparison_figures"]
RESULTS_DIR = DIRECTORY_PATHS["results"]


DEFAULT_ANALYSIS_PARAMS = {
    "p0_values": [0.3, 0.5, 0.7],
    "sigma_values": [1.0, 2.0, 3.0],
    "n_mcmc_values": [20, 40, 60],
    "L_target": 7,
    "N": 5000,
    "max_iter": 50,
}


DEFAULT_MCMC_PARAMS = {
    "p0": 0.5,
    "sigma": 1.0,
    "n_mcmc": 60,
    "L_target": 7,
    "N": 5000,
    "max_iter": 100,
}


REFERENCE_PARAMS = {"p0_ref": 0.5, "sigma_ref": 2.0, "n_mcmc_ref": 40}


CORE_CONSTANTS: Dict[str, float] = {
    "phi_multiplier": -1.0,
    "mcmc_proposal_mean": 0.0,
    "log_ratio_coefficient": -0.5,
    "acceptance_default": 0.0,
}


SMC_CONSTANTS: Dict[str, float] = {
    "initial_prob_estimate": 1.0,
    "percentile_scale": 100.0,
}


SMC_MESSAGES = {
    "start": ">>> Début du SMC adaptatif...",
    "threshold_reached": "    Seuil cible atteint.",
    "no_survivor": "    Aucune particule survivante à l'itération {iteration}",
    "failure": "    Le SMC n'a pas réussi à atteindre le seuil cible.",
}


ANALYSIS_MESSAGES = {
    "hyperparam_start": ">>> Début de l'analyse des hyperparamètres...",
    "hyperparam_end": ">>> Analyse des hyperparamètres terminée.",
    "single_start": ">>> Début de l'analyse spécifique du MCMC...",
    "single_params": "    Paramètres : p0={p0}, sigma={sigma}, n_mcmc={n_mcmc}, L_target={L_target}",
    "single_success": "    Succès : Seuil cible atteint à L_target = {L_target}.",
    "single_failure": "    Échec : Seuil final atteint = {threshold:.3f}.",
    "single_end": ">>> Analyse spécifique du MCMC terminée.",
    "smc_failure": "    Le SMC n'a pas réussi à atteindre le seuil cible.",
    "combination_failure": "Échec pour {key}: aucun résultat",
    "combination_exception": "{key} a généré une exception: {error}",
}


COMPARISON_CONFIG = {
    "paths": {
        "figures_dir": DIRECTORY_PATHS["comparison_figures"],
        "results_dir": DIRECTORY_PATHS["results"],
        "results_filename": "simulation_results.pkl",
        "figures": {
            "methods": "compare_methods_graph_all.png",
            "thresholds": "plot_thresholds_on_gaussian.png",
            "variances": "compare_variances_boxplot.png",
            "relative_errors": "compare_relative_errors_boxplot.png",
        },
    },
    "simulation": {
        "time_budget_seconds": 60.0,
        "particle_count": 5000,
        "p0": 0.5,
        "n_mcmc": 40,
        "sigma": 2.0,
        "max_iter": 50,
        "fixed_num_levels": 10,
        "demo_threshold": 5.0,
        "variance_runs": 30,
        "naive_batch_size": 10**6,
    "fixed_threshold_start": 0.0,
        "verbose": False,
    },
    "grid": {
        "l_min": 2.0,
        "l_max": 7.5,
        "l_count": 10,
    },
    "plots": {
        "dpi": 300,
        "methods_figsize": (10, 6),
        "boxplot_figsize": (8, 6),
        "threshold_figsize": (10, 6),
        "markers": {
            "naive": "o",
            "adaptive": "s",
            "fixed": "^",
        },
        "colors": {
            "naive": "C0",
            "adaptive": "C1",
            "fixed": "C2",
            "adaptive_threshold": "C1",
            "fixed_threshold": "C3",
            "theoretical": "green",
        },
        "linestyles": {
            "adaptive": "--",
            "fixed": "-.",
        },
        "grid": True,
        "density_range": {
            "start": -10.0,
            "stop": 10.0,
            "num": 1000,
        },
    },
    "labels": {
        "relative_error_title": "Comparaison des erreurs relatives en fonction de L",
        "threshold_title": "Visualisation des seuils sur la loi gaussienne initiale",
        "boxplot_title": "Comparaison des estimateurs (budget = {time_budget:.2f}s, L_target={L_target})",
        "relative_boxplot_title": "Comparaison des erreurs relatives (budget = {time_budget:.2f}s, L_target={L_target})",
        "relative_error_xlabel": "Seuil L",
        "relative_error_ylabel": "Erreur relative",
        "probability_ylabel": "Probabilité estimée",
        "density_xlabel": "x",
        "density_ylabel": "Densité",
    "gaussian_label": "N(0,1)",
        "naive_relative_label": "MC naïf - Erreur relative",
        "adaptive_relative_label": "SMC adaptatif - Erreur relative",
        "fixed_relative_label": "SMC fixe - Erreur relative",
        "naive_boxplot_label": "MC naïf",
        "adaptive_boxplot_label": "SMC adaptatif",
        "fixed_boxplot_label": "SMC fixe",
        "theoretical_label": "P(L_target={L_target:.2f}) théorique",
        "adaptive_threshold_label": "Seuil adaptatif",
        "fixed_threshold_label": "Seuil fixe",
    },
    "messages": {
        "analysis_start": "Début de l'analyse : {timestamp}",
        "header": "\n=== Analyse du SMC adaptatif avec différentes configurations d'hyperparamètres ===",
        "methods_simulation_start": ">>> Début de la simulation de la comparaison des méthodes...",
        "methods_simulation_end": ">>> Simulation de la comparaison des méthodes terminée.",
        "variance_simulation_start": ">>> Début de la simulation des variances...",
        "variance_simulation_end": ">>> Simulation des variances terminée.",
        "plotting_start": ">>> Début du tracé des graphiques...",
        "plotting_end": ">>> Tracé des graphiques terminé.",
        "simulation_complete": ">>> Simulation terminée.",
        "results_loaded": ">>> Résultats chargés depuis {filepath}",
        "results_saved": ">>> Résultats sauvegardés dans {filepath}",
        "results_computation": ">>> Début du calcul des résultats...",
        "fixed_no_survivor": "Aucune particule survivante pour le seuil fixe.",
        "adaptive_plot_failure": "Impossible de générer les seuils : le SMC adaptatif n'a pas atteint la cible.",
    },
    "prompts": {
        "existing_results": "Les résultats de simulation existent déjà. Voulez-vous relancer la simulation ? (o/n) : ",
        "affirmative": "o",
    },
}


MAIN_SETTINGS = {
    "results_filename": "adaptive_smc_analysis.pkl",
    "naive_time_multiplier": 3.0,
    "naive_time_batch_size": 10**5,
}


MAIN_MESSAGES = {
    "analysis_start": "Début de l'analyse : {timestamp}",
    "analysis_header": "\n=== Analyse du SMC adaptatif avec différentes configurations d'hyperparamètres ===",
    "mcmc_start": "\n>>> Début de l'analyse spécifique du MCMC pour p0={p0}, sigma={sigma}, nMCMC={n_mcmc}, L_target={L_target}...",
    "analysis_success": "\n=== Analyse terminée avec succès, resultat estimé à {result}",
    "theoretical_result": "résultat théorique: {value}",
    "analysis_error": "\n=== Erreur lors de l'analyse : {error} ===",
    "duration": "Durée totale : {seconds}s",
    "naive_result": "resultat avec le naif avec {multiplier} fois le budget de temps: {result}",
}
