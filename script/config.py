#!/usr/bin/env python3
"""
Configuration et constantes pour l'analyse SMC adaptatif.
"""
from pathlib import Path

# Répertoires de travail
BASE_DIR = Path("script")
FIGURES_DIR = Path("figures/analyse_adaptive_smc")
RESULTS_DIR = Path("results")

# Créer les répertoires s'ils n'existent pas
for directory in [FIGURES_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Paramètres par défaut pour l'analyse
DEFAULT_ANALYSIS_PARAMS = {
    "p0_values": [0.3, 0.5, 0.7],
    "sigma_values": [1.0, 2.0, 3.0],
    "n_mcmc_values": [20, 40, 60],
    "L_target": 7,
    "N": 5000,
    "max_iter": 50,
}

# Paramètres par défaut pour l'analyse MCMC spécifique
DEFAULT_MCMC_PARAMS = {
    "p0": 0.5,
    "sigma": 1.0,
    "n_mcmc": 60,
    "L_target": 7,
    "N": 5000,
    "max_iter": 100,
}

# Paramètres de référence pour les graphiques
REFERENCE_PARAMS = {"p0_ref": 0.5, "sigma_ref": 2.0, "n_mcmc_ref": 40}
