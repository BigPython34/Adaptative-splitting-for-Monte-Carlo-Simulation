#!/usr/bin/env python3
"""
Package pour l'analyse SMC adaptatif.

Ce package contient les modules suivants :
- config : Configuration et constantes
- core : Fonctions mathématiques de base
- smc : Algorithme SMC adaptatif
- analysis : Fonctions d'analyse des hyperparamètres
- visualization : Fonctions de visualisation
- main : Script principal
"""

__version__ = "1.0.0"

# Imports principaux pour faciliter l'utilisation
from .smc import adaptive_smc_run, AdaptiveSMCResult
from .analysis import analyze_hyperparameters, analyze_single_mcmc_run
from .visualization import plot_all_influences, plot_mcmc_traces
from .main import main

__all__ = [
    "adaptive_smc_run",
    "AdaptiveSMCResult",
    "analyze_hyperparameters",
    "analyze_single_mcmc_run",
    "plot_all_influences",
    "plot_mcmc_traces",
    "main",
]
