#!/usr/bin/env python3
"""
Script principal pour l'analyse du SMC adaptatif.
"""
import datetime
import pickle
from pathlib import Path
from scipy.stats import norm
from config import RESULTS_DIR, DEFAULT_ANALYSIS_PARAMS
from analysis import analyze_hyperparameters, analyze_single_mcmc_run
from smc import run_naive_mc_time, run_naive_mc_iterations
import numpy as np

L = DEFAULT_ANALYSIS_PARAMS["L_target"]


def load_or_compute_results(results_file: Path) -> dict:
    """
    Charge les résultats depuis un fichier ou les calcule s'ils n'existent pas.

    Args:
        results_file: Chemin vers le fichier de résultats

    Returns:
        Dictionnaire des résultats
    """
    if results_file.exists():
        with open(results_file, "rb") as f:
            results = pickle.load(f)
        print(f">>> Résultats chargés depuis {results_file}")
    else:
        print(">>> Début du calcul des résultats...")
        results = analyze_hyperparameters()
        with open(results_file, "wb") as f:
            pickle.dump(results, f)
        print(f">>> Résultats sauvegardés dans {results_file}")

    return results


def main():
    """Fonction principale d'analyse."""
    start_time = datetime.datetime.now()
    print(f"Début de l'analyse : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        "\n=== Analyse du SMC adaptatif avec différentes configurations d'hyperparamètres ==="
    )

    # Fichier de résultats
    results_file = RESULTS_DIR / "adaptive_smc_analysis.pkl"

    try:
        # Chargement ou calcul des résultats de l'analyse des hyperparamètres
        results = load_or_compute_results(results_file)

        # Analyse spécifique du MCMC pour une configuration donnée
        print(
            "\n>>> Début de l'analyse spécifique du MCMC pour p0=0.5, sigma=1.0, nMCMC=60, L_target=6..."
        )
        single_result = analyze_single_mcmc_run().to_dict()["prob_est"]

        print(f"\n=== Analyse terminée avec succès, resultat estimé à {single_result}")
        print(f"résultat théorique: { norm.cdf(-L) }")
    except Exception as e:
        print(f"\n=== Erreur lors de l'analyse : {e} ===")
        raise

    finally:
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f"Durée totale : {duration.total_seconds()}s")
    resultat_mc = run_naive_mc_time(L, time_budget=duration.total_seconds())
    print(f"resultat avec le naif avec le même budget de temps: {resultat_mc}")
    start_time = datetime.datetime.now()
    resultat_mc = run_naive_mc_time(L, time_budget=10 * duration.total_seconds())
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(
        f"resultat avec le naif avec 10x le budget (en {10*duration.total_seconds()}s): {resultat_mc}"
    )


if __name__ == "__main__":
    main()
