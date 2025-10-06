#!/usr/bin/env python3
"""Script principal pour l'analyse du SMC adaptatif."""

import datetime
import pickle
from pathlib import Path

from scipy.stats import norm

try:  # Support package and standalone execution
    from .config import (
        RESULTS_DIR,
        DEFAULT_ANALYSIS_PARAMS,
        DEFAULT_MCMC_PARAMS,
        MAIN_SETTINGS,
        MAIN_MESSAGES,
        COMPARISON_CONFIG,
    )
    from .analysis import analyze_hyperparameters, analyze_single_mcmc_run
    from .smc import run_naive_mc_time
except ImportError:  # pragma: no cover - fallback when run as script
    from config import (
        RESULTS_DIR,
        DEFAULT_ANALYSIS_PARAMS,
        DEFAULT_MCMC_PARAMS,
        MAIN_SETTINGS,
        MAIN_MESSAGES,
        COMPARISON_CONFIG,
    )
    from analysis import analyze_hyperparameters, analyze_single_mcmc_run
    from smc import run_naive_mc_time

L_TARGET = DEFAULT_ANALYSIS_PARAMS["L_target"]
COMPARISON_MESSAGES = COMPARISON_CONFIG["messages"]


def load_or_compute_results(results_file: Path) -> dict:
    """
    Charge les résultats depuis un fichier ou les calcule s'ils n'existent pas.

    Args:
        results_file: Chemin vers le fichier de résultats

    Returns:
        Dictionnaire des résultats
    """
    if results_file.exists():
        with open(results_file, "rb") as file_handle:
            results = pickle.load(file_handle)
        print(COMPARISON_MESSAGES["results_loaded"].format(filepath=results_file))
    else:
        print(COMPARISON_MESSAGES["results_computation"])
        results = analyze_hyperparameters()
        with open(results_file, "wb") as file_handle:
            pickle.dump(results, file_handle)
        print(COMPARISON_MESSAGES["results_saved"].format(filepath=results_file))

    return results


def main():
    """Fonction principale d'analyse."""
    start_time = datetime.datetime.now()
    print(
        MAIN_MESSAGES["analysis_start"].format(
            timestamp=start_time.strftime("%Y-%m-%d %H:%M:%S")
        )
    )
    print(MAIN_MESSAGES["analysis_header"])

    # Fichier de résultats
    results_file = RESULTS_DIR / MAIN_SETTINGS["results_filename"]

    try:
        # Chargement ou calcul des résultats de l'analyse des hyperparamètres
        load_or_compute_results(results_file)

        # Analyse spécifique du MCMC pour une configuration donnée
        print(
            MAIN_MESSAGES["mcmc_start"].format(
                p0=DEFAULT_MCMC_PARAMS["p0"],
                sigma=DEFAULT_MCMC_PARAMS["sigma"],
                n_mcmc=DEFAULT_MCMC_PARAMS["n_mcmc"],
                L_target=DEFAULT_MCMC_PARAMS["L_target"],
            )
        )
        single_result = analyze_single_mcmc_run().to_dict()["prob_est"]

        print(MAIN_MESSAGES["analysis_success"].format(result=single_result))
        print(
            MAIN_MESSAGES["theoretical_result"].format(
                value=norm.cdf(-L_TARGET)
            )
        )
    except Exception as e:
        print(MAIN_MESSAGES["analysis_error"].format(error=e))
        raise

    finally:
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(
            MAIN_MESSAGES["duration"].format(
                seconds=duration.total_seconds()
            )
        )
    time_multiplier = MAIN_SETTINGS["naive_time_multiplier"]
    batch_size_time = MAIN_SETTINGS["naive_time_batch_size"]
    naive_result = run_naive_mc_time(
        L_TARGET,
        time_budget=time_multiplier * duration.total_seconds(),
        batch_size=batch_size_time,
    )
    print(
        MAIN_MESSAGES["naive_result"].format(
            multiplier=time_multiplier, result=naive_result
        )
    )


if __name__ == "__main__":
    main()
