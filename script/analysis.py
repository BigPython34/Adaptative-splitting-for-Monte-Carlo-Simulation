#!/usr/bin/env python3
"""
Fonctions d'analyse pour l'étude des hyperparamètres du SMC adaptatif.
"""
import concurrent.futures
from typing import Dict, Tuple

try:  # Support package and standalone execution
    from .config import (
        DEFAULT_ANALYSIS_PARAMS,
        DEFAULT_MCMC_PARAMS,
        ANALYSIS_MESSAGES,
    )
    from .smc import adaptive_smc_run, AdaptiveSMCResult
except ImportError:  # pragma: no cover - fallback when run as script
    from config import (
        DEFAULT_ANALYSIS_PARAMS,
        DEFAULT_MCMC_PARAMS,
        ANALYSIS_MESSAGES,
    )
    from smc import adaptive_smc_run, AdaptiveSMCResult


def run_single_combination(params: Tuple) -> Tuple:
    """
    Fonction auxiliaire pour exécuter une combinaison de paramètres.

    Args:
        params: Tuple contenant (p0, sigma, n_mcmc, L_target, N, max_iter)

    Returns:
        Tuple (p0, sigma, n_mcmc, result)
    """
    p0, sigma, n_mcmc, L_target, N, max_iter = params
    result = adaptive_smc_run(N, p0, L_target, n_mcmc, sigma, max_iter, verbose=False)
    return (p0, sigma, n_mcmc, result)


def analyze_hyperparameters(
    p0_values=None,
    sigma_values=None,
    n_mcmc_values=None,
    L_target=None,
    N=None,
    max_iter=None,
) -> Dict[str, dict]:
    """
    Explore différentes configurations du SMC adaptatif en faisant varier :
      - p0 (seuil de sélection),
      - sigma (écart-type utilisé dans le kernel MCMC),
      - n_mcmc (nombre d'étapes dans Metropolis-Hastings).

    Args:
        p0_values: Liste des valeurs de p0 à tester
        sigma_values: Liste des valeurs de sigma à tester
        n_mcmc_values: Liste des valeurs de n_mcmc à tester
        L_target: Seuil cible
        N: Nombre de particules
        max_iter: Nombre maximum d'itérations

    Returns:
        Dictionnaire contenant les résultats pour chaque configuration.
    """
    # Utiliser les paramètres par défaut si non spécifiés
    params = DEFAULT_ANALYSIS_PARAMS.copy()
    if p0_values is not None:
        params["p0_values"] = p0_values
    if sigma_values is not None:
        params["sigma_values"] = sigma_values
    if n_mcmc_values is not None:
        params["n_mcmc_values"] = n_mcmc_values
    if L_target is not None:
        params["L_target"] = L_target
    if N is not None:
        params["N"] = N
    if max_iter is not None:
        params["max_iter"] = max_iter

    print(ANALYSIS_MESSAGES["hyperparam_start"])

    results = {}

    # Préparation de toutes les combinaisons
    param_list = []
    for p0 in params["p0_values"]:
        for sigma in params["sigma_values"]:
            for n_mcmc in params["n_mcmc_values"]:
                param_list.append(
                    (
                        p0,
                        sigma,
                        n_mcmc,
                        params["L_target"],
                        params["N"],
                        params["max_iter"],
                    )
                )

    # Exécution en parallèle de toutes les combinaisons
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_single_combination, param_combo): param_combo
            for param_combo in param_list
        }

        for future in concurrent.futures.as_completed(futures):
            p0, sigma, n_mcmc, _, _, _ = futures[future]
            key = f"p0={p0}_sigma={sigma}_nMCMC={n_mcmc}"

            try:
                p0_res, sigma_res, n_mcmc_res, result = future.result()
                if result is not None:
                    results[key] = result.to_dict()
                else:
                    print(
                        ANALYSIS_MESSAGES["combination_failure"].format(key=key)
                    )

            except Exception as exc:
                print(
                    ANALYSIS_MESSAGES["combination_exception"].format(
                        key=key, error=exc
                    )
                )

    print(ANALYSIS_MESSAGES["hyperparam_end"])
    return results


def analyze_single_mcmc_run(
    p0=None, sigma=None, n_mcmc=None, L_target=None, N=None, max_iter=None, verbose=True
) -> AdaptiveSMCResult:
    """
    Exécute une run du SMC adaptatif avec des paramètres fixes pour analyser
    le comportement MCMC d'une particule à chaque niveau.

    Args:
        p0: Seuil de sélection (défaut: 0.5)
        sigma: Écart-type MCMC (défaut: 1.0)
        n_mcmc: Nombre d'étapes MCMC (défaut: 60)
        L_target: Seuil cible (défaut: 6.0)
        N: Nombre de particules (défaut: 5000)
        max_iter: Nombre maximum d'itérations (défaut: 100)
        verbose: Afficher les informations de debug

    Returns:
        Résultat de l'analyse SMC
    """
    # Utiliser les paramètres par défaut si non spécifiés
    params = DEFAULT_MCMC_PARAMS.copy()
    if p0 is not None:
        params["p0"] = p0
    if sigma is not None:
        params["sigma"] = sigma
    if n_mcmc is not None:
        params["n_mcmc"] = n_mcmc
    if L_target is not None:
        params["L_target"] = L_target
    if N is not None:
        params["N"] = N
    if max_iter is not None:
        params["max_iter"] = max_iter

    if verbose:
        print(ANALYSIS_MESSAGES["single_start"])
        print(
            ANALYSIS_MESSAGES["single_params"].format(
                p0=params["p0"],
                sigma=params["sigma"],
                n_mcmc=params["n_mcmc"],
                L_target=params["L_target"],
            )
        )

    result = adaptive_smc_run(
        params["N"],
        params["p0"],
        params["L_target"],
        params["n_mcmc"],
        params["sigma"],
        params["max_iter"],
        verbose=verbose,
    )

    if result is None:
        print(ANALYSIS_MESSAGES["smc_failure"])
        raise RuntimeError("SMC failed to reach target threshold")
    else:
        if result.thresholds[-1] >= params["L_target"]:
            if verbose:
                print(
                    ANALYSIS_MESSAGES["single_success"].format(
                        L_target=params["L_target"]
                    )
                )
        else:
            if verbose:
                print(
                    ANALYSIS_MESSAGES["single_failure"].format(
                        threshold=result.thresholds[-1]
                    )
                )

    if verbose:
        print(ANALYSIS_MESSAGES["single_end"])

    return result
