#!/usr/bin/env python3
"""Algorithme SMC adaptatif."""

import numpy as np
from typing import List, Optional
import time

try:  # Support package and standalone execution
    from .core import phi, mcmc_kernel
    from .config import SMC_CONSTANTS, SMC_MESSAGES
except ImportError:  # pragma: no cover - fallback when run as script
    from core import phi, mcmc_kernel
    from config import SMC_CONSTANTS, SMC_MESSAGES


class AdaptiveSMCResult:
    """Classe pour stocker les résultats d'une exécution SMC."""

    def __init__(
        self,
        prob_est: float,
        thresholds: List[float],
        acc_rates: List[float],
        particle_means: List[float],
        particle_vars: List[float],
        mcmc_traces: List[Optional[List[float]]],
        n_iter: int,
    ):
        self.prob_est = prob_est
        self.thresholds = thresholds
        self.acc_rates = acc_rates
        self.particle_means = particle_means
        self.particle_vars = particle_vars
        self.mcmc_traces = mcmc_traces
        self.n_iter = n_iter

    def to_dict(self) -> dict:
        """Convertit le résultat en dictionnaire."""
        return {
            "prob_est": self.prob_est,
            "thresholds": self.thresholds,
            "acc_rates": self.acc_rates,
            "particle_means": self.particle_means,
            "particle_vars": self.particle_vars,
            "mcmc_traces": self.mcmc_traces,
            "n_iter": self.n_iter,
        }


def adaptive_smc_run(
    N: int,
    p0: float,
    L_target: float,
    n_mcmc: int,
    sigma: float,
    max_iter: int = 50,
    verbose: bool = False,
) -> Optional[AdaptiveSMCResult]:
    """
    Exécute une run de l'algorithme SMC adaptatif.

    Args:
        N: Nombre de particules
        p0: Seuil de sélection (proportion de particules conservées)
        L_target: Seuil cible à atteindre
        n_mcmc: Nombre d'étapes MCMC par particule
        sigma: Écart-type pour les propositions MCMC
        max_iter: Nombre maximum d'itérations
        verbose: Si True, affiche des informations de debug

    Returns:
        AdaptiveSMCResult si succès, None sinon
    """
    particles = np.random.randn(N)
    prob_est = SMC_CONSTANTS["initial_prob_estimate"]
    thresholds = []
    acc_rates = []
    particle_means = []
    particle_vars = []
    mcmc_traces = []
    n_iter = 0

    if verbose:
        print(SMC_MESSAGES["start"])

    while n_iter < max_iter:
        phi_vals = phi(particles)

        # Détermination du seuil adaptatif via le quantile (1-p0)
        L_current = np.percentile(
            phi_vals,
            (1.0 - p0) * SMC_CONSTANTS["percentile_scale"],
        )
        thresholds.append(L_current)
        n_iter += 1

        if verbose:
            print(f"    Itération {n_iter:2d} - L_current = {L_current:.3f}")

        if L_current >= L_target:
            if verbose:
                print(SMC_MESSAGES["threshold_reached"])
            break

        prob_est *= p0
        survivors = particles[phi_vals >= L_current]

        if survivors.size == 0:
            if verbose:
                print(SMC_MESSAGES["no_survivor"].format(iteration=n_iter))
            return None

        # Diagnostics sur les particules survivantes
        particle_means.append(np.mean(survivors))
        particle_vars.append(np.var(survivors))

        # Re-échantillonnage
        indices = np.random.choice(len(survivors), size=N, replace=True)
        particles = survivors[indices]

        # Étape MCMC
        new_particles = []
        acc_list = []
        mcmc_trace_iter = None

        for i, x in enumerate(particles):
            if i == 0:
                x_new, acc, trace = mcmc_kernel(
                    x, L_current, n_mcmc, sigma, return_trace=True
                )
                mcmc_trace_iter = trace
            else:
                x_new, acc, _ = mcmc_kernel(x, L_current, n_mcmc, sigma)

            new_particles.append(x_new)
            acc_list.append(acc)

        particles = np.array(new_particles)
        acc_rates.append(np.mean(acc_list))
        mcmc_traces.append(mcmc_trace_iter)

    # Estimation finale
    phi_vals = phi(particles)
    r = np.mean(phi_vals >= L_target)
    prob_est *= r

    return AdaptiveSMCResult(
        prob_est=prob_est,
        thresholds=thresholds,
        acc_rates=acc_rates,
        particle_means=particle_means,
        particle_vars=particle_vars,
        mcmc_traces=mcmc_traces,
        n_iter=n_iter,
    )


def run_naive_mc_time(L_target: float, time_budget: float, batch_size: int) -> float:
    """
    Exécute la simulation MC naïf pendant ``time_budget`` secondes.
    Retourne la probabilité estimée.
    """
    total_samples = 0
    total_success = 0
    start_time = time.time()
    while time.time() - start_time < time_budget:
        samples = np.random.randn(batch_size)
        successes = np.sum(samples <= -L_target)
        total_success += successes
        total_samples += batch_size
    estimated_prob = (
        total_success / total_samples
    if total_samples > 0
    else 0.0
    )
    return float(estimated_prob)


def run_naive_mc_iterations(L_target: float, iter: int):
    """
    Exécute la simulation MC naïf pendant 'time_budget' secondes.
    Retourne : (probabilité estimée, nb total d'échantillons, nb de succès, temps écoulé)
    """
    successes = 0
    for _ in range(iter):
        result = np.random.randn()
        if result <= -L_target:
            successes += 1
    estimated_prob = successes / iter
    return float(estimated_prob)
