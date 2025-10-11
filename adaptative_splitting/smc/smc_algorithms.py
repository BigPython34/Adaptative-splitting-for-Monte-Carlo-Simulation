#!/usr/bin/env python3
"""
Sequential Monte Carlo (SMC) algorithms implementation.

This module contains pure algorithm implementations for:
- Adaptive SMC (Algorithm 2 from article)
- Fixed-level SMC (Algorithm 1 from article)
- Naive Monte Carlo
"""
import time
from typing import List, Optional, Tuple, Dict
import numpy as np

from .core import phi, mcmc_kernel

class AdaptiveSMCResult:
    """Classe pour stocker les résultats d'une exécution SMC adaptative."""
    def __init__(
        self,
        prob_est: float,
        thresholds: List[float],
        acc_rates: List[float],
        mcmc_traces: List[Optional[List[float]]],
        n_iter: int,
        # On peut garder des diagnostics si on le souhaite
        particle_means: List[float],
        particle_vars: List[float],
    ):
        self.prob_est = prob_est
        self.thresholds = thresholds
        self.acc_rates = acc_rates
        self.mcmc_traces = mcmc_traces
        self.n_iter = n_iter
        self.particle_means = particle_means
        self.particle_vars = particle_vars

    def to_dict(self) -> Dict:
        """Convertit le résultat en dictionnaire."""
        # ... (inchangé)

class FixedSMCResult:
    """Stores results from a fixed-level SMC execution."""
    def __init__(
        self,
        prob_est: float,
        thresholds: np.ndarray,
        acc_rates: List[float],
    ):
        self.prob_est = prob_est
        self.thresholds = thresholds
        self.acc_rates = acc_rates


def adaptive_smc_run(
    N: int,
    p0: float,
    phi_function: callable,
    initial_sampler: callable,
    L_target: float,
    n_mcmc: int,
    sigma: float,
    max_iter: int = 50,
) -> Optional[AdaptiveSMCResult]:
    """Executes adaptive SMC algorithm (Algorithm 2 from article)."""
    
    particles = initial_sampler(N) 
    prob_est = 1.0
    
    thresholds, acc_rates, particle_means, particle_vars, mcmc_traces = [], [], [], [], []
    n_iter = 0

    while n_iter < max_iter:
        phi_vals = phi(particles)
        
        L_current = np.percentile(phi_vals, (1.0 - p0) * 100.0)
        thresholds.append(L_current)
        n_iter += 1

        if L_current >= L_target:
            break

        prob_est *= p0
        survivors = particles[phi_vals >= L_current]

        if survivors.size == 0:
            return None

        particle_means.append(np.mean(survivors))
        particle_vars.append(np.var(survivors))
        
        indices = np.random.choice(len(survivors), size=N, replace=True)
        particles = survivors[indices]
        
        new_particles = np.empty(N)
        acc_list = np.empty(N)
        mcmc_trace_iter = None

        for i in range(N):
            trace_this_particle = (i == 0)
            x_new, acc, trace = mcmc_kernel(
                particles[i], L_current, n_mcmc, sigma, return_trace=trace_this_particle
            )
            new_particles[i] = x_new
            acc_list[i] = acc
            if trace_this_particle:
                mcmc_trace_iter = trace
        
        particles = new_particles
        acc_rates.append(np.mean(acc_list))
        mcmc_traces.append(mcmc_trace_iter)

    r_final = np.mean(phi(particles) >= L_target)
    prob_est *= r_final

    return AdaptiveSMCResult(
        prob_est=prob_est,
        thresholds=thresholds,
        final_particles=particles,
        acc_rates=acc_rates,
        mcmc_traces=mcmc_traces,
        n_iter=n_iter,
        particle_means=particle_means,
        particle_vars=particle_vars,
    )


def fixed_smc_run(
    N: int,
    thresholds: np.ndarray,
    n_mcmc: int,
    sigma: float,
) -> Optional[FixedSMCResult]:
    """Executes fixed-level SMC algorithm (Algorithm 1 from article)."""
    
    particles = np.random.randn(N)
    prob_est = 1.0
    acc_rates: List[float] = []

    for l_k in thresholds:
        phi_vals = phi(particles)
        survivors = particles[phi_vals >= l_k]
        
        if survivors.size == 0:
            return None

        prob_est *= survivors.size / particles.size
        
        indices = np.random.choice(len(survivors), size=N, replace=True)
        particles = survivors[indices]
        
        new_particles = np.empty(N)
        acc_list = np.empty(N)

        for i in range(N):
            x_new, acc, _ = mcmc_kernel(particles[i], l_k, n_mcmc, sigma)
            new_particles[i] = x_new
            acc_list[i] = acc
            
        particles = new_particles
        acc_rates.append(np.mean(acc_list))
    
    # L'estimation finale est déjà calculée car le dernier seuil est L_target
    return FixedSMCResult(
        prob_est=prob_est,
        thresholds=thresholds,
        acc_rates=acc_rates,
    )

# --- Algorithme Naïf ---

def run_naive_mc(
    L_target: float,
    num_samples: int,
    batch_size: int = 10**6
) -> float:
    """
    Exécute une simulation Monte-Carlo naïve de manière efficace par lots.
    """
    if num_samples == 0:
        return 0.0
        
    total_success = 0
    remaining_samples = num_samples
    
    while remaining_samples > 0:
        current_batch_size = min(remaining_samples, batch_size)
        samples = np.random.randn(current_batch_size)
        # Note: phi(x) = -x, donc phi(x) > L est équivalent à x < -L
        total_success += np.sum(samples < -L_target)
        remaining_samples -= current_batch_size
        
    return total_success / num_samples