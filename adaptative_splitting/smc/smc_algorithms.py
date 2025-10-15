# smc_algorithms.py

import numpy as np
from typing import Callable, Optional, List

# --- Classe de Résultat Générique ---
class AdaptiveSMCResult:
    """Résultats génériques d'une exécution SMC."""
    def __init__(self, prob_est: float, thresholds: List[float], final_particles: np.ndarray, final_scores: np.ndarray):
        self.prob_est = prob_est
        self.thresholds = thresholds
        self.final_particles = final_particles
        self.final_scores = final_scores
        
class FixedSMCResult:
    """Résultats pour le SMC à niveaux fixes."""
    def __init__(self, prob_est: float):
        self.prob_est = prob_est

def fixed_smc_run(
    N: int,
    thresholds: np.ndarray,
    phi_function: Callable,
    initial_sampler: Callable,
    mcmc_kernel_func: Callable,
    n_mcmc: int,
    sigma: float
) -> Optional[FixedSMCResult]:
    """Exécute l'algorithme SMC à niveaux fixes (générique)."""
    particles = initial_sampler(N)
    prob_est = 1.0

    for l_k in thresholds:
        scores = phi_function(particles)
        survivors_mask = scores >= l_k
        num_survivors = np.sum(survivors_mask)
        
        if num_survivors == 0: return None

        prob_est *= num_survivors / N
        survivors = particles[survivors_mask]
        
        indices = np.random.choice(len(survivors), size=N, replace=True)
        particles = survivors[indices]
        
        new_particles = np.empty(N)
        for i in range(N):
            x_new, _, _ = mcmc_kernel_func(particles[i], l_k, n_mcmc, sigma)
            new_particles[i] = x_new
        particles = new_particles
        
    return FixedSMCResult(prob_est)

def run_naive_mc(
    L_target: float,
    phi_function: Callable,
    initial_sampler: Callable,
    num_samples: int,
    batch_size: int = 10**6
) -> float:
    """Exécute une simulation Monte-Carlo naïve (générique)."""
    if num_samples == 0: return 0.0
    
    total_success = 0
    remaining = num_samples
    while remaining > 0:
        batch = min(remaining, batch_size)
        samples = initial_sampler(batch)
        scores = phi_function(samples)
        total_success += np.sum(scores >= L_target)
        remaining -= batch
        
    return total_success / num_samples

# --- L'ALGORITHME SMC GÉNÉRIQUE FINAL ---
def adaptive_smc_run(
    N: int,
    p0: float,
    phi_function: Callable,
    initial_sampler: Callable,
    propagation_step: Callable, # L'étape de propagation est maintenant un argument !
    L_target: Optional[float] = None,
    max_iter: int = 100
) -> Optional[AdaptiveSMCResult]:
    """
    Exécute l'algorithme SMC adaptatif générique et final.
    """
    particles = initial_sampler(N)
    prob_est = 1.0
    thresholds = []

    for k in range(max_iter):
        scores = phi_function(particles)
        valid_scores = scores[~np.isnan(scores)]
        if len(valid_scores) == 0: return None

        L_current = np.percentile(valid_scores, (1 - p0) * 100)
        
        if L_target is not None and L_current >= L_target:
            print(f"Cible L_target={L_target:.2f} atteinte.")
            break
        
        if k > 0 and L_current <= thresholds[-1] + 1e-6:
            print(f"Convergence des seuils à l'itération {k}.")
            break
            
        thresholds.append(L_current)
        
        survivors_mask = (scores >= L_current) & (~np.isnan(scores))
        num_survivors = np.sum(survivors_mask)
        if num_survivors < 2: break
        
        prob_est *= num_survivors / len(valid_scores)
        survivors = particles[survivors_mask]
        
        # Rééchantillonnage
        indices = np.random.choice(len(survivors), size=N, replace=True)
        resampled_particles = survivors[indices]
        
        # Étape de PROPAGATION (maintenant entièrement personnalisable)
        particles = propagation_step(resampled_particles, survivors, L_current)

    final_scores = phi_function(particles)
    
    # Calcul de la probabilité finale si une cible est définie
    if L_target is not None:
        final_survival_rate = np.mean(final_scores >= L_target)
        final_probability = prob_est * final_survival_rate
    else:
        final_probability = prob_est

    return AdaptiveSMCResult(final_probability, thresholds, particles, final_scores)


# --- Usines à Stratégies de Propagation ---

def create_mcmc_propagation_step(mcmc_kernel_func: Callable, n_mcmc: int, sigma: float) -> Callable:
    """Crée une fonction de propagation basée sur un noyau MCMC (pour le cas simple)."""
    def propagation_step(resampled_particles, survivors, L_current):
        N = len(resampled_particles)
        new_particles = np.empty(N)
        for i in range(N):
            x_new, _, _ = mcmc_kernel_func(resampled_particles[i], L_current, n_mcmc, sigma)
            new_particles[i] = x_new
        return new_particles
    return propagation_step

def create_rejuvenation_propagation_step(initial_sampler: Callable, rejuvenation_ratio: float, mutation_std_ratio: float) -> Callable:
    """Crée une fonction de propagation basée sur la régénération (pour le cas risque)."""
    def propagation_step(resampled_particles, survivors, L_current):
        N = len(resampled_particles)
        particles = resampled_particles.copy()
        
        num_rejuvenate = int(N * rejuvenation_ratio)
        if num_rejuvenate > 0:
            particles[:num_rejuvenate] = initial_sampler(num_rejuvenate)
        
        num_mutate = N - num_rejuvenate
        if num_mutate > 0:
            mutation_std = np.std(survivors) * mutation_std_ratio
            if mutation_std > 1e-9:
                start_index = num_rejuvenate
                particles[start_index:] += np.random.normal(0, mutation_std, size=num_mutate)
        return particles
    return propagation_step