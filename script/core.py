#!/usr/bin/env python3
"""
Fonctions mathématiques de base pour l'analyse SMC.
"""
import numpy as np
from typing import Tuple, Optional, List


def phi(x: np.ndarray) -> np.ndarray:
    """
    Transformation phi(x) = -x.

    Args:
        x: Array d'entrée

    Returns:
        Transformation appliquée à x
    """
    return -x


def mcmc_kernel(
    x: float, L_current: float, n_steps: int, sigma: float, return_trace: bool = False
) -> Tuple[float, float, Optional[List[float]]]:
    """
    Effectue n_steps du kernel Metropolis-Hastings pour échantillonner la loi N(0,1)
    tronquée à x <= -L_current.

    Args:
        x: Valeur initiale de la particule
        L_current: Seuil courant (contrainte x <= -L_current)
        n_steps: Nombre d'étapes MCMC
        sigma: Écart-type pour les propositions
        return_trace: Si True, retourne la trace complète

    Returns:
        Tuple contenant:
        - x_final: Valeur finale de la particule
        - acceptance_rate: Taux d'acceptation
        - trace: Liste des valeurs (si return_trace=True), sinon None
    """
    x_current = x
    accepts = 0
    trace = [] if return_trace else None

    for _ in range(n_steps):
        proposal = x_current + np.random.normal(0, sigma)

        if proposal <= -L_current:
            log_ratio = -0.5 * (proposal**2 - x_current**2)
            alpha = np.exp(log_ratio)

            if np.random.rand() < alpha:
                x_current = proposal
                accepts += 1

        if return_trace:
            trace.append(x_current)

    acceptance_rate = accepts / n_steps if n_steps > 0 else 0.0

    if return_trace:
        return x_current, acceptance_rate, trace
    else:
        return x_current, acceptance_rate, None
