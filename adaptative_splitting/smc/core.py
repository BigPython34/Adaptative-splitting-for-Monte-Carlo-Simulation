# core.py

import numpy as np
from typing import Tuple, List, Optional

def phi(x: np.ndarray) -> np.ndarray:
    """Fonction de score simple pour P(X > L), donc Phi(X) = X."""
    return x

def mcmc_kernel(
    x: float,
    L_current: float,
    n_steps: int,
    sigma: float,
    return_trace: bool = False # Gardé pour la compatibilité, mais non utilisé
) -> Tuple[float, float, Optional[List[float]]]:
    """
    Noyau de Metropolis-Hastings pour un processus gaussien N(0,1)
    conditionné à être au-dessus du seuil L_current.
    """
    x_current = x
    accepts = 0
    
    for _ in range(n_steps):
        # 1. Proposition symétrique (un pas gaussien)
        proposal = x_current + np.random.normal(0, sigma)
        
        # 2. Condition de support : la proposition doit être dans la zone valide
        if proposal >= L_current:
            
            # 3. Calcul du ratio de Metropolis-Hastings
            # Ratio des densités cibles : pi(proposal) / pi(x_current)
            # Pour une cible N(0,1), pi(x) est proportionnel à exp(-x^2 / 2)
            log_ratio = -0.5 * (proposal**2 - x_current**2)
            
            # Le ratio d'acceptation alpha
            alpha = np.exp(log_ratio)
            # Note: pas besoin de min(1, alpha) car on compare à un tirage uniforme

            # 4. Décision d'acceptation
            if np.random.rand() < alpha:
                x_current = proposal
                accepts += 1
        # Si la proposition est hors de la zone valide, elle est automatiquement rejetée
        # et on reste sur place (x_current ne change pas).

    acceptance_rate = accepts / n_steps if n_steps > 0 else 0.0
    
    return x_current, acceptance_rate, None