# finance_lib/pricing_engine.py

import numpy as np
from typing import Dict
from .models import BaseShortRateModel

class MonteCarloDerivativesPricer:
    """
    Moteur de pricing pour dérivés sur taux d'intérêt par Monte Carlo.
    """
    def __init__(self, model: BaseShortRateModel):
        self.model = model

    def price_european_call_on_rate(
        self, T: float, strike: float, n_sims: int, n_steps: int
    ) -> Dict[str, float]:
        """
        Prix d'une option call européenne sur le taux court à maturité T.
        Payoff = max(r(T) - K, 0).
        
        CORRECTION : Cette fonction retourne maintenant un dictionnaire complet
        incluant le prix, l'erreur standard, et l'intervalle de confiance.
        """
        paths = self.model.simulate_euler(T, n_steps, n_sims)
        
        # Payoffs à maturité T
        final_rates = paths[:, -1]
        payoffs = np.maximum(final_rates - strike, 0)

        # Facteurs d'actualisation stochastiques
        dt = T / n_steps
        integrated_rates = np.sum(paths[:, :-1], axis=1) * dt
        discount_factors = np.exp(-integrated_rates)

        # L'estimateur Monte Carlo est le tableau des payoffs actualisés
        discounted_payoffs = discount_factors * payoffs
        
        # Calcul du prix (la moyenne)
        price = np.mean(discounted_payoffs)
        
        # Calcul de l'écart-type des estimateurs individuels
        std_dev = np.std(discounted_payoffs)
        
        # Calcul de l'erreur standard de la moyenne
        std_error = std_dev / np.sqrt(n_sims)
        # ------------------------

        return {
            "price": price,
            "std_error": std_error,
            "confidence_interval_95": (price - 1.96 * std_error, price + 1.96 * std_error),
        }

    def price_range_accrual_note(
        self, T: float, r_min: float, r_max: float, notional: float, n_observations: int, n_sims: int
    ) -> float:
        paths = self.model.simulate_euler(T, n_observations, n_sims)
        observed_paths = paths[:, 1:]
        in_range_counts = np.sum((observed_paths >= r_min) & (observed_paths <= r_max), axis=1)
        payoffs = notional * (in_range_counts / n_observations)
        dt = T / n_observations
        discount_factors = np.exp(-np.sum(paths[:, :-1] * dt, axis=1))
        price = np.mean(payoffs * discount_factors)
        return price