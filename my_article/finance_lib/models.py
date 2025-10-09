# finance_lib/models.py
import numpy as np
from scipy.integrate import quad
from typing import Tuple, Callable

class BaseShortRateModel:
    """Classe de base pour les modèles de taux courts stochastiques."""
    def __init__(self, r0: float):
        if r0 < 0:
            raise ValueError("Le taux initial r0 ne peut pas être négatif.")
        self.r0 = r0

    def _get_drift_diffusion(self, t: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne le drift et la diffusion. À implémenter par les sous-classes."""
        raise NotImplementedError

    def _enforce_constraints(self, r: np.ndarray) -> np.ndarray:
        """Applique les contraintes du modèle (ex: positivité)."""
        return r

    def simulate_euler(self, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """Simule des trajectoires du taux court par le schéma d'Euler-Maruyama."""
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        r_paths = np.zeros((n_paths, n_steps + 1))
        r_paths[:, 0] = self.r0

        for i in range(n_steps):
            t_current = times[i]
            r_current = r_paths[:, i]
            drift, diffusion_term = self._get_drift_diffusion(t_current, r_current)
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            r_next = r_current + drift * dt + diffusion_term * dW
            r_paths[:, i + 1] = self._enforce_constraints(r_next)
        return r_paths

    def bond_price_analytical(self, t: float, T: float, r_at_t: float) -> float:
        """Calcule le prix analytique d'une obligation zéro-coupon."""
        raise NotImplementedError

    def yield_curve(self, t: float, maturities: np.ndarray, r_at_t: float) -> np.ndarray:
        """Calcule la courbe des taux à l'instant t."""
        yields = np.zeros_like(maturities, dtype=float)
        for i, Ti in enumerate(maturities):
            if Ti <= t:
                yields[i] = np.nan
            else:
                price = self.bond_price_analytical(t, Ti, r_at_t)
                yields[i] = -np.log(price) / (Ti - t) if price > 0 else np.nan
        return yields

class CIRModel(BaseShortRateModel):
    """Implémentation du modèle Cox-Ingersoll-Ross (CIR)."""
    def __init__(self, b: float, beta: float, sigma: float, r0: float):
        super().__init__(r0)
        self.b, self.beta, self.sigma = b, beta, sigma
        self.feller_condition_met = (2 * b) >= (sigma**2)

    def _get_drift_diffusion(self, t: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r_safe = np.maximum(r, 0)
        return self.b - self.beta * r, self.sigma * np.sqrt(r_safe)

    def _enforce_constraints(self, r: np.ndarray) -> np.ndarray:
        return np.maximum(r, 0)

    def bond_price_analytical(self, t: float, T: float, r_at_t: float) -> float:
        tau = T - t
        if tau <= 0: return 1.0
        d = np.sqrt(self.beta**2 + 2 * self.sigma**2)
        exp_d_tau = np.exp(d * tau)
        B = 2 * (exp_d_tau - 1) / ((d + self.beta) * (exp_d_tau - 1) + 2 * d)
        num_A = 2 * d * np.exp((d + self.beta) * tau / 2)
        den_A = (d + self.beta) * (exp_d_tau - 1) + 2 * d
        A = (2 * self.b / self.sigma**2) * np.log(num_A / den_A)
        return np.exp(A - B * r_at_t)

class HullWhiteModel(BaseShortRateModel):
    """Implémentation du modèle Hull-White."""
    def __init__(self, beta: float, sigma: float, r0: float, b_function: Callable[[float], float]):
        super().__init__(r0)
        self.beta, self.sigma, self.b_function = beta, sigma, b_function

    def _get_drift_diffusion(self, t: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.b_function(t) - self.beta * r, self.sigma * np.ones_like(r)

    def bond_price_analytical(self, t: float, T: float, r_at_t: float) -> float:
        tau = T - t
        if tau <= 0: return 1.0
        B_val = (1 - np.exp(-self.beta * tau)) / self.beta if self.beta != 0 else tau
        
        def integrand(s):
            B_s = (1 - np.exp(-self.beta * (T - s))) / self.beta if self.beta != 0 else (T-s)
            # CORRECTION DÉFINITIVE : A(t,T) = integrale(0.5*sigma^2*B^2 - b*B)
            return 0.5 * self.sigma**2 * B_s**2 - self.b_function(s) * B_s
            
        A_val, _ = quad(integrand, t, T)
        return np.exp(A_val - B_val * r_at_t)