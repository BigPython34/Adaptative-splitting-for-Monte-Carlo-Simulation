"""
Script d'implémentation complète des modèles CIR et Hull-White
pour le pricing d'obligations zéro-coupon et de dérivés sur taux d'intérêt

Basé sur l'article : "Zero-Coupon Bond Pricing in Stochastic Interest Rate Models:
The Cox-Ingersoll-Ross and Hull-White Frameworks"

Auteur: Octave Cerclé - ISAE-Supaero
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2
from scipy.integrate import quad
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore")

# Configuration globale des graphiques
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class CIRModel:
    """
    Modèle Cox-Ingersoll-Ross pour le taux court
    dr(t) = (b - β*r(t))dt + σ*√r(t)*dW(t)
    """

    def __init__(self, b: float, beta: float, sigma: float, r0: float):
        """
        Paramètres:
        - b: paramètre de drift constant
        - beta: paramètre de mean-reversion
        - sigma: volatilité
        - r0: taux initial
        """
        self.b = b
        self.beta = beta
        self.sigma = sigma
        self.r0 = r0

        # Vérification de la condition de Feller
        self.feller_condition = b >= (sigma**2) / 2
        if not self.feller_condition:
            print(
                f"⚠️  Attention: Condition de Feller non respectée (b={b:.4f} < σ²/2={sigma**2/2:.4f})"
            )

    def simulate_euler(self, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """
        Simulation Euler du processus CIR avec troncature pour éviter les taux négatifs
        """
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)

        # Initialisation des chemins
        r_paths = np.zeros((n_paths, n_steps + 1))
        r_paths[:, 0] = self.r0

        for i in range(n_steps):
            # Génération des innovations gaussiennes
            dW = np.random.normal(0, np.sqrt(dt), n_paths)

            # Schéma d'Euler avec troncature
            r_current = np.maximum(r_paths[:, i], 1e-8)  # Éviter les valeurs négatives
            drift = (self.b - self.beta * r_current) * dt
            diffusion = self.sigma * np.sqrt(r_current) * dW

            r_paths[:, i + 1] = np.maximum(r_current + drift + diffusion, 1e-8)

        return r_paths

    def bond_price_analytical(self, t: float, T: float, r: float) -> float:
        """
        Prix analytique d'une obligation zéro-coupon dans le modèle CIR
        p(t,T) = exp(A(t,T) - B(t,T)*r(t))
        """
        tau = T - t
        if tau <= 0:
            return 1.0

        # Calcul des paramètres
        d = np.sqrt(self.beta**2 + 2 * self.sigma**2)

        # Fonction B(t,T)
        exp_d_tau = np.exp(d * tau)
        B = 2 * (exp_d_tau - 1) / ((d + self.beta) * (exp_d_tau - 1) + 2 * d)

        # Fonction A(t,T)
        numerator = 2 * d * np.exp((d + self.beta) * tau / 2)
        denominator = (d + self.beta) * (exp_d_tau - 1) + 2 * d
        A = (2 * self.b / self.sigma**2) * np.log(numerator / denominator)

        return np.exp(A - B * r)

    def yield_curve(self, t: float, maturities: np.ndarray, r: float) -> np.ndarray:
        """
        Calcul de la courbe des taux pour différentes maturités
        """
        yields = np.zeros_like(maturities)
        for i, T in enumerate(maturities):
            if T > t:
                price = self.bond_price_analytical(t, T, r)
                yields[i] = -np.log(price) / (T - t)
            else:
                yields[i] = r
        return yields


class HullWhiteModel:
    """
    Modèle Hull-White pour le taux court
    dr(t) = (b(t) - β*r(t))dt + σ*dW(t)
    """

    def __init__(self, beta: float, sigma: float, r0: float, b_function=None):
        """
        Paramètres:
        - beta: paramètre de mean-reversion
        - sigma: volatilité constante
        - r0: taux initial
        - b_function: fonction b(t) (par défaut constante)
        """
        self.beta = beta
        self.sigma = sigma
        self.r0 = r0
        self.b_function = (
            b_function if b_function else lambda t: 0.02
        )  # Drift constant par défaut

    def simulate_euler(self, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """
        Simulation Euler du processus Hull-White
        """
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)

        # Initialisation des chemins
        r_paths = np.zeros((n_paths, n_steps + 1))
        r_paths[:, 0] = self.r0

        for i in range(n_steps):
            t = times[i]

            # Génération des innovations gaussiennes
            dW = np.random.normal(0, np.sqrt(dt), n_paths)

            # Schéma d'Euler
            drift = (self.b_function(t) - self.beta * r_paths[:, i]) * dt
            diffusion = self.sigma * dW

            r_paths[:, i + 1] = r_paths[:, i] + drift + diffusion

        return r_paths

    def simulate_exact(self, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """
        Simulation exacte du processus Hull-White (pour b(t) constant)
        """
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)

        r_paths = np.zeros((n_paths, n_steps + 1))
        r_paths[:, 0] = self.r0

        for i in range(n_steps):
            t = times[i]
            dt_step = times[i + 1] - times[i]

            # Solution exacte pour b(t) constant
            b_val = self.b_function(t)
            exp_beta_dt = np.exp(-self.beta * dt_step)

            # Moyenne conditionnelle
            mean = r_paths[:, i] * exp_beta_dt + (b_val / self.beta) * (1 - exp_beta_dt)

            # Variance conditionnelle
            var = (self.sigma**2 / (2 * self.beta)) * (1 - exp_beta_dt**2)

            # Génération des taux
            r_paths[:, i + 1] = np.random.normal(mean, np.sqrt(var))

        return r_paths

    def bond_price_analytical(self, t: float, T: float, r: float) -> float:
        """
        Prix analytique d'une obligation zéro-coupon dans le modèle Hull-White
        """
        tau = T - t
        if tau <= 0:
            return 1.0

        if self.beta == 0:
            B = tau
        else:
            B = (1 - np.exp(-self.beta * tau)) / self.beta

        # Calcul de A(t,T) par intégration numérique
        def integrand(s):
            tau_s = T - s
            if self.beta == 0:
                B_s = tau_s
            else:
                B_s = (1 - np.exp(-self.beta * tau_s)) / self.beta
            return 0.5 * self.sigma**2 * B_s**2 + self.b_function(s) * B_s

        A, _ = quad(integrand, t, T)

        return np.exp(A - B * r)

    def yield_curve(self, t: float, maturities: np.ndarray, r: float) -> np.ndarray:
        """
        Calcul de la courbe des taux pour différentes maturités
        """
        yields = np.zeros_like(maturities)
        for i, T in enumerate(maturities):
            if T > t:
                price = self.bond_price_analytical(t, T, r)
                yields[i] = -np.log(price) / (T - t)
            else:
                yields[i] = r
        return yields


class MonteCarloDerivativesPricer:
    """
    Pricing Monte Carlo de dérivés sur taux d'intérêt
    """

    def __init__(self, model):
        self.model = model

    def price_call_on_rate(
        self,
        T: float,
        strike: float,
        n_sims: int = 50000,
        n_steps: int = 100,
        use_antithetic: bool = False,
    ) -> Dict[str, float]:
        """
        Prix d'un call sur le taux court: payoff = max(r(T) - K, 0)
        """
        # Simulation des chemins
        if use_antithetic:
            # Variables antithétiques
            n_sims_half = n_sims // 2

            # Première moitié avec seed fixe
            np.random.seed(42)
            paths1 = self.model.simulate_euler(T, n_steps, n_sims_half)

            # Deuxième moitié avec chemins antithétiques
            np.random.seed(42)
            # Réutiliser les mêmes nombres aléatoires mais avec signe opposé
            paths2 = self._simulate_antithetic(T, n_steps, n_sims_half)

            all_paths = np.vstack([paths1, paths2])
        else:
            all_paths = self.model.simulate_euler(T, n_steps, n_sims)

        # Calcul des payoffs
        final_rates = all_paths[:, -1]
        payoffs = np.maximum(final_rates - strike, 0)

        # Calcul des facteurs d'actualisation
        dt = T / n_steps
        discount_factors = np.exp(-np.sum(all_paths[:, :-1], axis=1) * dt)

        # Prix actualisé
        discounted_payoffs = discount_factors * payoffs
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))

        return {
            "price": price,
            "std_error": std_error,
            "confidence_interval": (price - 1.96 * std_error, price + 1.96 * std_error),
        }

    def _simulate_antithetic(self, T: float, n_steps: int, n_paths: int) -> np.ndarray:
        """
        Simulation avec variables antithétiques
        """
        dt = T / n_steps
        r_paths = np.zeros((n_paths, n_steps + 1))
        r_paths[:, 0] = self.model.r0

        for i in range(n_steps):
            # Utiliser les mêmes nombres aléatoires mais négatifs
            dW = -np.random.normal(0, np.sqrt(dt), n_paths)

            if isinstance(self.model, CIRModel):
                r_current = np.maximum(r_paths[:, i], 1e-8)
                drift = (self.model.b - self.model.beta * r_current) * dt
                diffusion = self.model.sigma * np.sqrt(r_current) * dW
                r_paths[:, i + 1] = np.maximum(r_current + drift + diffusion, 1e-8)
            else:  # Hull-White
                t = i * dt
                drift = (
                    self.model.b_function(t) - self.model.beta * r_paths[:, i]
                ) * dt
                diffusion = self.model.sigma * dW
                r_paths[:, i + 1] = r_paths[:, i] + drift + diffusion

        return r_paths


def compare_models_simulation():
    """
    Comparaison des simulations des modèles CIR et Hull-White
    """
    print("=" * 80)
    print("COMPARAISON DES MODÈLES CIR ET HULL-WHITE")
    print("=" * 80)

    # Paramètres communs
    T = 2.0
    n_steps = 200
    n_paths = 1000
    r0 = 0.03

    # Paramètres CIR
    b_cir = 0.02
    beta_cir = 0.3
    sigma_cir = 0.15

    # Paramètres Hull-White
    beta_hw = 0.3
    sigma_hw = 0.01
    b_hw = lambda t: 0.02 + 0.01 * np.sin(2 * np.pi * t)  # Drift saisonnier

    # Initialisation des modèles
    cir_model = CIRModel(b_cir, beta_cir, sigma_cir, r0)
    hw_model = HullWhiteModel(beta_hw, sigma_hw, r0, b_hw)

    print(f"Modèle CIR - Condition de Feller: {cir_model.feller_condition}")
    print(f"Paramètres CIR: b={b_cir}, β={beta_cir}, σ={sigma_cir}, r0={r0}")
    print(f"Paramètres Hull-White: β={beta_hw}, σ={sigma_hw}, r0={r0}")
    print()

    # Simulation des chemins
    print("Simulation des chemins de taux...")
    cir_paths = cir_model.simulate_euler(T, n_steps, n_paths)
    hw_paths = hw_model.simulate_euler(T, n_steps, n_paths)

    times = np.linspace(0, T, n_steps + 1)

    # Graphique 1: Quelques chemins représentatifs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # CIR
    for i in range(min(10, n_paths)):
        ax1.plot(times, cir_paths[i, :], alpha=0.6, linewidth=0.8)
    ax1.set_title("Modèle CIR - Chemins du taux court")
    ax1.set_xlabel("Temps (années)")
    ax1.set_ylabel("Taux d'intérêt")
    ax1.grid(True, alpha=0.3)

    # Hull-White
    for i in range(min(10, n_paths)):
        ax2.plot(times, hw_paths[i, :], alpha=0.6, linewidth=0.8)
    ax2.set_title("Modèle Hull-White - Chemins du taux court")
    ax2.set_xlabel("Temps (années)")
    ax2.set_ylabel("Taux d'intérêt")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "figures/comparaison/simulation_paths_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Statistiques des taux finaux
    cir_final = cir_paths[:, -1]
    hw_final = hw_paths[:, -1]

    print(f"Statistiques des taux à T={T}:")
    print(
        f"CIR - Moyenne: {np.mean(cir_final):.4f}, Écart-type: {np.std(cir_final):.4f}"
    )
    print(f"CIR - Min: {np.min(cir_final):.4f}, Max: {np.max(cir_final):.4f}")
    print(
        f"Hull-White - Moyenne: {np.mean(hw_final):.4f}, Écart-type: {np.std(hw_final):.4f}"
    )
    print(f"Hull-White - Min: {np.min(hw_final):.4f}, Max: {np.max(hw_final):.4f}")
    print()

    # Histogrammes des distributions finales
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.hist(cir_final, bins=50, alpha=0.7, density=True, color="blue", label="CIR")
    ax1.set_title(f"Distribution des taux à T={T} - Modèle CIR")
    ax1.set_xlabel("Taux d'intérêt")
    ax1.set_ylabel("Densité")
    ax1.axvline(
        np.mean(cir_final),
        color="red",
        linestyle="--",
        label=f"Moyenne: {np.mean(cir_final):.4f}",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(
        hw_final, bins=50, alpha=0.7, density=True, color="green", label="Hull-White"
    )
    ax2.set_title(f"Distribution des taux à T={T} - Modèle Hull-White")
    ax2.set_xlabel("Taux d'intérêt")
    ax2.set_ylabel("Densité")
    ax2.axvline(
        np.mean(hw_final),
        color="red",
        linestyle="--",
        label=f"Moyenne: {np.mean(hw_final):.4f}",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "figures/comparaison/final_rate_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    return cir_model, hw_model, cir_paths, hw_paths


def bond_pricing_analysis():
    """
    Analyse du pricing d'obligations zéro-coupon
    """
    print("=" * 80)
    print("ANALYSE DU PRICING D'OBLIGATIONS ZÉRO-COUPON")
    print("=" * 80)

    # Paramètres
    r_current = 0.03
    maturities = np.linspace(0.25, 10, 40)

    # Modèles
    cir_model = CIRModel(b=0.02, beta=0.3, sigma=0.15, r0=r_current)
    hw_model = HullWhiteModel(beta=0.3, sigma=0.01, r0=r_current)

    # Calcul des courbes de taux
    cir_yields = cir_model.yield_curve(0, maturities, r_current)
    hw_yields = hw_model.yield_curve(0, maturities, r_current)

    # Graphique des courbes de taux
    plt.figure(figsize=(12, 8))
    plt.plot(
        maturities,
        cir_yields * 100,
        "b-",
        linewidth=2,
        label="CIR",
        marker="o",
        markersize=4,
    )
    plt.plot(
        maturities,
        hw_yields * 100,
        "r-",
        linewidth=2,
        label="Hull-White",
        marker="s",
        markersize=4,
    )
    plt.xlabel("Maturité (années)")
    plt.ylabel("Rendement (%)")
    plt.title("Courbes de rendement - Comparaison CIR vs Hull-White")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        "figures/comparaison/yield_curves_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # Analyse de sensibilité au niveau du taux
    rate_levels = np.linspace(0.01, 0.08, 20)
    maturity_test = 5.0

    cir_prices = [
        cir_model.bond_price_analytical(0, maturity_test, r) for r in rate_levels
    ]
    hw_prices = [
        hw_model.bond_price_analytical(0, maturity_test, r) for r in rate_levels
    ]

    plt.figure(figsize=(12, 8))
    plt.plot(
        rate_levels * 100,
        cir_prices,
        "b-",
        linewidth=2,
        label="CIR",
        marker="o",
        markersize=4,
    )
    plt.plot(
        rate_levels * 100,
        hw_prices,
        "r-",
        linewidth=2,
        label="Hull-White",
        marker="s",
        markersize=4,
    )
    plt.xlabel("Taux court (%)")
    plt.ylabel("Prix de l'obligation")
    plt.title(f"Prix d'obligation (maturité {maturity_test} ans) vs Taux court")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        "figures/comparaison/bond_price_sensitivity.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print(f"Prix d'obligation à 5 ans pour r={r_current*100:.1f}%:")
    print(f"CIR: {cir_model.bond_price_analytical(0, 5, r_current):.6f}")
    print(f"Hull-White: {hw_model.bond_price_analytical(0, 5, r_current):.6f}")
    print()


def derivative_pricing_monte_carlo():
    """
    Pricing Monte Carlo de dérivés sur taux d'intérêt
    """
    print("=" * 80)
    print("PRICING MONTE CARLO DE DÉRIVÉS SUR TAUX D'INTÉRÊT")
    print("=" * 80)

    # Paramètres du dérivé
    T = 1.0  # Maturité
    strike = 0.035  # Strike du call
    n_sims = 100000  # Nombre de simulations

    # Modèles
    cir_model = CIRModel(b=0.02, beta=0.3, sigma=0.15, r0=0.03)
    hw_model = HullWhiteModel(beta=0.3, sigma=0.01, r0=0.03)

    # Pricers Monte Carlo
    cir_pricer = MonteCarloDerivativesPricer(cir_model)
    hw_pricer = MonteCarloDerivativesPricer(hw_model)

    print(f"Pricing d'un call sur le taux court:")
    print(f"Payoff: max(r({T}) - {strike}, 0)")
    print(f"Nombre de simulations: {n_sims:,}")
    print()

    # Pricing standard
    print("Pricing standard (Monte Carlo classique):")
    cir_result = cir_pricer.price_call_on_rate(T, strike, n_sims)
    hw_result = hw_pricer.price_call_on_rate(T, strike, n_sims)

    print(f"CIR - Prix: {cir_result['price']:.6f} ± {cir_result['std_error']:.6f}")
    print(
        f"CIR - IC 95%: [{cir_result['confidence_interval'][0]:.6f}, {cir_result['confidence_interval'][1]:.6f}]"
    )
    print(f"Hull-White - Prix: {hw_result['price']:.6f} ± {hw_result['std_error']:.6f}")
    print(
        f"Hull-White - IC 95%: [{hw_result['confidence_interval'][0]:.6f}, {hw_result['confidence_interval'][1]:.6f}]"
    )
    print()

    # Pricing avec variables antithétiques
    print("Pricing avec variables antithétiques:")
    cir_result_anti = cir_pricer.price_call_on_rate(
        T, strike, n_sims, use_antithetic=True
    )
    hw_result_anti = hw_pricer.price_call_on_rate(
        T, strike, n_sims, use_antithetic=True
    )

    print(
        f"CIR - Prix: {cir_result_anti['price']:.6f} ± {cir_result_anti['std_error']:.6f}"
    )
    print(
        f"CIR - Réduction variance: {(1 - cir_result_anti['std_error']**2/cir_result['std_error']**2)*100:.1f}%"
    )
    print(
        f"Hull-White - Prix: {hw_result_anti['price']:.6f} ± {hw_result_anti['std_error']:.6f}"
    )
    print(
        f"Hull-White - Réduction variance: {(1 - hw_result_anti['std_error']**2/hw_result['std_error']**2)*100:.1f}%"
    )
    print()

    # Analyse de convergence
    convergence_analysis(cir_pricer, hw_pricer, T, strike)


def convergence_analysis(cir_pricer, hw_pricer, T, strike):
    """
    Analyse de la convergence Monte Carlo
    """
    print("Analyse de convergence Monte Carlo:")

    # Différents nombres de simulations
    n_sims_list = [1000, 2000, 5000, 10000, 20000, 50000]
    cir_prices = []
    hw_prices = []
    cir_errors = []
    hw_errors = []

    for n_sims in n_sims_list:
        cir_result = cir_pricer.price_call_on_rate(T, strike, n_sims)
        hw_result = hw_pricer.price_call_on_rate(T, strike, n_sims)

        cir_prices.append(cir_result["price"])
        hw_prices.append(hw_result["price"])
        cir_errors.append(cir_result["std_error"])
        hw_errors.append(hw_result["std_error"])

    # Graphique de convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Prix
    ax1.plot(n_sims_list, cir_prices, "b-o", label="CIR", linewidth=2, markersize=6)
    ax1.plot(
        n_sims_list, hw_prices, "r-s", label="Hull-White", linewidth=2, markersize=6
    )
    ax1.set_xlabel("Nombre de simulations")
    ax1.set_ylabel("Prix du call")
    ax1.set_title("Convergence du prix Monte Carlo")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    # Erreur standard
    ax2.loglog(n_sims_list, cir_errors, "b-o", label="CIR", linewidth=2, markersize=6)
    ax2.loglog(
        n_sims_list, hw_errors, "r-s", label="Hull-White", linewidth=2, markersize=6
    )
    ax2.loglog(
        n_sims_list,
        [0.1 / np.sqrt(n) for n in n_sims_list],
        "k--",
        label="1/√N théorique",
        alpha=0.7,
    )
    ax2.set_xlabel("Nombre de simulations")
    ax2.set_ylabel("Erreur standard")
    ax2.set_title("Convergence de l'erreur Monte Carlo")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "figures/comparaison/monte_carlo_convergence.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def sensitivity_analysis():
    """
    Analyse de sensibilité des modèles aux paramètres
    """
    print("=" * 80)
    print("ANALYSE DE SENSIBILITÉ AUX PARAMÈTRES")
    print("=" * 80)

    # Paramètres de base
    T = 2.0
    r0 = 0.03

    # Analyse de sensibilité CIR
    print("Sensibilité du modèle CIR:")

    # Variation de sigma
    sigmas = np.linspace(0.05, 0.25, 10)
    cir_final_means = []
    cir_final_stds = []

    for sigma in sigmas:
        model = CIRModel(b=0.02, beta=0.3, sigma=sigma, r0=r0)
        paths = model.simulate_euler(T, 200, 5000)
        final_rates = paths[:, -1]
        cir_final_means.append(np.mean(final_rates))
        cir_final_stds.append(np.std(final_rates))

    # Graphiques de sensibilité
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # CIR - Sensibilité à sigma
    ax1.plot(sigmas, cir_final_means, "b-o", linewidth=2, markersize=6)
    ax1.set_xlabel("Volatilité σ")
    ax1.set_ylabel("Moyenne r(T)")
    ax1.set_title("CIR - Sensibilité à la volatilité (moyenne)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(sigmas, cir_final_stds, "r-o", linewidth=2, markersize=6)
    ax2.set_xlabel("Volatilité σ")
    ax2.set_ylabel("Écart-type r(T)")
    ax2.set_title("CIR - Sensibilité à la volatilité (dispersion)")
    ax2.grid(True, alpha=0.3)

    # Hull-White - Sensibilité à sigma
    hw_sigmas = np.linspace(0.005, 0.02, 10)
    hw_final_means = []
    hw_final_stds = []

    for sigma in hw_sigmas:
        model = HullWhiteModel(beta=0.3, sigma=sigma, r0=r0)
        paths = model.simulate_euler(T, 200, 5000)
        final_rates = paths[:, -1]
        hw_final_means.append(np.mean(final_rates))
        hw_final_stds.append(np.std(final_rates))

    ax3.plot(hw_sigmas, hw_final_means, "g-o", linewidth=2, markersize=6)
    ax3.set_xlabel("Volatilité σ")
    ax3.set_ylabel("Moyenne r(T)")
    ax3.set_title("Hull-White - Sensibilité à la volatilité (moyenne)")
    ax3.grid(True, alpha=0.3)

    ax4.plot(hw_sigmas, hw_final_stds, "orange", marker="o", linewidth=2, markersize=6)
    ax4.set_xlabel("Volatilité σ")
    ax4.set_ylabel("Écart-type r(T)")
    ax4.set_title("Hull-White - Sensibilité à la volatilité (dispersion)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "figures/comparaison/sensitivity_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def main():
    """
    Fonction principale exécutant toutes les analyses
    """
    print("🚀 DÉMARRAGE DU SCRIPT D'ANALYSE COMPLÈTE")
    print("Modèles de taux d'intérêt stochastiques CIR et Hull-White")
    print("=" * 80)

    # Créer les dossiers pour les figures si nécessaire
    import os

    os.makedirs("figures", exist_ok=True)
    os.makedirs("figures/comparaison", exist_ok=True)

    # Seed pour la reproductibilité
    np.random.seed(42)

    try:
        # 1. Comparaison des simulations
        cir_model, hw_model, cir_paths, hw_paths = compare_models_simulation()

        # 2. Analyse du pricing d'obligations
        bond_pricing_analysis()

        # 3. Pricing Monte Carlo de dérivés
        derivative_pricing_monte_carlo()

        # 4. Analyse de sensibilité
        sensitivity_analysis()

        print("=" * 80)
        print("✅ ANALYSE COMPLÈTE TERMINÉE AVEC SUCCÈS")
        print(
            "📊 Tous les graphiques ont été sauvegardés dans le dossier 'figures/comparaison/'"
        )
        print("=" * 80)

    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
