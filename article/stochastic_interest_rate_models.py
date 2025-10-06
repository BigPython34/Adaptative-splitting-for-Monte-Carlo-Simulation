"""
Implémentation et visualisation des modèles de taux courts stochastiques :
Cox-Ingersoll-Ross (CIR) et Hull-White, pour le pricing d'obligations
zéro-coupon et de produits dérivés.

Basé sur l'article : "Zero-Coupon Bond Pricing in Stochastic Interest Rate Models:
The Cox-Ingersoll-Ross and Hull-White Frameworks"

Auteur: Octave Cerclé - ISAE-Supaero Finance Project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from typing import Tuple, Optional, Callable, Dict, Any
import warnings
import os

# Supprimer les avertissements qui peuvent être bruyants pour l'utilisateur
warnings.filterwarnings("ignore")

# Configuration globale des graphiques
plt.style.use("seaborn-v0_8-darkgrid") # Style plus moderne et adapté
plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["lines.linewidth"] = 1.5


class BaseShortRateModel:
    """
    Classe de base abstraite pour les modèles de taux courts stochastiques.
    Définit l'interface commune pour la simulation et le pricing d'obligations.
    """

    def __init__(self, r0: float):
        if r0 < 0:
            raise ValueError("Le taux initial r0 ne peut pas être négatif.")
        self.r0 = r0

    def _get_drift_diffusion(self, t: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retourne le terme de drift et de diffusion du processus SDE.
        Doit être implémenté par les sous-classes.
        """
        raise NotImplementedError("La méthode _get_drift_diffusion doit être implémentée.")

    def simulate_euler(self, T: float, n_steps: int, n_paths: int = 1,
                       dt_override: Optional[float] = None,
                       dW_generator: Optional[Callable[[int, float], np.ndarray]] = None) -> np.ndarray:
        """
        Simule des trajectoires du taux court en utilisant le schéma d'Euler-Maruyama.
        Gère la troncature pour le modèle CIR.
        """
        dt = dt_override if dt_override is not None else T / n_steps
        times = np.linspace(0, T, n_steps + 1)

        r_paths = np.zeros((n_paths, n_steps + 1))
        r_paths[:, 0] = self.r0

        if dW_generator is None:
            dW_generator = lambda num_paths, step_dt: np.random.normal(0, np.sqrt(step_dt), num_paths)

        for i in range(n_steps):
            t_current = times[i]
            r_current = np.maximum(r_paths[:, i], 1e-10)

            drift, diffusion_term = self._get_drift_diffusion(t_current, r_current)
            dW = dW_generator(n_paths, dt)

            r_next = r_current + drift * dt + diffusion_term * dW

            if isinstance(self, CIRModel):
                r_paths[:, i + 1] = np.maximum(r_next, 1e-8)
            else:
                r_paths[:, i + 1] = r_next

        return r_paths

    def bond_price_analytical(self, t: float, T: float, r_at_t: float) -> float:
        """
        Calcule le prix analytique d'une obligation zéro-coupon.
        """
        raise NotImplementedError("La méthode bond_price_analytical doit être implémentée.")

    def yield_curve(self, t: float, maturities: np.ndarray, r_at_t: float) -> np.ndarray:
        """
        Calcule la courbe des taux à l'instant t pour différentes maturités.
        """
        yields = np.zeros_like(maturities, dtype=float)
        for i, Ti in enumerate(maturities):
            if Ti <= t:
                yields[i] = r_at_t
            else:
                price = self.bond_price_analytical(t, Ti, r_at_t)
                if price <= 0:
                    yields[i] = np.nan
                else:
                    yields[i] = -np.log(price) / (Ti - t)
        return yields


class CIRModel(BaseShortRateModel):
    """
    Implémentation du modèle Cox-Ingersoll-Ross (CIR).
    """

    def __init__(self, b: float, beta: float, sigma: float, r0: float):
        super().__init__(r0)
        self.b = b
        self.beta = beta
        self.sigma = sigma
        self.feller_condition_met = (2 * b) >= (sigma**2)

    def _get_drift_diffusion(self, t: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        drift = self.b - self.beta * r
        diffusion_term = self.sigma * np.sqrt(r)
        return drift, diffusion_term

    def bond_price_analytical(self, t: float, T: float, r_at_t: float) -> float:
        tau = T - t
        if tau <= 0: return 1.0
        d = np.sqrt(self.beta**2 + 2 * self.sigma**2)
        exp_d_tau = np.exp(d * tau)
        B = 2 * (exp_d_tau - 1) / ((d + self.beta) * (exp_d_tau - 1) + 2 * d)
        numerator_A = 2 * d * np.exp((d + self.beta) * tau / 2)
        denominator_A = (d + self.beta) * (exp_d_tau - 1) + 2 * d
        A = (2 * self.b / self.sigma**2) * np.log(numerator_A / denominator_A)
        return np.exp(A - B * r_at_t)


class HullWhiteModel(BaseShortRateModel):
    """
    Implémentation du modèle Hull-White.
    """

    def __init__(self, beta: float, sigma: float, r0: float,
                 b_function: Optional[Callable[[float], float]] = None):
        super().__init__(r0)
        self.beta = beta
        self.sigma = sigma
        self.b_function = b_function if b_function else lambda t: 0.02

    def _get_drift_diffusion(self, t: float, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        drift = self.b_function(t) - self.beta * r
        diffusion_term = self.sigma * np.ones_like(r)
        return drift, diffusion_term

    def bond_price_analytical(self, t: float, T: float, r_at_t: float) -> float:
        tau = T - t
        if tau <= 0: return 1.0
        B_val = (1 - np.exp(-self.beta * tau)) / self.beta if self.beta != 0 else tau

        def integrand(s_inner):
            tau_s = T - s_inner
            B_s_inner = (1 - np.exp(-self.beta * tau_s)) / self.beta if self.beta != 0 else tau_s
            return 0.5 * self.sigma**2 * B_s_inner**2 + self.b_function(s_inner) * B_s_inner

        A_val, _ = quad(integrand, t, T)
        return np.exp(A_val - B_val * r_at_t)


class MonteCarloDerivativesPricer:
    """
    Classe pour le pricing de dérivés sur taux d'intérêt par Monte Carlo.
    Prend en charge les techniques de réduction de variance.
    """

    def __init__(self, model: BaseShortRateModel):
        self.model = model

    def price_european_call_on_rate(
        self,
        T: float,
        strike: float,
        n_sims: int = 50000,
        n_steps: int = 100,
        use_antithetic_variates: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Prix d'une option call européenne sur le taux court à maturité T.
        Payoff = max(r(T) - K, 0).

        Args:
            T (float): Maturité de l'option.
            strike (float): Prix d'exercice.
            n_sims (int): Nombre de simulations Monte Carlo.
            n_steps (int): Nombre d'étapes de temps pour la simulation du taux.
            use_antithetic_variates (bool): Utiliser la réduction de variance par variables antithétiques.
            seed (Optional[int]): Graine pour la reproductibilité.

        Returns:
            Dict[str, float]: Dictionnaire contenant le prix, l'erreur standard et l'intervalle de confiance.
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps

        if use_antithetic_variates:
            n_sims_half = n_sims // 2
            # Générateur dW pour les chemins positifs
            dW_positive = lambda num_paths, step_dt: np.random.normal(0, np.sqrt(step_dt), num_paths)
            # Générateur dW pour les chemins antithétiques
            dW_negative = lambda num_paths, step_dt: -np.random.normal(0, np.sqrt(step_dt), num_paths)

            paths1 = self.model.simulate_euler(T, n_steps, n_sims_half, dt_override=dt, dW_generator=dW_positive)
            paths2 = self.model.simulate_euler(T, n_steps, n_sims_half, dt_override=dt, dW_generator=dW_negative)
            all_paths = np.vstack([paths1, paths2])
        else:
            all_paths = self.model.simulate_euler(T, n_steps, n_sims, dt_override=dt)

        # Calcul des payoffs à maturité T
        final_rates = all_paths[:, -1]
        payoffs = np.maximum(final_rates - strike, 0)

        # Calcul des facteurs d'actualisation stochastiques
        # Approximation de l'intégrale par somme de Riemann à gauche
        integrated_rates = np.sum(all_paths[:, :-1], axis=1) * dt
        discount_factors = np.exp(-integrated_rates)

        # Prix actualisé
        discounted_payoffs = discount_factors * payoffs
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))

        return {
            "price": price,
            "std_error": std_error,
            "confidence_interval_95": (price - 1.96 * std_error, price + 1.96 * std_error),
        }


# --- Fonctions de visualisation ---

def plot_short_rate_trajectories(model: BaseShortRateModel, model_name: str, T: float, n_steps: int, n_paths: int = 10):
    """
    Trace un échantillon de trajectoires du taux court simulées.
    """
    times = np.linspace(0, T, n_steps + 1)
    paths = model.simulate_euler(T, n_steps, n_paths)

    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(times, paths[i, :], alpha=0.7, lw=1.0)

    plt.title(f"{model_name} Model: Sampled Short-Rate Trajectories")
    plt.xlabel("Time (Years)")
    plt.ylabel("Short Rate r(t)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"figures/cir_hw_comparison/{model_name.lower().replace(' ', '_')}_trajectories.png")
    plt.show()

def plot_final_rate_distribution(model: BaseShortRateModel, model_name: str, T: float, n_sims: int = 10000):
    """
    Trace l'histogramme de la distribution des taux courts à maturité T.
    """
    paths = model.simulate_euler(T, 100, n_sims) # 100 steps pour la distribution finale
    final_rates = paths[:, -1]

    plt.figure(figsize=(10, 6))
    plt.hist(final_rates, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f"{model_name} Model: Distribution of Short Rate at T={T} Years")
    plt.xlabel(f"Short Rate r(T)")
    plt.ylabel("Density")
    plt.axvline(np.mean(final_rates), color='red', linestyle='--', label=f'Mean: {np.mean(final_rates):.4f}')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"figures/cir_hw_comparison/{model_name.lower().replace(' ', '_')}_final_distribution.png")
    plt.show()


def plot_yield_curves(cir_model: CIRModel, hw_model: HullWhiteModel, r_current: float, max_maturity: float = 10.0):
    """
    Compare les courbes de rendement générées par les modèles CIR et Hull-White.
    """
    maturities = np.linspace(0.1, max_maturity, 50) # Maturités jusqu'à 10 ans

    cir_yields = cir_model.yield_curve(0, maturities, r_current)
    hw_yields = hw_model.yield_curve(0, maturities, r_current)

    plt.figure(figsize=(10, 6))
    plt.plot(maturities, cir_yields * 100, label="CIR Model", marker='o', markersize=4, lw=1.5)
    plt.plot(maturities, hw_yields * 100, label="Hull-White Model", marker='s', markersize=4, lw=1.5)
    plt.title(f"Zero-Coupon Yield Curves at r(0)={r_current*100:.2f}%")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/cir_hw_comparison/yield_curves_comparison.png")
    plt.show()

def plot_bond_price_sensitivity_to_rate(cir_model: CIRModel, hw_model: HullWhiteModel, maturity_test: float = 5.0):
    """
    Montre la sensibilité du prix d'une obligation zéro-coupon au taux court initial.
    """
    rate_levels = np.linspace(0.005, 0.10, 50) # Taux initiaux de 0.5% à 10%

    cir_prices = [cir_model.bond_price_analytical(0, maturity_test, r) for r in rate_levels]
    hw_prices = [hw_model.bond_price_analytical(0, maturity_test, r) for r in rate_levels]

    plt.figure(figsize=(10, 6))
    plt.plot(rate_levels * 100, cir_prices, label="CIR Model", marker='o', markersize=4, lw=1.5)
    plt.plot(rate_levels * 100, hw_prices, label="Hull-White Model", marker='s', markersize=4, lw=1.5)
    plt.title(f"Zero-Coupon Bond Price Sensitivity (Maturity T={maturity_test} Years)")
    plt.xlabel("Initial Short Rate r(0) (%)")
    plt.ylabel("Bond Price p(0, T)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/cir_hw_comparison/bond_price_sensitivity.png")
    plt.show()

def plot_monte_carlo_convergence(pricer_cir: MonteCarloDerivativesPricer, pricer_hw: MonteCarloDerivativesPricer, T: float, strike: float):
    """
    Analyse et trace la convergence Monte Carlo pour le pricing d'option.
    """
    n_sims_list = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    cir_prices = []
    hw_prices = []
    cir_errors = []
    hw_errors = []

    # Utilisation d'une graine fixe pour chaque point afin de lisser la convergence
    # Bien que la théorie suggère que l'erreur doit diminuer avec sqrt(N), la variance des
    # estimateurs peut rendre le graphique "bruité" si les graines changent.
    base_seed = 42

    print("\n--- Analyse de Convergence Monte Carlo ---")
    print(f"Option: Call Européen sur r(T) avec T={T}, K={strike}")

    for n_sims in n_sims_list:
        cir_result = pricer_cir.price_european_call_on_rate(T, strike, n_sims, seed=base_seed)
        hw_result = pricer_hw.price_european_call_on_rate(T, strike, n_sims, seed=base_seed + 1) # Graine différente pour HW
        cir_prices.append(cir_result["price"])
        hw_prices.append(hw_result["price"])
        cir_errors.append(cir_result["std_error"])
        hw_errors.append(hw_result["std_error"])
        print(f"  N_sims={n_sims:<7}: CIR Price={cir_result['price']:.6f} (Err={cir_result['std_error']:.6f}) | HW Price={hw_result['price']:.6f} (Err={hw_result['std_error']:.6f})")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot des prix
    axes[0].plot(n_sims_list, cir_prices, '-o', label='CIR Price', markersize=5, lw=1.5)
    axes[0].plot(n_sims_list, hw_prices, '-s', label='Hull-White Price', markersize=5, lw=1.5)
    axes[0].set_xscale('log')
    axes[0].set_title('Monte Carlo Price Convergence')
    axes[0].set_xlabel('Number of Simulations (log scale)')
    axes[0].set_ylabel('Option Price')
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot des erreurs standard
    axes[1].plot(n_sims_list, cir_errors, '-o', label='CIR Standard Error', markersize=5, lw=1.5)
    axes[1].plot(n_sims_list, hw_errors, '-s', label='Hull-White Standard Error', markersize=5, lw=1.5)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log') # L'erreur décroit avec sqrt(N), donc linéaire en log-log
    axes[1].set_title('Monte Carlo Standard Error Convergence')
    axes[1].set_xlabel('Number of Simulations (log scale)')
    axes[1].set_ylabel('Standard Error (log scale)')
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("figures/monte_carlo/monte_carlo_convergence.png")
    plt.show()


def plot_variance_reduction_comparison(pricer: MonteCarloDerivativesPricer, model_name: str, T: float, strike: float, n_sims_max: int = 100000):
    """
    Compare l'efficacité de la réduction de variance (variables antithétiques)
    en traçant l'erreur standard en fonction du nombre de simulations.
    """
    n_sims_list = [1000, 2000, 5000, 10000, 20000, 50000, n_sims_max]
    errors_vanilla = []
    errors_antithetic = []
    prices_vanilla = []
    prices_antithetic = []

    base_seed = 100 # Graine pour la reproductibilité

    print(f"\n--- Comparaison Réduction de Variance ({model_name}) ---")
    print(f"Option: Call Européen sur r(T) avec T={T}, K={strike}")

    for n_sims in n_sims_list:
        # Sans réduction de variance
        result_vanilla = pricer.price_european_call_on_rate(T, strike, n_sims, use_antithetic_variates=False, seed=base_seed)
        errors_vanilla.append(result_vanilla["std_error"])
        prices_vanilla.append(result_vanilla["price"])

        # Avec réduction de variance
        result_antithetic = pricer.price_european_call_on_rate(T, strike, n_sims, use_antithetic_variates=True, seed=base_seed)
        errors_antithetic.append(result_antithetic["std_error"])
        prices_antithetic.append(result_antithetic["price"])
        
        print(f"  N_sims={n_sims:<7}: Vanilla Err={result_vanilla['std_error']:.6f} | Antithetic Err={result_antithetic['std_error']:.6f}")


    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot des erreurs standard
    axes[0].plot(n_sims_list, errors_vanilla, '-o', label='Standard MC', markersize=5, lw=1.5)
    axes[0].plot(n_sims_list, errors_antithetic, '-s', label='Antithetic Variates', markersize=5, lw=1.5)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title(f'{model_name} MC Error with Variance Reduction')
    axes[0].set_xlabel('Number of Simulations (log scale)')
    axes[0].set_ylabel('Standard Error (log scale)')
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot des prix (pour montrer qu'ils convergent vers la même valeur)
    axes[1].plot(n_sims_list, prices_vanilla, '-o', label='Standard MC Price', markersize=5, lw=1.5)
    axes[1].plot(n_sims_list, prices_antithetic, '-s', label='Antithetic Variates Price', markersize=5, lw=1.5)
    axes[1].set_xscale('log')
    axes[1].set_title(f'{model_name} MC Price with Variance Reduction')
    axes[1].set_xlabel('Number of Simulations (log scale)')
    axes[1].set_ylabel('Option Price')
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"figures/monte_carlo/{model_name.lower().replace(' ', '_')}_variance_reduction.png")
    plt.show()


def plot_b_function(b_func: Callable, T: float, model_name: str):
    """
    Trace la fonction de drift b(t) sur une période T.
    """
    times = np.linspace(0, T, 200)
    b_values = [b_func(t) for t in times]

    plt.figure(figsize=(10, 6))
    plt.plot(times, b_values, lw=2)
    plt.title(f"Fonction de Drift Exogène b(t) pour le modèle {model_name}")
    plt.xlabel("Temps (années)")
    plt.ylabel("Valeur de b(t)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"figures/cir_hw_comparison/{model_name.lower().replace(' ', '_')}_b_function.png")
    plt.show()

# La fonction principale est modifiée pour utiliser le b(t) saisonnier
def main_analysis_with_seasonal_drift():
    """
    Exécute une analyse comparative en utilisant un drift saisonnier pour
    le modèle Hull-White.
    """
    np.random.seed(42)
    r0_common = 0.025

    # --- Modèle CIR (inchangé, sert de référence) ---
    cir_model = CIRModel(b=0.006, beta=0.2, sigma=0.1, r0=r0_common)
    print("Modèle CIR de référence initialisé.")

    # --- MODIFICATION : Définition du modèle Hull-White avec b(t) saisonnier ---
    hw_beta = 0.2
    hw_sigma = 0.015

    def seasonal_b_function(t: float) -> float:
        """
        Une fonction b(t) qui modélise une réversion vers une moyenne
        elle-même saisonnière.
        theta(t) = long_term_mean + amplitude * sin(2*pi*t)
        b(t) = beta * theta(t)
        """
        long_term_mean_theta = 0.03  # La moyenne de long terme est de 3%
        seasonality_amplitude = 0.07 # L'amplitude des cycles annuels est de 7%
        
        # theta(t) est la moyenne vers laquelle le taux r(t) est attiré à l'instant t
        theta_t = long_term_mean_theta + seasonality_amplitude * np.sin(2 * np.pi * t)
        return hw_beta * theta_t

    hw_model_seasonal = HullWhiteModel(hw_beta, hw_sigma, r0_common, seasonal_b_function)
    print("\nModèle Hull-White avec drift b(t) SAISONNIER initialisé.")
    print(f"  Paramètres : β={hw_beta}, σ={hw_sigma}")
    print(f"  Moyenne de réversion theta(t) oscille entre {(0.03-0.015)*100:.1f}% et {(0.03+0.015)*100:.1f}%")
    print("-" * 80)

    # --- Section 1: Visualisation de la fonction b(t) et des trajectoires ---
    print("\n[SECTION 1] Visualisation de la dynamique du modèle saisonnier")
    T_sim = 5.0
    n_steps_sim = 250

    # Visualiser la fonction b(t) elle-même pour comprendre l'hypothèse
    plot_b_function(seasonal_b_function, T_sim, "Hull-White Saisonnalier")

    # Visualiser les trajectoires qui en résultent
    print(f"\nSimulation de 10 trajectoires sur {T_sim} ans...")
    # On peut aussi tracer le CIR pour comparer
    plot_short_rate_trajectories(cir_model, "CIR (Référence)", T_sim, n_steps_sim, n_paths=10)
    plot_short_rate_trajectories(hw_model_seasonal, "Hull-White Saisonnalier", T_sim, n_steps_sim, n_paths=10)
    print("  -> Les trajectoires du modèle HW devraient montrer des tendances saisonnières.")

    # --- Section 2: Impact sur la courbe des taux ---
    print("\n[SECTION 2] Impact du drift saisonnier sur la courbe des taux")
    
    # On crée aussi un modèle HW simple pour comparer
    hw_model_simple = HullWhiteModel(hw_beta, hw_sigma, r0_common, lambda t: hw_beta * 0.03)
    
    maturities = np.linspace(0.1, 10, 50)
    cir_yields = cir_model.yield_curve(0, maturities, r0_common)
    hw_simple_yields = hw_model_simple.yield_curve(0, maturities, r0_common)
    hw_seasonal_yields = hw_model_seasonal.yield_curve(0, maturities, r0_common)

    plt.figure(figsize=(12, 7))
    plt.plot(maturities, cir_yields * 100, label="CIR (Référence)", marker='o', markersize=4, ls=':')
    plt.plot(maturities, hw_simple_yields * 100, label="Hull-White (b constant)", marker='s', markersize=4, ls='--')
    plt.plot(maturities, hw_seasonal_yields * 100, label="Hull-White (b saisonnier)", marker='^', markersize=5, lw=2)
    plt.title(f"Impact d'un Drift Saisonnalier sur la Courbe des Taux (r(0)={r0_common*100:.2f}%)")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Rendement (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/cir_hw_comparison/yield_curve_seasonal_impact.png")
    plt.show()
    print("  -> La courbe des taux du modèle saisonnier peut présenter des formes plus complexes ('bosses').")
    print("-" * 80)
    
    print("\n✅ Analyse avec drift saisonnier terminée.")


if __name__ == "__main__":
    main_analysis_with_seasonal_drift()