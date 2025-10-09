# finance_lib/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from .models import BaseShortRateModel, CIRModel, HullWhiteModel
from .pricing_engine import MonteCarloDerivativesPricer
# Configuration globale des graphiques
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "figure.figsize": (12, 7), "font.size": 12, "axes.labelsize": 12,
    "axes.titlesize": 14, "legend.fontsize": 11, "lines.linewidth": 1.5
})

def plot_short_rate_trajectories(model: BaseShortRateModel, model_name: str, T: float, n_steps: int, n_paths: int):
    """Trace un échantillon de trajectoires du taux court."""
    times = np.linspace(0, T, n_steps + 1)
    paths = model.simulate_euler(T, n_steps, n_paths)
    plt.figure()
    plt.plot(times, paths.T, alpha=0.7, lw=1.0)
    plt.title(f"{model_name}: Trajectoires Simulées du Taux Court")
    plt.xlabel("Temps (années)"), plt.ylabel("Taux Court r(t)")
    plt.grid(True, linestyle="--"), plt.show()

def plot_final_rate_distribution(model: BaseShortRateModel, model_name: str, T: float, n_sims: int):
    """Trace l'histogramme de la distribution des taux à maturité T."""
    paths = model.simulate_euler(T, n_steps=100, n_paths=n_sims)
    final_rates = paths[:, -1]
    plt.figure()
    plt.hist(final_rates, bins=50, density=True, alpha=0.7, edgecolor='black')
    plt.title(f"{model_name}: Distribution de r(T) pour T={T} ans")
    plt.xlabel("Taux Court r(T)"), plt.ylabel("Densité")
    plt.axvline(np.mean(final_rates), color='red', ls='--', label=f'Moyenne: {np.mean(final_rates):.4f}')
    plt.legend(), plt.grid(True, linestyle="--"), plt.show()

def plot_yield_curves_comparison(models: dict, r_current: float, max_maturity: float):
    """Compare les courbes de rendement de plusieurs modèles."""
    maturities = np.linspace(0.1, max_maturity, 50)
    plt.figure()
    for name, model in models.items():
        yields = model.yield_curve(0, maturities, r_current)
        plt.plot(maturities, yields * 100, marker='o', markersize=4, label=name)
    plt.title(f"Comparaison des Courbes de Rendement (r(0)={r_current*100:.2f}%)")
    plt.xlabel("Maturité (années)"), plt.ylabel("Rendement (%)")
    plt.legend(), plt.grid(True, linestyle="--"), plt.show()

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


def plot_b_function(b_func: callable, T: float, model_name: str):
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