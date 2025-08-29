"""
Script de démonstration pratique des modèles CIR et Hull-White
Applications spécifiques : calibration, pricing de produits structurés, gestion de risque

Auteur: Octave Cerclé - ISAE-Supaero
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from script_for_article import CIRModel, HullWhiteModel, MonteCarloDerivativesPricer
import pandas as pd


def calibrate_cir_to_yield_curve():
    """
    Calibration du modèle CIR sur une courbe des taux observée
    """
    print("=" * 80)
    print("CALIBRATION DU MODÈLE CIR SUR COURBE DES TAUX")
    print("=" * 80)

    # Courbe des taux de marché observée (exemples réalistes)
    market_maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 30])
    market_yields = np.array([0.015, 0.018, 0.022, 0.025, 0.032, 0.035, 0.038])

    print("Courbe de taux de marché:")
    for mat, yield_rate in zip(market_maturities, market_yields):
        print(f"  {mat:5.2f} ans: {yield_rate*100:5.2f}%")
    print()

    # Fonction objectif pour la calibration
    def objective_function(params):
        b, beta, sigma = params
        if b <= 0 or beta <= 0 or sigma <= 0:
            return 1e6

        r0 = market_yields[2]  # Utiliser le taux 1 an comme proxy du taux court
        model = CIRModel(b, beta, sigma, r0)

        model_yields = model.yield_curve(0, market_maturities, r0)

        # Erreur quadratique moyenne
        mse = np.mean((model_yields - market_yields) ** 2)
        return mse * 10000  # Scaling pour l'optimisation

    # Paramètres initiaux
    initial_params = [0.03, 0.2, 0.15]
    bounds = [(0.001, 0.1), (0.01, 1.0), (0.01, 0.5)]

    print("Calibration en cours...")
    result = minimize(
        objective_function, initial_params, bounds=bounds, method="L-BFGS-B"
    )

    b_opt, beta_opt, sigma_opt = result.x
    calibrated_model = CIRModel(b_opt, beta_opt, sigma_opt, market_yields[2])

    print(f"Paramètres calibrés:")
    print(f"  b = {b_opt:.6f}")
    print(f"  β = {beta_opt:.6f}")
    print(f"  σ = {sigma_opt:.6f}")
    print(f"  Erreur finale: {result.fun:.6f}")
    print(f"  Condition de Feller: {calibrated_model.feller_condition}")
    print()

    # Comparaison marché vs modèle
    model_yields = calibrated_model.yield_curve(0, market_maturities, market_yields[2])

    plt.figure(figsize=(12, 8))
    plt.plot(
        market_maturities,
        market_yields * 100,
        "ro-",
        linewidth=2,
        markersize=8,
        label="Marché",
        markerfacecolor="white",
        markeredgewidth=2,
    )
    plt.plot(
        market_maturities,
        model_yields * 100,
        "bs-",
        linewidth=2,
        markersize=6,
        label="CIR calibré",
    )
    plt.xlabel("Maturité (années)")
    plt.ylabel("Rendement (%)")
    plt.title("Calibration CIR - Comparaison Marché vs Modèle")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("figures/cir_calibration.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Analyse des erreurs
    errors = (model_yields - market_yields) * 10000  # en points de base
    print("Erreurs de calibration (en points de base):")
    for mat, error in zip(market_maturities, errors):
        print(f"  {mat:5.2f} ans: {error:+6.2f} bp")
    print(f"  RMSE: {np.sqrt(np.mean(errors**2)):.2f} bp")
    print()

    return calibrated_model


def price_structured_products():
    """
    Pricing de produits structurés complexes
    """
    print("=" * 80)
    print("PRICING DE PRODUITS STRUCTURÉS")
    print("=" * 80)

    # Modèles calibrés
    cir_model = CIRModel(b=0.025, beta=0.3, sigma=0.12, r0=0.025)
    hw_model = HullWhiteModel(beta=0.3, sigma=0.008, r0=0.025)

    # 1. Range Accrual Note
    print("1. Range Accrual Note")
    print("   Payoff: Notionnel × Σ(1{r_min ≤ r(t) ≤ r_max}) / N")

    T = 2.0
    n_observations = 24  # Observations mensuelles
    r_min, r_max = 0.02, 0.04
    notional = 1000000  # 1M EUR

    # Simulation pour Range Accrual
    n_sims = 50000
    observation_times = np.linspace(0, T, n_observations + 1)[1:]  # Exclure t=0

    def price_range_accrual(model, model_name):
        # Simulation des chemins
        paths = model.simulate_euler(T, n_observations, n_sims)

        # Comptage des observations dans la range
        in_range_count = 0
        total_payoff = 0

        for i in range(n_sims):
            path = paths[i, 1:]  # Exclure le point initial
            in_range = np.sum((path >= r_min) & (path <= r_max))

            # Facteur d'actualisation
            dt = T / n_observations
            discount_factor = np.exp(-np.sum(paths[i, :-1]) * dt)

            # Payoff actualisé
            payoff = notional * (in_range / n_observations)
            total_payoff += payoff * discount_factor

        return total_payoff / n_sims

    price_cir = price_range_accrual(cir_model, "CIR")
    price_hw = price_range_accrual(hw_model, "Hull-White")

    print(f"   Prix CIR: {price_cir:,.2f} EUR")
    print(f"   Prix Hull-White: {price_hw:,.2f} EUR")
    print(
        f"   Différence: {abs(price_cir - price_hw):,.2f} EUR ({abs(price_cir - price_hw)/price_cir*100:.2f}%)"
    )
    print()

    # 2. Cliquet Option (ratchet)
    print("2. Cliquet Option")
    print("   Payoff: Σ max(r(t_i) - r(t_{i-1}), 0)")

    def price_cliquet(model, model_name):
        paths = model.simulate_euler(T, n_observations, n_sims)
        total_payoff = 0

        for i in range(n_sims):
            path = paths[i, :]
            # Calcul des increments positifs
            increments = np.diff(path)
            positive_increments = np.maximum(increments, 0)

            # Facteur d'actualisation
            dt = T / n_observations
            discount_factor = np.exp(-np.sum(path[:-1]) * dt)

            payoff = notional * np.sum(positive_increments)
            total_payoff += payoff * discount_factor

        return total_payoff / n_sims

    cliquet_cir = price_cliquet(cir_model, "CIR")
    cliquet_hw = price_cliquet(hw_model, "Hull-White")

    print(f"   Prix CIR: {cliquet_cir:,.2f} EUR")
    print(f"   Prix Hull-White: {cliquet_hw:,.2f} EUR")
    print(f"   Différence: {abs(cliquet_cir - cliquet_hw):,.2f} EUR")
    print()


def risk_management_analysis():
    """
    Analyse de gestion des risques : VaR, stress testing
    """
    print("=" * 80)
    print("ANALYSE DE GESTION DES RISQUES")
    print("=" * 80)

    # Portefeuille d'obligations
    portfolio = {
        "Bond_2Y": {"notional": 10000000, "maturity": 2.0},
        "Bond_5Y": {"notional": 15000000, "maturity": 5.0},
        "Bond_10Y": {"notional": 8000000, "maturity": 10.0},
    }

    # Modèles pour l'analyse
    base_cir = CIRModel(b=0.025, beta=0.3, sigma=0.12, r0=0.025)

    print("Portefeuille d'obligations:")
    total_notional = sum(bond["notional"] for bond in portfolio.values())
    for name, bond in portfolio.items():
        print(f"  {name}: {bond['notional']:,} EUR (maturité {bond['maturity']} ans)")
    print(f"Total: {total_notional:,} EUR")
    print()

    # Calcul de la valeur du portefeuille
    def portfolio_value(model, r_current):
        total_value = 0
        for bond in portfolio.values():
            bond_price = model.bond_price_analytical(0, bond["maturity"], r_current)
            total_value += bond["notional"] * bond_price
        return total_value

    # Valeur actuelle
    current_value = portfolio_value(base_cir, 0.025)
    print(f"Valeur actuelle du portefeuille: {current_value:,.2f} EUR")
    print()

    # Stress testing - chocs de taux parallèles
    rate_shocks = np.linspace(-0.02, 0.02, 21)  # Chocs de -200bp à +200bp
    portfolio_values = []

    for shock in rate_shocks:
        shocked_rate = 0.025 + shock
        if shocked_rate > 0:  # Éviter les taux négatifs pour CIR
            shocked_value = portfolio_value(base_cir, shocked_rate)
            portfolio_values.append(shocked_value)
        else:
            portfolio_values.append(np.nan)

    # Graphique de stress testing
    plt.figure(figsize=(12, 8))
    pnl = [(val - current_value) / 1000000 for val in portfolio_values]  # En millions
    plt.plot(rate_shocks * 10000, pnl, "b-o", linewidth=2, markersize=4)
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.7)
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.7)
    plt.xlabel("Choc de taux (points de base)")
    plt.ylabel("P&L (millions EUR)")
    plt.title("Stress Testing - Sensibilité du portefeuille aux chocs de taux")
    plt.grid(True, alpha=0.3)
    plt.savefig("figures/stress_testing.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Calcul de la duration modifiée (approximation numérique)
    rate_base = 0.025
    rate_up = 0.026
    rate_down = 0.024

    value_up = portfolio_value(base_cir, rate_up)
    value_down = portfolio_value(base_cir, rate_down)

    modified_duration = -(value_up - value_down) / (2 * current_value * 0.001)

    print(f"Duration modifiée du portefeuille: {modified_duration:.2f} années")
    print(
        f"Sensibilité à +100bp: {(value_up - current_value)/1000000:.2f} millions EUR"
    )
    print(
        f"Sensibilité à -100bp: {(value_down - current_value)/1000000:.2f} millions EUR"
    )
    print()

    # VaR Monte Carlo
    print("Calcul de la VaR par Monte Carlo (horizon 1 mois):")
    T_risk = 1 / 12  # 1 mois
    n_sims_var = 10000

    # Simulation des taux futurs
    future_rates = []
    current_rate = 0.025

    for _ in range(n_sims_var):
        path = base_cir.simulate_euler(T_risk, 20, 1)
        future_rates.append(path[0, -1])

    # Calcul des valeurs de portefeuille futures
    future_values = [portfolio_value(base_cir, r) for r in future_rates]
    pnl_distribution = [(val - current_value) for val in future_values]

    # VaR à différents niveaux de confiance
    confidence_levels = [0.95, 0.99, 0.999]

    for conf in confidence_levels:
        var = -np.percentile(pnl_distribution, (1 - conf) * 100)
        print(f"  VaR {conf*100:.1f}%: {var/1000000:.2f} millions EUR")

    # Expected Shortfall (CVaR)
    var_99 = -np.percentile(pnl_distribution, 1)
    worst_losses = [pnl for pnl in pnl_distribution if pnl <= -var_99]
    expected_shortfall = -np.mean(worst_losses)

    print(f"  Expected Shortfall (99%): {expected_shortfall/1000000:.2f} millions EUR")
    print()

    # Histogramme de la distribution P&L
    plt.figure(figsize=(12, 8))
    plt.hist(
        [pnl / 1000000 for pnl in pnl_distribution],
        bins=50,
        alpha=0.7,
        density=True,
        color="skyblue",
        edgecolor="black",
    )
    plt.axvline(
        -var_99 / 1000000,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"VaR 99%: {var_99/1000000:.2f}M EUR",
    )
    plt.axvline(
        -expected_shortfall / 1000000,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"ES 99%: {expected_shortfall/1000000:.2f}M EUR",
    )
    plt.xlabel("P&L (millions EUR)")
    plt.ylabel("Densité")
    plt.title("Distribution du P&L du portefeuille (horizon 1 mois)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        "figures/var_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def model_validation():
    """
    Validation et tests de robustesse des modèles
    """
    print("=" * 80)
    print("VALIDATION ET TESTS DE ROBUSTESSE")
    print("=" * 80)

    # Test de la convergence Monte Carlo
    print("1. Test de convergence Monte Carlo")

    n_sims_list = [1000, 2000, 5000, 10000, 20000, 50000]
    T = 1.0
    strike = 0.03

    cir_model = CIRModel(b=0.025, beta=0.3, sigma=0.12, r0=0.025)
    pricer = MonteCarloDerivativesPricer(cir_model)

    prices = []
    errors = []

    for n_sims in n_sims_list:
        result = pricer.price_call_on_rate(T, strike, n_sims)
        prices.append(result["price"])
        errors.append(result["std_error"])
        print(
            f"   {n_sims:6d} sims: Prix = {result['price']:.6f} ± {result['std_error']:.6f}"
        )

    print()

    # Test de stabilité des paramètres
    print("2. Test de stabilité des paramètres CIR")

    base_params = {"b": 0.025, "beta": 0.3, "sigma": 0.12}
    param_variations = {
        "b": [0.02, 0.025, 0.03],
        "beta": [0.2, 0.3, 0.4],
        "sigma": [0.1, 0.12, 0.15],
    }

    base_model = CIRModel(**base_params, r0=0.025)
    base_price = base_model.bond_price_analytical(0, 5, 0.025)

    print(f"   Prix de base (obligation 5 ans): {base_price:.6f}")

    for param_name, values in param_variations.items():
        print(f"   Sensibilité à {param_name}:")
        for value in values:
            test_params = base_params.copy()
            test_params[param_name] = value
            test_model = CIRModel(**test_params, r0=0.025)
            test_price = test_model.bond_price_analytical(0, 5, 0.025)
            diff = (test_price - base_price) / base_price * 100
            print(f"     {param_name}={value}: {test_price:.6f} ({diff:+.2f}%)")

    print()

    # Test de la condition de Feller
    print("3. Test de violation de la condition de Feller")

    feller_violation_model = CIRModel(b=0.01, beta=0.3, sigma=0.20, r0=0.025)
    print(f"   Condition respectée: {feller_violation_model.feller_condition}")

    # Simulation pour voir l'impact
    paths_normal = base_model.simulate_euler(1.0, 100, 1000)
    paths_violation = feller_violation_model.simulate_euler(1.0, 100, 1000)

    negative_rates_normal = np.sum(paths_normal < 0)
    negative_rates_violation = np.sum(paths_violation < 0)

    print(f"   Taux négatifs (Feller OK): {negative_rates_normal}")
    print(f"   Taux négatifs (Feller violée): {negative_rates_violation}")
    print()


def main():
    """
    Exécution des analyses pratiques avancées
    """
    print("🎯 ANALYSES PRATIQUES AVANCÉES")
    print("Modèles CIR et Hull-White - Applications en Finance Quantitative")
    print("=" * 80)

    np.random.seed(42)  # Reproductibilité

    try:
        # 1. Calibration
        calibrated_model = calibrate_cir_to_yield_curve()

        # 2. Produits structurés
        price_structured_products()

        # 3. Gestion des risques
        risk_management_analysis()

        # 4. Validation des modèles
        model_validation()

        print("=" * 80)
        print("✅ TOUTES LES ANALYSES PRATIQUES TERMINÉES")
        print("📈 Applications réalisées:")
        print("   • Calibration sur courbe de taux")
        print("   • Pricing de produits structurés")
        print("   • Gestion des risques (VaR, stress testing)")
        print("   • Validation et tests de robustesse")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
