"""
Applications avancées des modèles de taux courts stochastiques :
calibration, pricing de produits structurés et gestion des risques.

Ce script utilise les classes de modèles définies dans
'stochastic_interest_rate_models.py'.

Auteur: Octave Cerclé - ISAE-Supaero Finance Project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Import des classes de modèles et du pricer depuis le premier script
from stochastic_interest_rate_models import CIRModel, HullWhiteModel, BaseShortRateModel

# Configuration des graphiques (cohérente avec le script principal)
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 12


def calibrate_cir_model(market_maturities: np.ndarray, market_yields: np.ndarray, r0_proxy: float):
    """
    Calibre les paramètres (b, beta, sigma) du modèle CIR pour correspondre
    à une courbe de taux de marché observée via la minimisation de l'erreur quadratique.

    Returns:
        Un tuple: (modèle CIR calibré, rendements du modèle sur la courbe de marché).
    """
    print("-" * 80)
    print("1. Calibration du modèle Cox-Ingersoll-Ross (CIR)")
    print("-" * 80)

    def objective_function(params: np.ndarray) -> float:
        b, beta, sigma = params
        if b <= 0 or beta <= 0 or sigma <= 0.01:
            return 1e9

        model = CIRModel(b=b, beta=beta, sigma=sigma, r0=r0_proxy)
        model_yields = model.yield_curve(t=0, maturities=market_maturities, r_at_t=r0_proxy)
        
        # Erreur quadratique moyenne (MSE)
        mse = np.mean((model_yields - market_yields) ** 2)
        return mse * 1e5

    initial_params = np.array([0.03, 0.3, 0.1])
    bounds = [(0.001, 0.2), (0.01, 1.5), (0.01, 0.5)]

    result = minimize(objective_function, initial_params, method='L-BFGS-B', bounds=bounds)
    
    b_opt, beta_opt, sigma_opt = result.x
    calibrated_model = CIRModel(b=b_opt, beta=beta_opt, sigma=sigma_opt, r0=r0_proxy)

    print("\n--- Résultats de la Calibration CIR ---")
    print(f"  Paramètres optimaux: b={b_opt:.6f}, β={beta_opt:.6f}, σ={sigma_opt:.6f}")
    print(f"  Erreur finale: {result.fun:.6f}")
    print(f"  Condition de Feller respectée: {calibrated_model.feller_condition_met}")
    
    
    return calibrated_model


def calibrate_hw_model(market_maturities: np.ndarray, market_yields: np.ndarray, r0_proxy: float):
    """
    Calibre les paramètres (b, beta, sigma) du modèle Hull-White (avec un drift b constant)
    pour correspondre à une courbe de taux de marché observée.

    Returns:
        Un tuple: (modèle Hull-White calibré, rendements du modèle sur la courbe de marché).
    """
    print("-" * 80)
    print("2. Calibration du modèle Hull-White (drift constant)")
    print("-" * 80)

    def objective_function(params: np.ndarray) -> float:
        b_const, beta, sigma = params
        # Pour Hull-White, b peut être négatif, mais beta et sigma doivent être positifs
        if beta <= 0 or sigma <= 0.001:
            return 1e9

        # On utilise une fonction b(t) constante pour la calibration
        model = HullWhiteModel(beta=beta, sigma=sigma, r0=r0_proxy, b_function=lambda t: b_const)
        model_yields = model.yield_curve(t=0, maturities=market_maturities, r_at_t=r0_proxy)
        
        mse = np.mean((model_yields - market_yields) ** 2)
        return mse * 1e5

    # Les paramètres de Hull-White peuvent avoir des ordres de grandeur différents
    initial_params = np.array([0.03, 0.2, 0.01])
    bounds = [(-0.1, 0.2), (0.01, 1.5), (0.001, 0.1)] # b, beta, sigma

    result = minimize(objective_function, initial_params, method='L-BFGS-B', bounds=bounds)
    
    b_opt, beta_opt, sigma_opt = result.x
    calibrated_model = HullWhiteModel(beta=beta_opt, sigma=sigma_opt, r0=r0_proxy, b_function=lambda t: b_opt)

    print("\n--- Résultats de la Calibration Hull-White ---")
    print(f"  Paramètres optimaux: b_const={b_opt:.6f}, β={beta_opt:.6f}, σ={sigma_opt:.6f}")
    print(f"  Erreur finale: {result.fun:.6f}")
    
    return calibrated_model


def price_range_accrual_note(model: BaseShortRateModel, T: float, r_min: float, r_max: float,
                               notional: float, n_observations: int, n_sims: int) -> float:
    """
    Prix d'une 'Range Accrual Note' par simulation Monte Carlo.
    Le payoff est proportionnel au nombre d'observations où le taux court
    reste dans un intervalle [r_min, r_max].
    """
    print("\n--- Pricing d'une Range Accrual Note ---")
    print(f"Modèle: {model.__class__.__name__}, Maturité: {T} ans, Borne: [{r_min*100:.2f}%, {r_max*100:.2f}%]")

    paths = model.simulate_euler(T, n_observations, n_sims)
    
    # Exclure t=0 pour les observations
    observed_paths = paths[:, 1:]
    
    # Compter le nombre de fois où le taux est dans l'intervalle pour chaque trajectoire
    in_range_counts = np.sum((observed_paths >= r_min) & (observed_paths <= r_max), axis=1)
    
    # Calculer le payoff pour chaque trajectoire
    payoffs = notional * (in_range_counts / n_observations)
    
    # Calculer le facteur d'actualisation pour chaque trajectoire
    dt = T / n_observations
    discount_factors = np.exp(-np.sum(paths[:, :-1] * dt, axis=1))
    
    # Calculer la moyenne des payoffs actualisés
    price = np.mean(payoffs * discount_factors)
    
    print(f"Prix estimé : {price:,.2f} EUR")
    return price


def analyze_portfolio_risk(portfolio: dict, model: BaseShortRateModel):
    """
    Effectue une analyse de risque sur un portefeuille d'obligations zéro-coupon.
    1. Stress Testing : évalue l'impact de chocs de taux parallèles.
    2. VaR Monte Carlo : estime la perte maximale potentielle sur un horizon de temps donné.
    """
    print("-" * 80)
    print(f"3. Analyse de Risque du Portefeuille (Modèle: {model.__class__.__name__})")
    print("-" * 80)

    def get_portfolio_value(r_current: float, current_model: BaseShortRateModel) -> float:
        """Calcule la valeur totale du portefeuille pour un taux court donné."""
        total_value = 0
        for bond in portfolio.values():
            price = current_model.bond_price_analytical(t=0, T=bond['maturity'], r_at_t=r_current)
            total_value += bond['notional'] * price
        return total_value

    initial_rate = model.r0
    current_value = get_portfolio_value(initial_rate, model)
    print(f"Valeur actuelle du portefeuille: {current_value:,.2f} EUR (pour r(0) = {initial_rate*100:.2f}%)")

    # --- 1. Stress Testing ---
    print("\n--- Analyse de sensibilité (Stress Testing) ---")
    rate_shocks_bps = np.linspace(-150, 150, 31) # Chocs de -1.5% à +1.5%
    pnl_values = []
    
    for shock_bps in rate_shocks_bps:
        shocked_rate = initial_rate + shock_bps / 10000
        # Pour CIR, on ne teste que des taux positifs
        if isinstance(model, CIRModel) and shocked_rate <= 0:
            pnl_values.append(np.nan)
        else:
            shocked_value = get_portfolio_value(shocked_rate, model)
            pnl_values.append(shocked_value - current_value)

    plt.figure()
    plt.plot(rate_shocks_bps, np.array(pnl_values) / 1e6, 'o-', label="P&L du portefeuille")
    plt.title("Stress Testing : Sensibilité du Portefeuille aux Chocs de Taux")
    plt.xlabel("Choc de Taux Parallèle (points de base)")
    plt.ylabel("Profits et Pertes (en millions d'EUR)")
    plt.axhline(0, color='black', linestyle='--', lw=1)
    plt.axvline(0, color='black', linestyle='--', lw=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("figures/portfolio_stress_testing.png")
    plt.show()

    # --- 2. Value-at-Risk (VaR) et Expected Shortfall (ES) par Monte Carlo ---
    print("\n--- Calcul de la Value-at-Risk (VaR) et Expected Shortfall (ES) ---")
    risk_horizon_T = 1/12  # Horizon de risque de 1 mois
    n_sims_var = 20000     # Nombre de simulations pour la VaR
    
    # Simuler les taux à l'horizon de risque
    simulated_rates_at_horizon = model.simulate_euler(risk_horizon_T, n_steps=20, n_paths=n_sims_var)[:, -1]
    
    # Réévaluer le portefeuille pour chaque taux simulé
    future_portfolio_values = [get_portfolio_value(r, model) for r in simulated_rates_at_horizon]
    
    # Calculer la distribution des profits et pertes (P&L)
    pnl_distribution = np.array(future_portfolio_values) - current_value
    
    # Calculer la VaR et l'ES
    confidence_levels = [0.95, 0.99]
    print(f"Horizon de risque: {risk_horizon_T*12:.0f} mois, Simulations: {n_sims_var}")
    for conf in confidence_levels:
        var = -np.percentile(pnl_distribution, (1 - conf) * 100)
        # L'Expected Shortfall (ES) est la moyenne des pertes qui dépassent la VaR
        losses_beyond_var = pnl_distribution[pnl_distribution < -var]
        es = -np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else var
        
        print(f"  VaR à {conf*100:.0f}%: {var:,.2f} EUR")
        print(f"  ES  à {conf*100:.0f}%: {es:,.2f} EUR")

    # Visualisation de la distribution P&L
    var_99 = -np.percentile(pnl_distribution, 1)
    plt.figure()
    plt.hist(pnl_distribution / 1e6, bins=50, density=True, alpha=0.7, label="Distribution P&L")
    plt.axvline(-var_99 / 1e6, color='red', linestyle='--', lw=2, label=f'VaR 99% = {-var_99/1e6:.2f} M EUR')
    plt.title(f"Distribution P&L du Portefeuille (Horizon {risk_horizon_T*12:.0f} mois)")
    plt.xlabel("Profits et Pertes (en millions d'EUR)")
    plt.ylabel("Densité de probabilité")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("figures/portfolio_pnl_distribution_var.png")
    plt.show()


def main_advanced_applications():
    """
    Fonction principale pour exécuter les analyses avancées.
    """
    np.random.seed(42) # Assurer la reproductibilité

    # --- Données de marché pour la calibration (exemple) ---
    market_maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 20, 30])
    market_yields = np.array([0.022, 0.024, 0.026, 0.028, 0.032, 0.035, 0.038, 0.040])
    r0_proxy = market_yields[2] # On utilise le taux à 1 an comme proxy pour r(0)

    # 1. Calibration du modèle CIR
    calibrated_cir_model = calibrate_cir_model(market_maturities, market_yields, r0_proxy)

    # --- Pricing de produits structurés ---
    print("\n" + "-" * 80)
    print("2. Pricing de Produits Structurés par Monte Carlo")
    print("-" * 80)
    
    # Création d'un modèle Hull-White pour la comparaison
    hw_model_for_pricing = HullWhiteModel(beta=0.2, sigma=0.01, r0=r0_proxy)
    
    price_range_accrual_note(model=calibrated_cir_model, T=2.0, r_min=0.01, r_max=0.04,
                               notional=1_000_000, n_observations=24, n_sims=20000)
    price_range_accrual_note(model=hw_model_for_pricing, T=2.0, r_min=0.01, r_max=0.04,
                               notional=1_000_000, n_observations=24, n_sims=20000)

    # --- Analyse de risque de portefeuille ---
    # Définition d'un portefeuille d'obligations zéro-coupon
    bond_portfolio = {
        "Bond_2Y":  {"notional": 5_000_000, "maturity": 2.0},
        "Bond_5Y":  {"notional": 10_000_000, "maturity": 5.0},
        "Bond_10Y": {"notional": 8_000_000, "maturity": 10.0},
    }
    
    # L'analyse de risque est menée avec le modèle calibré sur le marché
    analyze_portfolio_risk(bond_portfolio, calibrated_cir_model)
    
    print("\n" + "=" * 80)
    print("Analyses avancées terminées.")
    print("=" * 80)


if __name__ == "__main__":
    main_advanced_applications()