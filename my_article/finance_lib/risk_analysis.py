# finance_lib/risk_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from .models import BaseShortRateModel

def _get_portfolio_value(portfolio: dict, r_current: float, model: BaseShortRateModel) -> float:
    """Fonction utilitaire pour calculer la valeur d'un portefeuille d'obligations."""
    total_value = 0
    for bond in portfolio.values():
        price = model.bond_price_analytical(t=0, T=bond['maturity'], r_at_t=r_current)
        total_value += bond['notional'] * price
    return total_value

def run_stress_test_analysis(portfolio: dict, model: BaseShortRateModel):
    """
    Évalue l'impact de chocs de taux parallèles sur la valeur du portefeuille.
    """
    print("\n--- Analyse de sensibilité (Stress Testing) ---")
    initial_rate = model.r0
    current_value = _get_portfolio_value(portfolio, initial_rate, model)
    print(f"Valeur actuelle du portefeuille: {current_value:,.2f} EUR (pour r(0) = {initial_rate*100:.2f}%)")
    
    rate_shocks_bps = np.linspace(-150, 150, 31)
    pnl_values = []
    
    for shock_bps in rate_shocks_bps:
        shocked_rate = initial_rate + shock_bps / 10000
        # Gérer la contrainte de positivité pour CIR
        if isinstance(model, type(model)) and hasattr(model, '_enforce_constraints') and shocked_rate <= 0:
             pnl_values.append(np.nan)
        else:
            shocked_value = _get_portfolio_value(portfolio, shocked_rate, model)
            pnl_values.append(shocked_value - current_value)
    
    # Visualisation du stress test (pourrait être déplacé dans visualization.py)
    plt.figure(figsize=(12, 7))
    plt.plot(rate_shocks_bps, np.array(pnl_values) / 1e6, 'o-')
    plt.title("Stress Testing : Sensibilité du Portefeuille aux Chocs de Taux")
    plt.xlabel("Choc de Taux Parallèle (points de base)")
    plt.ylabel("Profits et Pertes (en millions d'EUR)")
    plt.axhline(0, color='black', ls='--', lw=1), plt.axvline(0, color='black', ls='--', lw=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def run_monte_carlo_var_analysis(portfolio: dict, model: BaseShortRateModel):
    """
    Estime la Value-at-Risk (VaR) et l'Expected Shortfall (ES) par simulation.
    """
    print("\n--- Calcul de la Value-at-Risk (VaR) et Expected Shortfall (ES) ---")
    risk_horizon_T = 1/12  # 1 mois
    n_sims_var = 20000
    
    initial_value = _get_portfolio_value(portfolio, model.r0, model)
    
    # Simuler les taux à l'horizon de risque
    simulated_rates = model.simulate_euler(risk_horizon_T, n_steps=20, n_paths=n_sims_var)[:, -1]
    
    # Réévaluer le portefeuille pour chaque scénario de taux
    future_values = np.array([_get_portfolio_value(portfolio, r, model) for r in simulated_rates])
    pnl_distribution = future_values - initial_value
    
    # Calculer VaR et ES
    confidence_levels = [0.95, 0.99]
    print(f"Horizon: 1 mois, Simulations: {n_sims_var}")
    for conf in confidence_levels:
        var = -np.percentile(pnl_distribution, (1 - conf) * 100)
        losses_beyond_var = pnl_distribution[pnl_distribution < -var]
        es = -np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else var
        print(f"  VaR à {conf*100:.0f}%: {var:,.2f} EUR")
        print(f"  ES  à {conf*100:.0f}%: {es:,.2f} EUR")

    # Visualisation de la distribution P&L (pourrait être déplacé)
    var_99 = -np.percentile(pnl_distribution, 1)
    plt.figure(figsize=(12, 7))
    plt.hist(pnl_distribution / 1e6, bins=50, density=True, alpha=0.7)
    plt.axvline(-var_99 / 1e6, color='red', linestyle='--', lw=2, label=f'VaR 99% = {-var_99/1e6:.2f} M EUR')
    plt.title("Distribution P&L du Portefeuille (Horizon 1 mois)")
    plt.xlabel("Profits et Pertes (en millions d'EUR)"), plt.ylabel("Densité")
    plt.legend(), plt.grid(True), plt.show()