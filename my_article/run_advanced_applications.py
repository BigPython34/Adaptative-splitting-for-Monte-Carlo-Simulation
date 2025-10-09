#!/usr/bin/env python3
"""
Script de démonstration pour des applications financières avancées.

Ce script :
1. Calibre le modèle CIR sur des données de marché de référence.
2. Utilise le modèle calibré pour pricer un produit structuré (Range Accrual Note).
3. Mène une analyse de risque (Stress Test, VaR, ES) sur un portefeuille d'obligations.
"""

import numpy as np
from finance_lib.models import CIRModel
from finance_lib.calibration import calibrate_cir_model
from finance_lib.pricing_engine import MonteCarloDerivativesPricer
from finance_lib.risk_analysis import run_stress_test_analysis, run_monte_carlo_var_analysis

def main():
    """Orchestre les applications avancées."""
    print("--- Lancement du scénario d'applications avancées ---")

    # --- 1. Calibration du modèle sur des données de marché de référence ---
    # Pour la reproductibilité, nous utilisons des données fixes ici.
    market_mats = np.array([0.5, 1, 2, 5, 10, 20, 30])
    market_yields = np.array([0.022, 0.024, 0.026, 0.032, 0.035, 0.038, 0.040])
    r0 = 0.024 # Proxy basé sur le taux à 1 an

    calibrated_cir = calibrate_cir_model(market_mats, market_yields, r0)

    # --- 2. Pricing de Produits Structurés avec le modèle calibré ---
    pricer = MonteCarloDerivativesPricer(calibrated_cir)
    print("\n--- Pricing d'une Range Accrual Note ---")
    
    price = pricer.price_range_accrual_note(
        T=2.0, r_min=0.015, r_max=0.045, notional=1_000_000, 
        n_observations=24, n_sims=50000
    )
    print(f"Prix estimé de la note : {price:,.2f} EUR")

    # --- 3. Analyse de Risque d'un Portefeuille d'Obligations ---
    portfolio = {
        "Obligation_Court_Terme_2A":  {"notional": 5_000_000, "maturity": 2.0},
        "Obligation_Moyen_Terme_5A":  {"notional": 10_000_000, "maturity": 5.0},
        "Obligation_Long_Terme_10A": {"notional": 8_000_000, "maturity": 10.0},
    }

    # L'analyse est menée avec le modèle qui reflète le marché (le modèle calibré)
    run_stress_test_analysis(portfolio, calibrated_cir)
    run_monte_carlo_var_analysis(portfolio, calibrated_cir)
    
    print("\n--- Analyse avancée terminée. ---")

if __name__ == "__main__":
    main()