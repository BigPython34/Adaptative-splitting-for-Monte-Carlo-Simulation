#!/usr/bin/env python3
"""
Application du Monte-Carlo Séquentiel (SMC) à l'analyse de risque financier.
"""
from smc.smc_algorithms import adaptive_smc_run, create_rejuvenation_propagation_step
import numpy as np
from scipy.stats import t as student_t
from smc.finance import *


# ... (le reste du script, la fonction main, est identique) ...
def main():
    print("="*60)
    print("=== Analyse de Risque par Monte-Carlo Séquentiel (SMC) ===")
    print("="*60)
    print(f"Portefeuille initial: {V0:,.2f} EUR")
    
    N, p0 = 50000, 0.1
    student_df = 4
    
    def student_sampler(num_samples: int) -> np.ndarray:
        vol_horizon = sigma * np.sqrt(horizon)
        scale_adjustment = np.sqrt((student_df - 2) / student_df)
        return student_t.rvs(df=student_df, size=num_samples) * vol_horizon * scale_adjustment

    propagation_strategy = create_rejuvenation_propagation_step(
        initial_sampler=student_sampler,
        rejuvenation_ratio=0.1,
        mutation_std_ratio=0.2
    )

    # 2. LANCER LE MOTEUR GÉNÉRIQUE
    result = adaptive_smc_run(
        N=50000,
        p0=0.1,
        phi_function=portfolio_loss_function,
        initial_sampler=student_sampler,
        propagation_step=propagation_strategy
    )

    if result is None or len(result.thresholds) <= 1:
        print("\nL'analyse SMC a échoué ou n'a pas pu progresser.")
        return

    print("\n--- Résultats de l'Analyse de Risque ---")
    
    
    target_alpha = 0.999 # VaR à 99.9%
    target_prob = 1 - target_alpha
    
    achieved_probs = [p0**k for k in range(1, len(result.thresholds))]
    if not achieved_probs:
        print("Aucun niveau de probabilité n'a pu être atteint.")
        return

    closest_prob_idx = np.abs(np.array(achieved_probs) - target_prob).argmin()
    
    VaR_999 = result.thresholds[closest_prob_idx + 1]
    prob_at_VaR_999 = achieved_probs[closest_prob_idx]

    print(f"\nValue-at-Risk (VaR):")
    print(f"Estimation la plus proche pour une VaR à {target_alpha:.1%}:")
    print(f"  P(Perte > {VaR_999:,.2f} EUR) ≈ {prob_at_VaR_999:.2e}")
    print(f"  -> VaR à {target_alpha:.1%} ≈ {VaR_999:,.2f} EUR")


if __name__ == "__main__":
    main()