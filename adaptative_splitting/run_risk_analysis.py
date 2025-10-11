#!/usr/bin/env python3
"""
Application du Monte-Carlo Séquentiel (SMC) à l'analyse de risque financier.
"""

import numpy as np
from scipy.stats import t as student_t, norm
from typing import Callable, Optional, List, Dict

# --- Boîte à Outils Financière (Robuste) ---

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Calcule le prix d'une option européenne par Black-Scholes de manière robuste."""
    S = np.asarray(S)
    price = np.zeros_like(S, dtype=float)
    valid_mask = S > 1e-9 # On ne calcule que pour les prix strictement positifs

    if np.any(valid_mask):
        S_valid = S[valid_mask]
        
        # Pour éviter les problèmes si T=0 (à l'expiration)
        if T < 1e-9:
            if option_type == 'call':
                price[valid_mask] = np.maximum(0, S_valid - K)
            else:
                price[valid_mask] = np.maximum(0, K - S_valid)
            return price

        d1 = (np.log(S_valid / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            call_prices = (S_valid * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
            price[valid_mask] = call_prices
        else: # put
            put_prices = (K * np.exp(-r * T) * norm.cdf(-d2) - S_valid * norm.cdf(-d1))
            price[valid_mask] = put_prices
            
    return price

# --- Définition du Problème : Portefeuille et Perte ---

# Paramètres du portefeuille et du marché
S0 = 100.0
r = 0.01
sigma = 0.30
T_option = 0.5
K_option = 105
horizon = 10/252

POS_STOCK = 1000
POS_OPTION = -5000

V0_stock = POS_STOCK * S0
V0_option = POS_OPTION * black_scholes_price(S0, K_option, T_option, r, sigma)
V0 = V0_stock + V0_option

def portfolio_loss_function(X: np.ndarray) -> np.ndarray:
    """Fonction de score Phi(X) qui calcule la perte du portefeuille."""
    S_final = S0 * np.maximum(0, 1 + X) # Un prix d'action ne peut pas être négatif
    
    V_final_stock = POS_STOCK * S_final
    V_final_option = POS_OPTION * black_scholes_price(S_final, K_option, T_option - horizon, r, sigma)
    V_final = V_final_stock + V_final_option
    
    loss = -(V_final - V0)
    return loss

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    S = np.asarray(S)
    price = np.zeros_like(S, dtype=float)
    valid_mask = S > 1e-9
    if np.any(valid_mask):
        S_valid = S[valid_mask]
        if T < 1e-9:
            price[valid_mask] = np.maximum(0, S_valid - K if option_type == 'call' else K - S_valid)
            return price
        d1 = (np.log(S_valid / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price[valid_mask] = (S_valid * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            price[valid_mask] = (K * np.exp(-r * T) * norm.cdf(-d2) - S_valid * norm.cdf(-d1))
    return price

S0, r, sigma, T_option, K_option, horizon = 100.0, 0.01, 0.30, 0.5, 105, 10/252
POS_STOCK, POS_OPTION = 1000, -5000
V0_stock = POS_STOCK * S0
V0_option = POS_OPTION * black_scholes_price(S0, K_option, T_option, r, sigma)
V0 = V0_stock + V0_option

def portfolio_loss_function(X: np.ndarray) -> np.ndarray:
    S_final = S0 * np.maximum(0, 1 + X)
    V_final_stock = POS_STOCK * S_final
    V_final_option = POS_OPTION * black_scholes_price(S_final, K_option, T_option - horizon, r, sigma)
    V_final = V_final_stock + V_final_option
    return -(V_final - V0)

class RiskSMCResult:
    def __init__(self, prob_est: float, thresholds: List[float], final_particles_scores: np.ndarray):
        self.prob_est, self.thresholds, self.final_particles_scores = prob_est, thresholds, final_particles_scores

def adaptive_smc_for_risk(
    N: int, p0: float, phi_function: Callable, initial_sampler: Callable,
    rejuvenation_ratio: float = 0.1 # NOUVEAU: ratio de particules à régénérer
) -> Optional[RiskSMCResult]:
    """Version de l'algorithme SMC avec une étape de régénération."""
    particles = initial_sampler(N)
    prob_est = 1.0
    thresholds = [0.0]
    
    max_iter = 100
    for k in range(max_iter):
        scores = phi_function(particles)
        valid_scores = scores[~np.isnan(scores)]
        if len(valid_scores) == 0: return None

        L_next = np.percentile(valid_scores, (1 - p0) * 100)
        
        if L_next <= thresholds[-1] + 1.0: # Seuil en euros, tolérance de 1€
            # On arrête si on stagne DANS UNE ZONE DE PERTE RAISONNABLE
            if L_next > 1000: # Si on a déjà atteint des pertes significatives
                print(f"Convergence des seuils à l'itération {k}. Arrêt.")
                break
            
        thresholds.append(L_next)
        
        survivors_mask = (scores >= L_next) & (~np.isnan(scores))
        num_survivors = np.sum(survivors_mask)
        
        if num_survivors < 2: # Besoin d'au moins 2 survivants pour calculer un std
            print(f"Moins de 2 survivants à l'itération {k}. Arrêt.")
            break
        
        prob_est *= num_survivors / len(valid_scores)
        survivors = particles[survivors_mask]
        
        # Rééchantillonnage
        indices = np.random.choice(len(survivors), size=N, replace=True)
        particles = survivors[indices]
        
        # --- CORRECTION : Étape de Mutation + Régénération ---
        
        # 1. Régénération : on remplace une partie des particules par des neuves
        num_rejuvenate = int(N * rejuvenation_ratio)
        particles[:num_rejuvenate] = initial_sampler(num_rejuvenate)
        
        # 2. Mutation : on fait muter le reste des particules
        num_mutate = N - num_rejuvenate
        mutation_std = np.std(survivors) * 0.2 # Un peu plus de bruit
        particles[num_rejuvenate:] += np.random.normal(0, mutation_std, size=num_mutate)
        # --------------------------------------------------------

    final_scores = phi_function(particles)
    return RiskSMCResult(prob_est, thresholds, final_scores[~np.isnan(final_scores)])


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

    result = adaptive_smc_for_risk(N, p0, portfolio_loss_function, student_sampler)

    if result is None or len(result.thresholds) <= 1:
        print("\nL'analyse SMC a échoué ou n'a pas pu progresser.")
        return

    print("\n--- Résultats de l'Analyse de Risque ---")
    
    final_prob = result.prob_est
    VaR_estimate = result.thresholds[-1]
    
    # On choisit le niveau de confiance le plus pertinent trouvé
    # Par exemple, si p0=0.1 et on a fait 5 itérations, p_final = 0.1^5 = 1e-5
    # Donc alpha = 1 - 1e-5 = 99.999%
    # On peut chercher le seuil qui correspond à un alpha plus standard
    target_alpha = 0.999 # VaR à 99.9%
    target_prob = 1 - target_alpha
    
    # On cherche le seuil L_k pour lequel P(Loss > L_k) est le plus proche de target_prob
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

    ES_estimate = np.mean(result.final_particles_scores)
    
    print(f"\nExpected Shortfall (ES):")
    print(f"L'estimation de la perte moyenne dans la queue de distribution extrême est :")
    print(f"  ES ≈ {ES_estimate:,.2f} EUR")
    print(f"\nNote: L'ES est calculé sur des scénarios de perte encore plus rares que la VaR à {target_alpha:.1%},")
    print(f"donc il est normal que ES > VaR.")

if __name__ == "__main__":
    main()