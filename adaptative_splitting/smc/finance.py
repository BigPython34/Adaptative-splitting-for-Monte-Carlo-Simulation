import numpy as np
from scipy.stats import norm

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