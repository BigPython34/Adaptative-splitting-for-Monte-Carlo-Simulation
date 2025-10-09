# finance_lib/calibration.py
import numpy as np
from scipy.optimize import minimize
from .models import CIRModel, HullWhiteModel

def calibrate_cir_model(market_maturities, market_yields, r0_proxy):
    """Calibre les paramètres du modèle CIR sur une courbe de taux."""
    print("\n--- Calibration du modèle CIR ---")
    def objective_function(params):
        b, beta, sigma = params
        if b <= 0 or beta <= 0 or sigma <= 0.01: return 1e9
        model = CIRModel(b, beta, sigma, r0_proxy)
        model_yields = model.yield_curve(0, market_maturities, r0_proxy)
        return np.mean((model_yields - market_yields) ** 2)

    initial_params = [0.03, 0.3, 0.1]
    bounds = [(1e-3, 0.2), (1e-2, 1.5), (1e-2, 0.5)]
    result = minimize(objective_function, initial_params, method='L-BFGS-B', bounds=bounds)
    
    b_opt, beta_opt, sigma_opt = result.x
    calibrated_model = CIRModel(b_opt, beta_opt, sigma_opt, r0_proxy)
    print(f"Paramètres CIR optimaux: b={b_opt:.4f}, β={beta_opt:.4f}, σ={sigma_opt:.4f}")
    return calibrated_model

def calibrate_hw_model(market_maturities, market_yields, r0_proxy):
    """Calibre les paramètres du modèle Hull-White (drift constant) sur une courbe de taux."""
    print("\n--- Calibration du modèle Hull-White ---")
    def objective_function(params):
        b_const, beta, sigma = params
        if beta <= 0 or sigma <= 1e-3: return 1e9
        model = HullWhiteModel(beta, sigma, r0_proxy, b_function=lambda t: b_const)
        model_yields = model.yield_curve(0, market_maturities, r0_proxy)
        return np.mean((model_yields - market_yields) ** 2)

    initial_params = [0.03, 0.2, 0.01]
    bounds = [(-0.1, 0.2), (1e-2, 1.5), (1e-3, 0.1)]
    result = minimize(objective_function, initial_params, method='L-BFGS-B', bounds=bounds)
    
    b_opt, beta_opt, sigma_opt = result.x
    calibrated_model = HullWhiteModel(beta_opt, sigma_opt, r0_proxy, lambda t: b_opt)
    print(f"Paramètres HW optimaux: b={b_opt:.4f}, β={beta_opt:.4f}, σ={sigma_opt:.4f}")
    return calibrated_model

def calibrate_hw_model_flexible(market_maturities, market_yields, r0_proxy):
    """
    Calibre un modèle Hull-White avec un drift b(t) constant par morceaux.
    Ceci donne au modèle une bien plus grande flexibilité pour s'ajuster à la courbe.
    """
    print("\n--- Calibration du modèle Hull-White FLEXIBLE (drift par morceaux) ---")

    # On définit les "nœuds" temporels où b(t) peut changer de valeur
    time_knots = [2.0, 10.0] # Changement à 2 ans et 10 ans

    def objective_function(params):
        # Les paramètres sont maintenant : b1, b2, b3, beta, sigma
        b1, b2, b3, beta, sigma = params
        if beta <= 0 or sigma <= 1e-3: return 1e9

        # On crée la fonction b(t) constante par morceaux
        def b_func_piecewise(t):
            if t <= time_knots[0]:
                return b1
            elif t <= time_knots[1]:
                return b2
            else:
                return b3
        
        model = HullWhiteModel(beta, sigma, r0_proxy, b_function=b_func_piecewise)
        model_yields = model.yield_curve(0, market_maturities, r0_proxy)
        
        # On pénalise plus les erreurs sur les taux courts, qui sont plus importants
        weights = np.exp(-0.1 * market_maturities)
        
        return np.mean(weights * (model_yields - market_yields) ** 2)

    # 5 paramètres à optimiser
    initial_params = [0.03, 0.03, 0.03, 0.2, 0.01]
    bounds = [(-0.1, 0.2), (-0.1, 0.2), (-0.1, 0.2), (1e-2, 1.5), (1e-3, 0.1)]

    result = minimize(objective_function, initial_params, method='L-BFGS-B', bounds=bounds)
    
    b1_opt, b2_opt, b3_opt, beta_opt, sigma_opt = result.x
    
    def final_b_func(t):
        if t <= time_knots[0]: return b1_opt
        elif t <= time_knots[1]: return b2_opt
        else: return b3_opt
        
    calibrated_model = HullWhiteModel(beta_opt, sigma_opt, r0_proxy, final_b_func)
    
    print("Paramètres HW Flexibles optimaux:")
    print(f"  b1 (0-{time_knots[0]}a) = {b1_opt:.4f}")
    print(f"  b2 ({time_knots[0]}-{time_knots[1]}a) = {b2_opt:.4f}")
    print(f"  b3 (>{time_knots[1]}a) = {b3_opt:.4f}")
    print(f"  β = {beta_opt:.4f}, σ = {sigma_opt:.4f}")
    
    return calibrated_model