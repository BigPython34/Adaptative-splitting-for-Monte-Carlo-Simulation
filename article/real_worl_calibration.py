"""
Script pour la calibration des modèles CIR et Hull-White en utilisant
uniquement la partie courte (ex: 0-2 ans) de la courbe des taux réelle.

Ce script illustre le compromis entre un bon ajustement local (sur le court terme)
et l'incapacité du modèle à extrapoler correctement sur le long terme.
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

try:
    from advanced_applications import calibrate_cir_model, calibrate_hw_model
except ImportError:
    print("Erreur: Le fichier 'advanced_applications.py' doit être dans le même répertoire.")
    exit()

def fetch_fred_yield_curve() -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Récupère la courbe des taux des bons du Trésor américain (Constant Maturity)
    la plus récente depuis la base de données FRED.
    """
    print("--- Récupération des données de la courbe des taux depuis FRED ---")
    fred_series_map = {
        'DGS1MO': 1/12, 'DGS3MO': 3/12, 'DGS6MO': 6/12, 'DGS1': 1.0, 'DGS2': 2.0,
        
    }
    series_ids = list(fred_series_map.keys())
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)

    try:
        yield_curve_data = web.get_data_fred(series_ids, start=start_date, end=end_date)
        latest_complete_data = yield_curve_data.dropna()
        
        if latest_complete_data.empty:
            print("Erreur: Aucune donnée complète trouvée sur la période récente.")
            return None, None
            
        latest_yields_series = latest_complete_data.iloc[-1]
        report_date = latest_yields_series.name.strftime('%Y-%m-%d')
        print(f"Données de la courbe des taux récupérées pour la date : {report_date}")
        
        maturities = np.array([fred_series_map[col] for col in latest_yields_series.index])
        yields = latest_yields_series.values / 100.0
        
        sort_order = np.argsort(maturities)
        maturities, yields = maturities[sort_order], yields[sort_order]
        
        return maturities, yields
    except Exception as e:
        print(f"Une erreur est survenue lors de la récupération des données : {e}")
        return None, None

def main_short_term_calibration():
    """
    Fonction principale pour exécuter la calibration sur la partie courte de la courbe.
    """
    # 1. Récupérer la courbe des taux COMPLÈTE
    full_market_maturities, full_market_yields = fetch_fred_yield_curve()
    
    if full_market_maturities is None:
        return

    # <<< MODIFICATION PRINCIPALE : Filtrer les données pour la calibration >>>
    max_maturity_for_calibration = 2.0
    
    # Créer un masque pour ne sélectionner que les maturités <= max_maturity
    calibration_mask = full_market_maturities <= max_maturity_for_calibration
    
    calib_maturities = full_market_maturities[calibration_mask]
    calib_yields = full_market_yields[calibration_mask]

    if len(calib_maturities) < 3:
        print(f"Pas assez de points de données pour calibrer (besoin d'au moins 3) en dessous de {max_maturity_for_calibration} ans.")
        return

    print(f"\nUtilisation de la partie de la courbe jusqu'à {max_maturity_for_calibration} ans pour la calibration :")
    df_display = pd.DataFrame({'Maturité (années)': calib_maturities, 'Rendement (%)': calib_yields * 100})
    print(df_display.round(3))
    # <<< FIN DE LA MODIFICATION >>>

    # 2. Définir r(0)
    r0_proxy = calib_yields[-1] # Utiliser le taux le plus long de la partie courte
    print(f"\nUtilisation du taux à {calib_maturities[-1]:.2f} an(s) comme proxy pour r(0): {r0_proxy*100:.3f}%")

    # 3. Lancer les calibrations sur les DONNÉES FILTRÉES
    calibrated_cir, _ = calibrate_cir_model(calib_maturities, calib_yields, r0_proxy)
    calibrated_hw, _ = calibrate_hw_model(calib_maturities, calib_yields, r0_proxy)

    # 4. Calculer les courbes des modèles calibrés sur TOUTE la plage de maturités
    # Ceci va nous montrer comment les modèles EXTRAPOLENT
    cir_extrapolated_yields = calibrated_cir.yield_curve(0, full_market_maturities, r0_proxy)
    hw_extrapolated_yields = calibrated_hw.yield_curve(0, full_market_maturities, r0_proxy)

    # 5. Tracer le graphique comparatif
    print("\n--- Génération du graphique de calibration court terme et d'extrapolation long terme ---")
    plt.figure(figsize=(14, 8))
    # Courbe de marché complète
    plt.plot(full_market_maturities, full_market_yields * 100, 'o-', label="Courbe de Marché Complète", markersize=8, lw=2.5, color='black')
    # Courbes des modèles
    plt.plot(full_market_maturities, cir_extrapolated_yields * 100, 's--', label="CIR (Calibré 0-2 ans)", markersize=6)
    plt.plot(full_market_maturities, hw_extrapolated_yields * 100, '^--', label="Hull-White (Calibré 0-2 ans)", markersize=6)
    
    # Mettre en évidence la zone de calibration
    plt.axvspan(0, max_maturity_for_calibration, color='gray', alpha=0.2, label=f'Zone de Calibration (0-{int(max_maturity_for_calibration)} ans)')
    
    plt.title("Calibration sur le Court Terme et Extrapolation sur le Long Terme")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Rendement (%)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.ylim(bottom=min(0, np.min(full_market_yields) * 100 * 1.1)) # Ajuster les limites de l'axe Y
    plt.savefig("figures/short_term_calibration_extrapolation.png")
    plt.show()

if __name__ == "__main__":
    main_short_term_calibration()