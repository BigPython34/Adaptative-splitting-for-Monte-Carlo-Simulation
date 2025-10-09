# finance_lib/data_fetcher.py
import numpy as np
import pandas_datareader.data as web
from datetime import datetime, timedelta

def fetch_fred_yield_curve() -> tuple[np.ndarray | None, np.ndarray | None]:
    """Récupère la courbe des taux des bons du Trésor US depuis FRED."""
    print("--- Récupération des données de la courbe des taux depuis FRED ---")
    fred_series = {'DGS1MO': 1/12, 'DGS3MO': 3/12, 'DGS6MO': 6/12, 'DGS1': 1.0, 
                   'DGS2': 2.0, 'DGS3': 3.0, 'DGS5': 5.0, 'DGS7': 7.0, 
                   'DGS10': 10.0, 'DGS20': 20.0, 'DGS30': 30.0}
    try:
        data = web.get_data_fred(list(fred_series.keys()), start=datetime.now() - timedelta(days=30))
        latest_yields = data.dropna().iloc[-1]
        print(f"Données récupérées pour la date : {latest_yields.name.strftime('%Y-%m-%d')}")
        maturities = np.array([fred_series[col] for col in latest_yields.index])
        yields = latest_yields.values / 100.0
        return np.sort(maturities), yields[np.argsort(maturities)]
    except Exception as e:
        print(f"Erreur lors de la récupération des données FRED : {e}")
        return None, None