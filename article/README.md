# Analyse des Modèles de Taux d'Intérêt Stochastiques

## Cox-Ingersoll-Ross (CIR) et Hull-White

**Auteur:** Octave Cerclé - ISAE-Supaero  
**Date:** Août 2025

---

## 📋 Description du Projet

Ce projet présente une implémentation complète et une analyse comparative des modèles Cox-Ingersoll-Ross (CIR) et Hull-White pour le pricing d'obligations zéro-coupon et de dérivés sur taux d'intérêt. Il accompagne l'article académique "Zero-Coupon Bond Pricing in Stochastic Interest Rate Models: The Cox-Ingersoll-Ross and Hull-White Frameworks".

### 🎯 Objectifs

- **Implémentation** des modèles CIR et Hull-White avec solutions analytiques
- **Simulation Monte Carlo** pour le pricing de dérivés complexes
- **Calibration** des paramètres sur données de marché
- **Analyse comparative** des performances et caractéristiques
- **Applications pratiques** en gestion des risques

---

## 🏗️ Structure du Projet

```
Script/
├── 📁 figures/comparaison/          # Graphiques générés
├── 📁 report/                      # Rapports d'analyse
├── 📁 script/                      # Scripts SMC existants
├── 📄 script_for_article.py        # Script principal d'analyse
├── 📄 advanced_applications.py     # Applications avancées
├── 📄 generate_report.py          # Génération de rapports
└── 📄 README.md                   # Ce fichier
```

---

## 🚀 Installation et Exécution

### Prérequis
- Python 3.8+
- Environnement virtuel recommandé

### Installation des dépendances
```bash
pip install numpy matplotlib scipy pandas seaborn
```

### Exécution des analyses

#### 1. Analyse principale (script_for_article.py)
```bash
python script_for_article.py
```
**Contenu:**
- Comparaison des simulations CIR vs Hull-White
- Pricing analytique d'obligations zéro-coupon
- Monte Carlo pour dérivés sur taux
- Analyses de convergence et sensibilité

#### 2. Applications avancées (advanced_applications.py)
```bash
python advanced_applications.py
```
**Contenu:**
- Calibration CIR sur courbe de taux
- Pricing de produits structurés (Range Accrual, Cliquet)
- Gestion des risques (VaR, stress testing)
- Tests de robustesse et validation

#### 3. Génération de rapport (generate_report.py)
```bash
python generate_report.py
```
**Contenu:**
- Rapport HTML complet
- Résumé Markdown
- Tableaux de performance CSV

---

## 📊 Principales Fonctionnalités

### Modèle Cox-Ingersoll-Ross (CIR)
- **Équation:** `dr = (b - βr)dt + σ√r dW`
- **Caractéristiques:** Positivité garantie, volatilité stochastique
- **Pricing analytique:** Formule affine fermée
- **Applications:** Environnements de taux élevés

### Modèle Hull-White
- **Équation:** `dr = (b(t) - βr)dt + σ dW`
- **Caractéristiques:** Calibration flexible, volatilité constante
- **Pricing analytique:** Formule affine avec intégration
- **Applications:** Calibration précise, taux bas/négatifs

### Monte Carlo Avancé
- **Simulation:** Euler et schémas exacts
- **Variance reduction:** Variables antithétiques
- **Dérivés complexes:** Calls sur taux, produits structurés
- **Convergence:** Analyse O(1/√N)

---

## 📈 Résultats Clés

### Métriques de Performance

| Métrique | CIR | Hull-White | Commentaire |
|----------|-----|------------|-------------|
| **Prix obligation 5Y** | 79.35% | 108.74% | Différence significative |
| **Call sur taux** | 1.11% | 0.60% | CIR plus conservateur |
| **VaR 99% (1 mois)** | 0.85M€ | - | Portefeuille 33M€ |
| **Duration modifiée** | 2.22 ans | - | Sensibilité globale |

### Calibration
- **RMSE:** 35.95 points de base
- **Défi:** Condition de Feller souvent violée
- **Solution:** Optimisation contrainte recommandée

### Produits Structurés
- **Range Accrual:** Écart de 48.7% entre modèles
- **Cliquet Options:** Comportements différents
- **Conclusion:** Choix du modèle critique

---

## 🔍 Analyses Réalisées

### 1. Comparaison Fondamentale
- ✅ Chemins simulés et distributions finales
- ✅ Courbes de rendement comparatives
- ✅ Sensibilité aux paramètres

### 2. Pricing et Valorisation
- ✅ Obligations zéro-coupon (analytique vs MC)
- ✅ Dérivés sur taux d'intérêt
- ✅ Produits structurés complexes

### 3. Gestion des Risques
- ✅ Value-at-Risk (VaR) Monte Carlo
- ✅ Stress testing (chocs parallèles)
- ✅ Expected Shortfall (CVaR)

### 4. Validation et Robustesse
- ✅ Convergence Monte Carlo
- ✅ Stabilité des paramètres
- ✅ Impact condition de Feller

---

## 📁 Fichiers Générés

### Graphiques (figures/comparaison/)
- `simulation_paths_comparison.png` - Chemins simulés
- `final_rate_distributions.png` - Distributions finales
- `yield_curves_comparison.png` - Courbes de rendement
- `monte_carlo_convergence.png` - Convergence MC
- `cir_calibration.png` - Résultats calibration
- `stress_testing.png` - Tests de stress
- `var_distribution.png` - Distribution VaR

### Rapports (report/)
- `rapport_complet.html` - Rapport détaillé interactif
- `resume.md` - Résumé exécutif
- `performance_summary.csv` - Métriques tabulées

---

## 🎯 Recommandations d'Usage

### Choix du Modèle

#### CIR Recommandé Pour:
- 🔹 Environnements de taux élevés (>2%)
- 🔹 Analyses théoriques rigoureuses
- 🔹 Produits sensibles à la volatilité stochastique
- 🔹 Respect de la positivité critique

#### Hull-White Recommandé Pour:
- 🔹 Calibration précise sur courbes observées
- 🔹 Environnements de taux bas/négatifs
- 🔹 Implémentation rapide et stable
- 🔹 Produits linéaires en taux

### Bonnes Pratiques

#### Implémentation
- ✅ Toujours vérifier la condition de Feller (CIR)
- ✅ Utiliser N ≥ 50,000 simulations pour la production
- ✅ Valider par comparaison analytique/Monte Carlo
- ✅ Implémenter des tests de robustesse

#### Calibration
- ✅ Contraindre les paramètres physiquement
- ✅ Utiliser plusieurs maturités pour la stabilité
- ✅ Valider sur données out-of-sample
- ✅ Surveiller la stabilité temporelle

---

## 🛠️ Code Source

### Classes Principales

#### `CIRModel`
```python
class CIRModel:
    def __init__(self, b, beta, sigma, r0)
    def simulate_euler(self, T, n_steps, n_paths)
    def bond_price_analytical(self, t, T, r)
    def yield_curve(self, t, maturities, r)
```

#### `HullWhiteModel`
```python
class HullWhiteModel:
    def __init__(self, beta, sigma, r0, b_function)
    def simulate_euler(self, T, n_steps, n_paths)
    def simulate_exact(self, T, n_steps, n_paths)
    def bond_price_analytical(self, t, T, r)
```

#### `MonteCarloDerivativesPricer`
```python
class MonteCarloDerivativesPricer:
    def price_call_on_rate(self, T, strike, n_sims, use_antithetic)
    def _simulate_antithetic(self, T, n_steps, n_paths)
```

---

## 📚 Références

### Article Principal
- **Titre:** "Zero-Coupon Bond Pricing in Stochastic Interest Rate Models: The Cox-Ingersoll-Ross and Hull-White Frameworks"
- **Auteur:** Octave Cerclé
- **Institution:** ISAE-Supaero

### Littérature
- Cox, J.C., Ingersoll, J.E., Ross, S.A. (1985) - Modèle CIR original
- Hull, J., White, A. (1990) - Modèle Hull-White
- Brigo, D., Mercurio, F. (2006) - "Interest Rate Models: Theory and Practice"

### Implémentation
- **Langage:** Python 3.8+
- **Librairies:** NumPy, SciPy, Matplotlib, Pandas
- **Performance:** Optimisé pour calculs intensifs

---

## 🤝 Contact et Support

**Auteur:** Octave Cerclé  
**Institution:** ISAE-Supaero  
**Email:** [contact académique]

Pour questions techniques ou collaborations, merci de consulter d'abord la documentation complète dans `report/rapport_complet.html`.

---

## 📄 Licence

Ce projet est développé dans un cadre académique à des fins éducatives et de recherche.

---

*Dernière mise à jour: Août 2025*
