
# Rapport d'Analyse - Modèles CIR et Hull-White

**Auteur:** Octave Cerclé - ISAE-Supaero  
**Date:** 01/08/2025 17:35

## Résumé Exécutif

Cette analyse compare les modèles Cox-Ingersoll-Ross (CIR) et Hull-White pour le pricing d'instruments de taux d'intérêt.

### Résultats Clés

- **Pricing d'obligations:** Différences significatives entre les modèles (CIR: 79.35%, Hull-White: 108.74% pour une obligation 5 ans)
- **Dérivés sur taux:** CIR donne des prix plus élevés (1.11% vs 0.60% pour un call)
- **Gestion des risques:** VaR 99% estimée à 0.85M€ pour un portefeuille de 33M€
- **Calibration:** Difficultés avec la condition de Feller pour CIR

### Recommandations

1. **CIR** pour les environnements de taux élevés et l'analyse théorique
2. **Hull-White** pour la calibration précise et les taux bas/négatifs
3. Validation systématique par comparaison analytique/Monte Carlo
4. Tests de robustesse indispensables

### Graphiques Générés

- Comparaisons de chemins simulés
- Courbes de rendement
- Analyses de sensibilité
- Tests de stress
- Distributions de risque

**Code source complet disponible dans les scripts Python développés.**
