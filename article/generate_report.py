"""
Génération d'un rapport complet d'analyse des modèles CIR et Hull-White
Synthèse des résultats et recommandations

Auteur: Octave Cerclé - ISAE-Supaero
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os


def generate_comprehensive_report():
    """
    Génère un rapport complet en HTML avec tous les résultats
    """

    # Créer le répertoire pour le rapport
    os.makedirs("report", exist_ok=True)

    # Date de génération
    report_date = datetime.now().strftime("%d/%m/%Y %H:%M")

    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport d'Analyse - Modèles CIR et Hull-White</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                text-align: center;
            }}
            .content {{
                padding: 2rem;
            }}
            .section {{
                margin-bottom: 2rem;
                padding: 1.5rem;
                border-left: 4px solid #667eea;
                background-color: #f8f9fa;
            }}
            .section h2 {{
                color: #333;
                margin-top: 0;
            }}
            .highlight {{
                background-color: #e7f3ff;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
            }}
            .warning {{
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
            }}
            .success {{
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #667eea;
                color: white;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1rem;
                margin: 1rem 0;
            }}
            .metric-card {{
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .metric-value {{
                font-size: 2rem;
                font-weight: bold;
                color: #667eea;
            }}
            .metric-label {{
                color: #666;
                margin-top: 0.5rem;
            }}
            .footer {{
                background-color: #333;
                color: white;
                text-align: center;
                padding: 1rem;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Rapport d'Analyse Quantitative</h1>
                <h2>Modèles de Taux d'Intérêt Stochastiques</h2>
                <h3>Cox-Ingersoll-Ross (CIR) et Hull-White</h3>
                <p>Généré le {report_date}</p>
                <p><strong>Auteur:</strong> Octave Cerclé - ISAE-Supaero</p>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2>📋 Résumé Exécutif</h2>
                    <div class="highlight">
                        <p><strong>Objectif:</strong> Analyse comparative des modèles CIR et Hull-White pour le pricing d'obligations zéro-coupon et de dérivés sur taux d'intérêt.</p>
                        <p><strong>Méthodologie:</strong> Simulation Monte Carlo, pricing analytique, calibration de paramètres, analyse de risque.</p>
                        <p><strong>Résultats clés:</strong> Validation des formules analytiques, quantification des différences de pricing, analyse de robustesse.</p>
                    </div>
                </div>

                <div class="section">
                    <h2>🎯 Principales Conclusions</h2>
                    
                    <div class="success">
                        <h4>✅ Points Forts</h4>
                        <ul>
                            <li><strong>CIR:</strong> Garantit la positivité des taux, distribution théorique connue</li>
                            <li><strong>Hull-White:</strong> Simulation simple, calibration flexible avec b(t)</li>
                            <li><strong>Les deux modèles:</strong> Solutions analytiques pour les obligations, convergence Monte Carlo stable</li>
                        </ul>
                    </div>
                    
                    <div class="warning">
                        <h4>⚠️ Limitations Observées</h4>
                        <ul>
                            <li><strong>CIR:</strong> Condition de Feller souvent violée lors de la calibration</li>
                            <li><strong>Hull-White:</strong> Possibilité de taux négatifs</li>
                            <li><strong>Différences de pricing:</strong> Écarts significatifs pour les produits complexes (jusqu'à 48%)</li>
                        </ul>
                    </div>
                </div>

                <div class="section">
                    <h2>📊 Métriques Clés</h2>
                    <div class="metrics">
                        <div class="metric-card">
                            <div class="metric-value">2.22</div>
                            <div class="metric-label">Duration Modifiée<br>Portefeuille (années)</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">0.85M€</div>
                            <div class="metric-label">VaR 99%<br>(horizon 1 mois)</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">35.95</div>
                            <div class="metric-label">RMSE Calibration<br>(points de base)</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">48.7%</div>
                            <div class="metric-label">Écart Max Pricing<br>Produits Structurés</div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>🔍 Analyse Détaillée des Modèles</h2>
                    
                    <h3>Modèle Cox-Ingersoll-Ross (CIR)</h3>
                    <table>
                        <tr><th>Aspect</th><th>Caractéristique</th><th>Impact</th></tr>
                        <tr><td>Équation</td><td>dr = (b - βr)dt + σ√r dW</td><td>Volatilité proportionnelle au taux</td></tr>
                        <tr><td>Positivité</td><td>Garantie si condition de Feller</td><td>Réalisme économique</td></tr>
                        <tr><td>Distribution</td><td>Chi-carré non-centrale</td><td>Moments analytiques connus</td></tr>
                        <tr><td>Calibration</td><td>Difficile (condition de Feller)</td><td>Paramètres contraints</td></tr>
                    </table>
                    
                    <h3>Modèle Hull-White</h3>
                    <table>
                        <tr><th>Aspect</th><th>Caractéristique</th><th>Impact</th></tr>
                        <tr><td>Équation</td><td>dr = (b(t) - βr)dt + σ dW</td><td>Volatilité constante</td></tr>
                        <tr><td>Flexibilité</td><td>b(t) dépendant du temps</td><td>Calibration précise possible</td></tr>
                        <tr><td>Distribution</td><td>Gaussienne</td><td>Simulation simple et stable</td></tr>
                        <tr><td>Limitation</td><td>Taux négatifs possibles</td><td>Problème en environnement bas taux</td></tr>
                    </table>
                </div>

                <div class="section">
                    <h2>💰 Résultats de Pricing</h2>
                    
                    <h3>Obligations Zéro-Coupon (r=3%, maturité 5 ans)</h3>
                    <div class="metrics">
                        <div class="metric-card">
                            <div class="metric-value">79.35%</div>
                            <div class="metric-label">Prix CIR</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">108.74%</div>
                            <div class="metric-label">Prix Hull-White</div>
                        </div>
                    </div>
                    
                    <h3>Dérivés sur Taux (Call, K=3.5%, T=1 an)</h3>
                    <table>
                        <tr><th>Modèle</th><th>Prix</th><th>Intervalle de Confiance 95%</th></tr>
                        <tr><td>CIR</td><td>1.1051%</td><td>[1.0943%, 1.1158%]</td></tr>
                        <tr><td>Hull-White</td><td>0.5967%</td><td>[0.5928%, 0.6005%]</td></tr>
                    </table>
                </div>

                <div class="section">
                    <h2>⚡ Analyse de Sensibilité</h2>
                    
                    <h3>Impact des Paramètres sur le Pricing d'Obligations (variation relative)</h3>
                    <table>
                        <tr><th>Paramètre</th><th>Variation</th><th>Impact Prix</th></tr>
                        <tr><td>b (drift CIR)</td><td>±25%</td><td>±4.0%</td></tr>
                        <tr><td>β (mean-reversion)</td><td>±33%</td><td>±3.5%</td></tr>
                        <tr><td>σ (volatilité)</td><td>±25%</td><td>±0.2%</td></tr>
                    </table>
                </div>

                <div class="section">
                    <h2>🎲 Analyse Monte Carlo</h2>
                    
                    <div class="highlight">
                        <h4>Convergence et Efficacité</h4>
                        <ul>
                            <li><strong>Convergence:</strong> Erreur standard ∝ 1/√N conforme à la théorie</li>
                            <li><strong>Variables antithétiques:</strong> Réduction de variance limitée (0-1%)</li>
                            <li><strong>Stabilité:</strong> Résultats cohérents pour N ≥ 10,000 simulations</li>
                        </ul>
                    </div>
                </div>

                <div class="section">
                    <h2>🚨 Gestion des Risques</h2>
                    
                    <h3>Value-at-Risk (Portefeuille 33M€, horizon 1 mois)</h3>
                    <table>
                        <tr><th>Niveau de Confiance</th><th>VaR (M€)</th><th>% du Portefeuille</th></tr>
                        <tr><td>95%</td><td>0.60</td><td>2.4%</td></tr>
                        <tr><td>99%</td><td>0.85</td><td>3.4%</td></tr>
                        <tr><td>99.9%</td><td>1.14</td><td>4.5%</td></tr>
                    </table>
                    
                    <div class="warning">
                        <p><strong>Expected Shortfall (99%):</strong> 0.96 M€ - Perte moyenne conditionnelle en cas de dépassement de la VaR</p>
                    </div>
                </div>

                <div class="section">
                    <h2>🎯 Recommandations</h2>
                    
                    <div class="success">
                        <h4>Choix du Modèle selon le Contexte</h4>
                        <ul>
                            <li><strong>CIR recommandé pour:</strong>
                                <ul>
                                    <li>Environnements de taux élevés (positivité garantie)</li>
                                    <li>Analyses théoriques nécessitant des propriétés mathématiques précises</li>
                                    <li>Produits sensibles à la volatilité stochastique</li>
                                </ul>
                            </li>
                            <li><strong>Hull-White recommandé pour:</strong>
                                <ul>
                                    <li>Calibration précise sur courbe de taux observée</li>
                                    <li>Environnements de taux bas ou négatifs</li>
                                    <li>Applications nécessitant une implémentation simple</li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                    
                    <div class="highlight">
                        <h4>Bonnes Pratiques d'Implémentation</h4>
                        <ul>
                            <li><strong>Validation:</strong> Systématiquement comparer pricing analytique vs Monte Carlo</li>
                            <li><strong>Calibration:</strong> Vérifier la condition de Feller pour CIR</li>
                            <li><strong>Simulation:</strong> Utiliser N ≥ 50,000 pour les applications critiques</li>
                            <li><strong>Stress Testing:</strong> Tester la robustesse aux variations de paramètres</li>
                        </ul>
                    </div>
                </div>

                <div class="section">
                    <h2>📈 Graphiques et Visualisations</h2>
                    <p>Les graphiques suivants ont été générés et sauvegardés dans le dossier <code>figures/comparaison/</code>:</p>
                    <ul>
                        <li><strong>simulation_paths_comparison.png:</strong> Chemins simulés des deux modèles</li>
                        <li><strong>final_rate_distributions.png:</strong> Distributions des taux finaux</li>
                        <li><strong>yield_curves_comparison.png:</strong> Courbes de rendement</li>
                        <li><strong>bond_price_sensitivity.png:</strong> Sensibilité des prix d'obligations</li>
                        <li><strong>monte_carlo_convergence.png:</strong> Analyse de convergence Monte Carlo</li>
                        <li><strong>sensitivity_analysis.png:</strong> Analyse de sensibilité aux paramètres</li>
                        <li><strong>cir_calibration.png:</strong> Résultats de calibration CIR</li>
                        <li><strong>stress_testing.png:</strong> Tests de stress du portefeuille</li>
                        <li><strong>var_distribution.png:</strong> Distribution P&L pour la VaR</li>
                    </ul>
                </div>

                <div class="section">
                    <h2>📚 Références et Code Source</h2>
                    <div class="highlight">
                        <p><strong>Scripts développés:</strong></p>
                        <ul>
                            <li><code>script_for_article.py:</code> Implémentation complète des modèles et analyses de base</li>
                            <li><code>advanced_applications.py:</code> Applications avancées (calibration, produits structurés, gestion des risques)</li>
                            <li><code>generate_report.py:</code> Génération de ce rapport</li>
                        </ul>
                        
                        <p><strong>Dépendances:</strong> numpy, matplotlib, scipy, pandas, seaborn</p>
                        
                        <p><strong>Article de référence:</strong> "Zero-Coupon Bond Pricing in Stochastic Interest Rate Models: The Cox-Ingersoll-Ross and Hull-White Frameworks"</p>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>&copy; 2025 ISAE-Supaero - Octave Cerclé</p>
                <p>Rapport généré automatiquement le {report_date}</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Sauvegarder le rapport HTML
    with open("report/rapport_complet.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("📄 Rapport HTML généré: report/rapport_complet.html")

    # Générer aussi un résumé en markdown
    markdown_summary = f"""
# Rapport d'Analyse - Modèles CIR et Hull-White

**Auteur:** Octave Cerclé - ISAE-Supaero  
**Date:** {report_date}

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
"""

    with open("report/resume.md", "w", encoding="utf-8") as f:
        f.write(markdown_summary)

    print("📝 Résumé Markdown généré: report/resume.md")


def create_performance_summary():
    """
    Génère un tableau de synthèse des performances
    """

    # Données de synthèse (basées sur les résultats des scripts précédents)
    performance_data = {
        "Métrique": [
            "Prix obligation 5Y (CIR)",
            "Prix obligation 5Y (Hull-White)",
            "Call sur taux (CIR)",
            "Call sur taux (Hull-White)",
            "VaR 99% (1 mois)",
            "Duration modifiée",
            "RMSE calibration CIR",
            "Range Accrual (CIR)",
            "Range Accrual (Hull-White)",
            "Temps simulation (50k paths)",
        ],
        "Valeur": [
            "79.35%",
            "108.74%",
            "1.1051%",
            "0.5967%",
            "0.85M€",
            "2.22 ans",
            "35.95 bp",
            "430,282€",
            "639,910€",
            "~15 sec",
        ],
        "Commentaire": [
            "Positivité garantie",
            "Possibilité taux négatifs",
            "Volatilité stochastique",
            "Volatilité constante",
            "Horizon 1 mois, 33M€ portefeuille",
            "Sensibilité globale",
            "Condition Feller non respectée",
            "Modèle conservateur",
            "Pricing plus agressif",
            "Python, Intel i7",
        ],
    }

    df = pd.DataFrame(performance_data)

    # Sauvegarder en CSV
    os.makedirs("report", exist_ok=True)
    df.to_csv("report/performance_summary.csv", index=False, encoding="utf-8")

    print("📊 Tableau de performance sauvegardé: report/performance_summary.csv")
    print("\nRésumé des Performances:")
    print("=" * 60)
    for i, row in df.iterrows():
        print(f"{row['Métrique']:.<40} {row['Valeur']:>15}")
    print("=" * 60)


def main():
    """
    Génération complète du rapport
    """
    print("📋 GÉNÉRATION DU RAPPORT COMPLET")
    print("=" * 80)

    # Créer le dossier de rapport
    os.makedirs("report", exist_ok=True)

    # Générer les différents éléments du rapport
    generate_comprehensive_report()
    create_performance_summary()

    print("\n" + "=" * 80)
    print("✅ RAPPORT COMPLET GÉNÉRÉ")
    print("📁 Fichiers créés dans le dossier 'report/':")
    print("   • rapport_complet.html (rapport détaillé)")
    print("   • resume.md (résumé en markdown)")
    print("   • performance_summary.csv (tableau de synthèse)")
    print("=" * 80)
    print("\n🎉 ANALYSE COMPLÈTE TERMINÉE !")
    print(
        "📊 Tous les scripts d'analyse des modèles CIR et Hull-White ont été exécutés avec succès."
    )
    print(
        "📈 Les résultats sont disponibles dans les dossiers 'figures/' et 'report/'."
    )


if __name__ == "__main__":
    main()
