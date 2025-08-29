#!/usr/bin/env python3
"""
Calcul de probabilités pour la loi normale standard N(0,1).
"""
import numpy as np
from scipy.stats import norm


def prob_normal_less_than(L: float) -> float:
    """
    Calcule P(N(0,1) < L) où N(0,1) est la loi normale standard.

    Args:
        L: Seuil

    Returns:
        Probabilité P(N(0,1) < L)
    """
    return norm.cdf(L)


def prob_normal_greater_than(L: float) -> float:
    """
    Calcule P(N(0,1) > L) où N(0,1) est la loi normale standard.

    Args:
        L: Seuil

    Returns:
        Probabilité P(N(0,1) > L)
    """
    return 1 - norm.cdf(L)


# Exemples d'utilisation
if __name__ == "__main__":
    L_values = [0, 1, 2, 3, 6]

    print("Probabilités pour la loi normale standard N(0,1):")
    print("=" * 50)

    for L in L_values:
        prob_less = prob_normal_less_than(L)
        prob_greater = prob_normal_greater_than(L)

        print(f"L = {L:2.0f}:")
        print(f"  P(N(0,1) < {L}) = {prob_less:.6f}")
        print(f"  P(N(0,1) > {L}) = {prob_greater:.6f}")
        print(f"  Vérification: {prob_less + prob_greater:.6f} (doit être ≈ 1.0)")
        print()

    # Cas spécial pour L = 6 (comme dans votre code SMC)
    L = 6.0
    prob_less_6 = prob_normal_less_than(L)
    prob_greater_6 = prob_normal_greater_than(L)

    print(f"Cas spécial L = 6:")
    print(f"P(N(0,1) < 6) = {prob_less_6:.10f}")
    print(f"P(N(0,1) > 6) = {prob_greater_6:.2e}")
