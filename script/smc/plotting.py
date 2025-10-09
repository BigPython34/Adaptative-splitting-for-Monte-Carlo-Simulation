#!/usr/bin/env python3
"""
Fonctions de traçage pour l'analyse des résultats des simulations SMC.
Ce module est autonome et reçoit sa configuration via les arguments des fonctions.
"""
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from pathlib import Path

def save_figure(fig, filename: str, config: dict) -> None:
    """Sauvegarde une figure en appliquant les réglages de la configuration."""
    figures_dir = config["paths"]["figures_dir"]
    dpi = config["plots"]["dpi"]
    
    filepath = Path(figures_dir) / filename
    fig.tight_layout()
    fig.savefig(filepath, dpi=dpi)
    plt.close(fig)

def plot_methods_graph_all(sim_data: Dict[str, List[float]], config: dict) -> None:
    """Trace les erreurs relatives pour chaque méthode."""
    plots_cfg = config["plots"]
    labels_cfg = config["labels"]
    
    fig = plt.figure(figsize=plots_cfg["methods_figsize"])
    plt.plot(
        sim_data["L_values"],
        sim_data["naive_rel_errors"],
        marker=plots_cfg["markers"]["naive"],
        color=plots_cfg["colors"]["naive"],
        label=labels_cfg["naive_relative_label"],
    )
    plt.plot(
        sim_data["L_values"],
        sim_data["smc_rel_errors"],
        marker=plots_cfg["markers"]["adaptive"],
        color=plots_cfg["colors"]["adaptive"],
        label=labels_cfg["adaptive_relative_label"],
    )
    plt.plot(
        sim_data["L_values"],
        sim_data["fixed_rel_errors"],
        marker=plots_cfg["markers"]["fixed"],
        color=plots_cfg["colors"]["fixed"],
        label=labels_cfg["fixed_relative_label"],
    )
    plt.xlabel(labels_cfg["relative_error_xlabel"])
    plt.ylabel(labels_cfg["relative_error_ylabel"])
    plt.title(labels_cfg["relative_error_title"])
    plt.legend()
    plt.grid(plots_cfg["grid"])
    save_figure(fig, config["paths"]["figures"]["methods"], config)

def plot_thresholds_on_gaussian(
    adaptive_thresholds: List[float],
    fixed_thresholds: np.ndarray,
    config: dict,
) -> None:
    """Superpose les seuils adaptatifs et fixes sur la densité gaussienne."""
    plots_cfg = config["plots"]
    labels_cfg = config["labels"]
    
    x_values = np.linspace(
        plots_cfg["density_range"]["start"],
        plots_cfg["density_range"]["stop"],
        plots_cfg["density_range"]["num"],
    )
    y_values = norm.pdf(x_values)
    
    fig = plt.figure(figsize=plots_cfg["threshold_figsize"])
    plt.plot(
        x_values, y_values,
        label=labels_cfg["gaussian_label"],
        color=plots_cfg["colors"]["naive"],
    )

    for i, threshold in enumerate(adaptive_thresholds):
        plt.axvline(
            x=-threshold,
            color=plots_cfg["colors"]["adaptive_threshold"],
            linestyle=plots_cfg["linestyles"]["adaptive"],
            label=labels_cfg["adaptive_threshold_label"] if i == 0 else None,
        )

    for i, threshold in enumerate(fixed_thresholds):
        plt.axvline(
            x=-threshold,
            color=plots_cfg["colors"]["fixed_threshold"],
            linestyle=plots_cfg["linestyles"]["fixed"],
            label=labels_cfg["fixed_threshold_label"] if i == 0 else None,
        )

    plt.xlabel(labels_cfg["density_xlabel"])
    plt.ylabel(labels_cfg["density_ylabel"])
    plt.title(labels_cfg["threshold_title"])
    plt.legend()
    plt.grid(plots_cfg["grid"])
    save_figure(fig, config["paths"]["figures"]["thresholds"], config)


def plot_variances_boxplot(var_data: Dict[str, List[float]], config: dict) -> None:
    """Affiche un boxplot des estimateurs."""
    plots_cfg = config["plots"]
    labels_cfg = config["labels"]
    sim_cfg = config["simulation"]

    time_budget = var_data.get("time_budget", sim_cfg["time_budget_seconds"])
    L_target = var_data.get("L_target", sim_cfg["demo_threshold"])
    theoretical_prob = norm.cdf(-L_target)

    fig = plt.figure(figsize=plots_cfg["boxplot_figsize"])
    plt.boxplot(
        [var_data["naive_estimates"], var_data["adaptive_estimates"], var_data["fixed_estimates"]],
        labels=[
            labels_cfg["naive_boxplot_label"],
            labels_cfg["adaptive_boxplot_label"],
            labels_cfg["fixed_boxplot_label"],
        ],
    )
    plt.axhline(
        theoretical_prob,
        color=plots_cfg["colors"]["theoretical"],
        linestyle=plots_cfg["linestyles"]["adaptive"],
        label=labels_cfg["theoretical_label"].format(L_target=L_target),
    )
    plt.ylabel(labels_cfg["probability_ylabel"])
    plt.title(labels_cfg["boxplot_title"].format(time_budget=time_budget, L_target=L_target))
    plt.legend()
    plt.grid(plots_cfg["grid"])
    save_figure(fig, config["paths"]["figures"]["variances"], config)


def plot_relative_errors_boxplot(var_data: Dict[str, List[float]], config: dict) -> None:
    """Affiche un boxplot des erreurs relatives."""
    plots_cfg = config["plots"]
    labels_cfg = config["labels"]
    sim_cfg = config["simulation"]
    
    time_budget = var_data.get("time_budget", sim_cfg["time_budget_seconds"])
    L_target = var_data.get("L_target", sim_cfg["demo_threshold"])
    theoretical_prob = norm.cdf(-L_target)

    if theoretical_prob == 0.0:
        rel_errors = [[np.nan] * len(var_data["naive_estimates"])] * 3
    else:
        rel_errors = [
            [abs(est - theoretical_prob) / theoretical_prob for est in var_data[key]]
            for key in ["naive_estimates", "adaptive_estimates", "fixed_estimates"]
        ]

    fig = plt.figure(figsize=plots_cfg["boxplot_figsize"])
    plt.boxplot(
        rel_errors,
        labels=[
            labels_cfg["naive_boxplot_label"],
            labels_cfg["adaptive_boxplot_label"],
            labels_cfg["fixed_boxplot_label"],
        ],
    )
    plt.ylabel(labels_cfg["relative_error_ylabel"])
    plt.title(labels_cfg["relative_boxplot_title"].format(time_budget=time_budget, L_target=L_target))
    plt.grid(plots_cfg["grid"])
    save_figure(fig, config["paths"]["figures"]["relative_errors"], config)