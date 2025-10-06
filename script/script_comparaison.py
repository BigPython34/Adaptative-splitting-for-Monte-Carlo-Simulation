#!/usr/bin/env python3

import datetime
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# optional tqdm for nicer progress bars; fallback to a noop if unavailable
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:  # Support package and standalone execution
    from .config import COMPARISON_CONFIG, SMC_CONSTANTS
    from .core import phi, mcmc_kernel
    from .smc import adaptive_smc_run
except ImportError:  # pragma: no cover - fallback when run as script
    from config import COMPARISON_CONFIG, SMC_CONSTANTS
    from core import phi, mcmc_kernel
    from smc import adaptive_smc_run

COMPARISON_PATHS = COMPARISON_CONFIG["paths"]
COMPARISON_SIMULATION = COMPARISON_CONFIG["simulation"]
COMPARISON_GRID = COMPARISON_CONFIG["grid"]
COMPARISON_PLOTS = COMPARISON_CONFIG["plots"]
COMPARISON_LABELS = COMPARISON_CONFIG["labels"]
COMPARISON_MESSAGES = COMPARISON_CONFIG["messages"]
COMPARISON_PROMPTS = COMPARISON_CONFIG["prompts"]

FIGURES_DIR = COMPARISON_PATHS["figures_dir"]
RESULTS_DIR = COMPARISON_PATHS["results_dir"]
RESULTS_FILENAME = COMPARISON_PATHS["results_filename"]
FIGURE_FILENAMES = COMPARISON_PATHS["figures"]
MARKERS = COMPARISON_PLOTS["markers"]
COLORS = COMPARISON_PLOTS["colors"]
LINESTYLES = COMPARISON_PLOTS["linestyles"]
DENSITY_RANGE = COMPARISON_PLOTS["density_range"]
GRID_ENABLED = COMPARISON_PLOTS["grid"]


def fixed_smc_run(
    particle_count: int,
    L_target: float,
    fixed_num_levels: int,
    n_mcmc: int,
    sigma: float,
    verbose: bool,
) -> Tuple[float | None, np.ndarray, int, List[float]]:
    """Exécute une itération de l'algorithme SMC à seuil fixe."""

    thresholds = np.linspace(
        COMPARISON_SIMULATION["fixed_threshold_start"],
        L_target,
        fixed_num_levels,
    )
    particles = np.random.randn(particle_count)
    prob_est = SMC_CONSTANTS["initial_prob_estimate"]
    acc_rates: List[float] = []

    for threshold in thresholds:
        phi_vals = phi(particles)
        survivors = particles[phi_vals >= threshold]
        if survivors.size == 0:
            if verbose:
                print(COMPARISON_MESSAGES["fixed_no_survivor"])
            return None, thresholds, len(thresholds), acc_rates

        survivors_count = survivors.size
        prob_est *= survivors_count / particles.size
        indices = np.random.choice(
            survivors_count, size=particle_count, replace=True
        )
        particles = survivors[indices]
        new_particles: List[float] = []
        level_acc: List[float] = []

        for particle in particles:
            x_new, acc = mcmc_kernel(particle, threshold, n_mcmc, sigma)
            new_particles.append(x_new)
            level_acc.append(acc)

        particles = np.array(new_particles)
        acc_rates.append(np.mean(level_acc))

    return prob_est, thresholds, fixed_num_levels, acc_rates


def run_naive_mc(
    L_target: float, time_budget: float, batch_size: int
) -> Tuple[float, int, int, float]:
    """Exécute une simulation MC naïve pendant un budget de temps donné."""

    total_samples = 0
    total_success = 0
    start_time = time.time()
    while time.time() - start_time < time_budget:
        samples = np.random.randn(batch_size)
        successes = np.sum(samples <= -L_target)
        total_success += successes
        total_samples += batch_size
    estimated_prob = (
        total_success / total_samples if total_samples > 0 else 0.0
    )
    elapsed = time.time() - start_time
    return float(estimated_prob), total_samples, total_success, elapsed


def run_smc(
    L_target: float,
    time_budget: float,
    particle_count: int,
    p0: float,
    n_mcmc: int,
    sigma: float,
    max_iter: int,
    verbose: bool,
) -> Dict[str, float]:
    """Répète le SMC adaptatif sur un budget de temps et agrège les diagnostics."""

    run_estimates: List[float] = []
    run_iters: List[int] = []
    run_accs: List[float] = []
    start_time = time.time()
    futures = []
    submitted = 0
    completed = 0

    with ProcessPoolExecutor() as executor:
        # submit tasks until budget exhausted
        while time.time() - start_time < time_budget:
            futures.append(
                executor.submit(
                    adaptive_smc_run,
                    particle_count,
                    p0,
                    L_target,
                    n_mcmc,
                    sigma,
                    max_iter,
                    verbose,
                )
            )
            submitted += 1
            if verbose and submitted % 10 == 0:
                print(f"    [SMC submit] {submitted} jobs queued (elapsed {time.time()-start_time:.1f}s)")

        # collect results as they complete (optionally with tqdm)
        if tqdm and verbose:
            it = tqdm(as_completed(futures), total=len(futures), desc="SMC jobs")
        else:
            it = as_completed(futures)

        for future in it:
            completed += 1
            try:
                result = future.result()
            except Exception as e:
                if verbose:
                    print(f"    [SMC error] a job raised an exception: {e}")
                continue

            if result is not None:
                run_estimates.append(result.prob_est)
                run_iters.append(result.n_iter)
                run_accs.append(
                    np.mean(result.acc_rates) if result.acc_rates else np.nan
                )

    total_time = time.time() - start_time
    if run_estimates:
        avg_estimate = float(np.mean(run_estimates))
        avg_iters = float(np.mean(run_iters))
        avg_acc = float(np.mean(run_accs))
        avg_run_time = total_time / len(run_estimates)
    else:
        avg_estimate = np.nan
        avg_iters = np.nan
        avg_acc = np.nan
        avg_run_time = np.nan

    return {
        "avg_estimate": avg_estimate,
        "num_runs": len(run_estimates),
        "avg_iters": avg_iters,
        "avg_acc": avg_acc,
        "avg_run_time": avg_run_time,
        "total_time": total_time,
    }


def run_fixed_smc(
    L_target: float,
    time_budget: float,
    particle_count: int,
    fixed_num_levels: int,
    n_mcmc: int,
    sigma: float,
    verbose: bool,
) -> Dict[str, float]:
    """Répète le SMC à seuil fixe sur un budget temporel donné."""

    run_estimates: List[float] = []
    run_levels: List[int] = []
    run_accs: List[float] = []
    start_time = time.time()
    futures = []

    with ProcessPoolExecutor() as executor:
        while time.time() - start_time < time_budget:
            futures.append(
                executor.submit(
                    fixed_smc_run,
                    particle_count,
                    L_target,
                    fixed_num_levels,
                    n_mcmc,
                    sigma,
                    verbose,
                )
            )
        # optionally use a progress iterator
        completed_iter = tqdm(as_completed(futures), total=len(futures), desc="Fixed SMC") if tqdm and verbose else as_completed(futures)
        for future in completed_iter:
            prob_est, thresholds, n_levels, acc_rates = future.result()
            if prob_est is not None:
                run_estimates.append(prob_est)
                run_levels.append(n_levels)
                run_accs.append(np.mean(acc_rates) if acc_rates else np.nan)
            if verbose and not (tqdm and verbose):
                # if tqdm is active it already shows progress
                print(f"    [Fixed SMC done] collected {len(run_estimates)} results so far")

    total_time = time.time() - start_time
    if run_estimates:
        avg_estimate = float(np.mean(run_estimates))
        avg_levels = float(np.mean(run_levels))
        avg_acc = float(np.mean(run_accs))
        avg_run_time = total_time / len(run_estimates)
    else:
        avg_estimate = np.nan
        avg_levels = np.nan
        avg_acc = np.nan
        avg_run_time = np.nan

    return {
        "avg_estimate": avg_estimate,
        "num_runs": len(run_estimates),
        "avg_levels": avg_levels,
        "avg_acc": avg_acc,
        "avg_run_time": avg_run_time,
        "total_time": total_time,
    }


def save_figure(fig, filename: str) -> None:
    """Sauvegarde une figure en appliquant les réglages globaux."""

    filepath = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(filepath, dpi=COMPARISON_PLOTS["dpi"])
    plt.close(fig)


def simulate_variances_fixed_time(
    L_target: float,
    n_runs: int,
    time_budget: float,
    particle_count: int,
    p0: float,
    n_mcmc: int,
    sigma: float,
    fixed_num_levels: int,
    max_iter: int,
    batch_size: int,
    verbose: bool,
) -> Dict[str, List[float]]:
    """Compare les estimateurs sous contrainte de temps fixe."""

    naive_estimates: List[float] = []
    adaptive_estimates: List[float] = []
    fixed_estimates: List[float] = []

    for _ in range(n_runs):
        naive_prob, _, _, _ = run_naive_mc(L_target, time_budget, batch_size)
        adaptive_result = run_smc(
            L_target,
            time_budget,
            particle_count,
            p0,
            n_mcmc,
            sigma,
            max_iter,
            verbose,
        )
        fixed_result = run_fixed_smc(
            L_target,
            time_budget,
            particle_count,
            fixed_num_levels,
            n_mcmc,
            sigma,
            verbose,
        )
        if tqdm and verbose:
            # if tqdm available, show a short summary instead of per-iteration prints
            print(f"    [Variance sim] run {len(naive_estimates)+1}/{n_runs} done (naive={naive_prob:.3e}, adaptive={adaptive_result['avg_estimate']:.3e})")
        elif verbose:
            run_index = len(naive_estimates) + 1
            print(f"    [Variance sim] run {run_index}/{n_runs} done (naive={naive_prob:.3e}, adaptive={adaptive_result['avg_estimate']:.3e})")
        naive_estimates.append(naive_prob)
        adaptive_estimates.append(adaptive_result["avg_estimate"])
        fixed_estimates.append(fixed_result["avg_estimate"])

    return {
        "time_budget": time_budget,
        "n_runs": n_runs,
        "naive_estimates": naive_estimates,
        "adaptive_estimates": adaptive_estimates,
        "fixed_estimates": fixed_estimates,
    }


def plot_methods_graph_all(sim_data: Dict[str, List[float]]) -> None:
    """Trace les erreurs relatives pour chaque méthode."""

    L_values = sim_data["L_values"]
    naive_rel_errors = sim_data["naive_rel_errors"]
    smc_rel_errors = sim_data["smc_rel_errors"]
    fixed_rel_errors = sim_data["fixed_rel_errors"]

    fig = plt.figure(figsize=COMPARISON_PLOTS["methods_figsize"])
    plt.plot(
        L_values,
        naive_rel_errors,
        marker=MARKERS["naive"],
        color=COLORS["naive"],
        label=COMPARISON_LABELS["naive_relative_label"],
    )
    plt.plot(
        L_values,
        smc_rel_errors,
        marker=MARKERS["adaptive"],
        color=COLORS["adaptive"],
        label=COMPARISON_LABELS["adaptive_relative_label"],
    )
    plt.plot(
        L_values,
        fixed_rel_errors,
        marker=MARKERS["fixed"],
        color=COLORS["fixed"],
        label=COMPARISON_LABELS["fixed_relative_label"],
    )
    plt.xlabel(COMPARISON_LABELS["relative_error_xlabel"])
    plt.ylabel(COMPARISON_LABELS["relative_error_ylabel"])
    plt.title(COMPARISON_LABELS["relative_error_title"])
    plt.legend()
    plt.grid(GRID_ENABLED)
    save_figure(fig, FIGURE_FILENAMES["methods"])


def plot_thresholds_on_gaussian(
    L_target: float,
    adaptive_thresholds: List[float],
    fixed_thresholds: np.ndarray,
) -> None:
    """Superpose les seuils adaptatifs et fixes sur la densité gaussienne."""

    x_values = np.linspace(
        DENSITY_RANGE["start"],
        DENSITY_RANGE["stop"],
        DENSITY_RANGE["num"],
    )
    y_values = norm.pdf(x_values)
    fig = plt.figure(figsize=COMPARISON_PLOTS["threshold_figsize"])
    plt.plot(
        x_values,
        y_values,
        label=COMPARISON_LABELS["gaussian_label"],
        color=COLORS["naive"],
    )

    for index, threshold in enumerate(adaptive_thresholds):
        plt.axvline(
            x=-threshold,
            color=COLORS["adaptive_threshold"],
            linestyle=LINESTYLES["adaptive"],
            label=(
                COMPARISON_LABELS["adaptive_threshold_label"]
                if index == 0
                else None
            ),
        )

    for index, threshold in enumerate(fixed_thresholds):
        plt.axvline(
            x=-threshold,
            color=COLORS["fixed_threshold"],
            linestyle=LINESTYLES["fixed"],
            label=(
                COMPARISON_LABELS["fixed_threshold_label"]
                if index == 0
                else None
            ),
        )

    plt.xlabel(COMPARISON_LABELS["density_xlabel"])
    plt.ylabel(COMPARISON_LABELS["density_ylabel"])
    plt.title(COMPARISON_LABELS["threshold_title"])
    plt.legend()
    plt.grid(GRID_ENABLED)
    save_figure(fig, FIGURE_FILENAMES["thresholds"])


def plot_variances_boxplot(var_data: Dict[str, List[float]]) -> None:
    """Affiche un boxplot des estimateurs."""

    naive_estimates = var_data["naive_estimates"]
    adaptive_estimates = var_data["adaptive_estimates"]
    fixed_estimates = var_data["fixed_estimates"]
    time_budget = var_data.get(
        "time_budget", COMPARISON_SIMULATION["time_budget_seconds"]
    )
    L_target = var_data.get("L_target", COMPARISON_SIMULATION["demo_threshold"])
    theoretical_prob = norm.cdf(-L_target)

    fig = plt.figure(figsize=COMPARISON_PLOTS["boxplot_figsize"])
    plt.boxplot(
        [naive_estimates, adaptive_estimates, fixed_estimates],
        labels=[
            COMPARISON_LABELS["naive_boxplot_label"],
            COMPARISON_LABELS["adaptive_boxplot_label"],
            COMPARISON_LABELS["fixed_boxplot_label"],
        ],
    )
    plt.axhline(
        theoretical_prob,
        color=COLORS["theoretical"],
        linestyle=LINESTYLES["adaptive"],
        label=COMPARISON_LABELS["theoretical_label"].format(
            L_target=L_target
        ),
    )
    plt.ylabel(COMPARISON_LABELS["probability_ylabel"])
    plt.title(
        COMPARISON_LABELS["boxplot_title"].format(
            time_budget=time_budget, L_target=L_target
        )
    )
    plt.legend()
    plt.grid(GRID_ENABLED)
    save_figure(fig, FIGURE_FILENAMES["variances"])


def plot_relative_errors_boxplot(var_data: Dict[str, List[float]]) -> None:
    """Affiche un boxplot des erreurs relatives."""

    naive_estimates = var_data["naive_estimates"]
    adaptive_estimates = var_data["adaptive_estimates"]
    fixed_estimates = var_data["fixed_estimates"]
    time_budget = var_data.get(
        "time_budget", COMPARISON_SIMULATION["time_budget_seconds"]
    )
    L_target = var_data.get("L_target", COMPARISON_SIMULATION["demo_threshold"])
    theoretical_prob = norm.cdf(-L_target)

    if theoretical_prob == 0.0:
        naive_rel_errors = [np.nan for _ in naive_estimates]
        adaptive_rel_errors = [np.nan for _ in adaptive_estimates]
        fixed_rel_errors = [np.nan for _ in fixed_estimates]
    else:
        naive_rel_errors = [
            abs(estimate - theoretical_prob) / theoretical_prob
            for estimate in naive_estimates
        ]
        adaptive_rel_errors = [
            abs(estimate - theoretical_prob) / theoretical_prob
            for estimate in adaptive_estimates
        ]
        fixed_rel_errors = [
            abs(estimate - theoretical_prob) / theoretical_prob
            for estimate in fixed_estimates
        ]

    fig = plt.figure(figsize=COMPARISON_PLOTS["boxplot_figsize"])
    plt.boxplot(
        [naive_rel_errors, adaptive_rel_errors, fixed_rel_errors],
        labels=[
            COMPARISON_LABELS["naive_boxplot_label"],
            COMPARISON_LABELS["adaptive_boxplot_label"],
            COMPARISON_LABELS["fixed_boxplot_label"],
        ],
    )
    plt.ylabel(COMPARISON_LABELS["relative_error_ylabel"])
    plt.title(
        COMPARISON_LABELS["relative_boxplot_title"].format(
            time_budget=time_budget, L_target=L_target
        )
    )
    plt.grid(GRID_ENABLED)
    save_figure(fig, FIGURE_FILENAMES["relative_errors"])


def simulate_methods_graph_all(
    simulation_config: Dict[str, float],
    grid_config: Dict[str, float],
) -> Dict[str, List[float]]:
    """Calcule les erreurs relatives pour une grille de seuils."""

    L_values = np.linspace(
        grid_config["l_min"],
        grid_config["l_max"],
        grid_config["l_count"],
    )
    naive_rel_errors: List[float] = []
    smc_rel_errors: List[float] = []
    fixed_rel_errors: List[float] = []
    theoretical_probs: List[float] = []

    for threshold in L_values:
        theoretical_prob = norm.cdf(-threshold)
        theoretical_probs.append(theoretical_prob)

        naive_prob, _, _, _ = run_naive_mc(
            threshold,
            simulation_config["time_budget_seconds"],
            simulation_config["naive_batch_size"],
        )
        if theoretical_prob == 0.0:
            naive_rel_errors.append(np.nan)
        else:
            naive_rel_errors.append(
                abs(naive_prob - theoretical_prob) / theoretical_prob
            )

        adaptive_result = run_smc(
            threshold,
            simulation_config["time_budget_seconds"],
            simulation_config["particle_count"],
            simulation_config["p0"],
            simulation_config["n_mcmc"],
            simulation_config["sigma"],
            simulation_config["max_iter"],
            simulation_config["verbose"],
        )
        adaptive_prob = adaptive_result["avg_estimate"]
        if theoretical_prob == 0.0:
            smc_rel_errors.append(np.nan)
        else:
            smc_rel_errors.append(
                abs(adaptive_prob - theoretical_prob) / theoretical_prob
            )

        fixed_result = run_fixed_smc(
            threshold,
            simulation_config["time_budget_seconds"],
            simulation_config["particle_count"],
            simulation_config["fixed_num_levels"],
            simulation_config["n_mcmc"],
            simulation_config["sigma"],
            simulation_config["verbose"],
        )
        fixed_prob = fixed_result["avg_estimate"]
        if theoretical_prob == 0.0:
            fixed_rel_errors.append(np.nan)
        else:
            fixed_rel_errors.append(
                abs(fixed_prob - theoretical_prob) / theoretical_prob
            )

        if simulation_config.get("verbose", False):
            idx = len(naive_rel_errors)
            if tqdm:
                # tqdm will show a progress bar; provide a concise line per step
                print(f"    [Methods sim] threshold {idx+1}/{len(L_values)} (L={threshold:.2f}): naive={naive_prob:.3e}, adaptive={adaptive_result['avg_estimate']:.3e}, fixed={fixed_result['avg_estimate']:.3e}")
            else:
                print(f"    [Methods sim] threshold {idx+1}/{len(L_values)} (L={threshold:.2f}): naive={naive_prob:.3e}, adaptive={adaptive_result['avg_estimate']:.3e}, fixed={fixed_result['avg_estimate']:.3e}")

    return {
        "L_values": L_values,
        "naive_rel_errors": naive_rel_errors,
        "smc_rel_errors": smc_rel_errors,
        "fixed_rel_errors": fixed_rel_errors,
        "theoretical_probs": theoretical_probs,
    }


def main() -> None:
    """Point d'entrée principal pour la comparaison des méthodes."""

    start_time = datetime.datetime.now()
    print(
        COMPARISON_MESSAGES["analysis_start"].format(
            timestamp=start_time.strftime("%Y-%m-%d %H:%M:%S")
        )
    )
    print(COMPARISON_MESSAGES["header"])

    results_file = RESULTS_DIR / RESULTS_FILENAME
    verbose = COMPARISON_SIMULATION["verbose"]

    if results_file.exists():
        choice = input(COMPARISON_PROMPTS["existing_results"])
        redo_simulation = (
            choice.lower() == COMPARISON_PROMPTS["affirmative"]
        )
    else:
        redo_simulation = True

    if redo_simulation:
        sim_results: Dict[str, Dict[str, List[float]]] = {}
        print(COMPARISON_MESSAGES["methods_simulation_start"])
        sim_results["methods_graph"] = simulate_methods_graph_all(
            COMPARISON_SIMULATION,
            COMPARISON_GRID,
        )
        print(COMPARISON_MESSAGES["methods_simulation_end"])

        print(COMPARISON_MESSAGES["variance_simulation_start"])
        sim_results["variances"] = simulate_variances_fixed_time(
            L_target=COMPARISON_SIMULATION["demo_threshold"],
            n_runs=COMPARISON_SIMULATION["variance_runs"],
            time_budget=COMPARISON_SIMULATION["time_budget_seconds"],
            particle_count=COMPARISON_SIMULATION["particle_count"],
            p0=COMPARISON_SIMULATION["p0"],
            n_mcmc=COMPARISON_SIMULATION["n_mcmc"],
            sigma=COMPARISON_SIMULATION["sigma"],
            fixed_num_levels=COMPARISON_SIMULATION["fixed_num_levels"],
            max_iter=COMPARISON_SIMULATION["max_iter"],
            batch_size=COMPARISON_SIMULATION["naive_batch_size"],
            verbose=verbose,
        )
        print(COMPARISON_MESSAGES["variance_simulation_end"])

        with open(results_file, "wb") as file_handle:
            pickle.dump(sim_results, file_handle)
    else:
        with open(results_file, "rb") as file_handle:
            sim_results = pickle.load(file_handle)

    print(COMPARISON_MESSAGES["plotting_start"])
    plot_methods_graph_all(sim_results["methods_graph"])

    result_adapt = adaptive_smc_run(
        COMPARISON_SIMULATION["particle_count"],
        COMPARISON_SIMULATION["p0"],
        COMPARISON_SIMULATION["demo_threshold"],
        COMPARISON_SIMULATION["n_mcmc"],
        COMPARISON_SIMULATION["sigma"],
        COMPARISON_SIMULATION["max_iter"],
        verbose=verbose,
    )
    if result_adapt is None:
        raise RuntimeError(COMPARISON_MESSAGES["adaptive_plot_failure"])

    adaptive_thresholds = result_adapt.thresholds
    fixed_thresholds = np.linspace(
        COMPARISON_SIMULATION["fixed_threshold_start"],
        COMPARISON_SIMULATION["demo_threshold"],
        COMPARISON_SIMULATION["fixed_num_levels"],
    )
    plot_thresholds_on_gaussian(
        COMPARISON_SIMULATION["demo_threshold"],
        adaptive_thresholds,
        fixed_thresholds,
    )

    var_data = sim_results["variances"]
    var_data["L_target"] = COMPARISON_SIMULATION["demo_threshold"]
    plot_variances_boxplot(var_data)
    plot_relative_errors_boxplot(var_data)
    print(COMPARISON_MESSAGES["plotting_end"])
    print(COMPARISON_MESSAGES["simulation_complete"])


if __name__ == "__main__":
    main()
