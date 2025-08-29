#!/usr/bin/env python3
import numpy as np
import time
from pathlib import Path
import pickle
from scipy.stats import norm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime

# Création des répertoires de sauvegarde
BASE_DIR = Path("script")
FIGURES_DIR = Path("figures/comparaison")
RESULTS_DIR = Path("results")
for d in [FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)




# ---------------- Fonctions communes ----------------

def phi(x: np.ndarray) -> np.ndarray:
    """Transformation phi(x) = -x."""
    return -x

def mcmc_kernel(x: float, L_current: float, n_steps: int, sigma: float):
    """
    Kernel MCMC (Metropolis-Hastings) pour échantillonner une loi N(0,1)
    tronquée à x <= -L_current.
    """
    x_current = x
    accepts = 0
    for _ in range(n_steps):
        proposal = x_current + np.random.normal(0, sigma)
        if proposal <= -L_current:
            log_ratio = -0.5 * (proposal**2 - x_current**2)
            alpha = np.exp(log_ratio)
            if np.random.rand() < alpha:
                x_current = proposal
                accepts += 1
    return x_current, accepts / n_steps if n_steps > 0 else 0.0

# ---------------- SMC adaptatif ----------------

def adaptive_smc_run(N: int, p0: float, L_target: float, n_mcmc: int, sigma: float,
                       max_iter: int = 50, verbose: bool = False):
    """
    Exécute une run de l'algorithme SMC adaptatif.
    Retourne : (prob_est, thresholds, n_iter, acc_rates)
    """
    particles = np.random.randn(N)
    prob_est = 1.0
    thresholds = []
    acc_rates = []
    n_iter = 0

    while n_iter < max_iter:
        phi_vals = phi(particles)
        # Seuil adaptatif : quantile (1-p0)
        L_current = np.percentile(phi_vals, (1 - p0) * 100)
        thresholds.append(L_current)
        n_iter += 1

        if verbose:
            print(f"Iter {n_iter:2d} - L_current = {L_current:.3f}")

        if L_current >= L_target:
            break

        prob_est *= p0
        survivors = particles[phi_vals >= L_current]
        if survivors.size == 0:
            if verbose:
                print("Aucune particule survivante.")
            return None, thresholds, n_iter, acc_rates

        indices = np.random.choice(len(survivors), size=N, replace=True)
        particles = survivors[indices]

        new_particles = []
        level_acc = []
        for x in particles:
            x_new, acc = mcmc_kernel(x, L_current, n_mcmc, sigma)
            new_particles.append(x_new)
            level_acc.append(acc)
        particles = np.array(new_particles)
        acc_rates.append(np.mean(level_acc))

    phi_vals = phi(particles)
    r = np.mean(phi_vals >= L_target)
    prob_est *= r
    return prob_est, thresholds, n_iter, acc_rates

# ---------------- SMC à seuil fixe ----------------

def fixed_smc_run(N: int, L_target: float, fixed_num_levels: int, n_mcmc: int, sigma: float, verbose: bool = False):
    """
    Exécute une run de l'algorithme SMC à seuil fixe.
    On fixe une grille de seuils linéairement espacés entre 0 et L_target.
    Retourne : (prob_est, thresholds, num_levels, acc_rates)
    """
    thresholds = np.linspace(0, L_target, fixed_num_levels)
    particles = np.random.randn(N)
    prob_est = 1.0
    acc_rates = []
    for L in thresholds:
        phi_vals = phi(particles)
        survivors = particles[phi_vals >= L]
        if survivors.size == 0:
            if verbose:
                print("Aucune particule survivante pour le seuil fixe.")
            return None, thresholds, len(thresholds), acc_rates
        fraction = survivors.size / particles.size
        prob_est *= fraction
        indices = np.random.choice(survivors.size, size=N, replace=True)
        particles = survivors[indices]
        new_particles = []
        level_acc = []
        for x in particles:
            x_new, acc = mcmc_kernel(x, L, n_mcmc, sigma)
            new_particles.append(x_new)
            level_acc.append(acc)
        particles = np.array(new_particles)
        acc_rates.append(np.mean(level_acc))
    return prob_est, thresholds, fixed_num_levels, acc_rates




# ---------------- Boucles de simulation ----------------

def run_naive_mc(L_target: float, time_budget: float = 60.0, batch_size: int = 10**6):
    """
    Exécute la simulation MC naïf pendant 'time_budget' secondes.
    Retourne : (probabilité estimée, nb total d'échantillons, nb de succès, temps écoulé)
    """
    total_samples = 0
    total_success = 0
    start_time = time.time()
    while time.time() - start_time < time_budget:
        samples = np.random.randn(batch_size)
        successes = np.sum(samples <= -L_target)
        total_success += successes
        total_samples += batch_size
    estimated_prob = total_success / total_samples if total_samples > 0 else 0.0
    elapsed = time.time() - start_time
    return estimated_prob, total_samples, total_success, elapsed

def run_smc(L_target: float, time_budget: float = 60.0,
            N: int = 5000, p0: float = 0.5, n_mcmc: int = 40, sigma: float = 2.0,
            max_iter: int = 50, verbose: bool = False):
    """
    Exécute autant de runs SMC adaptatif que possible pendant 'time_budget' secondes (en parallèle).
    Retourne un dictionnaire résumant les résultats.
    """
    run_estimates = []
    run_iters = []
    run_accs = []
    start_time = time.time()
    futures = []
    with ProcessPoolExecutor() as executor:
        while time.time() - start_time < time_budget:
            futures.append(executor.submit(adaptive_smc_run, N, p0, L_target, n_mcmc, sigma, max_iter, verbose))
        for future in as_completed(futures):
            result = future.result()
            if result[0] is not None:
                prob_est, thresholds, n_iter, acc_rates = result
                run_estimates.append(prob_est)
                run_iters.append(n_iter)
                run_accs.append(np.mean(acc_rates) if acc_rates else np.nan)
    total_time = time.time() - start_time
    if run_estimates:
        avg_estimate = np.mean(run_estimates)
        avg_iters = np.mean(run_iters)
        avg_acc = np.mean(run_accs)
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
        "total_time": total_time
    }

def run_fixed_smc(L_target: float, time_budget: float = 60.0,
                  N: int = 5000, fixed_num_levels: int = 10, n_mcmc: int = 40, sigma: float = 2.0,
                  verbose: bool = False):
    """
    Exécute autant de runs SMC à seuil fixe que possible pendant 'time_budget' secondes (en parallèle).
    Retourne un dictionnaire résumant les résultats.
    """
    run_estimates = []
    run_levels = []
    run_accs = []
    start_time = time.time()
    futures = []
    with ProcessPoolExecutor() as executor:
        while time.time() - start_time < time_budget:
            futures.append(executor.submit(fixed_smc_run, N, L_target, fixed_num_levels, n_mcmc, sigma, verbose))
        for future in as_completed(futures):
            result = future.result()
            if result[0] is not None:
                prob_est, thresholds, n_levels, acc_rates = result
                run_estimates.append(prob_est)
                run_levels.append(n_levels)
                run_accs.append(np.mean(acc_rates) if acc_rates else np.nan)
    total_time = time.time() - start_time
    if run_estimates:
        avg_estimate = np.mean(run_estimates)
        avg_levels = np.mean(run_levels)
        avg_acc = np.mean(run_accs)
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
        "total_time": total_time
    }




# ---------------- Enregistrement des figures ----------------

def save_figure(fig, filename: str):
    """Sauvegarde la figure dans FIGURES_DIR avec le nom donné."""
    filepath = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)




# ---------------- Comparaison des variances avec temps fixe ----------------

def simulate_variances_fixed_time(L_target: float, N_runs: int = 30, time_budget: float = 60.0,
                                  N: int = 5000, p0: float = 0.5, n_mcmc: int = 40, sigma: float = 2.0,
                                  fixed_num_levels: int = 10):
    """
    Exécute N_runs pour chaque algorithme en allouant 'time_budget' secondes à chacun.
    Retourne un dictionnaire avec les estimations.
    """
    naive_estimates = []
    adaptive_estimates = []
    fixed_estimates = []
    for _ in range(N_runs):
        naive_prob, _, _, _ = run_naive_mc(L_target, time_budget=time_budget, batch_size=10**6)
        adaptive_result = run_smc(L_target, time_budget=time_budget, N=N, p0=p0, n_mcmc=n_mcmc, sigma=sigma, max_iter=50, verbose=False)
        fixed_result = run_fixed_smc(L_target, time_budget=time_budget, N=N, fixed_num_levels=fixed_num_levels, n_mcmc=n_mcmc, sigma=sigma, verbose=False)
        naive_estimates.append(naive_prob)
        adaptive_estimates.append(adaptive_result["avg_estimate"])
        fixed_estimates.append(fixed_result["avg_estimate"])
    return {"time_budget": time_budget,
            "N_runs": N_runs,
            "naive_estimates": naive_estimates,
            "adaptive_estimates": adaptive_estimates,
            "fixed_estimates": fixed_estimates}




# ---------------- Fonctions de tracé ----------------

def plot_methods_graph_all(sim_data):
    """
    Trace l'erreur relative en fonction de L à partir des données simulées.
    """
    L_values = sim_data["L_values"]
    naive_rel_errors = sim_data["naive_rel_errors"]
    smc_rel_errors = sim_data["smc_rel_errors"]
    fixed_rel_errors = sim_data["fixed_rel_errors"]

    fig = plt.figure(figsize=(10, 6))
    plt.plot(L_values, naive_rel_errors, marker='o', label="MC naïf - Erreur relative")
    plt.plot(L_values, smc_rel_errors, marker='s', label="SMC adaptatif - Erreur relative")
    plt.plot(L_values, fixed_rel_errors, marker='^', label="SMC fixe - Erreur relative")
    plt.xlabel("Seuil L")
    plt.ylabel("Erreur relative")
    plt.title("Comparaison des erreurs relatives en fonction de L")
    plt.legend()
    plt.grid(True)
    save_figure(fig, "compare_methods_graph_all.png")

def plot_thresholds_on_gaussian(L_target: float, adaptive_thresholds, fixed_thresholds):
    """
    Trace la densité de N(0,1) et superpose les seuils issus des méthodes adaptative et fixe.
    """
    x = np.linspace(-10, 10, 1000)
    y = norm.pdf(x)
    fig = plt.figure(figsize=(10,6))
    plt.plot(x, y, label="N(0,1)")
    for i, L in enumerate(adaptive_thresholds):
        plt.axvline(x=-L, color='blue', linestyle='--', label="Seuil adaptatif" if i==0 else "")
    for i, L in enumerate(fixed_thresholds):
        plt.axvline(x=-L, color='red', linestyle='-.', label="Seuil fixe" if i==0 else "")
    plt.xlabel("x")
    plt.ylabel("Densité")
    plt.title("Visualisation des seuils sur la loi gaussienne initiale")
    plt.legend()
    plt.grid(True)
    save_figure(fig, "plot_thresholds_on_gaussian.png")

def plot_variances_boxplot(var_data):
    """
    Affiche un boxplot comparant les estimateurs des trois méthodes avec la probabilité théorique.
    """
    naive_estimates = var_data["naive_estimates"]
    adaptive_estimates = var_data["adaptive_estimates"]
    fixed_estimates = var_data["fixed_estimates"]
    time_budget = var_data.get("time_budget", 60.0)
    L_target = var_data.get("L_target", "N/A")

    # Calcul de la probabilité théorique pour L_target
    theoretical_prob = norm.cdf(-L_target)

    fig = plt.figure(figsize=(8,6))
    plt.boxplot([naive_estimates, adaptive_estimates, fixed_estimates],
                labels=["MC naïf", "SMC adaptatif", "SMC fixe"])
    plt.axhline(theoretical_prob, color='green', linestyle='--', label=f"P(L_target={L_target:.2f}) théorique")
    plt.ylabel("Probabilité estimée")
    plt.title(f"Comparaison des estimateurs (budget = {time_budget:.2f}s, L_target={L_target})")
    plt.legend()
    plt.grid(True)
    save_figure(fig, "compare_variances_boxplot.png")

def plot_relative_errors_boxplot(var_data):
    """
    Affiche un boxplot des erreurs relatives comparant les trois méthodes.
    """
    naive_estimates = var_data["naive_estimates"]
    adaptive_estimates = var_data["adaptive_estimates"]
    fixed_estimates = var_data["fixed_estimates"]
    time_budget = var_data.get("time_budget", 60.0)
    L_target = var_data.get("L_target", "N/A")

    # Calcul de la probabilité théorique pour L_target
    theoretical_prob = norm.cdf(-L_target)

    # Calcul des erreurs relatives
    naive_rel_errors = [abs(est - theoretical_prob) / theoretical_prob for est in naive_estimates]
    adaptive_rel_errors = [abs(est - theoretical_prob) / theoretical_prob for est in adaptive_estimates]
    fixed_rel_errors = [abs(est - theoretical_prob) / theoretical_prob for est in fixed_estimates]

    fig = plt.figure(figsize=(8,6))
    plt.boxplot([naive_rel_errors, adaptive_rel_errors, fixed_rel_errors],
                labels=["MC naïf", "SMC adaptatif", "SMC fixe"])
    plt.ylabel("Erreur relative")
    plt.title(f"Comparaison des erreurs relatives (budget = {time_budget:.2f}s, L_target={L_target})")
    plt.grid(True)
    save_figure(fig, "compare_relative_errors_boxplot.png")




# ---------------- Simulation pour la comparaison classique ----------------

def simulate_methods_graph_all():
    """
    Pour une série de valeurs de L, exécute les simulations pour chaque méthode 
    (MC naïf, SMC adaptatif et SMC fixe) et renvoie un dictionnaire contenant les résultats.
    """
    L_values = np.linspace(2.0, 7.5, 10)
    naive_rel_errors = []
    smc_rel_errors = []
    fixed_rel_errors = []
    theoretical_probs = []

    time_budget = 60.0  # 1 minute pour chaque méthode
    N = 5000
    p0 = 0.5
    n_mcmc = 40
    sigma = 2.0
    max_iter = 50
    fixed_num_levels = 10

    for L in L_values:
        P_th = norm.cdf(-L)
        theoretical_probs.append(P_th)

        naive_prob, _, _, _ = run_naive_mc(L, time_budget, batch_size=10**6)
        err_naive = abs(naive_prob - P_th) / P_th if P_th != 0 else np.nan
        naive_rel_errors.append(err_naive)

        adaptive_results = run_smc(L, time_budget, N=N, p0=p0, n_mcmc=n_mcmc, sigma=sigma, max_iter=max_iter, verbose=False)
        adaptive_prob = adaptive_results["avg_estimate"]
        err_adaptive = abs(adaptive_prob - P_th) / P_th if P_th != 0 else np.nan
        smc_rel_errors.append(err_adaptive)

        fixed_results = run_fixed_smc(L, time_budget, N=N, fixed_num_levels=fixed_num_levels, n_mcmc=n_mcmc, sigma=sigma, verbose=False)
        fixed_prob = fixed_results["avg_estimate"]
        err_fixed = abs(fixed_prob - P_th) / P_th if P_th != 0 else np.nan
        fixed_rel_errors.append(err_fixed)

    return {"L_values": L_values,
            "naive_rel_errors": naive_rel_errors,
            "smc_rel_errors": smc_rel_errors,
            "fixed_rel_errors": fixed_rel_errors,
            "theoretical_probs": theoretical_probs}




# ---------------- Main ----------------

def main():
    start_time = datetime.datetime.now()
    print(f"Début de l'analyse : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Définition des hyperparamètres
    time_budget = 60.0
    N = 5000
    p0 = 0.5
    n_mcmc = 40
    sigma = 2.0
    max_iter = 50
    fixed_num_levels = 10
    L_target_demo = 5.0

    # Construction du nom de fichier de résultats incluant les hyperparamètres
    #results_filename = f"simulation_results_N{N}_p0{p0}_nMCMC{n_mcmc}_sigma{sigma}_fixedLevels{fixed_num_levels}_L{L_target_demo}_time{time_budget}.pkl"
    results_filename = f"simulation_results.pkl"
    results_file = RESULTS_DIR / results_filename

    # Si le fichier existe déjà, demander si l'utilisateur souhaite relancer la simulation
    if results_file.exists():
        choice = input("Les résultats de simulation existent déjà. Voulez-vous relancer la simulation ? (o/n) : ")
        redo_simulation = (choice.lower() == 'o')
    else:
        redo_simulation = True

    if redo_simulation:
        sim_results = {}
        print("\n\n>>> Début de la simulation de la comparaison des méthodes...")
        sim_results["methods_graph"] = simulate_methods_graph_all()
        print(">>> Simulation de la comparaison des méthodes terminée.")

        print(">>> Début de la simulation des variances...")
        sim_results["variances"] = simulate_variances_fixed_time(L_target_demo, N_runs=30, time_budget=time_budget,
                                                                N=N, p0=p0, n_mcmc=n_mcmc, sigma=sigma, fixed_num_levels=fixed_num_levels)
        print(">>> Simulation des variances terminée.")

        with open(results_file, "wb") as f:
            pickle.dump(sim_results, f)
    else:
        with open(results_file, "rb") as f:
            sim_results = pickle.load(f)

    print(">>> Début du tracé des graphiques...")
    # Tracé de la comparaison classique des méthodes et des seuils
    plot_methods_graph_all(sim_results["methods_graph"])
    result_adapt = adaptive_smc_run(N=N, p0=p0, L_target=L_target_demo, n_mcmc=n_mcmc, sigma=sigma,
                                    max_iter=max_iter, verbose=False)
    adaptive_thresholds = result_adapt[1]
    fixed_thresholds = np.linspace(0, L_target_demo, fixed_num_levels)
    plot_thresholds_on_gaussian(L_target_demo, adaptive_thresholds, fixed_thresholds)

    # Tracé de la comparaison des variances et des erreurs relatives
    var_data = sim_results["variances"]
    var_data["L_target"] = L_target_demo
    plot_variances_boxplot(var_data)
    plot_relative_errors_boxplot(var_data)
    print(">>> Tracé des graphiques terminé.")

    print(">>> Simulation terminée.")

if __name__ == '__main__':
    main()
