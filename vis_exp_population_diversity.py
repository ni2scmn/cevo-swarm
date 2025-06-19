import csv
import ast
from math import exp
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import os
import re
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm
import sys


def read_weights_from_csv(file_path):
    weight_sets = []
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        reader.__next__()  # Skip the header row
        for row in reader:
            ts_weights = []
            for weight_str in row:
                try:
                    weights = np.fromstring(weight_str.strip('[]'), sep=',')
                    if isinstance(weights, list) or isinstance(weights, np.ndarray):
                        ts_weights.append(weights)
                except (ValueError, SyntaxError):
                    continue  # Skip header or malformed entries
            
            if ts_weights:
                # Convert the list of lists to a numpy array
                ts_weights = np.array(ts_weights, dtype=float)
                weight_sets.append(ts_weights)
    return weight_sets

    return  gens

def diversity_pairwise_distance(population):
    """
    Computes the average pairwise Euclidean distance between all genomes in the population.
    
    Args:
        population (np.ndarray): A 2D array of shape (n_individuals, n_genes).
    
    Returns:
        float: The average pairwise distance.
    """
    # print(population.shape)

    pop_c1_size = int(population.shape[0] * fraction_c1)


    distances_ic = cdist(population[0:pop_c1_size], population[pop_c1_size:10], metric='cosine')
    distances_c1 = pdist(population[0:pop_c1_size], metric='cosine')
    distances_c2 = pdist(population[pop_c1_size:10], metric='cosine')
    distances = pdist(population, metric='cosine')
    #print(f"ic: {np.mean(distances_ic)}, c1: {np.mean(distances_c1)}, c2: {np.mean(distances_c2)}, all: {np.mean(distances)}")
    return (np.mean(distances_ic), np.mean(distances_c1), np.mean(distances_c2), np.mean(distances))

def plot_diversity_over_timesteps_multi(runs_data):
    """
    Plots the average and stddev of pairwise distances over timesteps for multiple runs.

    Args:
        runs_data (list): List of lists, each inner list is [(timestep, population), ...] for a run.
    """
    # Find the minimum length of runs to align timesteps
    min_len = min(len(run) for run in runs_data)
    all_distances_ic = []
    all_distances_c1 = []
    all_distances_c2 = []
    all_distances = []

    # Compute diversity for each run and timestep
    for ridx, run in enumerate(runs_data):
        run_distances_ic = []
        run_distances_c1 = []
        run_distances_c2 = []
        run_distances = []
        for timestep, population in run[:min_len]:
            avg_distance_ic, avg_distance_c1, avg_distance_2, avg_distance = diversity_pairwise_distance(np.array(population))
            run_distances_ic.append(avg_distance_ic)
            run_distances_c1.append(avg_distance_c1)
            run_distances_c2.append(avg_distance_2)
            run_distances.append(avg_distance)
        all_distances_ic.append(run_distances_ic)
        all_distances_c1.append(run_distances_c1)
        all_distances_c2.append(run_distances_c2)
        all_distances.append(run_distances)
        # print(f"Run {ridx}, Avg Distance = {run_distances}")

    all_distances_ic = np.array(all_distances_ic)  # shape: (n_runs, n_timesteps)
    all_distances_c1 = np.array(all_distances_c1)  # shape: (n_runs, n_timesteps)
    all_distances_c2 = np.array(all_distances_c2)  # shape:
    all_distances = np.array(all_distances)  # shape: (n_runs, n_timesteps)
    avg_distances_ic = np.mean(all_distances_ic, axis=0)
    avg_distances_c1 = np.mean(all_distances_c1, axis=0)
    std_distances_c1 = np.std(all_distances_c1, axis=0)
    avg_distances = np.mean(all_distances, axis=0)
    std_distances_ic = np.std(all_distances_ic, axis=0)
    std_distances_c1 = np.std(all_distances_c1, axis=0)
    avg_distances_c2 = np.mean(all_distances_c2, axis=0)
    std_distances = np.std(all_distances, axis=0)
    timesteps = list(range(min_len))
    timesteps = [t * 10 for t in timesteps]  # Adjust to capture the 10x timesteps

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, avg_distances_ic, marker='o', label='IC Average Diversity')
    plt.plot(timesteps, avg_distances_c1, marker='o', label='C1 Average Diversity')
    plt.plot(timesteps, avg_distances_c2, marker='o', label='C2 Average Diversity')
    plt.plot(timesteps, avg_distances, marker='o', label='Average Diversity')
    plt.fill_between(timesteps, avg_distances_c1 - std_distances_c1, avg_distances_c1 + std_distances_c1, alpha=0.3, label='C1 Std Dev')
    plt.fill_between(timesteps, avg_distances_c2 - std_distances_c1, avg_distances_c2 + std_distances_c1, alpha=0.3, label='C2 Std Dev')
    plt.fill_between(timesteps, avg_distances - std_distances, avg_distances + std_distances, alpha=0.3, label='Std Dev')
    plt.title('Average Pairwise Distance Over Timesteps (Multiple Runs)')
    plt.xlabel('Timestep')
    plt.ylabel('Average Pairwise Distance')
    plt.grid()
    plt.legend()
    #plt.show()
    plt.savefig("diversity_over_timesteps_multi.png")


if __name__ == "__main__":
    # Find all nn_weights_*.csv files in the directory
    run_dir = sys.argv[1]
    fraction_c1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    run_files = sorted([f for f in os.listdir(run_dir) if re.match(r"nn_weights_\d+\.csv", f)])
    print("WARINING: SELECTING FIRST 6 FILES FOR PERFORMANCE")
    run_files = run_files[:6]  # Limit to first  6 files for performance
    runs_data = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(read_weights_from_csv, [os.path.join(run_dir, fname) for fname in run_files]),
                total=len(run_files),
                desc="Loading weight files",
            )
        )

    for idx, weight in enumerate(results):
        # weights = read_weights_from_csv(os.path.join(run_dir, fname))
        runs_data.append([(i, w) for i, w in enumerate(weight)])
        print(f"Loaded {len(weight)} weight sets from {run_files[idx]}")

    plot_diversity_over_timesteps_multi(runs_data)