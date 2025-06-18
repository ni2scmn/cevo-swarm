import csv
import ast
from math import exp
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
import re
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm


def read_weights_from_csv(file_path):
    weight_sets = []
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        reader.__next__()  # Skip the header row
        for row in reader:
            ts_weights = []
            for weight_str in row:
                try:
                    # Try to parse the string as a Python list
                    weights = ast.literal_eval(weight_str)
                    if isinstance(weights, list):
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
    distances = pdist(population, metric='cosine')
    return np.mean(distances)

def plot_diversity_over_timesteps(data):
    """
    Plots the average pairwise distance for each timestep.

    Args:
        data (list): A list of tuples containing timestep and population.
    """
    avg_distances = []
    timesteps = []

    for timestep, population in data:
        avg_distance = diversity_pairwise_distance(np.array(population))
        avg_distances.append(avg_distance)
        timesteps.append(timestep)

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, avg_distances, marker='o')
    plt.title('Average Pairwise Distance Over Timesteps')
    plt.xlabel('Timestep')
    plt.ylabel('Average Pairwise Distance')
    plt.grid()
    plt.show()


def plot_diversity_over_timesteps_multi(runs_data):
    """
    Plots the average and stddev of pairwise distances over timesteps for multiple runs.

    Args:
        runs_data (list): List of lists, each inner list is [(timestep, population), ...] for a run.
    """
    # Find the minimum length of runs to align timesteps
    min_len = min(len(run) for run in runs_data)
    all_distances = []

    # Compute diversity for each run and timestep
    for ridx, run in enumerate(runs_data):
        run_distances = []
        for timestep, population in run[:min_len]:
            avg_distance = diversity_pairwise_distance(np.array(population))
            run_distances.append(avg_distance)
        all_distances.append(run_distances)
        # print(f"Run {ridx}, Avg Distance = {run_distances}")

    all_distances = np.array(all_distances)  # shape: (n_runs, n_timesteps)
    avg_distances = np.mean(all_distances, axis=0)
    std_distances = np.std(all_distances, axis=0)
    timesteps = list(range(min_len))

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, avg_distances, marker='o', label='Average Diversity')
    plt.fill_between(timesteps, avg_distances - std_distances, avg_distances + std_distances, alpha=0.3, label='Std Dev')
    plt.title('Average Pairwise Distance Over Timesteps (Multiple Runs)')
    plt.xlabel('Timestep')
    plt.ylabel('Average Pairwise Distance')
    plt.grid()
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # Find all nn_weights_*.csv files in the directory
    run_dir = "data/e_nn_1/1750244174"
    run_files = sorted([f for f in os.listdir(run_dir) if re.match(r"nn_weights_\d+\.csv", f)])
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