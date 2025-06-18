from math import exp
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
import re
import matplotlib.pyplot as plt

from tqdm import tqdm

def read_population_file(file_path):
    """
    Reads a population file and returns a list of numpy arrays.
    
    Args:
        file_path (str): Path to the population CSV file.
        
    Returns:
        list: A list of numpy arrays representing the population.
    """
    population = []
    with open(file_path, 'r') as f:
        f.readline()  # Skip the header line
        for line in f:
            # Convert each value to float and wrap in a NumPy array
            float_array = np.array([float(value) for value in line.strip().split(',')], dtype=float)
            population.append(float_array)
    return population

def read_fitness_file(file_path):
    """
    Reads a fitness file and returns a list of fitness values.

    Args:
        file_path (str): Path to the fitness CSV file.

    Returns:
        list: A list of fitness values.
    """
    fitness = []
    with open(file_path, 'r') as f:
        f.readline()  # Skip the header line
        for line in f:
            fitness.append(float(line.strip()))
    return fitness


def read_pops(base_dir, exp_id , desc=True):
    """
    Parse the population CSV file and return a list of numpy arrays.
    """

    exp_filter = re.compile(r"^" + exp_id + r"_train_(\d+)$")

    exp_dirs = [f for f in os.listdir(os.path.normpath(base_dir)) if exp_filter.match(f)]

    gens = []

    for d in tqdm(exp_dirs):
        train_data_path = os.path.join(base_dir, d)
        if not os.path.exists(train_data_path):
            print(f"Directory {train_data_path} does not exist.")
            continue
        
        # Check if population.csv and fitness.csv exist
        population_file = os.path.join(train_data_path, "population.csv")
        fitness_file = os.path.join(train_data_path, "fitness.csv")
        
        if not os.path.exists(population_file) or not os.path.exists(fitness_file):
            print(f"Required files not found in {train_data_path}. Skipping this directory.")
            continue

        # Parse gen from the path by regex
        gen_match = exp_filter.match(d)
        if gen_match:
            gen = gen_match.group(1)
        else:
            print(f"Could not parse generation from directory name {d}. Skipping.")
            continue   

        # Read population and fitness data
        population = read_population_file(population_file)
        fitness = read_fitness_file(fitness_file)

        # Sort the population based on fitness values
        sorted_indices = np.argsort(fitness)
        if desc:
            sorted_indices = sorted_indices[::-1]
        # Reverse for descending order
        sorted_fitness = [fitness[i] for i in sorted_indices]
        sorted_population = [population[i] for i in sorted_indices]

        gens.append((gen, sorted_population, sorted_fitness))

    return  gens

def diversity_pairwise_distance(population):
    """
    Computes the average pairwise Euclidean distance between all genomes in the population.
    
    Args:
        population (np.ndarray): A 2D array of shape (n_individuals, n_genes).
    
    Returns:
        float: The average pairwise distance.
    """
    distances = pdist(population, metric='cosine')
    return np.mean(distances)


def plot_diversity_over_generations(gens):
    """
    Plots the average pairwise distance for each generation.

    Args:
        gens (list): A list of tuples containing generation number, population, and fitness.
    """


    avg_distances = []
    generations = []

    for gen, population, fitness in gens:
        avg_distance = diversity_pairwise_distance(np.array(population))
        avg_distances.append(avg_distance)
        generations.append(gen)

    plt.figure(figsize=(10, 5))
    plt.plot(generations, avg_distances, marker='o')
    plt.title('Average Pairwise Distance Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Average Pairwise Distance')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    base_dir = "data/pretrain"
    exp_id = "1749941416"
    gens = read_pops(base_dir, exp_id)
    
    plot_diversity_over_generations(gens)

