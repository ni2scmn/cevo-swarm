import os
import csv
import re
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_fitness_over_generations(root_dir, id, invalid_value=-10000.0, plot=True):
    generation_stats = {}

    # Regex to extract generation number and match on id
    pattern = re.compile(rf"{id}_train_(\d+)$")
#    pattern = re.compile(r".+_train_(\d+)$")

    for folder_name in os.listdir(root_dir):
        match = pattern.match(folder_name)
        if match:
            generation = int(match.group(1))
            fitness_path = os.path.join(root_dir, folder_name, "fitness.csv")
            if os.path.isfile(fitness_path):
                with open(fitness_path, newline='') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    values = [float(row[0]) for row in reader if float(row[0]) != invalid_value]

                    if values:
                        generation_stats[generation] = {
                            "max_fitness": max(values),
                            "avg_fitness": sum(values) / len(values)
                        }
                    else:
                        generation_stats[generation] = {
                            "max_fitness": None,
                            "avg_fitness": None
                        }

    # Sort by generation
    generation_stats = dict(sorted(generation_stats.items()))

    if plot:
        generations = list(generation_stats.keys())
        max_values = [generation_stats[gen]["max_fitness"] for gen in generations]
        avg_values = [generation_stats[gen]["avg_fitness"] for gen in generations]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, max_values, label="Max Fitness", marker='o')
        plt.plot(generations, avg_values, label="Avg Fitness", marker='x')
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Over Generations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return generation_stats

# Example usage:
if __name__ == "__main__":
    # Replace with the path to your runs directory
    stats = analyze_fitness_over_generations("data/pretrain","1748799218")
    print(stats)