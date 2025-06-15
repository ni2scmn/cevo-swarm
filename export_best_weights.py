import numpy as np

def sort_best(train_data_path, desc=True):
    """
    Parse the population CSV file and return a list of numpy arrays.
    """
    population = []
    with open(train_data_path.strip("/") + "/population.csv", 'r') as f:
        f.readline()  # Skip the header line
        for line in f:
            # Convert each value to float and wrap in a NumPy array
            float_array = np.array([float(value) for value in line.strip().split(',')], dtype=float)
            population.append(float_array)

    with open(train_data_path.strip("/") + "/fitness.csv", 'r') as f:
        f.readline() # Skip the header line
        fitness = [float(line.strip()) for line in f]
    
    # Sort the population based on fitness values
    sorted_indices = np.argsort(fitness)
    if desc:
        sorted_indices = sorted_indices[::-1]  # Reverse for descending order
    
    sorted_fitness = [fitness[i] for i in sorted_indices]
    sorted_population = [population[i] for i in sorted_indices]
    
    return (sorted_population, sorted_fitness)


if __name__ == "__main__":

    ex_id = "1749922427"
    gen = "99"

    p, f = sort_best("../data/{}/{}_train_{}".format(ex_id, ex_id, gen))
    print("Best fitness:", f[0:10])

    with open("best_weights.txt", "w") as file:
        for idx in range(min(10, len(p))):
            file.write(str(f[idx]) + ": [" + ", ".join([str(w) for w in p[idx]]) + "]\n")
    print("Best weights saved to best_weights.txt")