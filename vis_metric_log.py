import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Configuration

ex_id = "1750002798"
metric = "x_axis"

csv_files = glob.glob(f"data/e_1/{ex_id}/{metric}*.csv")  # Update the path
column_name = "0"  # Replace with the actual column name

# Read and aggregate data
all_runs = []

for file in csv_files:
    df = pd.read_csv(file)
    metric_series = df[column_name].reset_index(drop=True)  # Ensure same length
    all_runs.append(metric_series)

# Combine into a DataFrame: each column is a run, each row is a timestep
data = pd.DataFrame(all_runs).T  # Transpose so rows = timesteps, columns = runs

# Calculate statistics
mean = data.mean(axis=1)
std = data.std(axis=1)
lower = mean - std
upper = mean + std

# Plotting
plt.figure(figsize=(12, 6))
timesteps = range(len(mean))

plt.plot(timesteps, mean, label='Mean', color='blue')
plt.fill_between(timesteps, lower, upper, alpha=0.3, label='±1 Std Dev', color='blue')

plt.title("Metric Distribution Over Timesteps")
plt.xlabel("Timestep")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
