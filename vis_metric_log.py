import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Configuration

# grab ex_id from command line or set manually
import sys
if len(sys.argv) > 1:
    ex_id = sys.argv[1]
else:
    raise ValueError("Please provide an experiment ID as a command line argument.")

if len(sys.argv) > 2:
    metric = sys.argv[2]
else:
    metric = "x_axis"

csv_files = glob.glob(f"data/{ex_id}/{metric}*.csv")  # Update the path
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

plt.ylim(0, 1)

plt.legend()
plt.grid(True)
plt.tight_layout()
fig = plt.gcf()
fig.savefig(f"{ex_id}_{metric}_distribution.png".replace("/", "_"), dpi=300, bbox_inches='tight')
plt.show()
