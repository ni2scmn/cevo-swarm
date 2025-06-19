import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- Config ---
base_folder = sys.argv[1]
file_pattern = "social_transmission_*.csv"

# --- Define Groups ---
def get_group(agent_id):
    return 0 if agent_id <= 4 else 1

group_names = {
    (0, 0): "G1 → G1",
    (0, 1): "G1 → G2",
    (1, 0): "G2 → G1",
    (1, 1): "G2 → G2"
}

# --- Accumulation Function ---
def accumulate_transmissions(df, full_index):
    df['from_group'] = df['from'].apply(get_group)
    df['to_group'] = df['to'].apply(get_group)
    df['group_pair'] = list(zip(df['from_group'], df['to_group']))
    
    grouped = (
        df.groupby(['timestep', 'group_pair'])
        .size()
        .unstack(fill_value=0)
    )
    
    # Reindex to full timestep index to fill missing timesteps with 0
    grouped = grouped.reindex(full_index, fill_value=0)
    
    return grouped.cumsum()

# --- Read Files and Get Timestep Range ---
all_files = glob.glob(os.path.join(base_folder, file_pattern))
timestep_set = set()

# First pass: get all timesteps
for f in all_files:
    df = pd.read_csv(f, header=None, names=["timestep", "from", "to"])
    timestep_set.update(df["timestep"].unique())

full_timesteps = sorted(timestep_set)
full_index = pd.Index(full_timesteps, name="timestep")

# --- Process Files with Padding ---
accumulated_list = []
for f in all_files:
    df = pd.read_csv(f, header=None, names=["timestep", "from", "to"])
    acc = accumulate_transmissions(df, full_index)
    accumulated_list.append(acc)

# --- Normalize and Combine ---
df_all = pd.concat(accumulated_list)
df_all = df_all.groupby(level=0).sum()
df_all /= len(all_files)

# --- Fill missing group pairs with 0 ---
for pair in group_names:
    if pair not in df_all.columns:
        df_all[pair] = 0

df_all = df_all.sort_index()

# --- Plotting ---
plt.figure(figsize=(10, 6))
for pair, label in group_names.items():
    # plot with jitter for better visibility
    jitter = np.random.normal(0, 0.05, size=df_all[pair].shape)
    plt.plot(df_all.index, df_all[pair] + jitter, label=label)

plt.title("Average Accumulated Social Transmissions Over Time")
plt.xlabel("Timestep")
plt.ylabel("Cumulative Transmissions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
