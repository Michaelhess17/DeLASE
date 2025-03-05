# System imports
import os
import pathlib
from tqdm.auto import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Numerical processing imports
import matplotlib

matplotlib.use("kitcat")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")
import numpy as np
import jax.numpy as jnp
import pandas as pd
from scipy import signal, interpolate, stats
import torch

# Local imports
from ddfa_node import (
    embed_data,
    takens_embedding,
    change_trial_length,
    split_data,
    phaser,
    stats as statistics,
    jax_utils,
)
from ddfa_node.stability.delase_utils import get_aics, get_λs
import delase


project_path = pathlib.Path("/mnt/Mouse_Face_Project/Desktop/Data/Python/delase/")
window_length = 30


# Create a function to load the data
def load_data(data_path="outputs/VDP_oscillators.npy", dt=1 / 20):
    data = jnp.load(project_path.joinpath(data_path))[:, :, :]
    data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3])

    # Apply convolution to all trials and features at once
    # new_data = jax_utils.convolve_trials(data[:, 5000:, 1:])

    new_data = data[:, 5000:10000, 1:]
    # Standardize the data
    new_data = (new_data - jnp.mean(new_data, axis=1)[:, None, :]) / jnp.std(
        new_data, axis=1
    )[:, None, :]
    return new_data


# Load the data
data = load_data()
print(f"Data shape: {data.shape}")

# Set up the parameters
n_delays = None
matrix_size = 500
delay_interval = 1
rank = None
rank_thresh = None
rank_explained_variance = None
lamb = 0
dt = 0.02
N_time_bins = None
max_freq = (1 / dt) // 2
max_unstable_freq = (1 / dt) // 2
device = torch.device("cuda")
verbose = True
data = torch.from_numpy(np.array(data)).to(device)

# Get the AICs
matrix_sizes = np.array([50, 200, 500])
ranks = np.array([10, 25, 50, 100, 500])
# matrix_sizes = np.array([10, 20, 50, 100, 200, 300, 500, 750, 1000])
# ranks = np.array([3, 5, 10, 25, 50, 75, 100, 125, 150, 200, *range(250, 801, 50), 900, 1000])
# matrix_sizes = np.array([50])
# ranks = np.array([50])

aics = get_aics(
    data,
    matrix_sizes,
    ranks,
    dt=dt,
    max_freq=max_freq,
    max_unstable_freq=max_unstable_freq,
    device=device,
    delay_interval=delay_interval,
    N_time_bins=N_time_bins,
)
# Get the λs
dataset_size = data.shape[1]
n_splits = 4

full_output = True
top_percent = 10
if full_output:
    top_percent = None

trial_lens = np.logspace(9.3, 11.3, n_splits, base=2).astype(int)
# skip = trial_lens // 3
skip_values = np.array([20, 100, 500])  # Add different skip values to test

# Create storage for r² values
r_squared_values = np.zeros((
    len(skip_values),
    len(matrix_sizes),
    len(ranks),
    data.shape[0]  # number of subjects/trials
))

# Test each combination
for skip_idx, skip_val in enumerate(skip_values):
    skip = skip_val * np.ones_like(trial_lens)
    all_λs = np.empty((len(trial_lens), *data.shape[0:1], len(matrix_sizes), len(ranks)), dtype=object)
    
    for idx, trial_len in enumerate(trial_lens):
        print(f"Trial Length: {trial_len}, Skip: {skip_val}")
        λs = get_λs(
            data,
            aics,
            matrix_sizes,
            ranks,
            full_output=full_output,
            top_percent=top_percent,
            dt=dt,
            max_freq=None,
            max_unstable_freq=None,
            device=device,
            trial_len=trial_len,
            skip=skip[idx],
            n_delays=None,
            delay_interval=1,
            N_time_bins=None,
        )
        all_λs[idx] = λs

    # Process λs for each parameter combination
    one_over_n_splits = (1 / np.arange(2, n_splits + 2, 1)) / dataset_size
    
    for matrix_idx in range(len(matrix_sizes)):
        for rank_idx in range(len(ranks)):
            for subject in range(data.shape[0]):
                # Extract λs for this combination
                λs_subset = np.zeros((all_λs.shape[0], all_λs.shape[1]), dtype=object)
                for i in range(all_λs.shape[0]):
                    λs_subset[i] = all_λs[i, subject, matrix_idx, rank_idx]
                
                # Filter λs
                filtered_λs = np.zeros_like(λs_subset)
                for i in range(λs_subset.shape[0]):
                    for j in range(λs_subset.shape[1]):
                        filtered_λs[i, j] = np.stack(
                            delase.filter_λs(
                                λs_subset[i, j],
                                delase.get_shape_mode(λs_subset[i, j]).mode
                            )
                        )
                
                # Analyze convergence and store r² value
                results = delase.analyze_lambda_convergence(
                    filtered_λs, one_over_n_splits, eig=0, max_splits=3, max_poly_degree=3
                )
                r_squared_values[skip_idx, matrix_idx, rank_idx, subject] = (
                    results['subject_0']['linear']['r_squared']
                )

# Save results
np.save(project_path.joinpath("outputs/r_squared_values.npy"), r_squared_values)

# Optional: Plot heatmap of average r² values across subjects
for skip_idx, skip_val in enumerate(skip_values):
    plt.figure(figsize=(10, 8))
    mean_r2 = np.mean(r_squared_values[skip_idx], axis=-1)  # Average across subjects
    sns.heatmap(mean_r2, xticklabels=ranks, yticklabels=matrix_sizes,
                annot=True, fmt='.3f', cmap='viridis')
    plt.title(f'Mean R² values (skip={skip_val})')
    plt.xlabel('Rank')
    plt.ylabel('Matrix Size')
    plt.savefig(project_path.joinpath(f"figures/r2_heatmap_skip_{skip_val}.png"))
    plt.close()
