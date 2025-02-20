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
from ddfa_node import embed_data, takens_embedding, change_trial_length, split_data, phaser, stats as statistics, jax_utils
from ddfa_node.stability.delase_utils import get_aics, get_λs
import delase


project_path = pathlib.Path("/mnt/Mouse_Face_Project/Desktop/Data/Python/delase/")
window_length = 30

# Create a function to load the data
def load_data(data_path="outputs/VDP_oscillators.npy", dt=1/20):
    data = jnp.load(project_path.joinpath(data_path))[:, :, ::2]
    data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])

    # Apply convolution to all trials and features at once
    new_data = jax_utils.convolve_trials(data[:10, :2**12, :1])

    # Standardize the data
    new_data = (new_data - jnp.mean(new_data, axis=1)[:, None, :]) / jnp.std(new_data, axis=1)[:, None, :]
    return new_data

# Load the data
data = load_data()
print(f"Data shape: {data.shape}")

# Set up the parameters
n_delays = 1
matrix_size = 10
delay_interval = 1
rank = None
rank_thresh = None
rank_explained_variance = None
lamb=0
dt = 0.004
N_time_bins = 50
max_freq=(1/dt)//2
max_unstable_freq=(1/dt)//2
device = torch.device("cuda")
verbose = True
data = torch.from_numpy(np.array(data)).to(device)

# Get the AICs
# matrix_sizes = np.array([10, 20, 50, 100, 200, 300, 500, 750, 1000])
# ranks = np.array([3, 5, 10, 25, 50, 75, 100, 125, 150, 200, *range(250, 801, 50), 900, 1000])
matrix_sizes = np.array([5, 10, 20, 50, 100, 200, 500, 1000])
ranks = np.array([3, 5, 10, 20, 50, 100, 200, 500, 1000])

aics = get_aics(data, matrix_sizes, ranks, dt=dt, max_freq=max_freq, max_unstable_freq=max_unstable_freq, device=device, delay_interval=delay_interval, N_time_bins=N_time_bins)

# Get the λs
dataset_size = data.shape[1]
n_splits = 4

full_output = True
top_percent = 10
if full_output:
    top_percent = None

trial_lens = np.logspace(8, 8+n_splits-1, n_splits, base=2).astype(int)
# skip = trial_lens // 2
skip = 50*np.ones_like(trial_lens)
all_λs = np.empty((len(trial_lens), data.shape[0]), dtype=object)
for idx, trial_len in enumerate(trial_lens):
    print(f"Trial Length: {trial_len}")
    λs = get_λs(data, aics, matrix_sizes, ranks, full_output=full_output,
           top_percent=top_percent, dt=dt, max_freq=None, max_unstable_freq=None,
           device=torch.device("cuda"), trial_len=trial_len, skip=skip[idx],  n_delays=None,
                                delay_interval=1, N_time_bins=50)
    for jdx in range(data.shape[0]):
        all_λs[idx, jdx] = λs[jdx]

λs = np.zeros((all_λs.shape[0], all_λs.shape[1]), dtype=object)
for idx in range(all_λs.shape[0]):
    for jdx in range(all_λs.shape[1]):
        λs[idx, jdx] = np.stack(delase.filter_λs(all_λs[idx, jdx], delase.get_shape_mode(all_λs[idx, jdx]).mode))

# Plot an example of the λs scaling
one_over_n_splits = (1 / np.arange(2, n_splits+2, 1)) / dataset_size
results = delase.analyze_lambda_convergence(λs, one_over_n_splits, eig=0, max_splits=4, max_poly_degree=3)
delase.plot_lambda_convergence(λs, one_over_n_splits, results, eig=0, max_splits=4, subject=0)
