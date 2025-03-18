from delase.delase_phase import DeLASEPhaser
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

gait_phase = np.loadtxt("data/gaitphase_label.csv", delimiter=",")
speeds = np.loadtxt("data/speeds_label.csv", delimiter=",")
subject_num = np.loadtxt("data/subject_num_label.csv", delimiter=",")
subjects = pd.read_csv("data/subjects_label.csv", header=None)
trial_type = np.loadtxt("data/trialtype_label.csv", delimiter=",")

data = np.loadtxt("data/DataStorage_6000_2D_kinematics_clean.csv", delimiter=",")
num_subjects = 150
num_timesteps = 6000
data = data.reshape(data.shape[0], num_subjects, num_timesteps).swapaxes(0, 1).swapaxes(1, 2)
data -= data.mean(axis=1, keepdims=True)  # Center the data
data /= data.std(axis=1, keepdims=True)  # Normalize the data
all_all_params = np.zeros(data.shape[0], dtype=object)
for subject in tqdm(range(data.shape[0])):
    subject_data = data[subject]
    dt = 1/100
    ts = np.linspace(0, num_timesteps*dt, num_timesteps)

    delase = DeLASEPhaser(subject_data,
                    n_delays=10,
                    matrix_size=None,
                    delay_interval=1,
                    rank=10,
                    rank_thresh=None,
                    rank_explained_variance=None,
                    lamb=0,
                    dt=ts[1]-ts[0],
                    N_time_bins=None,
                    max_freq=None,
                    max_unstable_freq=None,
                    device=torch.device("cuda"),
                    n_segments=101,
                    verbose=False)

    steps = np.arange(start=0, stop=101, step=100)
    all_params, all_freqs = np.zeros(len(steps) - 1, dtype=object), np.zeros(len(steps) - 1, dtype=object)
    for idx, (start, stop) in enumerate(zip(steps[:-1], steps[1:])):
        params, freqs = delase.fit_phase_range(phase_start=start, phase_end=stop)
        all_params[idx] = params.cpu()
        all_freqs[idx] = freqs.cpu()
    all_all_params[subject] = all_params

all_all_params = np.stack(np.stack(all_all_params).squeeze())

# Plot the eigenspectra for each subject
for ii in range(all_all_params.shape[0]):
    plt.scatter(range(all_all_params[ii].shape[0]), all_all_params[ii], label=f"Subject {ii+1}", alpha=0.5, marker="o")

plt.savefig("figures/tmp.png")