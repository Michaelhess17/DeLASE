from delase import DeLASE
import numpy as np
import torch

data = np.loadtxt("data/DataStorage_6000_2D_kinematics_clean.csv", delimiter=",")

num_subjects = 150
num_timesteps = 6000
data = data.reshape(data.shape[0], num_subjects, num_timesteps).swapaxes(0, 1).swapaxes(1, 2)
all_all_params = np.zeros(data.shape[0], dtype=object)
for subject in range(data.shape[0]):
    subject_data = data[subject]
    dt = 1/100
    ts = np.linspace(0, num_timesteps*dt, num_timesteps)

    delase = DeLASE(subject_data,
                    n_delays=None,
                    matrix_size=50,
                    delay_interval=1,
                    rank=50,
                    rank_thresh=None,
                    rank_explained_variance=None,
                    lamb=0,
                    dt=ts[1]-ts[0],
                    N_time_bins=None,
                    max_freq=None,
                    max_unstable_freq=None,
                    device=torch.device("cuda"),
                    verbose=False)

