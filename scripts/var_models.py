import statsmodels
from statsmodels.tsa.api import VAR
import numpy as np
import matplotlib.pyplot as plt
import delase

data = np.loadtxt("data/DataStorage_6000_2D_kinematics_clean.csv", delimiter=",")
speeds = np.loadtxt("data/speeds_label.csv", delimiter=",")

num_subjects = 150
num_timesteps = 6000
data = data.reshape(data.shape[0], num_subjects, num_timesteps).swapaxes(0, 1).swapaxes(1, 2)

results = {}
for trial in range(data.shape[0]):
    model = VAR(data[trial, :, :500])
    res = model.fit(maxlags=100, ic='bic', verbose=True)
    # Eigenvalues
    A = res.coefs
    companion_matrix = statsmodels.tsa.vector_ar.util.comp_matrix(A)

    eigvals_comp = np.linalg.eigvals(companion_matrix)
    
    results[trial] = eigvals_comp

all_eigenvalues = np.concatenate([np.array(results[trial])[None, :] for trial in range(data.shape[0])], axis=1).squeeze()

c_speeds = np.concatenate([speeds[trial]*np.ones_like(results[trial], dtype=float) for trial in range(data.shape[0])]).squeeze()

delase.utils.plot_eigvals_on_unit_circle(all_eigenvalues.flatten(), c_speeds)