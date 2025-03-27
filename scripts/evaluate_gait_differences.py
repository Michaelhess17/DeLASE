from delase.delase_phase import DeLASEPhaser
from delase import DeLASE
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.interpolate import make_smoothing_spline
import seaborn as sns
import copy

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

dt_orig = 1/100  # Original time step
dt_new_values = [1/x for x in np.arange(100, 2500, 400)]
matrix_sizes = np.arange(10, 500, 40)
ranks = np.arange(10, 500, 40)

ts_orig = np.linspace(0, num_timesteps*dt_orig, num_timesteps)

# Create storage for results
results = {
    'dt_new': [],
    'matrix_size': [],
    'rank': [],
    'subject': [],
    'stability_params': [],
    'stability_freqs': []
}

# Calculate valid combinations (excluding cases where rank > matrix_size)
valid_combinations = 0
for matrix_size in matrix_sizes:
    for rank in ranks:
        if rank <= matrix_size:
            valid_combinations += 1

# Main loop over parameters
total_iterations = len(dt_new_values) * valid_combinations # * data.shape[0]
progress_bar = tqdm(total=total_iterations, desc="Processing parameters")

for dt in dt_new_values:
    ts_new = np.linspace(0, num_timesteps*dt_orig, int(num_timesteps*dt_orig/dt))
    xdata = np.zeros((data.shape[0], len(ts_new), data.shape[2]))
    
    for subject in range(num_subjects):
        xdata_orig = data[subject, :, :]
        for feat in range(xdata_orig.shape[1]):
            interp = make_smoothing_spline(ts_orig, xdata_orig[:, feat], lam=0.00001)
            xdata[subject, :, feat] = interp(ts_new)
    
    for matrix_size in matrix_sizes:
        for rank in ranks:
            if rank > matrix_size:
                continue  # Skip invalid combinations without updating progress bar
                
            # for subject in range(data.shape[0]):
            subject = 0
            subject_data = xdata[subject]
            
            delase = DeLASE(subject_data,
                        n_delays=None,
                        matrix_size=int(matrix_size),
                        delay_interval=1,
                        rank=int(rank),
                        rank_thresh=None,
                        rank_explained_variance=None,
                        lamb=0,
                        dt=dt,
                        N_time_bins=None,
                        max_freq=None,
                        max_unstable_freq=None,
                        device=torch.device("cuda"),
                        verbose=False)

            delase.fit()
            params = delase.stability_params.cpu().numpy()
            freqs = delase.stability_freqs.cpu().numpy()
            
            results['dt_new'].append(dt)
            results['matrix_size'].append(matrix_size)
            results['rank'].append(rank)
            results['subject'].append(subject)
            results['stability_params'].append(params)
            results['stability_freqs'].append(freqs)
            
            progress_bar.update(1)

progress_bar.close()

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Save results
np.save("outputs/parameter_sweep_results.npy", results)

# Plotting
plt.figure(figsize=(20, 15))

# Plot 1: Matrix Size vs Rank for different dt_new
plt.subplot(2, 2, 1)
for dt_idx, dt in enumerate(dt_new_values):
    df_subset = df_results[df_results['dt_new'] == dt]
    mean_stability = df_subset.groupby(['matrix_size', 'rank'])['stability_params'].apply(
        lambda x: np.mean([np.max(params) for params in x])
    ).reset_index()
    
    plt.scatter(mean_stability['matrix_size'], mean_stability['rank'],
                c=np.log(mean_stability['stability_params']), s=100,
                label=f'dt={dt:.4f}')

plt.colorbar(label='log(Mean Max Stability)')
plt.xlabel('Matrix Size')
plt.ylabel('Rank')
plt.title('Stability vs Matrix Size and Rank')
plt.legend()

# Plot 2: Heatmap of average stability for each dt_new
plt.subplot(2, 2, 2)
pivot_data = df_results.groupby(['rank'])['stability_params'].apply(
    lambda x: np.mean([np.max(params) for params in x])
).reset_index()
plt.plot(pivot_data['rank'], pivot_data['stability_params'], 'o-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('rank')
plt.ylabel('Mean Max Stability')
plt.title('Average Stability vs rank')

# Plot 3: Distribution of stability parameters
plt.subplot(2, 2, 3)
all_max_stabilities = [np.log(np.max(params)) for params in df_results['stability_params']]
sns.histplot(all_max_stabilities)
plt.xlabel('log(Max Stability Parameter)')
plt.ylabel('Count')
plt.title('Distribution of Maximum Stability Parameters')

# Plot 4: 3D scatter plot
ax = plt.subplot(2, 2, 4, projection='3d')
scatter = ax.scatter(df_results['dt_new'],
                    df_results['matrix_size'],
                    df_results['rank'],
                    c=[np.log(np.max(params)) for params in df_results['stability_params']],
                    cmap='viridis')
ax.set_xlabel('dt_new')
ax.set_ylabel('Matrix Size')
ax.set_zlabel('Rank')
plt.colorbar(scatter, label='log(Max Stability)')

plt.tight_layout()
plt.savefig("figures/parameter_sweep_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# Additional analysis: Create separate plots for each dt value
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.ravel()

for idx, dt in enumerate(dt_new_values[:4]):
    df_subset = copy.deepcopy(df_results[df_results['dt_new'] == dt])
    df_subset['stability_max_abs'] = df_subset['stability_params'].apply(lambda x: np.max(x))
    
    pivot_table = pd.pivot_table(df_subset, values='stability_max_abs', index=['matrix_size'], columns=['rank'])
    
    sns.heatmap(pivot_table, ax=axes[idx], cmap='viridis')
    axes[idx].set_title(f'dt_new = {dt:.4f}')
    axes[idx].set_xlabel('Rank')
    axes[idx].set_ylabel('Matrix Size')

plt.tight_layout()
plt.savefig("figures/parameter_sweep_heatmaps.png", dpi=300, bbox_inches='tight')
plt.close()
