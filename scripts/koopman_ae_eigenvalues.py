from koopman_ae import KoopmanAE, train_koopman_ae, evaluate_mse
import numpy as np
import torch
from scipy.interpolate import make_smoothing_spline
from delase import embed_signal_torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import seaborn as sns
from rich.progress import Progress
import pandas as pd
from ddfa_node import phaser
from scipy.signal import find_peaks


train_size = 5000
test_size = 1000
device = "cuda"

# Prepare data
data = np.loadtxt("data/DataStorage_6000_2D_kinematics_clean.csv", delimiter=",")
subject_num = np.loadtxt("data/subject_num_label.csv", delimiter=",").astype(int)
trial_type = np.loadtxt("data/trialtype_label.csv", delimiter=",")
speeds = np.loadtxt("data/speeds_label.csv", delimiter=",")

# Create a mapping from original subject numbers to sequential indices
unique_subjects = np.unique(subject_num)
subject_to_index = {subj: idx for idx, subj in enumerate(unique_subjects)}
subject_indices = np.array([subject_to_index[subj] for subj in subject_num])

# Normalize data
data = data.reshape(data.shape[0], 150, 6000).swapaxes(0, 1).swapaxes(1, 2)[:1]
data -= data.mean(axis=1, keepdims=True)
data /= data.std(axis=1, keepdims=True)

num_trials = data.shape[0]

# Interpolate data
dt_orig = 1/100
dt_new = 1/100
ts_orig = np.linspace(0, data.shape[1]*dt_orig, data.shape[1])
ts_new = np.linspace(0, data.shape[1]*dt_orig, int(data.shape[1]*dt_orig/dt_new))

new_data = np.zeros((num_trials, len(ts_new), data.shape[2]))

for subject in range(num_trials):
    for feat in range(data.shape[2]):
        interp = make_smoothing_spline(ts_orig, data[subject, :, feat], lam=0.00001)
        new_data[subject, :, feat] = interp(ts_new)

# Delay embed data
n_delays = 30
τ = 1
train_data, test_data = [], []
for subject in range(num_trials):
    out = embed_signal_torch(new_data[subject], n_delays, τ).to(device)
    train_data.append(out[:train_size][None, :])
    test_data.append(out[train_size:train_size+test_size][None, :])
train_data = torch.cat(train_data, dim=0)
test_data = torch.cat(test_data, dim=0)

# # Compute phase of signals and interpolate all to same length
# phis_train = torch.zeros((train_data.shape[0], train_data.shape[1]))
# phis_test = torch.zeros((test_data.shape[0], test_data.shape[1]))
# for a in range(data.shape[0]):
#     raw_train = train_data[a, :, ::n_delays].T.detach().cpu().numpy()
#     raw_test = test_data[a, :, ::n_delays].T.detach().cpu().numpy()

#     # Assess phase using Phaser
#     phr = phaser.Phaser([ raw_train ])
#     phi = phr.phaserEval( raw_train ) # extract phase
#     phis_train[a, :] = torch.from_numpy(phi[0, :] % 2*np.pi)

#     phr = phaser.Phaser([ raw_test ])
#     phi = phr.phaserEval( raw_test ) # extract phase
#     phis_test[a, :] = torch.from_numpy(phi[0, :] % 2*np.pi)

# samples_per_cycle = 100
# distance = 20
# height = 0.85
# height *= 2*np.pi
# new_train_data = []
# new_test_data = []
# subject_ids = []
# for subject in range(num_trials):
#     peaks_train = find_peaks(phis_train[subject].cpu().numpy(), distance=distance, height=height)[0]
#     curr_data = np.zeros((samples_per_cycle, len(peaks_train)-1, train_data.shape[2]))
#     for feat in range(train_data.shape[2]):
#         for i in range(len(peaks_train)-1):
#             start, end = peaks_train[i], peaks_train[i+1]
#             dat_train = train_data[subject, start:end, feat].cpu().numpy()
#             # interpolate from size of dat_train to samples_per_cycle
#             ts_new = np.linspace(0, len(dat_train), samples_per_cycle)
#             ts_orig = np.linspace(0, len(dat_train), len(dat_train))
#             interp = np.interp(ts_new, ts_orig, dat_train)
#             curr_data[:, i, feat] = interp

#     new_train_data.append(torch.from_numpy(curr_data).to(device)) 

#     peaks_test = find_peaks(phis_test[subject].cpu().numpy(), distance=distance, height=height)[0]
#     curr_data = np.zeros((samples_per_cycle, len(peaks_test)-1, data.shape[2]))
#     for feat in range(data.shape[2]):
#         for i in range(len(peaks_test)-1):
#             start, end = peaks_test[i], peaks_test[i+1]
#             dat_test = test_data[subject, start:end, feat].cpu().numpy()
#             # interpolate from size of dat_train to samples_per_cycle
#             ts_new = np.linspace(0, len(dat_test), samples_per_cycle)
#             ts_orig = np.linspace(0, len(dat_test), len(dat_test))
#             interp = np.interp(ts_new, ts_orig, dat_test)
#             curr_data[:, i, feat] = interp
    
#     new_test_data.append(torch.from_numpy(curr_data).to(device)) 
# for subject in range(num_trials):
#     subject_tensor_train = torch.full((samples_per_cycle, new_train_data[subject].shape[1]), subject_to_index[subject_num[subject]], dtype=torch.int64, device=device)
#     subject_ids.append(subject_tensor_train)


# # train_data = torch.cat(new_train_data, dim=0)
# # test_data = torch.cat(new_test_data, dim=0)
# train_data = torch.cat(new_train_data, dim=1)
# test_data = torch.cat(new_test_data, dim=1)
# subject_ids = torch.cat(subject_ids, dim=1)

# subject_ids = torch.full((train_data.shape[0], train_data.shape[1]), subject_indices, dtype=torch.int64, device=device)
subject_ids = torch.tile(torch.from_numpy(subject_indices), (train_data.shape[1], 1)).T.to(device)

train_data = train_data.reshape(-1, train_data.shape[-1]).float()
subject_ids = subject_ids.reshape(-1)

perm = torch.randperm(train_data.shape[0])
train_data = train_data[perm]
subject_ids = subject_ids[perm]

train_size = 500_000
train_size = min(train_size, train_data.shape[0]-1)
X = train_data[:train_size, :]
subject_ids = subject_ids[:train_size]
Y = train_data[1:train_size+1, :]

# Initialize and train model
model = KoopmanAE(
    input_dim=X.shape[1],
    hidden_dim=64,
    output_dim=10,
    # num_subjects=len(np.unique(subject_num)),
    num_subjects=num_trials,
    depth=2,
    device=torch.device(device)
)

print(torch.linalg.eigvals(model.A[0]), model.A.shape)

losses = train_koopman_ae(
    model,
    X,
    Y,
    subject_ids,
    n_epochs=2000,
    batch_size=128,
    learning_rate=5e-3,
    verbose=True
)

def predict(model, test_data, indices, n_steps=100):
    """Predict future states using the KoopmanAE model"""
    model.eval()
    with torch.no_grad():
        x = test_data[:1]
        predictions = [x]
        for _ in range(n_steps):
            x = model(predictions[-1], indices)
            predictions.append(x)
        predictions = torch.stack(predictions[1:]).squeeze()
    return predictions

# Evaluate MSE
test_data, test_indices = test_data[0], torch.zeros(1, dtype=torch.int64).to(model.device)
prediction = predict(model, test_data, test_indices, n_steps=test_size)
mse = torch.mean((prediction[:test_data.shape[0]-1] - test_data[1:test_size+1])**2).item()
# mse = evaluate_mse(model, test_data, n_steps=test_size)

all_eigenvalues = np.array([torch.linalg.eigvals(model.A[subject]).detach().cpu().numpy() for subject in range(len(np.unique(subject_num)))])

# all_eigenvalues = np.array([results[subject]["eigenvalues"].detach().cpu().numpy() for subject in range(num_subjects)])

# Figure 1

# Create colormap based on extremes of speeds

norm = mcolors.Normalize(vmin=min(speeds), vmax=max(speeds))
cmap = cm.get_cmap("viridis")

fig = plt.figure(figsize=(10, 10))
# Plot in complex plane, colored by speed
ax = plt.subplot(1, 1, 1)
for ii in range(num_trials):
    color = cmap(norm(speeds[ii]))
    sc = ax.scatter(all_eigenvalues[ii, :].real, all_eigenvalues[ii, :].imag, color=[color for _ in range(all_eigenvalues.shape[1])], s=5)

ax.axis('equal')

# Add colorbar using the last scatter plot
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label('Speed', fontsize=14)

# Add unit circle
ax.add_patch(plt.Circle((0, 0), 1, color='black', fill=False))

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r"$\Re(\lambda)$", fontsize=14)
plt.ylabel(r"$\Im(\lambda)$", fontsize=14)
plt.savefig("figures/koopman_ae_eigenvalues.png")
plt.close()

# Figure 2

df = {"speed": speeds, "trial_type": trial_type, "subject_num": subject_num, "subject_index": subject_indices}
df = pd.DataFrame(df)

# Create a mapping of subject_num → speed when trial_type == 3
ss_speed_mapping = df.loc[df["trial_type"] == 3, ["subject_num", "speed"]].set_index("subject_num")["speed"]

# Map the SS_speed values back to all rows based on subject_num
df["SS_speed"] = df["subject_num"].map(ss_speed_mapping)
df["Δ_SS_speed"] = df["speed"] - df["SS_speed"]


num_colors = len(df["subject_num"].unique())
palette = sns.color_palette("husl", num_colors)  # Husl provides maximally distinct colors

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

# eigs = np.array([result['eigenvalues'].detach().cpu().numpy() for result in results.values()])
eigs = all_eigenvalues
df['eig'] = list(eigs)

for i, subject in enumerate(df["subject_num"].unique()):
    sub_df = df[df["subject_num"] == subject]
    ax.plot(sub_df["Δ_SS_speed"], sub_df["eig"].apply(lambda x: np.max(np.abs(x)).max()), label=f"Subject {subject}", color=palette[i], alpha=0.7, marker="o", linestyle="--")

Β, b = np.polyfit(df["Δ_SS_speed"].to_numpy(), np.max(np.abs(eigs), axis=1), 1)
x = np.linspace(df["Δ_SS_speed"].min(), df["Δ_SS_speed"].max(), 10)
ax.plot(x, Β*x + b, color="k", label="Linear fit", linestyle="--")
ax.set_xlabel(r"$\Delta$ SS speed", fontsize=14)
ax.set_ylabel(r"$\max(|\lambda|)$", fontsize=14)
# plt.legend()
plt.tight_layout()
plt.savefig("figures/koopman_ae_eigenvalues_speed.png")
plt.close()

# Save results and dataframe
df.to_csv("outputs/koopman_ae_eigenvalues.csv", index=False)