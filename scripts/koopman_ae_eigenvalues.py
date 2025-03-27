from scripts.koopman_ae import KoopmanAE, train_koopman_ae, evaluate_mse
import numpy as np
import torch
from scipy.interpolate import make_smoothing_spline
from delase import embed_signal_torch
import matplotlib.pyplot as plt
import seaborn as sns
from rich.progress import Progress
import pandas as pd

train_size = 2000
test_size = 300

data = np.loadtxt("data/DataStorage_6000_2D_kinematics_clean.csv", delimiter=",")
subject_num = np.loadtxt("data/subject_num_label.csv", delimiter=",")
trial_type = np.loadtxt("data/trialtype_label.csv", delimiter=",")

data = data.reshape(data.shape[0], 150, 6000).swapaxes(0, 1).swapaxes(1, 2)
speeds = np.loadtxt("data/speeds_label.csv", delimiter=",")
num_subjects = 150

# Prepare train and test data
train_data = data[:, :train_size, :]
train_data -= train_data.mean(axis=1, keepdims=True)
train_data /= train_data.std(axis=1, keepdims=True)

test_data = data[:, train_size:train_size+test_size, :]
test_data -= test_data.mean(axis=1, keepdims=True)
test_data /= test_data.std(axis=1, keepdims=True)

# Interpolate data
dt_orig = 1/100
dt_new = 1/100
ts_orig_train = np.linspace(0, train_data.shape[1]*dt_orig, train_data.shape[1])
ts_new_train = np.linspace(0, train_data.shape[1]*dt_orig, int(train_data.shape[1]*dt_orig/dt_new))
ts_orig_test = np.linspace(0, test_data.shape[1]*dt_orig, test_data.shape[1])
ts_new_test = np.linspace(0, test_data.shape[1]*dt_orig, int(test_data.shape[1]*dt_orig/dt_new))

new_train_data = np.zeros((num_subjects, len(ts_new_train), train_data.shape[2]))
new_test_data = np.zeros((num_subjects, len(ts_new_test), test_data.shape[2]))

for subject in range(num_subjects):
    for feat in range(train_data.shape[2]):
        interp = make_smoothing_spline(ts_orig_train, train_data[subject, :, feat], lam=0.00001)
        new_train_data[subject, :, feat] = interp(ts_new_train)
        
        interp = make_smoothing_spline(ts_orig_test, test_data[subject, :, feat], lam=0.00001)
        new_test_data[subject, :, feat] = interp(ts_new_test)

with Progress() as progress:
    task = progress.add_task("[cyan]Processing subjects...", total=num_subjects)
    results = {subject: {} for subject in range(num_subjects)}
    for subject in range(num_subjects):
        # Delay embed data
        n_delays = 10
        τ = 1
        train_data = embed_signal_torch(new_train_data[subject], n_delays, τ).to("cuda")
        test_data = embed_signal_torch(new_test_data[subject], n_delays, τ).to("cuda")

        # Initialize and train model
        model = KoopmanAE(
            input_dim=train_data.shape[1],
            hidden_dim=64,
            output_dim=10,
            depth=2,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        losses = train_koopman_ae(
            model,
            train_data,
            n_epochs=500,
            batch_size=128,
            learning_rate=1e-3,
            verbose=False
        )

        # Evaluate MSE
        mse = evaluate_mse(model, test_data, n_steps=test_size)
        results[subject]["mse"] = mse
        results[subject]["eigenvalues"] = torch.linalg.eigvals(model.A.weight)
        progress.update(task, advance=1)

all_eigenvalues = np.array([results[subject]["eigenvalues"].detach().cpu().numpy() for subject in range(num_subjects)])

# Figure 1

# Create colormap based on extremes of speeds
import matplotlib.colors as mcolors
from matplotlib import cm

norm = mcolors.Normalize(vmin=min(speeds), vmax=max(speeds))
cmap = cm.get_cmap("viridis")

fig = plt.figure(figsize=(10, 10))
# Plot in complex plane, colored by speed
ax = plt.subplot(1, 1, 1)
for ii in range(num_subjects):
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

df = {"speed": speeds, "trial_type": trial_type, "subject_num": subject_num}
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

eigs = np.array([result['eigenvalues'].detach().cpu().numpy() for result in results.values()])
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