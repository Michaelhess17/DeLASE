import statsmodels
from statsmodels.tsa.api import VAR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import delase
from rich.progress import Progress
import pandas as pd
import torch
from sklearn.manifold import MDS
import os

trials_per_group = 50
    
# Load data
df = pd.read_pickle("data/all_human_data.pkl")
data = np.concatenate([a[None, :] for a in df["data"]], axis=0)
speeds = df["speed"].to_numpy()

# Select trials from each type
ya_inds = np.where(df["subject"].str[:2] == "YA")[0]
st_inds = np.where(df["subject"].str[:2] == "ST")[0]

# separate low-functioning and high-functioning trials based on lowest speed
st_subjects = pd.unique(df.iloc[st_inds, :]["subject"])
lf_or_hf = {}
for st_subject in st_subjects:
    if np.min(df[df["subject"] == st_subject]["speed"]) < 40:
        lf_or_hf[st_subject] = "lf"
    else:
        lf_or_hf[st_subject] = "hf"

df["lf_or_hf"] = [lf_or_hf[subject] if subject in lf_or_hf else "AB" for subject in df["subject"]]
lf_inds = np.where(df["lf_or_hf"] == "lf")[0]
hf_inds = np.where(df["lf_or_hf"] == "hf")[0]
ab_inds = np.where(df["lf_or_hf"] == "AB")[0]

ya_inds = np.random.permutation(ya_inds)
st_inds = np.random.permutation(st_inds)
lf_inds = np.random.permutation(lf_inds)
hf_inds = np.random.permutation(hf_inds)
ab_inds = np.random.permutation(ab_inds)

df_subset = df.iloc[np.concatenate([ab_inds[:trials_per_group], lf_inds[:trials_per_group], hf_inds[:trials_per_group]])]
data = np.concatenate([a[None, :] for a in df_subset["data"]], axis=0)
speeds = df_subset["speed"].to_numpy()

num_subjects, num_timesteps, features = data.shape

lags = 50
device = "cpu"
results = [{"eigenvalues": {}, "results": {}} for trial in range(data.shape[0])]

with Progress() as progress:
    task = progress.add_task("[cyan]Fitting VAR models...", total=num_subjects)
    for trial in range(data.shape[0]):
        model = VAR(data[trial, :1000, :])
        # res = model.fit(maxlags=100, ic='bic', verbose=False)
        res = model.fit(lags)
        # Eigenvalues
        A = res.coefs
        companion_matrix = statsmodels.tsa.vector_ar.util.comp_matrix(A)

        eigvals_comp = np.linalg.eigvals(companion_matrix)

        results[trial]["eigenvalues"] = eigvals_comp
        results[trial]["results"] = res

        # plot results
        pred = res.forecast(data[trial, :lags, :], 200)
        fig = plt.figure(figsize=(10, 10))
        for idx in range(6):
            plt.subplot(3, 2, idx+1)
            plt.plot(pred[:, idx], label="Predicted")
            plt.plot(data[trial, lags:lags+200, idx], label="True")
            if idx == 0:
                plt.legend()
        plt.tight_layout()
        
        if not os.path.exists("figures/var_models"):
            os.makedirs("figures/var_models")
        plt.savefig(f"figures/var_models/var_model_{trial}.png")
        plt.close()
        progress.update(task, advance=1)


all_eigenvalues = np.concatenate([np.array(results[trial]['eigenvalues'])[None, :] for trial in range(data.shape[0])], axis=1).squeeze()

c_speeds = np.concatenate([speeds[trial]*np.ones_like(results[trial]['eigenvalues'], dtype=float) for trial in range(data.shape[0])]).squeeze()

delase.utils.plot_eigvals_on_unit_circle(all_eigenvalues.flatten(), c_speeds)

dist_scores = np.zeros((data.shape[0], data.shape[0]))

with Progress() as progress:
    task = progress.add_task("[magenta]Computing similarity scores...", total=num_subjects**2)
    for ii in range(data.shape[0]):
        for jj in range(data.shape[0]):
            if ii == jj:
                dist_scores[ii, jj] = 0
            elif ii > jj:
                dist_scores[ii, jj] = dist_scores[jj, ii]
            else:
                A = results[ii]["results"].coefs
                B = results[jj]["results"].coefs
                companion_matrix = statsmodels.tsa.vector_ar.util.comp_matrix(A)
                companion_matrix2 = statsmodels.tsa.vector_ar.util.comp_matrix(B)


                companion_matrix = torch.from_numpy(companion_matrix).float().to(device)
                companion_matrix2 = torch.from_numpy(companion_matrix2).float().to(device)

                sim_model = delase.simdist.SimilarityTransformDist(iters=500, lr=5e-3, device=device, verbose=False)
                sim_model.fit(companion_matrix, companion_matrix2)

                dist_scores[ii, jj] = sim_model.score(score_method="wasserstein")
            progress.update(task, advance=1)
plt.close("all")
plt.figure(figsize=(10, 10))
sns.heatmap(dist_scores, cmap="viridis")
# Show groups
plt.xticks(np.arange(len(df_subset["subject"]))+0.5, df_subset["subject"], rotation=90)
plt.yticks(np.arange(len(df_subset["subject"]))+0.5, df_subset["subject"], rotation=0)
plt.show()

# Create a MDS embedding from distance matrix
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
pos = mds.fit_transform(dist_scores)

plt.close("all")
fig = plt.figure(figsize=(10, 10))
groups = df_subset["lf_or_hf"].str[:2]
for group in groups.unique():
    plt.scatter(pos[groups == group, 0], pos[groups == group, 1], label=group, cmap="viridis")
plt.legend()
plt.show()
