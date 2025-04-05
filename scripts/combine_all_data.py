import pandas as pd
import numpy as np
from scipy.interpolate import make_smoothing_spline

dt = 1/100


data_6000 = np.loadtxt("data/DataStorage_6000_2D_kinematics_clean.csv", delimiter=",").reshape(6, 150, 6000).swapaxes(0, 1).swapaxes(1, 2)
data_1500 = np.loadtxt("data/SpeedSigs_data.csv", delimiter=",").reshape(149, 1500, 6)

speeds_6000 = np.loadtxt("data/speeds_label.csv", delimiter=",")
speeds_1500 = np.loadtxt("data/SpeedSigs_speeds.csv", delimiter=",")

labels_6000 = np.loadtxt("data/subject_num_label.csv", delimiter=",", dtype=str)
labels_6000 = np.array(["YA" + str(i) if len(str(i)) == 2 else "YA0" + str(i) for i in labels_6000])
labels_1500 = np.loadtxt("data/SpeedSigs_labels.csv", delimiter=",", dtype=str)


all_data = np.concatenate((data_6000[:, :1500, :], data_1500), axis=0)
all_speeds = np.concatenate((speeds_6000, speeds_1500), axis=0)
all_labels = np.concatenate((labels_6000, labels_1500), axis=0)

for subject in range(all_data.shape[0]):
    for feat in range(all_data.shape[2]):
        interp = make_smoothing_spline(np.arange(0, 1500*dt, step=dt), all_data[subject, :1500, feat], lam=0.00001)
        all_data[subject, :, feat] = interp(np.arange(0, 1500*dt, step=dt))

df = pd.DataFrame({
    "subject": all_labels,
    "speed": all_speeds,
    "data": [data for data in all_data]
})

# Save the DataFrame to a CSV file
df.to_pickle("data/all_human_data.pkl")