from koopman_ae import KoopmanAE, train_koopman_ae, evaluate_mse, predict
import numpy as np
import torch
from scipy.interpolate import make_smoothing_spline
from delase import embed_signal_torch
import matplotlib.pyplot as plt

train_size = 300
test_size = 500

# Load data
data = np.loadtxt("data/DataStorage_6000_2D_kinematics_clean.csv", delimiter=",")
data = data.reshape(data.shape[0], 150, 6000).swapaxes(0, 1).swapaxes(1, 2)


# Prepare train and test data
train_data = data[0, :train_size, :]
train_data -= train_data.mean(axis=1, keepdims=True)
train_data /= train_data.std(axis=1, keepdims=True)

test_data = data[0, train_size:train_size+test_size, :]
test_data -= test_data.mean(axis=1, keepdims=True)
test_data /= test_data.std(axis=1, keepdims=True)

# Interpolate data
dt_orig = 1/100
dt_new = 1/100
ts_orig_train = np.linspace(0, train_data.shape[0]*dt_orig, train_data.shape[0])
ts_new_train = np.linspace(0, train_data.shape[0]*dt_orig, int(train_data.shape[0]*dt_orig/dt_new))
ts_orig_test = np.linspace(0, test_data.shape[0]*dt_orig, test_data.shape[0])
ts_new_test = np.linspace(0, test_data.shape[0]*dt_orig, int(test_data.shape[0]*dt_orig/dt_new))

new_train_data = np.zeros((len(ts_new_train), train_data.shape[1]))
new_test_data = np.zeros((len(ts_new_test), test_data.shape[1]))

for feat in range(train_data.shape[1]):
    interp = make_smoothing_spline(ts_orig_train, train_data[:, feat], lam=0.00001)
    new_train_data[:, feat] = interp(ts_new_train)
    
    interp = make_smoothing_spline(ts_orig_test, test_data[:, feat], lam=0.00001)
    new_test_data[:, feat] = interp(ts_new_test)

# Delay embed data
n_delays = 100
τ = 1
train_data = embed_signal_torch(new_train_data, n_delays, τ).to("cuda")
test_data = embed_signal_torch(new_test_data, n_delays, τ).to("cuda")

# Initialize and train model
model = KoopmanAE(
    input_dim=train_data.shape[1],
    hidden_dim=64,
    output_dim=6,
    num_subjects=1,
    depth=3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print(torch.linalg.eigvals(model.A[0]), model.A.shape)

subject_indices = torch.zeros(train_data.shape[0]-1, dtype=torch.int64).to(model.device)

losses = train_koopman_ae(
    model,
    train_data,
    subject_indices,
    n_epochs=500,
    batch_size=128,
    learning_rate=1e-3
)

# Evaluate MSE
mse = evaluate_mse(model, test_data, n_steps=test_size)

# Make plots
prediction = predict(model, test_data, n_steps=test_size)[:test_data.shape[0]-1]
fig = plt.figure(figsize=(10, 6))
for ii in range(6):
    plt.subplot(3, 2, ii+1)
    if ii == 0:
        plt.plot(ts_new_test[:prediction.shape[0]], test_data[1:, ii*n_delays].cpu().numpy(), label="True")
        plt.plot(ts_new_test[:prediction.shape[0]], prediction[:, ii*n_delays].cpu().numpy(), label="Predicted")
        plt.title(f"Feature {ii}")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()

    else:
        plt.plot(ts_new_test[:prediction.shape[0]], test_data[1:, ii*n_delays].cpu().numpy())
        plt.plot(ts_new_test[:prediction.shape[0]], prediction[:, ii*n_delays].cpu().numpy())
        plt.title(f"Feature {ii}")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")

plt.tight_layout()
plt.savefig("figures/koopman_ae_pred.pdf")
plt.close()