from koopman_ae import KoopmanAE, train_koopman_ae, evaluate_mse, predict
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
from delase import embed_signal_torch
import torch
import numpy as np

if __name__ == "__main__":
    # Load data
    data = np.loadtxt("data/DataStorage_6000_2D_kinematics_clean.csv", delimiter=",")
    data = data.reshape(data.shape[0], 150, 6000).swapaxes(0, 1).swapaxes(1, 2)
    
    # Parameters for evaluation
    train_sizes = [150, 300, 500, 750, 1000]
    n_replicates = 10
    test_size = 150
    results = {size: [] for size in train_sizes}
    
    actual_sizes = []

    for train_size in train_sizes:
        print(f"\nEvaluating train_size: {train_size}")
        
        for rep in range(n_replicates):
            print(f"Replicate {rep + 1}/{n_replicates}")
            
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
                output_dim=10,
                depth=2,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            losses = train_koopman_ae(
                model,
                train_data,
                n_epochs=500,
                batch_size=128,
                learning_rate=1e-3
            )
            
            # Evaluate MSE
            mse = evaluate_mse(model, test_data, n_steps=test_size)
            results[train_size].append(mse)

        actual_size = train_data.shape[0]
        actual_sizes.append(actual_size)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    means = [np.mean(results[size]) for size in train_sizes]
    stds = [np.std(results[size]) for size in train_sizes]
    
    plt.errorbar(actual_sizes, means, yerr=stds, fmt='o-', capsize=5)
    plt.xlabel('Training Set Size')
    plt.ylabel('Test MSE')
    plt.title('Test MSE vs Training Set Size')
    plt.grid(True)
    plt.savefig("figures/koopman_ae_train_size_analysis.pdf")
    plt.close()
    
    # Print numerical results
    print("\nNumerical Results:")
    for size in train_sizes:
        print(f"Train size {size}: MSE = {np.mean(results[size]):.6f} ± {np.std(results[size]):.6f}")