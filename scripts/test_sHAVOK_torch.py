import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import make_smoothing_spline
import torch
from delase.stability_estimation import compute_DDE_chroots
from rich.progress import Progress

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
verbose = False
torch.set_grad_enabled(False)  # Globally disable gradients for the entire script
if verbose:
    print(f"Using device: {device}")

num_subjects = 150
num_timesteps = 6000
r = 250
n_delays = 101
dt_orig = 1/100
dt_new = 1/1000
short = 1/3
full = 0.6

# Load and preprocess data
data_np = np.loadtxt("data/DataStorage_6000_2D_kinematics_clean.csv", delimiter=",")
data_np = data_np.reshape(data_np.shape[0], num_subjects, num_timesteps).swapaxes(0, 1).swapaxes(1, 2)
data_np -= data_np.mean(axis=1, keepdims=True)  # Center the data
data_np /= data_np.std(axis=1, keepdims=True)  # Normalize the data
data = torch.tensor(data_np, dtype=DTYPE).to(device)

# Function definitions
def create_cmap(color1, color2, color3):
    colors = [color1, color2, color3]
    cmap_name = 'list'
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cmap

def build_hankel(data, n_delays, cols):
    X = torch.empty((n_delays, cols), dtype=data.dtype, device=data.device)
    for k in range(n_delays):
        X[k, :] = data[k:cols + k]
    return X

def HAVOK(X, dt, r, norm, center=False, return_uv=False):
    if center: 
        m = X.shape[0]
        X̄ = X - X[m//2,:]
        U, Σ, Vh = torch.linalg.svd(X̄, full_matrices=False)
    else:
        U, Σ, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.T
    polys = true_polys(X.shape[0], dt, r, center)
    for _i in range(r):
        if torch.dot(U[:,_i], polys[:,_i]) < 0:
            U[:,_i] *= -1
            V[:,_i] *= -1
    V1 = V[:-1,:r]
    V2 = V[1:,:r]
    A = (V2.T @ torch.pinverse(V1.T) - torch.eye(r, device=X.device)) / (norm * dt)
    if return_uv: 
        return A, U, Σ, V
    return A

def sHAVOK(X, dt, r, norm, return_uv=False):
    X1 = X[:,:-1]
    X2 = X[:,1:]
    U1,Σ,Vh1 = torch.linalg.svd(X1, full_matrices=False)
    U2,Σ,Vh2 = torch.linalg.svd(X2, full_matrices=False)
    V1 = Vh1.T
    V2 = Vh2.T
    polys = true_polys(X.shape[0], dt, r, center=False)
    for _i in range(r):
        if torch.dot(U1[:,_i], polys[:,_i]) < 0:
            V1[:,_i] *= -1
        if torch.dot(U2[:,_i], polys[:,_i]) < 0:
            V2[:,_i] *= -1
    A = ((V2.T @ V1)[:r,:r] - torch.eye(r, device=X.device)) / (norm * dt)
    if return_uv: 
        return A, U, Σ, V
    return A

def true_polys(n_delays, dt, r, center): 
    m = n_delays // 2
    Ut = torch.linspace(-m*dt, m*dt, n_delays, device=device, dtype=DTYPE)
    poly_stack = []
    for j in range(r):
        if center:
            poly_stack.append(Ut ** (j + 1))
        else: 
            poly_stack.append(Ut ** j)
    poly_stack = torch.vstack(poly_stack).T
    Q = torch.zeros((n_delays, r), dtype=DTYPE, device=device)  # Initialize Q for Gram-Schmidt
    for j in range(r): 
        v = poly_stack[:, j].clone()
        for k in range(j - 1): 
            r_jk = Q[:, k:k+1].T @ poly_stack[:, j]
            v -= (r_jk * Q[:, k])
        r_jj = torch.linalg.norm(v)
        Q[:, j] = v / r_jj
    return Q

def reconstruct_v(A, Vh_full, r, dt):
    Ā = A[:-1, :-1]
    B = A[:-1, -1].reshape(-1, 1)
    t0 = np.arange(Vh_full.shape[1]) * dt
    sys = signal.StateSpace(Ā.cpu().numpy(), B.cpu().numpy(), np.eye(r-1), np.zeros(r-1).reshape(-1, 1))
    tout, y, x = signal.lsim(sys, Vh_full[r-1, :].cpu().numpy(), t0, X0=Vh_full[:r-1, 0].cpu().numpy())
    Vh_rec = torch.tensor(y.T, dtype=Vh_full.dtype, device=Vh_full.device)
    return Vh_rec

def compute_jacobians(A_havok_dmd, dt):
    Js = torch.zeros(n_delays, n, n).to(A_havok_dmd.device)
    for i in range(n_delays):
        if i == 0:
            Js[i] = (A_havok_dmd[:n, i*n:(i + 1)*n] - torch.eye(n).to(A_havok_dmd.device))/dt
        else:
            Js[i] = A_havok_dmd[:n, i*n:(i + 1)*n]/dt
    return Js    

def filter_chroots(chroots, max_freq=None, max_unstable_freq=None):
    max_freq = max_freq if max_freq is None else max_freq
    max_unstable_freq = max_unstable_freq if max_unstable_freq is None else max_unstable_freq
    stability_params = torch.real(chroots)
    freqs = torch.imag(chroots)/(2*torch.pi)

    if max_freq is not None:
        filtered_inds = torch.abs(freqs) <= max_freq
        stability_params = stability_params[filtered_inds]
        freqs = freqs[filtered_inds]
    
    if max_unstable_freq is not None:
        filtered_inds = torch.logical_or(torch.abs(freqs) <= max_unstable_freq, stability_params <= 0)
        stability_params = stability_params[filtered_inds]
        freqs = freqs[filtered_inds]

    return stability_params, freqs

def get_stability(A_havok_dmd, dt, max_freq=None, max_unstable_freq=None):
    if verbose:
        print("Computing jacobians...")
    Js = compute_jacobians(A_havok_dmd=A_havok_dmd, dt=dt)

    if verbose:
        print("Computing DDE characteristic roots...")
    chroots = compute_DDE_chroots(Js, dt, N=n_delays, device=Js.device)

    if isinstance(chroots, np.ndarray):
        chroots = torch.from_numpy(chroots).to(device)
    chroots = chroots[torch.argsort(torch.real(chroots)).flip(dims=(0,))]

    if verbose:
        print("Characteristic root computation complete!")
        print("Filtering characteristic roots...")
    chroots = filter_chroots(chroots, max_freq, max_unstable_freq)

    if verbose:
        print("Stability analysis complete")
    return chroots    

# Main analysis loop
results = {}
cmap = create_cmap('tab:blue', 'white', 'tab:orange')

with Progress() as progress:
    task = progress.add_task("[cyan]Processing subjects...", total=num_subjects)

    for subject in range(num_subjects):
        if verbose:
            print(f"Processing subject {subject+1}/{num_subjects}")
        subject_data = data[subject]
        xdata_orig = subject_data[:, :].cpu().numpy()

        ts_orig = np.linspace(0, num_timesteps*dt_orig, num_timesteps)
        ts_new = np.linspace(0, num_timesteps*dt_orig, int(num_timesteps*dt_orig/dt_new))

        xdata = np.zeros((len(ts_new), xdata_orig.shape[1]))
        for feat in range(xdata_orig.shape[1]):
            interp = make_smoothing_spline(ts_orig, xdata_orig[:, feat], lam=0.00001)
            xdata[:, feat] = interp(ts_new)

        pendulum_short = dict()
        pendulum_full = dict()
        pendulum_short['n'] = int(xdata.shape[0] * short)
        pendulum_full['n'] = int(xdata.shape[0] * full)

        xdata_tensor = torch.tensor(xdata, dtype=DTYPE).to(device)

        for i, elem in enumerate((pendulum_short, pendulum_full)):
            if verbose:
                print(f"Processing element {i+1} with n={elem['n']}")
            
            elem['xdata'] = xdata_tensor[:elem['n']]
            
            for idx in range(elem['xdata'].shape[1]):
                if "H" not in elem.keys():
                    elem['H'] = build_hankel(elem['xdata'][:, idx], n_delays, elem['xdata'][:, idx].shape[0] - n_delays + 1)
                else:
                    H_curr = build_hankel(elem['xdata'][:, idx], n_delays, elem['xdata'][:, idx].shape[0] - n_delays + 1)
                    elem['H'] = torch.cat((elem['H'], H_curr), axis=0)
            if verbose:
                print("Hankel matrix built")
            
            elem['Vh'] = torch.linalg.svd(elem['H'], full_matrices=False)[2]
            if verbose:
                print("SVD computed")

            elem['A1'], U, Σ, V = HAVOK(elem['H'], dt_new, r, 1, return_uv=True)
            elem['A1'] = elem['A1'].cpu()  # Move A1 to CPU
            U, Σ, V = U.cpu(), Σ.cpu(), V.cpu()  # Move U and V to CPU
            elem['Vh'] = V.T
            elem['S1'] = Σ
            elem['U1'] = U
            if verbose:
                print("HAVOK dynamics matrix computed")

            elem['A2'], U, Σ, _ = sHAVOK(elem['H'], dt_new, r, 1, return_uv=True)
            U, Σ = U.cpu(), Σ.cpu()  # Move U and V to CPU
            elem['S2'] = Σ
            elem['U2'] = U
            if verbose:
                print("sHAVOK dynamics matrix computed")
            
            elem['ω1'] = torch.linalg.eig(elem['A1'])[0].cpu()
            elem['ω2'] = torch.linalg.eig(elem['A2'])[0].cpu()
            if verbose:
                print("Eigenvalues computed")

            # Compute stability
            matrix_size = elem['H'].shape[0]  # Number of columns in the Hankel matrix
            A_v = elem['A1']  # Use the HAVOK dynamics matrix for DMD
            n = xdata.shape[-1]  # Number of features (columns) in the data
            S = elem['S1']  # Singular values from the HAVOK SVD
            s = len(S)  # Number of singular values
            U = elem['U1']  # Left singular vectors from the HAVOK SVD
            dim = U.shape[1]  # Dimension of the system
            S_mat = torch.zeros(dim, dim, dtype=DTYPE)
            S_mat_inv = torch.zeros(dim, dim, dtype=DTYPE)
            S_mat[np.arange(s), np.arange(s)] = S
            S_mat_inv[np.arange(s), np.arange(s)] = 1 / S
            A_havok_dmd = U @ S_mat[:dim, :r] @ A_v @ S_mat_inv[:r, :dim] @ U.T

            chroots, freqs = get_stability(A_havok_dmd=A_havok_dmd, dt=dt_new, max_freq=None, max_unstable_freq=None)
            elem['chroots'] = chroots.cpu()  # Move chroots to CPU
            elem['chroots_freqs'] = freqs.cpu()  # Move freqs to CPU
            if verbose:
                print("Stability analysis complete")

        results[subject] = {'short': pendulum_short, 'full': pendulum_full}
        progress.update(task, advance=1)

torch.save(results, "outputs/sHAVOK_results.pkl") # Save results to a file