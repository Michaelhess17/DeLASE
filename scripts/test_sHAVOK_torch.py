import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import CubicSpline, make_smoothing_spline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
torch.set_grad_enabled(False)  # Globally disable gradients for the entire script
print(f"Using device: {device}")

data_np = np.loadtxt("data/DataStorage_6000_2D_kinematics_clean.csv", delimiter=",")
num_subjects = 150
num_timesteps = 6000
data_np = data_np.reshape(data_np.shape[0], num_subjects, num_timesteps).swapaxes(0, 1).swapaxes(1, 2)
data_np -= data_np.mean(axis=1, keepdims=True)  # Center the data
data_np /= data_np.std(axis=1, keepdims=True)  # Normalize the data
data = torch.tensor(data_np, dtype=DTYPE).to(device)
all_all_params = np.zeros(data.shape[0], dtype=object)

subject = 0  # Change this to select a specific subject for testing
subject_data = data[subject]

def create_cmap(color1, color2, color3):
    colors = [color1, color2, color3]
    cmap_name = 'list'
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cmap

cmap = create_cmap('tab:blue', 'white', 'tab:orange')

def build_hankel(data, rows, cols):
    X = torch.empty((rows, cols), dtype=data.dtype, device=data.device)
    for k in range(rows):
        X[k, :] = data[k:cols + k]
    return X

def HAVOK(X, dt, r, norm, center=False, return_uv=False):
    if (center): 
        m = X.shape[0]
        X̄ = X - X[m//2,:]
        U, Σ, Vh = torch.linalg.svd(X̄, full_matrices=False)
    else:
        U, Σ, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.T
    polys = true_polys(X.shape[0], dt, r, center)
    for _i in range(r):
        if (torch.dot(U[:,_i], polys[:,_i]) < 0):
            U[:,_i] *= -1
            V[:,_i] *= -1
    V1 = V[:-1,:r]
    V2 = V[1:,:r]
    A = (V2.T @ torch.pinverse(V1.T) - torch.eye(r, device=X.device)) / (norm * dt)
    if (return_uv): 
        return A, U, V
    return A

def sHAVOK(X, dt, r, norm):
    X1 = X[:,:-1]
    X2 = X[:,1:]
    U1,_,Vh1 = torch.linalg.svd(X1, full_matrices=False)
    U2,_,Vh2 = torch.linalg.svd(X2, full_matrices=False)
    V1 = Vh1.T
    V2 = Vh2.T
    polys = true_polys(X.shape[0], dt, r, center=False)
    for _i in range(r):
        if (torch.dot(U1[:,_i], polys[:,_i]) < 0):
            V1[:,_i] *= -1
        if (torch.dot(U2[:,_i], polys[:,_i]) < 0):
            V2[:,_i] *= -1
    A = ((V2.T @ V1)[:r,:r] - torch.eye(r, device=X.device)) / (norm * dt)
    return A

def true_polys(rows, dt, r, center): 
    m = rows // 2
    Ut = torch.linspace(-m*dt, m*dt, rows, device=device, dtype=DTYPE)
    poly_stack = []
    for j in range(r):
        if (center):
            poly_stack.append(Ut ** (j + 1))
        else: 
            poly_stack.append(Ut ** j)
    poly_stack = torch.vstack(poly_stack).T
    Q = np.empty((rows, r), dtype=np.float32 if DTYPE == torch.float32 else np.float64)  # Initialize Q for Gram-Schmidt
    Q = torch.from_numpy(Q).to(device) # Convert to tensor and move to device
    # Q = torch.empty((rows, r), device=device) # Perform Gram-Schmidt
    print(Q.shape)
    for j in range(r): 
        v = poly_stack[:, j]
        for k in range(j - 1): 
            r_jk = Q[:, k:k+1].T @ poly_stack[:, j]
            v -= (r_jk * Q[:, k])
        r_jj = torch.norm(v)
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

xdata_orig = subject_data[:, :].cpu().numpy()

dt_orig = 1/100
dt_new = 1/2000
ts_orig = np.linspace(0, num_timesteps*dt_orig, num_timesteps)
ts_new = np.linspace(0, num_timesteps*dt_orig, int(num_timesteps*dt_orig/dt_new))

xdata = np.zeros((len(ts_new), xdata_orig.shape[1]))
for feat in range(xdata_orig.shape[1]):
    # interp = CubicSpline(ts_orig, xdata_orig, axis=0)
    interp = make_smoothing_spline(ts_orig, xdata_orig[:, feat], lam=0.00001)
    xdata[:, feat] = interp(ts_new)

# Plot the original and interpolated data
orig_plot_steps = 100
new_plot_steps = int(orig_plot_steps * (dt_orig / dt_new))
plt.subplot(1, 2, 1)
plt.plot(ts_orig[:orig_plot_steps], xdata_orig[:orig_plot_steps], color='gray', label='Original Data')
plt.title('Original Data')
plt.subplot(1, 2, 2)
plt.plot(ts_new[:new_plot_steps], xdata[:new_plot_steps], color='black', label='Interpolated Data')
plt.title('Interpolated Data')
plt.tight_layout()
plt.savefig('figures/interpolated_data.png', bbox_inches='tight')

# Define parameters 
r = 200
rows = 1001
dt = dt_new

pendulum_short = dict()
pendulum_full = dict()
# pendulum_recon = dict()
# pendulum_reconf = dict()
short = 1/5
full = 1
pendulum_short['n'] = int(xdata.shape[0] * short)
pendulum_full['n'] = int(xdata.shape[0]*full)
# pendulum_recon['n'] = int(xdata.shape[0] * short)
# pendulum_reconf['n'] = int(xdata.shape[0] * full)

xdata_tensor = torch.tensor(xdata, dtype=DTYPE).to(device)

for i, elem in enumerate((pendulum_short, pendulum_full)):
    print(f"Processing element {i+1} with n={elem['n']}")
    elem['xdata'] = xdata_tensor[:elem['n']]
    for idx in range(elem['xdata'].shape[1]):
        if "H" not in elem.keys():
            elem['H'] = build_hankel(elem['xdata'][:, idx],rows,elem['xdata'][:, idx].shape[0]-rows+1)
        else:
            H_curr = build_hankel(elem['xdata'][:, idx],rows,elem['xdata'][:, idx].shape[0]-rows+1)
            elem['H'] = torch.cat((elem['H'], H_curr), axis=0)
    print("Hankel matrix built")
    elem['Vh'] = torch.linalg.svd(elem['H'], full_matrices=False)[2]
    print("SVD computed")
    elem['A1'], U, V = HAVOK(elem['H'],dt,r,1,return_uv=True)
    print("HAVOK dynamics matrix computed")
    elem['Vh'] = V.T
    elem['A2'] = sHAVOK(elem['H'],dt,r,1)
    print("sHAVOK dynamics matrix computed")
    elem['ω1'] = torch.linalg.eig(elem['A1'])[0]
    elem['ω2'] = torch.linalg.eig(elem['A2'])[0]
    
# Plot results 
plt.figure(figsize=(18,3),dpi=200)

# Picture
plt.subplot(1,5,1)
# img = mpimg.imread('doublependulum.png')
# plt.imshow(img)
plt.axis('off')
plt.xticks([])
plt.yticks([])

# Time series 
ax = plt.subplot(1,5,2)
y_shift = 0.4
plt.plot(np.arange(pendulum_full['n'])*dt, pendulum_full['xdata'].cpu().numpy(), color='gray')
plt.plot(np.arange(pendulum_short['n'])*dt, pendulum_short['xdata'].cpu().numpy(),color='black')
plt.ylim(np.min(pendulum_full['xdata'].cpu().numpy()*1.3),np.max(pendulum_full['xdata'].cpu().numpy()*1.3))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.ylabel(r'$\sin(\theta_2)$', fontsize=14, color='white')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# HAVOK dynamics matrix 
plt.subplot(1,5,3)
plt.imshow(pendulum_short['A1'].cpu().numpy(),cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.grid('on',color='k')

# sHAVOK dynamics matrix 
plt.subplot(1,5,4)
plt.imshow(pendulum_short['A2'].cpu().numpy(),cmap=cmap)
plt.xticks([])
plt.yticks([])

# Eigenvalues 
ax = plt.subplot(1,5,5)
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
plt.xticks([])
plt.yticks([])
plt.plot(pendulum_short['ω1'].real.cpu().numpy(),pendulum_short['ω1'].imag.cpu().numpy(),'o',color='teal',markersize=14,label='HAVOK',alpha=0.5)
plt.plot(pendulum_short['ω2'].real.cpu().numpy(),pendulum_short['ω2'].imag.cpu().numpy(),'o',color='maroon',markersize=14,label='sHAVOK',alpha=0.5)
plt.plot(pendulum_full['ω1'].real.cpu().numpy(),pendulum_full['ω1'].imag.cpu().numpy(),'+',color='k',markersize=14,mew=2,label='true',linewidth=20)
plt.tight_layout()
plt.savefig('figures/fig9a_torch.pdf', bbox_inches='tight')

# Reconstruction
plt.figure(figsize=(10,2))

plt.subplot(1,1,1)
plt.plot(pendulum_full['Vh'][0,:].cpu().numpy(),color='k',label='Truth',linestyle='dashed')
plt.plot(reconstruct_v(pendulum_short['A1'],pendulum_full['Vh'],r,dt)[0,:].cpu().numpy(),color='teal',label='HAVOK')
plt.plot(reconstruct_v(pendulum_short['A2'],pendulum_full['Vh'],r,dt)[0,:].cpu().numpy(),color='maroon',label='sHAVOK')
plt.xticks([])
plt.yticks([])
plt.ylim([-0.02,0.02])
plt.legend()

plt.tight_layout()
plt.savefig('figures/fig9c_torch.pdf', bbox_inches='tight')