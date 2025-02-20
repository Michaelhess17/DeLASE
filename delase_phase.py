from delase.dmd import DMD
from ddfa_node import get_phased_signals
import numpy as np
import torch

from delase.stability_estimation import compute_DDE_chroots

class DeLASEPhaser:
    def __init__(
            self,
            data,
            n_delays=None,
            matrix_size=None,
            delay_interval=1,
            rank=None,
            rank_thresh=None,
            rank_explained_variance=None,
            lamb=0,
            dt=None,
            N_time_bins=None,
            max_freq=None,
            max_unstable_freq=None,
            device=None,
            n_segments=101,
            verbose=False
        ):
        self.data = data
        self.window = data.shape[-2]
        self.n = data.shape[-1]
        if n_delays is not None and matrix_size is not None:
            raise ValueError("Cannot provide both n_delays and matrix_size!!")
        elif n_delays is None:
            self.matrix_size = matrix_size
            self.n_delays = int(np.ceil(matrix_size/self.n))
        else: # matrix_size is None
            self.n_delays = n_delays
            self.matrix_size = self.n*self.n_delays

        self.delay_interval = delay_interval
        self.dt = dt
        self.N_time_bins = N_time_bins
        self.max_freq = max_freq
        self.max_unstable_freq = max_unstable_freq
        self.device = device
        self.verbose = verbose
        
        self.DMD = DMD(
            self.data,
            n_delays=self.n_delays,
            delay_interval=self.delay_interval,
            rank=rank,
            rank_thresh=rank_thresh,
            rank_explained_variance=rank_explained_variance,
            lamb=lamb,
            device=self.device,
            verbose=self.verbose,
        )

        self.Js = None

        self.N_time_bins = None
        self.chroots = None
        self.stability_params = None
        self.stability_freqs = None
        self.n_segments = n_segments

    
    def assign_data_to_percent_segments(self, phaser_feats=None, trim_cycles=1, height=0.85, distance=80):
        # TODO: remove trim_cycles and use all data
        if isinstance(self.data, torch.Tensor):
            raw = self.data.T.cpu().numpy()
        else:
            raw = self.data.T
        # Assess phase using Phaser
        cycle_list, _ = get_phased_signals(raw, phaser_feats, trim_cycles, self.n_segments, height, distance)
        self.phased_signals = cycle_list # will have shape (nCycles, nFeatures, nSegments)

    def fit_phase_range(self, phase_start=0, phase_end=100, height=0.85, distance=80):
        self.assign_data_to_percent_segments()
        data_in_range = self.phased_signals[:, :, phase_start:phase_end].swapaxes(1, 2)
        Vt_minus = data_in_range[:, :, :-1].reshape(-1, data_in_range.shape[-1])
        Vt_plus = data_in_range[:, :, 1:].reshape(-1, data_in_range.shape[-1])
        self.DMD.fit(Vt_minus=Vt_minus, Vt_plus=Vt_plus)
        params, freqs = self.get_stability()

    def compute_jacobians(
            self,
            dt=None
        ):
        if dt is None:
            if self.dt is None:
                raise ValueError("Time step dt required for computation!")
            else:
                dt = self.dt
        else:
            # overwrite saved dt
            self.dt = dt
        Js = torch.zeros(self.n_delays, self.n, self.n).to(self.device)
        for i in range(self.n_delays):
            if i == 0:
                Js[i] = (self.DMD.A_havok_dmd[:self.n, i*self.n:(i + 1)*self.n] - torch.eye(self.n).to(self.device))/dt
            else:
                Js[i] = self.DMD.A_havok_dmd[:self.n, i*self.n:(i + 1)*self.n]/dt
        
        self.Js = Js

    def filter_chroots(
            self,
            max_freq=None,
            max_unstable_freq=None
        ):
        #print(max_freq, max_unstable_freq)
        self.max_freq = self.max_freq if max_freq is None else max_freq
        self.max_unstable_freq = self.max_unstable_freq if max_unstable_freq is None else max_unstable_freq
        stability_params = torch.real(self.chroots)
        freqs = torch.imag(self.chroots)/(2*torch.pi)

        if self.max_freq is not None:
            filtered_inds = torch.abs(freqs) <= self.max_freq
            stability_params = stability_params[filtered_inds]
            freqs = freqs[filtered_inds]
        
        if self.max_unstable_freq is not None:
            filtered_inds = torch.logical_or(torch.abs(freqs) <= self.max_unstable_freq, stability_params <= 0)
            stability_params = stability_params[filtered_inds]
            freqs = freqs[filtered_inds]

        self.stability_params = stability_params
        self.stability_freqs = freqs
        return stability_params, freqs
    
    def get_stability(
            self,
            N_time_bins=None,
            max_freq=None,
            max_unstable_freq=None
        ):
        self.N_time_bins = self.N_time_bins if N_time_bins is None else N_time_bins
        self.max_freq = self.max_freq if max_freq is None else max_freq
        self.max_unstable_freq = self.max_unstable_freq if max_unstable_freq is None else max_unstable_freq

        if self.verbose:
            print("Computing jacbians...")
        self.compute_jacobians()
            # raise ValueError("Jacobians are needed for stability estimation! Run compute_jacobians first")

        if self.verbose:
            print("Computing DDE characteristic roots...")
        if N_time_bins is None:
            N_time_bins = self.n_delays
        self.N_time_bins = N_time_bins

        chroots = compute_DDE_chroots(self.Js, self.dt*self.delay_interval, N=N_time_bins, device=self.device)

        if isinstance(chroots, np.ndarray):
            chroots = torch.from_numpy(chroots).to(self.device)
        chroots = chroots[torch.argsort(torch.real(chroots)).flip(dims=(0,))]
        self.chroots = chroots

        if self.verbose:
            print("Characteristic root computation complete!")
            print("Filtering characteristic roots...")

        params, freqs = self.filter_chroots(max_freq, max_unstable_freq)

        if self.verbose:
            print("Stability analysis complete")
        return params, freqs
 
    def fit(
        self,
        data=None,
        n_delays=None,
        delay_interval=None,
        rank=None,
        rank_thresh=None,
        rank_explained_variance=None,
        lamb=None,
        dt=None,
        N_time_bins=None,
        max_freq=None,
        max_unstable_freq=None,
        device=None,
        verbose=None,
    ):
        self.device = self.device if device is None else device
        self.verbose = self.verbose if verbose is None else verbose

        self.DMD.fit(
            data=data,
            n_delays=n_delays,
            delay_interval=delay_interval,
            rank=rank,
            rank_thresh=rank_thresh,
            rank_explained_variance=rank_explained_variance,
            lamb=lamb,
            device=device,
            verbose=verbose  
        )

        self.get_stability(
            N_time_bins=N_time_bins,
            max_freq=max_freq,
            max_unstable_freq=max_unstable_freq
        )
    
    def to(self, device):
        self.DMD.to(device)
        if self.Js is not None:
            self.Js.to(device)
        if self.chroots is not None:
            self.chroots.to(device)
        if self.stability_params is not None:
            self.stability_params.to(device)
        if self.stability_freqs is not None:
            self.stability_freqs.to(device)

ts = np.linspace(0, 100, 10000)
data = np.stack([np.sin(ts), np.cos(ts)]).T

delase = DeLASEPhaser(data,
            n_delays=None,
            matrix_size=20,
            delay_interval=1,
            rank=20,
            rank_thresh=None,
            rank_explained_variance=None,
            lamb=0,
            dt=ts[1]-ts[0],
            N_time_bins=None,
            max_freq=None,
            max_unstable_freq=None,
            device=torch.device("cpu"),
            n_segments=101,
            verbose=True)
delase.fit_phase_range(phase_start=0, phase_end=100)