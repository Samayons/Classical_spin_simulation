import numpy as np
import h5py
import os
#import time
from tqdm import tqdm  # Add this for progress bar
from coupling import get_coupling_constants, get_spin_timescale
from system_parameters import (
    r_vector, N, gamma, h_bar, H_ext, h_chem, active_spins
)

current_dir = os.getcwd() # get the current directory

"""

Setting_1: isotropic J-coupling with chemical shift
Setting_2: dipole-dipole with chemical shift
Setting_3: dipole-dipole without chemical shift

"""

if N == 8:
    num_initialized_spins = "eight"
    r_vector = r_vector[active_spins]  # Use only the active spins for 8 spins
elif N == 16:
    num_initialized_spins = "all"
    r_vector = r_vector  # Use all spins for 16 spins

time_units = 2000
num_real = 100
setting = 3 # 1, 2, or 3

if setting == 1:
  choose_coupling_type = "J-coupling"
  choose_shift_type = "with-chemical"
elif setting == 2:
  choose_coupling_type = "dipole-dipole"
  choose_shift_type = "with-chemical"
elif setting == 3:
  choose_coupling_type = "dipole-dipole"
  choose_shift_type = "without-chemical"
else:
  raise ValueError("Please select a valid setting: 1, 2, or 3.")

# File paths
file_path_to_simulations = f"{current_dir}/simulations/setting_{setting}"
file_path_to_results = f"{current_dir}/results/setting_{setting}"

os.makedirs(file_path_to_simulations, exist_ok=True)  # creates folder if it doesn’t exist
os.makedirs(file_path_to_comparisons, exist_ok=True)

class Simulation:
    def __init__(self, coupling_type, shift_type, num_realizations=50):
        # Configuration
        self.coupling_type = coupling_type
        self.shift_type = shift_type
        self.num_realizations = num_realizations

        # Coupling matrices and timescales
        self.Jx, self.Jy, self.Jz = get_coupling_constants(
            r_vector, H_ext, gamma, h_bar, coupling_type=self.coupling_type
        )
        print(f"Largest coupling constant is Jz = {np.max(np.abs(self.Jz))} (Hz) in setting {setting}")
        print(f"Jz has shape {self.Jz.shape}")

        tau_list = [
            get_spin_timescale(i, self.Jx, self.Jy, self.Jz, self.shift_type)
            for i in range(N)
        ]
        spin_index = np.argmin(tau_list)
        tau_min = np.nanmin(tau_list)
        print(f"Smallest timescale: {tau_list[spin_index]} (s) for spin {spin_index + 1}")

        # Time settings
        self.dt = 1e-2 * tau_min
        self.T = time_units * tau_min
        self.steps = int(self.T / self.dt)
        self.gamma = gamma
        self.h_chem = h_chem
        self.h_bar = h_bar
        self.N = N

    def random_unit_vector(self, size: int):
        """Generate random unit vectors (size × 3)."""
        S = np.sqrt(3) / 2
        phi = np.random.uniform(0, 2 * np.pi, size)
        cost = np.random.uniform(-1, 1, size)
        sint = np.sqrt(1 - cost**2)
        vecs = S * np.column_stack((sint * np.cos(phi), sint * np.sin(phi), cost))
        return vecs

    def dSdt(self, spins, H_eff: str):
        """Vectorised time derivative for all spins (N × 3)."""
        h_ex = np.stack([
            self.Jx @ spins[:, 0],
            self.Jy @ spins[:, 1],
            self.Jz @ spins[:, 2]
        ], axis=1)
        if H_eff == "with-chemical":
            h_eff = (self.gamma * self.h_chem - h_ex)
        elif H_eff == "without-chemical":
            h_eff = -h_ex
        else:
            raise ValueError("Please select a setting, with/without chemical")
        return np.cross(spins, h_eff)

    def rk4_step(self, spins, H_eff: str):
        """Single RK4 integration step."""
        k1 = self.dSdt(spins, H_eff=H_eff)
        k2 = self.dSdt(spins + 0.5 * self.dt * k1, H_eff=H_eff)
        k3 = self.dSdt(spins + 0.5 * self.dt * k2, H_eff=H_eff)
        k4 = self.dSdt(spins + self.dt * k3, H_eff=H_eff)
        S_new = spins + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Check for numerical issues
        if np.any(np.isnan(S_new)) or np.any(np.isinf(S_new)):
            raise RuntimeError("NaN or inf in spin values. Try reducing dt or scaling J.")

        return S_new

    def compute_autocorrelation_directly(self, autocorr_path, max_fraction=0.1, lag_step=10):
        """
        Simulate and compute autocorrelation of z-component directly.
        Vectorized over lags and uses circular buffer. Matches original buffer logic.
        """

        max_lag = int(max_fraction * self.steps)
        lags_idx = np.arange(0, max_lag + 1, lag_step, dtype=int)

        with h5py.File(autocorr_path, "w") as out:
            out.create_dataset("tau", data=lags_idx * self.dt)
            out.create_dataset("Cz", shape=(self.N, len(lags_idx)), dtype='f8')

            C = np.zeros((self.N, len(lags_idx)))

            for r in tqdm(range(self.num_realizations), desc="Simulations", unit="realization"):
                spins = self.random_unit_vector(self.N)
                buffer = np.zeros((max_lag + 1, self.N))
                buffer_ptr = 0

                for t in range(self.steps):
                    z = spins[:, 2]
                    buffer[buffer_ptr] = z

                    # Only compute lags that are valid at this timestep
                    valid_mask = lags_idx <= t
                    if np.any(valid_mask):
                        valid_lags = lags_idx[valid_mask]
                        lag_indices = (buffer_ptr - valid_lags) % (max_lag + 1)
                        past_z = buffer[lag_indices]  # shape: (len(valid_lags), N)
                        C[:, valid_mask] += z[:, None] * past_z.T  # vectorized dot products

                    buffer_ptr = (buffer_ptr + 1) % (max_lag + 1)
                    spins = self.rk4_step(spins, H_eff=self.shift_type)

            # Normalize
            normalization = self.num_realizations * (self.steps - lags_idx[:, None])
            C /= normalization.T
            C /= C[:, [0]]  # Normalize so that C(0) = 1
            out["Cz"][:] = C

        print(f"Autocorrelation saved to {autocorr_path}")



def main():
    # get_trajectories_to_disk(f"simulations/setting_{setting}/spin_{spin_type}/trajectories_{num_real}_reals.h5", num_real, choose_coupling_type, choose_shift_type)
    # get_autocorrelation(f"simulations/setting_{setting}/spin_{spin_type}/trajectories_{num_real}_reals.h5",
    #     autocorr_path=f"simulations/setting_{setting}/spin_{spin_type}/autocorrelation_{num_real}_reals.h5",
    #     lag_step=10, dt=dt)
    sim = Simulation(choose_coupling_type, choose_shift_type, num_real)
    sim.compute_autocorrelation_directly(f"{file_path_to_simulations}/{num_initialized_spins}_spins_initialized/autocorrelation_{num_real}_reals_{time_units}_time_units.h5", 0.1, 10)
    os.system('speaker-test -t sine -f 700 -p 10 -l 1')



if __name__ == "__main__":
    main()
