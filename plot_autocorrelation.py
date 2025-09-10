import matplotlib.pyplot as plt
import numpy as np
import h5py
from main import num_real, file_path_to_simulations, time_units, num_initialized_spins, setting, file_path_to_results
from system_parameters import active_spins, N

plt.rcParams.update({
    'font.size': 28,        # All text
    #'axes.labelweight': 'bold',  # Bold axis labels (optional)
    'figure.dpi': 300       # High resolution for export
})

spin_mapping = np.array(["2", "3", "7", "8", "12", "13", "17", "18", "2'", "3'", "7'", "8'", "12'", "13'", "17'", "18'"])


if N == 8:
    selected_spins = spin_mapping[active_spins]  # Use only the first 8 spins for 8 spins case
elif N == 16:
    selected_spins = spin_mapping  # Use all spins for 16 spins case

"""Plot autocorrelation"""

with h5py.File(f"{file_path_to_simulations}/{num_initialized_spins}_spins_initialized/autocorrelation_{num_real}_reals_{time_units}_time_units.h5", "r") as f:
    tau = f["tau"][:]
    Cz = f["Cz"][:]

print(f"tau shape: {tau.shape}, Cz shape: {Cz.shape}")
for i in range(Cz.shape[0]):
  plt.figure(figsize=(10, 6))
  plt.plot(tau, Cz[i, :], label=f"{N} spins $C^z-{selected_spins[i]}/{selected_spins[i]}$")
  plt.xlabel("Time (s)")
  plt.ylabel(f"$C_{{{selected_spins[i]},{selected_spins[i]}}}$")
  plt.legend()
  plt.ticklabel_format(useOffset=False, style='plain', axis='y')
  plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
  plt.grid(True)
  plt.savefig(f"{file_path_to_results}/autocorrelation_{num_real}_reals_{time_units}_time_units_spin{selected_spins[i]}.pdf", dpi=300)
  print(f"Plot saved to {file_path_to_results}/autocorrelation_{num_real}_reals_{time_units}_time_units_spin{selected_spins[i]}.pdf")

