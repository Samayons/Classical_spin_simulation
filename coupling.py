import numpy as np
from system_parameters import gamma, h_chem, active_spins, N

'''Calculation of the coupling constants using the formula for magn. dip. interaction'''

def get_coupling_constants(positions, H0, g, h, coupling_type: str):
  # Determine the number of spins based on the input positions
  N_local = positions.shape[0]
  
  if coupling_type == "dipole-dipole":
    r_mn = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
  
    # Norm of magn. field vector
    norm_H0 = np.linalg.norm(H0)
  
    # Norm of each separation vector |r_i - r_j|
    norms_r_mn = np.linalg.norm(r_mn, axis=-1)
  
    # Dot products with H0 for each separation vector
    H0_dot_r_mn = np.dot(r_mn, H0)
  
    # Safe cosine values
    cos_thetas = np.zeros((N_local, N_local))
    nonzero = (norms_r_mn > 0) & (norm_H0 > 0)
    cos_thetas[nonzero] = H0_dot_r_mn[nonzero] / (norms_r_mn[nonzero] * norm_H0)
  
    # Clamp for safety
    cos_thetas = np.clip(cos_thetas, -1.0, 1.0)
  
    # avoid self-interaction: set r_ii → ∞  so 1/r^3 = 0
    np.fill_diagonal(norms_r_mn, np.inf)
  
    # 1 / |r|^3  (shape (N,N))
    inv_norms_r_mn_cube = 1.0 / norms_r_mn**3
  
    # central dipolar factor  (1 - 3 cos²θ)
    dipolar_factor = 1.0 - 3.0 * cos_thetas**2
  
    # J_z matrix
    Jz = g**2 * h * dipolar_factor * inv_norms_r_mn_cube #/ 1e3 # [J] = [E/h] divided by 1000 for scaling purposes units: [J] = [kHz]
  
    # J_x = J_y = −½ J_z
    Jx = Jy = -0.5 * Jz
  
    return Jx, Jy, Jz
    
  # elif coupling_type == "J-coupling":
  #   Jxx, Jyy, Jzz = -0.41, -0.41, 0.82          # anisotropic nearest-neighbour couplings in kHz
  #   # #Coupling matrices (periodic BC)
  #   Jx = np.zeros((N_local, N_local)); Jy = Jx.copy(); Jz = Jx.copy()
  #   for i in range(N_local):
  #     Jx[i, (i+1)%N_local] = Jx[i, (i-1)%N_local] = Jxx
  #     Jy[i, (i+1)%N_local] = Jy[i, (i-1)%N_local] = Jyy
  #     Jz[i, (i+1)%N_local] = Jz[i, (i-1)%N_local] = Jzz
  #   return Jx, Jy, Jz
  elif coupling_type == "J-coupling":
    coupling_constants = np.array([
      [0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 300.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 300.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0]
    ])*(2 * np.pi / 10)  # Convert kHz to Hz
    if N_local == 16:
      Jx = coupling_constants.copy()
      Jy = coupling_constants.copy()
      Jz = coupling_constants.copy()
    elif N_local == 8:
      Jx = coupling_constants[np.ix_(active_spins, active_spins)].copy()
      Jy = coupling_constants[np.ix_(active_spins, active_spins)].copy()
      Jz = coupling_constants[np.ix_(active_spins, active_spins)].copy()
    else:
      raise ValueError("Invalid number of spins. Expected 8 or 16.")
    return Jx, Jy, Jz
    
  else:
    raise Exception("Please provide a coupling type: dipole-dipole/J-coupling.")


"""Timescale calculation"""
def get_spin_timescale(i, Jxx, Jyy, Jzz, shift_type: str):

  if not (Jxx.shape == Jyy.shape == Jzz.shape):
    raise ValueError("All three matrices must have the same shape.")
  if i < 0 or i >= Jzz.shape[0]:
    raise IndexError("spin index out of range")

  J_squared_row = 0
  J_chem = 0
  
  if shift_type == "with-chemical":
    J_chem = gamma * h_chem[i, 2]
    for j in range(Jzz.shape[0]):
      if j != i:
        J_squared_row += Jxx[i, j]**2 + Jyy[i, j]**2 + Jzz[i, j]**2
  elif shift_type == "without-chemical":
    for j in range(Jzz.shape[0]):
      if j != i:
        J_squared_row += Jxx[i, j]**2 + Jyy[i, j]**2 + Jzz[i, j]**2
  else:
    raise Exception("Please provide a chemical-type: with/without-chemical")    
  
  J_total = (0.25 * J_squared_row) + J_chem**2


  tau = J_total**(-0.5)

  return tau
