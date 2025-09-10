import numpy as np

r_vector = 1e-8*np.array([
    [0, 0, 0], #2
    [0.34, 2.20, 0], #3
    [6.38, 1.01, 0], #7
    [8.72, 0.22, 0], #8
    [7.38, -5.04, 0], #12
    [5.71, -7.05, 0], #13
    [0.65, -7.05, 0], #17
    [0, -5.00, 0], #18
    [0.67, -1.30, 3.50], #2'
    [2.66, -2.30, 3.50], #3'
    [5.19, 3.31, 3.50], #7'
    [5.91, 5.67, 3.50], #8'
    [0.85, 7.63, 3.50], #12'
    [-1.76, 7.43, 3.50], #13'
    [-4.69, 3.32, 3.50], #17'
    [-3.40, 1.60, 3.50] #18'
]) # cm

chemical_shift = np.array([
    134.3, #2
    193.5, #3
    46.1, #7
    32.1, #8
    119.5, #12
    189.4, #13
    29.9, #17
    49.1, #18
    136.0, #2'
    195.6, #3'
    48.3, #7'
    28.9, #8'
    128.8, #12'
    187.5, #13'
    28.4, #17'
    50.9 #18'
    ])*(1e-6) #ppm

active_spins = [0,1,6,7,8,9,14,15]
N = r_vector[active_spins].shape[0]        # number of spins: r_vector[active_spins].shape[0] to consider only eight spins
gamma  = 6.7262e+3 # rad/s*G
h_bar = 6.626e-27 / (2*np.pi) # g*cm^2/s
H0 = 47000 # Gauss(G)
H_ext  = np.array([0.0, 0.0, H0])

if N == 8:
    h_chem = chemical_shift[active_spins][:, np.newaxis] * H_ext
elif N == 16:
    h_chem = chemical_shift[:, np.newaxis] * H_ext
