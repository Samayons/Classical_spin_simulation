
After downloading this project, the first thing is choosing how many spins to be simulated: 8 or 16.
The default is set to 8 spins. To change this, open "systems_parameters.py" and replace in line 42 "N = r_vector[active_spins].shape[0]" to "N = r_vector.shape[0]".

To set the type of interaction hamiltonian(setting1/2/3), go to line 30 in "main.py" and change it to your preferred setting type.

- setting 1: J-coupling with chemical shift
- setting 2: dipole-dipole coupling with chemical shift
- setting 3: dipole-dipole coupling without chemical shift

The simulation time and number of ensembles can also be changed here, line 28,29.

Run "main.py" to get the whole trajectory, which will be saved to the simulation directory, then run "plot_autocorrelation.py" to get the plots which will be saved as a pdf to the results directory.
