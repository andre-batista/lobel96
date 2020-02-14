import numpy as np
import model as md

# Main parameters
setup_name = 'setup_exp01_basic'
filepath = ''
I, J = 100, 100
dx, dy = 1e-3, 1e-3
epsrb, sigb = 1., .0
wv_frequency, sampled_frequencies = 1e9, np.array([1e9])
dtheta, Rs = 12, .07
magnitude_signal = 1e1
time_window = 1.4e-8

# Create model object
setup_model = md.Model(setup_name,I,J,dx,dy,epsrb,sigb,wv_frequency,
                             sampled_frequencies,dtheta,Rs,magnitude_signal,
                             time_window)

# Generate setup data - structs, incident field and Green functions
setup_model.generate_setup_data(setup_name,filepath)