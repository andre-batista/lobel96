# Importing general libraries
import numpy as np
import copy as cp
import pickle
from scipy import special as spl
from scipy.spatial import distance as dst
from matplotlib import pyplot as plt

# Importing FDTD2D library
import fdtd2d as slv
import domain as dm
import boundary as bd
import waveform as wv
import source as sc
import probe as pb

# Model parameters
Lx, Ly = 1e-1, 1e-1
I, J = 100, 100
dx, dy = Lx/I, Ly/J
epsrb, sigb = 1., 0.
sampled_frequencies = np.array([1e9])
wv_frequency = 1e9
dtheta = 12
M = round(360/dtheta)
Rs = 0.070
ls_x, ls_y = 4*dx, 4*dy
magnitude = 1e1
time_window = 1.4e-8

N = I*J
lambda_b = 1/sampled_frequencies/np.sqrt(slv.mu_0*epsrb*slv.eps_0)
kb = 2*np.pi*sampled_frequencies*np.sqrt(slv.mu_0*slv.eps_0*epsrb)

Rx = pb.get_rx_array(Rs,dtheta,sampled_frequencies.size,save_signal=True)

mymodel = slv.FDTD2D(dm.Domain(dx,dy,I,J),epsrb,sigb,sampled_frequencies,ls_x,
                     ls_y,time_window,probes=Rx)

Tx = sc.Source(sc.get_tx_position(Rs,dtheta,0),
               wv.GaussianSignal(dx,dy,wv_frequency),magnitude)

epsr = epsrb*np.ones((I,J))
# epsr[np.ix_(range(round(I/2)-round(.1*I),round(I/2)+round(.1*I)),
#             range(round(J/2)-round(.1*J),round(J/2)+round(.1*J)))] = 9.8

mymodel.run(Tx,epsr=epsr)
mymodel.plot_probes(index=15)
# mymodel.save_field_figure('test','jpeg')
