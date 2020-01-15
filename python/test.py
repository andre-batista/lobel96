# Importing general libraries
import numpy as np
import copy as cp
import pickle
from scipy import special as spl
from scipy.spatial import distance as dst
import time as tm

# Importing FDTD2D library
import fdtd2d as slv
import domain as dm
import boundary as bd
import waveform as wv
import source as sc
import probe as pb

# Model parameters
expname = 'basic'
I, J = 50, 50   # Number of cells in x,y-axis
N = I*J
dx, dy = 5e-3, 5e-3
epsrb, sigb = 1., 0.
sampled_frequencies = np.array([800e6])
wv_frequency = 800e6
dtheta = 12
M = round(360/dtheta)
Rs = 18e-2
ls_x, ls_y = 4*dx, 4*dy
magnitude = 1e2
time_window = 5e-8
lambda_b = 1/sampled_frequencies/np.sqrt(slv.mu_0*epsrb*slv.eps_0)
kb = 2*np.pi*sampled_frequencies*np.sqrt(slv.mu_0*slv.eps_0*epsrb)

Rx = pb.get_rx_array(Rs,dtheta,sampled_frequencies.size,save_signal=True)
Tx = sc.Source(sc.get_tx_position(Rs,dtheta,0),
               wv.GaussianSignal(dx,dy,wv_frequency),magnitude)

mymodel = slv.FDTD2D(dm.Domain(dx,dy,I,J),epsrb,sigb,sampled_frequencies,ls_x,
                     ls_y,time_window,probes=Rx)

i = 19

Tx = sc.Source(sc.get_tx_position(Rs,dtheta,i),
               wv.GaussianSignal(dx,dy,wv_frequency),magnitude)

epsr = epsrb*np.ones((I,J))
sig = sigb*np.ones((I,J))
# epsr[np.ix_(range(round(I/4)-round(.1*I),round(I/4)+round(.1*I)),
#             range(round(J/4)-round(.1*J),round(J/4)+round(.1*J)))] = 5.0

mymodel.run(Tx,epsr=epsr,sig=sig)
mymodel.plot_probes(index=3)
print(mymodel.et[0,0,0
                 ])
# mymodel.save_field_figure('test','jpeg')
