# # Importing general libraries
# import numpy as np
# import copy as cp
# import pickle
# from scipy import special as spl
# from scipy.spatial import distance as dst
# import time as tm

# # Importing FDTD2D library
# import fdtd2d as slv
# import domain as dm
# import boundary as bd
# import waveform as wv
# import source as sc
# import probe as pb

# # Model parameters
# expname = 'basic'
# I, J = 50, 50   # Number of cells in x,y-axis
# N = I*J
# dx, dy = 5e-3, 5e-3
# epsrb, sigb = 1., 0.
# sampled_frequencies = np.array([800e6])
# wv_frequency = 800e6
# dtheta = 12
# M = round(360/dtheta)
# Rs = 18e-2
# ls_x, ls_y = 4*dx, 4*dy
# magnitude = 1e2
# time_window = 5e-8
# lambda_b = 1/sampled_frequencies/np.sqrt(slv.mu_0*epsrb*slv.eps_0)
# kb = 2*np.pi*sampled_frequencies*np.sqrt(slv.mu_0*slv.eps_0*epsrb)

# Rx = pb.get_rx_array(Rs,dtheta,sampled_frequencies.size,save_signal=True)
# Tx = sc.Source(sc.get_tx_position(Rs,dtheta,0),
#                wv.GaussianSignal(dx,dy,wv_frequency),magnitude)

# mymodel = slv.FDTD2D(dm.Domain(dx,dy,I,J),epsrb,sigb,sampled_frequencies,ls_x,
#                      ls_y,time_window,probes=Rx)

# i = 19

# Tx = sc.Source(sc.get_tx_position(Rs,dtheta,i),
#                wv.GaussianSignal(dx,dy,wv_frequency),magnitude)

# epsr = epsrb*np.ones((I,J))
# sig = sigb*np.ones((I,J))
# # epsr[np.ix_(range(round(I/4)-round(.1*I),round(I/4)+round(.1*I)),
# #             range(round(J/4)-round(.1*J),round(J/4)+round(.1*J)))] = 5.0

# mymodel.run(Tx,epsr=epsr,sig=sig)
# mymodel.plot_probes(index=3)
# print(mymodel.et[0,0,0
#                  ])
# # mymodel.save_field_figure('test','jpeg')

import numpy as np
from numba import jit
import time as tm

@jit(nopython=True)
def weighted_laplacian(c,br,bi):
    I,J = c.shape
    re = np.zeros((I,J))
    im = np.zeros((I,J))
    for i in range(1,I-1):
        for j in range(1,J-1):
            re[i,j] = -(2*br[i,j] + br[i-1,j] + br[i,j-1])*np.real(c[i,j]) + br[i,j]*np.real(c[i,j+1]) + br[i,j]*np.real(c[i+1,j]) + br[i,j-1]*np.real(c[i,j-1]) + br[i-1,j]*np.real(c[i-1,j])
            im[i,j] = -(2*bi[i,j] + bi[i-1,j] + bi[i,j-1])*np.imag(c[i,j]) + bi[i,j]*np.imag(c[i,j+1]) + bi[i,j]*np.imag(c[i+1,j]) + bi[i,j-1]*np.imag(c[i,j-1]) + bi[i-1,j]*np.imag(c[i-1,j])
    re = re.reshape((I*J,1))
    im = im.reshape((I*J,1))
    return re, im

c = np.random.rand(1000,1000)
br = np.random.rand(1000,1000)
bi = np.random.rand(1000,1000)

t1 = tm.time()
re, im = weighted_laplacian(c,br,bi)
print('Time elapsed %f' %(tm.time()-t1))