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

eps0, mu0, c = 8.8541878128e-12, 4e-7*np.pi, 299792458.

# Model class
class Model:
    
    expname = ''
    contrast = float()
    frequency = float()
    object_size = float()
    number_sources = int()
    
    def __init__(self,
                 expname,
                 contrast,
                 frequency,
                 object_size,
                 number_sources):
        
        self.expname = expname
        
        self.epsr_b, self.sig_b = 1., 0.
        self.epsr_o, self.sig_o = contrast*self.epsr_b, 0.
        
        
        
        self.I, self.J = I, J
        self.dx, self.dy = dx, dy
        self.epsrb, self.sigb = epsrb, sigb
        self.sampled_frequencies = sampled_frequencies
        self.wv_frequency = wv_frequency
        self.dtheta    
    




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

Rx = pb.get_rx_array(Rs,dtheta,sampled_frequencies.size)
Tx = sc.Source(sc.get_tx_position(Rs,dtheta,0),
               wv.GaussianSignal(dx,dy,wv_frequency),magnitude)

mymodel = slv.FDTD2D(dm.Domain(dx,dy,I,J),epsrb,sigb,sampled_frequencies,ls_x,
                     ls_y,time_window,probes=Rx)

ei = np.zeros((I,J,M),dtype=complex)
es = np.zeros((M,M),dtype=complex)

print('Runing incident field...')
for i in range(M):
   Tx = sc.Source(sc.get_tx_position(Rs,dtheta,i),
               wv.GaussianSignal(dx,dy,wv_frequency),magnitude)
   print('%d' %(i+1) + ' of %d' %M)
   mymodel.run(Tx)
   ei[:,:,i] = np.squeeze(mymodel.get_intern_field())
   for j in range(M):
      es[j,i] = mymodel.probes[j].get_field_freq(0)

epsr = epsrb*np.ones((I,J))
sig = sigb*np.ones((I,J))
epsr[np.ix_(range(round(I/2)-round(.1*I),round(I/2)+round(.1*I)),
            range(round(J/2)-round(.1*J),round(J/2)+round(.1*J)))] = 2.0

et = np.zeros((I,J,M),dtype=complex)

print('Runing total field...')
for i in range(M):
   Tx = sc.Source(sc.get_tx_position(Rs,dtheta,i),
               wv.GaussianSignal(dx,dy,wv_frequency),magnitude)
   print('%d' %(i+1) + ' of %d' %M)
   mymodel.run(Tx,epsr=epsr,sig=sig)
   et[:,:,i] = np.squeeze(mymodel.get_intern_field())
   for j in range(M):
      es[j,i] = mymodel.probes[j].get_field_freq(0)-es[j,i]

et = et.reshape((N,M))
ei = ei.reshape((N,M))
x, y = mymodel.get_intern_coordinates()
rx_xy = np.zeros((M,2))
for i in range(M):
   rx_xy[i,0], rx_xy[i,1] = Rx[i].position[0], Rx[i].position[1]

deltasn = dx*dy
an = np.sqrt(deltasn/np.pi)

xn, yn = np.meshgrid(x,y,indexing='ij')
xn, yn = xn.reshape(N), yn.reshape(N)
R = dst.cdist(rx_xy,np.stack((xn,yn),axis=1),'euclidean')

gs = np.zeros(R.shape,dtype=complex)
gs[R!=0] = 1j/2*np.pi*kb*an*spl.jv(1,kb*an)*spl.hankel2(0,kb*R[R!=0])
gs[R==0] = 1j/2*(np.pi*kb*an*spl.hankel2(1,kb*an)-2j)
gs = -gs

R = dst.cdist(np.stack((xn,yn),axis=1),np.stack((xn,yn),axis=1),'euclidean')
gd = np.zeros(R.shape,dtype=complex)
gd[R!=0] = 1j/2*np.pi*kb*an*spl.jv(1,kb*an)*spl.hankel2(0,kb*R[R!=0])
gd[R==0] = 1j/2*(np.pi*kb*an*spl.hankel2(1,kb*an)-2j)
gd = -gd

data = {
   'dx':dx,
   'dy':dy,
   'I':I,
   'J':J,
   'epsr':epsr,
   'sig':sig,
   'epsrb':epsrb,
   'sigb':sigb,
   'frequency':sampled_frequencies,
   'waveform_frequency':wv_frequency,
   'dtheta':dtheta,
   'Rs':Rs,
   'ls_x':ls_x,
   'ls_y':ls_y,
   'magnitude':magnitude,
   'time_window':time_window,
   'Rx':Rx,
   'kb':kb,
   'lambda_b':lambda_b,
   'ei':ei,
   'et':et,
   'es':es,
   'x':x,
   'y':y,
   'gs':gs,
   'gd':gd
}

with open(expname,'wb') as datafile:
   pickle.dump(data,datafile)