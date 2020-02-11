# Importing general libraries
import numpy as np
import copy as cp
import pickle
from scipy import special as spl
from scipy.spatial import distance as dst
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

# Importing FDTD2D library
import fdtd2d as slv
import domain as dm
import boundary as bd
import waveform as wv
import source as sc
import probe as pb

class Model:
    
    modelname = ''
    I, J = int(), int()
    dx, dy = float(), float()
    epsrb, sigb = float(), float()
    wv_frequency, sampled_frequencies = float(), np.array([])
    dtheta, Rs = float(), float()
    magnitude, time_window = float(), float()
    
    def __init__(self, modelname, I, J, dx, dy, epsrb, sigb, wv_frequency,
                 sampled_frequencies, dtheta, Rs, magnitude, time_window):
        
        self.modelname = modelname
        self.I, self.J = I, J
        self.dx, self.dy = dx, dy
        self.epsrb, self.sigb = epsrb, sigb
        self.wv_frequency = wv_frequency
        self.sampled_frequencies = sampled_frequencies
        self.dtheta, self.Rs = dtheta, Rs
        self.magnitude, self.time_window = magnitude, time_window
        
        self.N = I*J
        self.M = round(360/dtheta)
        self.ls_x, self.ls_y = 4*dx, 4*dy
        self.lambda_b = 1/sampled_frequencies/np.sqrt(slv.mu_0*epsrb*slv.eps_0)
        self.kb = 2*np.pi*sampled_frequencies*np.sqrt(slv.mu_0*slv.eps_0*epsrb)
        
        self.Rx = pb.get_rx_array(Rs,dtheta,sampled_frequencies.size)
        self.Tx = sc.Source(sc.get_tx_position(Rs,dtheta,0),
                            wv.GaussianSignal(dx,dy,wv_frequency),magnitude)
        
        # self.mymodel = slv.FDTD2D(dm.Domain(dx,dy,I,J),epsrb,sigb,
        #                           sampled_frequencies,self.ls_x,self.ls_y,
        #                           time_window,probes=self.Rx)
        
    # def compute_incident_field(self):
        
    #     self.ei = np.zeros((self.I,self.J,self.M),dtype=complex)
    #     print('Runing incident field...')
    #     for i in range(self.M):
    #         Tx = sc.Source(sc.get_tx_position(self.Rs,self.dtheta,i),
    #                        wv.GaussianSignal(self.dx,self.dy,self.wv_frequency),
    #                        self.magnitude)
    #         print('%d' %(i+1) + ' of %d' %self.M)
    #         self.mymodel.run(Tx)
    #         self.ei[:,:,i] = np.squeeze(self.mymodel.get_intern_field())
    
    def run_fdtd_iteration(self,i):
        
        mymodel = slv.FDTD2D(dm.Domain(self.dx,self.dy,self.I,self.J),self.epsrb,self.sigb,
                             self.sampled_frequencies,self.ls_x,self.ls_y,
                             self.time_window,probes=self.Rx)

        Tx = sc.Source(sc.get_tx_position(self.Rs,self.dtheta,i),
                           wv.GaussianSignal(self.dx,self.dy,self.wv_frequency),
                           self.magnitude)
        # print('%d' %(i+1) + ' of %d' %self.M)
        mymodel.run(Tx)
        return np.squeeze(mymodel.get_intern_field())
                
    def compute_incident_field(self):
        
        self.ei = np.zeros((self.I,self.J,self.M),dtype=complex)
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.run_fdtd_iteration)(i) for i in range(self.M))
         
"""

                HOW I TRIES TO RUN IT PARALLELY
                
    def run_fdtd_iteration(self,i):
        Tx = sc.Source(sc.get_tx_position(self.Rs,self.dtheta,i),
                           wv.GaussianSignal(self.dx,self.dy,self.wv_frequency),
                           self.magnitude)
        print('%d' %(i+1) + ' of %d' %self.M)
        self.mymodel.run(Tx)
        return np.squeeze(self.mymodel.get_intern_field())
                
    def compute_incident_field(self):
        
        self.ei = np.zeros((self.I,self.J,self.M),dtype=complex)
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.run_fdtd_iteration)(i) for i in range(2))


                MODEL FOR PARALELLISM

# what are your inputs, and what operation do you want to 
# perform on each input. For example...
inputs = range(10) 
def processInput(i):
    return i * i
 
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)

"""     
