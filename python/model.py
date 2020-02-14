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
    
    def __init__(self,modelname,*args):
    
    # def __init__(self, modelname, I, J, dx, dy, epsrb, sigb, wv_frequency,
    #              sampled_frequencies, dtheta, Rs, magnitude, time_window):

    # def __init__(self, modelname, eifile, filepath):
    
        if len(args) == 1 or len(args) == 2:
            
            if isinstance(args[0],str):
                if len(args) == 1:
                    with open(args[0],'rb') as datafile:
                        data = pickle.load(datafile)
                else:
                    with open(args[0]+args[1],'rb') as datafile:
                        data = pickle.load(datafile)
            else:
                data = args[0]
            
            self.__load_incident_file(data)
            
        elif len(args) == 12:
            
            self.modelname = modelname
            self.I, self.J = args[0], args[1]
            self.dx, self.dy = args[2], args[3]
            self.epsrb, self.sigb = args[4], args[5]
            self.wv_frequency = args[6]
            self.sampled_frequencies = args[7]
            self.dtheta, self.Rs = args[8], args[9]
            self.magnitude, self.time_window = args[10], args[11]
        
        else:
            print('Error: wrong number of inputs!')
            exit()
        
        self.N = self.I*self.J
        self.M = round(360/self.dtheta)
        self.ls_x, self.ls_y = 4*self.dx, 4*self.dy
        self.lambda_b = 1/self.sampled_frequencies/np.sqrt(slv.mu_0*self.epsrb*slv.eps_0)
        self.kb = 2*np.pi*self.sampled_frequencies*np.sqrt(slv.mu_0*slv.eps_0*self.epsrb)
        
        self.Rx = pb.get_rx_array(self.Rs,self.dtheta,self.sampled_frequencies.size)
        self.Tx = sc.Source(sc.get_tx_position(self.Rs,self.dtheta,0),
                            wv.GaussianSignal(self.dx,self.dy,self.wv_frequency),self.magnitude)
        
    def __run_fdtd_iteration(self,i):
        
        mymodel = slv.FDTD2D(dm.Domain(self.dx,self.dy,self.I,self.J),self.epsrb,self.sigb,
                             self.sampled_frequencies,self.ls_x,self.ls_y,
                             self.time_window,probes=self.Rx)

        Tx = sc.Source(sc.get_tx_position(self.Rs,self.dtheta,i),
                           wv.GaussianSignal(self.dx,self.dy,self.wv_frequency),
                           self.magnitude)

        mymodel.run(Tx,epsr=self.epsr,sig=self.sig)
        probes = np.zeros(self.M,dtype=complex)
        for m in range(self.M):
            probes[m] = mymodel.probes[m].get_field_freq(0)
        return np.squeeze(mymodel.get_intern_field()), probes
                
    def compute_incident_field(self):
        
        self.epsr = self.epsrb*np.ones((self.I,self.J))
        self.sig = self.sigb*np.ones((self.I,self.J))
        self.ei = np.zeros((self.I,self.J,self.M),dtype=complex)
        self.ei_pr = np.zeros((self.M,self.M),dtype=complex)
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.__run_fdtd_iteration)(i) for i in range(self.M))
        for m in range(self.M):
            self.ei[:,:,m] = results[m][0]
            self.ei_pr[m,:] = results[m][1]

    def compute_total_field(self,epsr=None,sig=None):
        
        if epsr is None:
            self.epsr = self.epsrb*np.ones((self.I,self.J))
        else:
            self.epsr = epsr
        
        if sig is None:
            self.sig = self.sigb*np.ones((self.I,self.J))
        else:
            self.sig = sig
            
        self.et = np.zeros((self.I,self.J,self.M),dtype=complex)
        self.et_pr = np.zeros((self.M,self.M),dtype=complex)
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.__run_fdtd_iteration)(i) for i in range(self.M))
        for m in range(self.M):
            self.et[:,:,m] = results[m][0]
            self.et_pr[m,:] = results[m][1]
        self.et = self.et.reshape((self.N,self.M))
            
    def compute_scattered_field(self,ei_pr=None):
        
        if ei_pr is None:
            ei_pr = self.ei_pr
        
        self.es = self.et_pr - ei_pr
        
    def compute_green_function_s(self):
        
        mymodel = slv.FDTD2D(dm.Domain(self.dx,self.dy,self.I,self.J),self.epsrb,self.sigb,
                             self.sampled_frequencies,self.ls_x,self.ls_y,
                             self.time_window,probes=self.Rx)
        
        self.x, self.y = mymodel.compute_intern_coordinates()
        rx_xy = np.zeros((self.M,2))
        for i in range(self.M):
            rx_xy[i,0], rx_xy[i,1] = self.Rx[i].position[0], self.Rx[i].position[1]

        deltasn = self.dx*self.dy
        an = np.sqrt(deltasn/np.pi)

        xn, yn = np.meshgrid(self.x,self.y,indexing='ij')
        xn, yn = xn.reshape(self.N), yn.reshape(self.N)
        R = dst.cdist(rx_xy,np.stack((xn,yn),axis=1),'euclidean')

        self.gs = np.zeros(R.shape,dtype=complex)
        self.gs[R!=0] = 1j/2*np.pi*self.kb*an*spl.jv(1,self.kb*an)*spl.hankel2(0,self.kb*R[R!=0])
        self.gs[R==0] = 1j/2*(np.pi*self.kb*an*spl.hankel2(1,self.kb*an)-2j)
        self.gs = -self.gs
        
    def compute_green_function_d(self):
        
        mymodel = slv.FDTD2D(dm.Domain(self.dx,self.dy,self.I,self.J),self.epsrb,self.sigb,
                             self.sampled_frequencies,self.ls_x,self.ls_y,
                             self.time_window,probes=self.Rx)
        
        self.x, self.y = mymodel.compute_intern_coordinates()
        deltasn = self.dx*self.dy
        an = np.sqrt(deltasn/np.pi)

        xn, yn = np.meshgrid(self.x,self.y,indexing='ij')
        xn, yn = xn.reshape(self.N), yn.reshape(self.N)
        R = dst.cdist(np.stack((xn,yn),axis=1),np.stack((xn,yn),axis=1),'euclidean')
        
        self.gd = np.zeros(R.shape,dtype=complex)
        self.gd[R!=0] = 1j/2*np.pi*self.kb*an*spl.jv(1,self.kb*an)*spl.hankel2(0,self.kb*R[R!=0])
        self.gd[R==0] = 1j/2*(np.pi*self.kb*an*spl.hankel2(1,self.kb*an)-2j)
        self.gd = -self.gd
        
    def generate_setup_data(self,filename,filepath=''):
        
        self.compute_incident_field()
        self.compute_green_function_s()
        self.compute_green_function_d()

        data = {
            'dx':self.dx,
            'dy':self.dy,
            'I':self.I,
            'J':self.J,
            'epsrb':self.epsrb,
            'sigb':self.sigb,
            'sampled_frequencies':self.sampled_frequencies,
            'wv_frequency':self.wv_frequency,
            'dtheta':self.dtheta,
            'Rs':self.Rs,
            'ls_x':self.ls_x,
            'ls_y':self.ls_y,
            'magnitude':self.magnitude,
            'time_window':self.time_window,
            'Rx':self.Rx,
            'kb':self.kb,
            'lambda_b':self.lambda_b,
            'ei':self.ei,
            'ei_pr':self.ei_pr,
            'x':self.x,
            'y':self.y,
            'gs':self.gs,
            'gd':self.gd
        }

        with open(filepath + filename,'wb') as datafile:
            pickle.dump(data,datafile)
    
    def gerenate_case_data(self,epsr=None,sig=None,ei_data=None,filepath=''):
        
        if ei_data is None and self.ei_pr is None:
            self.compute_incident_field()
        elif isinstance(ei_data,str):
            with open(ei_data,'rb') as datafile:
                data = pickle.load(datafile)
            self.__load_incident_file(data)
        else:
            self.__load_incident_file(ei_data)
            
        self.compute_total_field(epsr=epsr,sig=sig)
        self.compute_scattered_field()
        
        data = {
            'et': self.et,
            'epsr': epsr,
            'sig': sig,
            'es': self.es
        }
        
        with open(filepath + self.modelname,'wb') as datafile:
            pickle.dump(data,datafile)
        
    def __load_incident_file(self,data):
                
        self.I, self.J = data['I'], data['J']
        self.dx, self.dy = data['dx'], data['dy']
        self.x, self.y = data['x'], data['y']
        self.epsrb, self.sigb = data['epsrb'], data['sigb']
        self.wv_frequency = data['wv_frequency']
        self.sampled_frequencies = data['sampled_frequencies']
        self.dtheta, self.Rs = data['dtheta'], data['Rs']
        self.magnitude, self.time_window = data['magnitude'], data['time_window']
        self.ei_pr = data['ei_pr']
    
    def compute_time_window(self):
        
        v = 1/np.sqrt(slv.mu_0*self.epsrb*slv.eps_0)
        self.time_window = 1.1*self.Rs/v