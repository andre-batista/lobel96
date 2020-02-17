import numpy as np
import copy as cp
import pickle
import time
from scipy import sparse as sps
from numpy import linalg as lag
import matplotlib
import matplotlib.pyplot as plt
from numba import jit

matplotlib.use('Agg')

class Method:
    
    def __init__(self):
        pass
    def solve(self):
        pass
    
class Lobel96(Method):
    
    n_iterations = 10
    initialization_method = 2
    unidim_opt_method = 1
    
    def __init__(self,n_iterations=None,initialization_method=None,
                 unidim_opt_method=None):
        
        if n_iterations is not None:
            self.n_iterations = n_iterations
        if initialization_method is not None:
            self.initialization_method = initialization_method
        if unidim_opt_method is not None:
            self.unidim_opt_method = unidim_opt_method
            
    def load_data(self,filename):
        with open(filename,'rb') as datafile:
            data = pickle.load(datafile)
        return data
    
    def __read_model_variables(self,data):
        self.dx, self.dy = data['dx'], data['dy']
        self.Nx, self.Ny = data['I'], data['J']
        self.epsrb, self.sigb = data['epsrb'], data['sigb']
        self.f = data['frequency']
        self.kb, self.lambda_b = data['kb'], data['lambda_b']
        self.ei = data['ei']
        self.x, self.y = data['x'], data['y']
        self.gs, self.gd = data['gs'], data['gd']
        
    def __read_data_variables(self,data):
        self.et, self.es = data['et'], data['es']
    
    def __init_background(self,N):
        C = sps.dia_matrix((N,N),dtype=complex)
        d = np.zeros((N,1),dtype=complex)
        g = np.ones((N,1),dtype=complex)
        return C, d, g
    
    def __init_backpropagation(self,N,L,gs,es,ei):
        gamma = (lag.norm(np.reshape(gs.conj().T@es,(-1,1)))**2
                 /lag.norm(np.reshape(gs@gs.conj().T@es,(-1,1)))**2)
        w0 = gamma*gs.conj().T@es
        C = sps.dia_matrix(np.diag(1/L*np.sum(w0/ei,1)),dtype=complex)
        d = np.zeros((N,1),dtype=complex)
        g = np.ones((N,1),dtype=complex)
        return C, d, g
    
    def __init_initialguess(self,N,epsr,sig,omega,epsrb,sigb):
        eps0 = 8.85418782e-12
        C = sps.dia_matrix(
            np.diag(np.reshape((epsr-1j*sig/omega/eps0/epsrb)
                               -(epsrb-1j*sigb/omega/eps0/epsrb),-1)),
            dtype=complex
        )
        d = np.zeros((N,1),dtype=complex)
        g = np.ones((N,1),dtype=complex)
        return C, d, g
    
    def __inner(self,v1,v2,d):
        return complex(d*v1.T@np.conj(v2))

    def __gsmethod(self,rho,v):
        gmean = 0.618
        delta = 1.0e-3
        a, b = self.__getinterval(rho,v)   
        xa = b - gmean*(b-a)
        xb = a + gmean*(b-a)
        fxa = lag.norm(np.reshape(rho-xa*v,(-1,1)))**2
        fxb = lag.norm(np.reshape(rho-xb*v,(-1,1)))**2
    
        while (b - a) > delta:
            if fxa > fxb:
                a = xa
                xa = xb
                fxa = fxb
                xb = a + gmean*(b - a)
                fxb = lag.norm(np.reshape(rho-xb*v,(-1,1)))**2
            else:
                b = xb
                xb = xa
                fxb = fxa
                xa = b - gmean*(b-a)
                fxa = lag.norm(np.reshape(rho-xa*v,(-1,1)))**2

        alpha = (b+a)/2
        return alpha
    
    def __getinterval(self,rho,v):
        step0 = 1.0e-03
        a = 0
        Fa = lag.norm(rho.reshape(-1))**2
        b = step0
        Fb = lag.norm(np.reshape(rho-step0*v,(-1,1)))**2
        stepsize = step0
        acceleration = 2
    
        while (Fa > Fb):
            stepsize = acceleration*stepsize
            b = b + stepsize
            Fb = lag.norm(np.reshape(rho-b*v,(-1,1)))**2
        return a,b
    
    def solve(self,model_info,data_info,n_iterations=None,
              initialization_method=None,unidim_opt_method=None,
              initial_guess_epsr=None,initial_guess_sig=None):
                    
        print('========== The Conjugated Gradient Method ==========')                    
        
        if isinstance(model_info,str):
            model_dict = self.load_data(model_info)
            self.__read_model_variables(model_dict)
        else:
            self.__read_model_variables(model_info)
        
        if isinstance(data_info,str):
            data_dict = self.load_data(data_info)
            self.__read_data_variables(data_dict)
        else:
            self.__read_data_variables(data_info)
            
        if n_iterations is not None:
            maxit = n_iterations
        else:
            maxit = self.n_iterations
            
        if initialization_method is not None:
            initopt = initialization_method
        else:
            initopt = self.initialization_method
            
        if unidim_opt_method is not None:
            alphaopt = unidim_opt_method
        else:
            alphaopt = self.unidim_opt_method

        M, L = self.es.shape        # M measurements, L sources
        N = self.ei.shape[0]        # N points within the mesh
        dS = self.dx*self.dy        # Surface element [m^2]
        eps0 = 8.85418782e-12       # Vaccum permittivity [F/m]
        omega = 2*np.pi*self.f      # Angular frequency [rad/sec]
        
        if initopt is 1:
            C, d, g = self.__init_background(N)
        elif initopt is 3:
            C, d, g = self.__init_initialguess(N,initial_guess_epsr,
                                               initial_guess_sig,omega,
                                               self.epsrb,self.sigb)
        else:
            C, d, g = self.__init_backpropagation(N,L,self.gs,self.es,self.ei)
        
        # Initializing variables
        cnvg = np.zeros((maxit+1,2)) # Convergence data
        I = sps.eye(N,dtype=complex) # Identity matrix
        LC = lag.inv(I-self.gd@C) # Initial inversion
        rho = self.es-self.gs@C@LC@self.ei # Initial residual
        
        # Printing first solution
        print('Iteration: 0 - Cost function: %.2e' %lag.norm(rho.reshape(-1))**2)
        
        if initopt is not 2:
            cnvg[0,:] = np.array([lag.norm(rho.reshape(-1))**2,lag.norm(g)])
        else:
            cnvg[0,:] = np.array([lag.norm(rho.reshape(-1))**2,.0])

        totaltime = time.time()
        
        # Iterations
        for it in range(maxit):
    
            tic = time.time()
    
            # Computing the gradient
            gradJ = np.zeros((N,1),dtype=complex)
            for l in range(L):
                gsrho = self.gs.conj().T@rho[:,l]
                gradJ = gradJ-2*np.conj(sps.spdiags(LC@self.ei[:,l],0,N,N)@LC)@gsrho
    
            g_last = np.copy(g)
            g = -gradJ
    
            # Computing the optimum direction
            d = g + self.__inner(g,g-g_last,self.dx)/lag.norm(g_last)**2*d
            D = sps.spdiags(d.reshape(-1),0,N,N)

            # Computing v matrix
            v = self.gs@LC.T@D@LC@self.ei
    
            # Computing step
            if alphaopt is 1:
                alpha = 0
                for l in range(L):
                    alpha = alpha + self.__inner(rho[:,l],v[:,l],self.dx)
                alpha = alpha/lag.norm(v.reshape(-1))**2
            else:
                alpha = self.__gsmethod(rho,v)
      
            # Computing next contrast
            C = C + alpha*D
    
            # Computing the inverse matriz
            LC = lag.inv(I-self.gd@C)
    
            # Computing the residual
            # rho = es-gs@C@LC@ei
            rho = rho-alpha*v
    
            # Computing the objective function
            J = lag.norm(rho.reshape(-1))**2
    
            # Printing iteration
            t = time.time()-tic
            print('Iteration: %d' %(it+1)
                  + ' - Cost function: %.2e' %J
                  + ' - norm(g): %.2e' %lag.norm(g)
                  + ' - time: %.1f sec' %t)
    
            # Saving objetive function and gradient magnitude
            cnvg[it,:] = np.array([J,lag.norm(g)])
            
        totaltime = time.time()-totaltime
        print('Total time: %f' %totaltime + ' seconds')

        # Recovering dielectric properties
        tau = np.reshape(C.data,(self.Nx,self.Ny))           # Constrast fuction
        self.epsr = np.real(tau) + self.epsrb           # Retrieved relative permittivity
        self.sig = -omega*eps0*self.epsrb*np.imag(tau)  # Relative conductivity [S/m]
        
        return cp.deepcopy(self.epsr), cp.deepcopy(self.sig)
    
    def plot_results(self):
        
        plt.imshow(self.epsr, extent = [self.x[0], self.x[-1], 
                                        self.y[0], self.y[-1]])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Recovered Relative Permittivity')
        cbar = plt.colorbar()
        cbar.set_label(r'$|\epsilon_r|$')
        # plt.savefig(expname +'lobel96fig', format = 'jpeg')
        plt.show()
        
    def compute_error(self,epsr_ori,sig_ori):
        
        error_epsr = (1/self.epsr.size
                      *np.sqrt(np.sum(((epsr_ori-self.epsr)/epsr_ori)**2))*100)
        error_sig = (1/self.sig.size
                     *np.sqrt(np.sum(((sig_ori-self.sig)/sig_ori)**2))*100)
        return error_epsr, error_sig
