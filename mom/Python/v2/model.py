'''
  Library  : Method of Moments embedded in a class for electromagnetic foward
             problem.  
  Author   : Jose Olger Vargas and Andre Costa Batista (translation to Python
             and object orientation).
  Date     : OCTOBER 2019, Universidade Federal de Minas Gerais  
  Function : To compute the electric field for scattering
             by an arbitrary shape object with Method of Moments (MoM) 
             and Conjugate Gradient- Fast Fourier Trasnsform (CG-FFT).
'''

# Libraries
import sys
import time
import pickle
import numpy as np
from numpy import linalg as LA
from numpy import fft
import scipy.constants as ct
import scipy.special as spc
import domain as dm
from joblib import Parallel, delayed
import multiprocessing
import matplotlib as mpl
# mpl.use('Agg') # Avoiding error when using ssh protocol
import matplotlib.pyplot as plt

class Model:
    """ MODEL: 
        Class to represent a microwave imaging problem with a solver for 
        sythesize data.  
    
    Constructor arguments:
    -- domain: an object of class Domain
    -- model_name: string
    -- frequencies: either a single or a numpy array of frequencies for 
       measurements [Hz]
    -- incident_field_magnitude: magnitude of time-harmonic incident wave [V/m]
    -- epsilon_r_background: relative permittivity of background medium. 
       Default: 1.0
    -- sigma_background: conductivity of background medium [S/m]. Default: 0.0 
       [S/m]
    -- maximum_iterations: maximum number of iterations allowed for solver
       algorithm. Default: 100
    -- tolerance: maximum error allowed for stop criterion of the solver. 
       Default: 1e-6
    """
    
    domain = dm.Domain(.0,.0,.0,0,0) # Imaging domain object
    model_name = ''
    f = np.array([]) # Frequencies of measuremnt [Hz]
    E0 = float() # Magnitude of incident wave
    max_it, TOL = int(), float() # Maximum number of iterations and error
    # tolerance
    
    def __init__(self,domain,model_name=None,frequencies=None,
                 incident_field_magnitude=1.,epsilon_r_background=1.,
                 sigma_background=.0,maximum_iterations=100, tolerance=1e-6):
        
        self.domain = domain
        self.E0 = incident_field_magnitude
        self.epsilon_rb = epsilon_r_background
        self.sigma_b = sigma_background
        
        if frequencies is not None:
            self.f = frequencies
        if model_name is not None:
            self.model_name = model_name
            
        self.max_it = maximum_iterations
        self.TOL = tolerance
            
    def solve(self,epsilon_r=None,sigma=None,frequencies=None,
              maximum_iterations=None, tolerance=None,file_name=None,
              file_path=''):
        """ Solve the scattered field for a given dielectric map. Either 
        epsilon_r or sigma or both matrices must be given. You may or not give
        a new single or a set of frequencies. If you haven't given yet, you
        must do it now. You may change the maximum number of iterations and
        tolerance error."""
        
        # Check inputs
        if epsilon_r is None and sigma is None:
            print('SOLVE ERROR: Either epsilon_r or sigma or both must be given!')
            sys.exit()
        elif sigma is None:
            Nx, Ny = epsilon_r.shape
            sigma = self.sigma_b*np.ones(epsilon_r.shape)
        elif epsilon_r is None:
            Nx, Ny = sigma.shape
            epsilon_r = self.epsilon_rb*np.ones(sigma.shape)
        else:
            Nx, Ny = epsilon_r.shape
            
        if frequencies is None and (not(isinstance(self.f,float)) 
                                    and self.f.size <= 0):
            print('SOLVE ERROR: Either a frequency or a set of frequencies' +
                  ' must be given!')
            sys.exit()
        elif frequencies is not None:
            self.f = frequencies
        
        if isinstance(self.f,float):
            MONO_FREQUENCY = True
        else:
            MONO_FREQUENCY = False
            
        if maximum_iterations is not None:
            self.max_it = maximum_iterations
        
        if tolerance is not None:
            self.TOL = tolerance
        
        # Mesh configuration
        xm, ym = get_coordinates(self.domain.R_obs,self.domain.M) # measurement points
        xmin, xmax = get_bounds(self.domain.Lx)
        ymin, ymax = get_bounds(self.domain.Ly)
        dx, dy = self.domain.Lx/Nx, self.domain.Ly/Ny
        x, y = get_domain_coordinates(dx,dy,xmin,xmax,ymin,ymax)
        
        lambda_b = get_wavelength(self.f,epsilon_r=self.epsilon_rb)
        kb = get_wavenumber(self.f,epsilon_r=self.epsilon_rb)
        deltasn = dx*dy             # area of the cell
        an = np.sqrt(deltasn/np.pi) # radius of the equivalent circle
        omega = 2*np.pi*self.f
        Xr = get_contrast_map(epsilon_r,sigma,self.epsilon_rb,self.sigma_b,
                              omega)
        
        # Using circular convolution
        [xe,ye] = np.meshgrid(np.arange(xmin-(Nx/2-1)*dx,xmax+Nx/2*dx,dx),
                              np.arange(ymin-(Ny/2-1)*dy,ymax+Ny/2*dy,dy)) # extended domain (2N-1)x(2N-1)
        
        Rmn = np.sqrt(xe**2 + ye**2) # distance between the cells
        Z = self.__get_extended_matrix(Rmn,kb,an,Nx,Ny)
        Ei = self.get_incident_field(Nx,Ny)
        
        if MONO_FREQUENCY:
            b = np.tile(Xr.reshape((-1,1)),(1,self.domain.L))*Ei

        else:
            b = np.zeros((Nx*Ny,self.domain.L,self.f.size),dtype=complex)
            for f in range(self.f.size):
                b[:,:,f] = np.tile(Xr[:,:,f].reshape((-1,1)),(1,self.domain.L))*Ei[:,:,f]

        if MONO_FREQUENCY:
            tic = time.time()
            J,niter,error = self.__CG_FFT(Z,b,Nx,Ny,self.domain.L,Xr,
                                          self.max_it,self.TOL)
            time_cg_fft=time.time()-tic
            
        else:
            J = np.zeros((Nx*Ny,self.domain.L,self.f.size),dtype=complex)
            niter = np.zeros(self.f.size)
            error = np.zeros((self.max_it,self.f.size))
            num_cores = multiprocessing.cpu_count()
            results = (Parallel(n_jobs=num_cores)
                       (delayed(self.__CG_FFT)(np.squeeze(Z[:,:,f]),
                                               np.squeeze(b[:,:,f]),
                                               Nx,Ny,self.domain.L,
                                               np.squeeze(Xr[:,:,f]),
                                               self.max_it,self.TOL) 
                        for f in range(self.f.size)))
            
            for f in range(self.f.size):
                J[:,:,f] = results[f][0]
                niter[f] = results[f][1]
                error[:,f] = results[f][2]
        
        xg = np.tile(xm.reshape((-1,1)),(1,Nx*Ny))
        yg = np.tile(ym.reshape((-1,1)),(1,Nx*Ny))
        Rscat = np.sqrt((xg-np.tile(np.reshape(x,(Nx*Ny,1)).T,
                                    (self.domain.M,1)))**2 
                        + (yg-np.tile(np.reshape(y,(Nx*Ny,1)).T,
                                      (self.domain.M,1)))**2) # Ns x N*N
        
        if MONO_FREQUENCY:
            ## Scattered Field
            Zscat = -1j*kb*np.pi*an/2*spc.jv(1,kb*an)*spc.hankel2(0,kb*Rscat) # Ns x N^2
            Esc_z = Zscat@J  # Ns x Ni
            Et = J/np.tile(Xr.reshape((-1,1)),(1,self.domain.L))
        
        else:
            
            Esc_z = np.zeros((self.domain.M,self.domain.L,self.f.size),
                             dtype=complex)
            Et = np.zeros((Nx*Ny,self.domain.L,self.f.size),dtype=complex)
            
            for f in range(self.f.size):
                Zscat = (-1j*kb[f]*np.pi*an/2*spc.jv(1,kb[f]*an)
                         * spc.hankel2(0,kb[f]*Rscat))
                Esc_z[:,:,f] = Zscat@J[:,:,f]
                Et[:,:,f] = J[:,:,f]/np.tile(Xr[:,:,f].reshape((-1,1)),
                                             (1,self.domain.L))
        
        if file_name is not None:
            self.__save_data(dx,dy,x,y,Ei,Esc_z,Et,Zscat,lambda_b,kb,epsilon_r,
                             sigma,file_name,file_path)
        
        if MONO_FREQUENCY:
            self.Et = np.copy(Et.reshape((Nx,Ny,self.domain.L)))
        else:
            self.Et = np.copy(Et.reshape((Nx,Ny,self.domain.L,self.f.size)))
        
        return Esc_z
        
    def get_incident_field(self,Nx,Ny):
        """ Compute the incident field for a given mesh."""

        phi = get_angles(self.domain.L)
        xmin, xmax = get_bounds(self.domain.Lx)
        ymin, ymax = get_bounds(self.domain.Ly)
        dx, dy = self.domain.Lx/Nx, self.domain.Ly/Ny
        x, y = get_domain_coordinates(dx,dy,xmin,xmax,ymin,ymax)
        kb = get_wavenumber(self.f,epsilon_r=self.epsilon_rb)

        if isinstance(kb,float):
            Ei = self.E0*np.exp(-1j*kb*(x.reshape((-1,1))
                                        @ np.cos(phi.reshape((1,-1))) 
                                        + y.reshape((-1,1))
                                        @ np.sin(phi.reshape((1,-1)))))
        else:
            Ei = np.zeros((Nx*Ny,self.domain.L,kb.size),dtype=complex)
            for f in range(kb.size):
                Ei[:,:,f] = self.E0*np.exp(-1j*kb[f]*(
                    x.reshape((-1,1))@np.cos(phi.reshape((1,-1))) 
                    + y.reshape((-1,1))@np.sin(phi.reshape((1,-1)))
                ))

        return Ei

    def __get_extended_matrix(self,Rmn,kb,an,Nx,Ny):
        """ Return the extended matrix of Method of Moments"""

        if isinstance(kb,float):

            # Matrix elements for off-diagonal entries
            Zmn = ((1j*np.pi*kb*an)/2)*spc.jv(1,kb*an)*spc.hankel2(0,kb*Rmn) # m=/n
            # Matrix elements for diagonal entries 
            Zmn[Nx-1,Ny-1]= ((1j*np.pi*kb*an)/2)*spc.hankel2(1,kb*an)+1 # m==n

            # Extended matrix (2N-1)x(2N-1) 
            Z = np.zeros((2*Nx-1,2*Ny-1),dtype=complex)
            Z[:Nx,:Ny] = Zmn[Nx-1:2*Nx-1,Ny-1:2*Ny-1]
            Z[Nx:2*Nx-1,Ny:2*Ny-1] = Zmn[:Nx-1,:Ny-1]
            Z[:Nx,Ny:2*Ny-1] = Zmn[Nx-1:2*Nx-1,:Ny-1]
            Z[Nx:2*Nx-1,:Ny] = Zmn[:Nx-1,Ny-1:2*Ny-1]
            
        else:
            
            Z = np.zeros((2*Nx-1,2*Ny-1,kb.size),dtype=complex)
            
            for f in range(kb.size):
                
                # Matrix elements for off-diagonal entries
                Zmn = (((1j*np.pi*kb[f]*an)/2)*spc.jv(1,kb[f]*an)
                       * spc.hankel2(0,kb[f]*Rmn)) # m=/n
                # Matrix elements for diagonal entries 
                Zmn[Nx-1,Ny-1]= ((1j*np.pi*kb[f]*an)/2)*spc.hankel2(1,kb[f]*an)+1 # m==n
                
                Z[:Nx,:Ny,f] = Zmn[Nx-1:2*Nx-1,Ny-1:2*Ny-1]
                Z[Nx:2*Nx-1,Ny:2*Ny-1,f] = Zmn[:Nx-1,:Ny-1]
                Z[:Nx,Ny:2*Ny-1,f] = Zmn[Nx-1:2*Nx-1,:Ny-1]
                Z[Nx:2*Nx-1,:Ny,f] = Zmn[:Nx-1,Ny-1:2*Ny-1]
    
        return Z
        
    def __CG_FFT(self,Z,b,Nx,Ny,Ni,Xr,max_it,TOL):
        '''
                Congugate-Gradient Method (CGM)

            inputs:
            * Z:       extended matrix     (2N-1)x(2N-1)
            * b:       excitation source    N^2 x Ni
            * N:       DOI size             1x1
            * Ni:      number of incidences 1x1
            * Xr:      contrast function    NxN
            * max_it:  number of iterations (integer number)
            * TOL:     error tolerance      

            output:  
            * J:       current density N^2xNi
        
        '''

        Jo = np.zeros((Nx*Ny,Ni),dtype=complex) # initial guess
        ro = self.__fft_A(Jo,Z,Nx,Ny,Ni,Xr)-b # ro = A.Jo - b;
        go = self.__fft_AH(ro,Z,Nx,Ny,Ni,Xr) # Complex conjugate AH
        po = -go
        error_res = np.zeros(max_it)

        for n in range(max_it):
    
            alpha = -1*(np.sum(np.conj(self.__fft_A(po,Z,Nx,Ny,Ni,Xr))
                              *(self.__fft_A(Jo,Z,Nx,Ny,Ni,Xr)-b),axis=0)
                        / LA.norm(np.reshape(self.__fft_A(po,Z,Nx,Ny,Ni,Xr),
                                             (Nx*Ny*Ni,1),order='F'),
                                  ord='fro')**2) # 1 x Ni
                                                  
            J = Jo + np.tile(alpha,(Nx*Ny,1))*po 
            r = self.__fft_A(J,Z,Nx,Ny,Ni,Xr)-b
            g = self.__fft_AH(r,Z,Nx,Ny,Ni,Xr) 
    
            error = LA.norm(r)/LA.norm(b) # error tolerance
            print('%.4e' %error)
            error_res[n] = error
            if error < TOL: # stopping criteria
                break
    
            beta = np.sum(np.conj(g)*(g-go),axis=0)/np.sum(np.abs(go)**2,axis=0) 
            p    = -g + np.tile(beta,(Nx*Ny,1))*po 
        
            po = p 
            Jo = J 
            go = g 

        return J,n,error_res

    def __fft_A(self,J,Z,Nx,Ny,Ni,Xr):
        """ Compute Matrix-vector product by using two-dimensional FFT."""

        J = np.reshape(J,(Nx,Ny,Ni))
        Z = np.tile(Z[:,:,np.newaxis],(1,1,Ni))
        e = fft.ifft2(fft.fft2(Z,axes=(0,1))
                      * fft.fft2(J,axes=(0,1),s=(2*Nx-1,2*Ny-1)),axes=(0,1))
        e = e[:Nx,:Ny,:]
        e = np.reshape(e,(Nx*Ny,Ni))
        e = np.reshape(J,(Nx*Ny,Ni)) + np.tile(Xr.reshape((-1,1)),(1,Ni))*e

        return e

    def __fft_AH(self,J,Z,Nx,Ny,Ni,Xr):
        """ Compute Matrix-vector product by using two-dimensional FFT*
            complex conjugate operator."""

        J = np.reshape(J,(Nx,Ny,Ni))
        Z = np.tile(Z[:,:,np.newaxis],(1,1,Ni))
        e = fft.ifft2(fft.fft2(np.conj(Z),axes=(0,1))
                      *fft.fft2(J,axes=(0,1),s=(2*Nx-1,2*Ny-1)),axes=(0,1))
        e = e[:Nx,:Ny,:]
        e = np.reshape(e,(Nx*Ny,Ni))
        e = (np.reshape(J,(Nx*Ny,Ni)) 
             + np.conj(np.tile(Xr.reshape((-1,1)),(1,Ni)))*e)

        return e

    def __save_data(self,dx,dy,x,y,Ei,Es,Et,Zscat,lambda_b,kb,epsilon_r,sigma,
                    filename,filepath):
        """ Save simulation data in a pickle file."""
        
        data = {
            'model_name': self.model_name,
            'Lx':self.domain.Lx, 'Ly':self.domain.Ly,
            'radius_observation':self.domain.R_obs,
            'number_measurements':self.domain.M,
            'number_sources':self.domain.L,
            'dx':dx, 'dy':dy,
            'x':x, 'y':y,
            'incident_field':Ei,
            'scattered_field':Es,
            'total_field':Et,
            'green_function_s':Zscat,
            'wavelength':lambda_b,
            'wavenumber':kb,
            'relative_permittivity_map':epsilon_r,
            'conductivity_map':sigma,
            'relative_permittivity_background':self.epsilon_rb,
            'conductivity_background':self.sigma_b,
            'maximum_number_iterations':self.max_it,
            'error_tolerance':self.TOL
        }
        
        with open(filepath + filename,'wb') as datafile:
            pickle.dump(data,datafile)

    def plot_total_field(self,file_name=None,file_path='',file_format='png',
                         frequency_index=None,source_index=None):

        xmin, xmax = get_bounds(self.domain.Lx)
        ymin, ymax = get_bounds(self.domain.Ly)
        lambda_b = get_wavelength(self.f,epsilon_r=self.epsilon_rb)
        
        xmin, xmax = xmin/lambda_b, xmax/lambda_b
        ymin, ymax = ymin/lambda_b, ymax/lambda_b
      
        if frequency_index is not None and source_index is not None:
            
            plt.imshow(np.abs(self.Et[:,:,source_index,frequency_index]),
                       extent=[xmin[frequency_index],xmax[frequency_index],
                               ymin[frequency_index],ymax[frequency_index]])
            plt.title('Intern field, Source = %d' %(source_index+1) 
                      + ', Frequency = %.3f' %(self.f[frequency_index]/1e9) 
                      + ' [GHz]')
            
            plt.xlabel(r'x [$\lambda_b$]')
            plt.ylabel(r'y [$\lambda_b$]')
            cbar = plt.colorbar()
            cbar.set_label(r'$|E_z^t|$ [V/m]')
             
        elif source_index is not None and isinstance(self.f,float):
            
            plt.imshow(np.abs(self.Et[:,:,source_index]),
                       extent=[xmin,xmax,ymin,ymax])
            plt.title('Intern field, Source = %d' %(source_index+1) 
                       + ', Frequency = %.3f' %(self.f/1e9) + ' [GHz]')
            
            plt.xlabel(r'x [$\lambda_b$]')
            plt.ylabel(r'y [$\lambda_b$]')
            cbar = plt.colorbar()
            cbar.set_label(r'$|E_z^t|$ [V/m]')
            
        elif source_index is None and isinstance(self.f,float):
            
            fig = plt.figure()
            nrow = np.floor(np.sqrt(self.domain.L)).astype(int)
            ncol = np.ceil(self.domain.L/nrow).astype(int)
            for ifig in range(self.domain.L):
                ax = fig.add_subplot(nrow, ncol, ifig+1)
                ax.imshow(np.abs(self.Et[:,:,ifig]),extent=[xmin,xmax,ymin,ymax])
                ax.set_xlabel(r'x [$\lambda_b$]')
                ax.set_ylabel(r'y [$\lambda_b$]')
                cbar = plt.colorbar(ax=ax)
                cbar.set_label(r'$|E_z^t|$ [V/m]')
                ax.set_title('Intern field, Source = %d' %(ifig+1) 
                             + ', Frequency = %.3f' %(self.f/1e9) + ' [GHz]')
                
        elif source_index is None and frequency_index is None:
            
            fig = []
            nrow = np.floor(np.sqrt(self.domain.L)).astype(int)
            ncol = np.ceil(self.domain.L/nrow).astype(int)
            for f in range(self.f.size):
                fig.append(plt.figure())
                for ifig in range(self.domain.L):
                    ax = fig[-1].add_subplot(nrow, ncol, ifig+1)
                    ax.imshow(np.abs(self.Et[:,:,ifig,f]),
                              extent=[xmin[f],xmax[f],ymin[f],ymax[f]])
                    ax.set_xlabel(r'x [$\lambda_b$]')
                    ax.set_ylabel(r'y [$\lambda_b$]')
                    cbar = plt.colorbar()
                    cbar.set_label(r'$|E_z^t|$ [V/m]')
                    ax.set_title('Intern field, Source = %d' %(ifig+1) 
                                 + ', Frequency = %.3f' %(self.f[f]/1e9) 
                                 + ' [GHz]')
        
        else:
            
            fig = plt.figure()
            nrow = np.floor(np.sqrt(len(self.domain.L))).astype(int)
            ncol = np.ceil(self.domain.L/nrow).astype(int)
            for ifig in range(self.domain.L):
                ax = fig.add_subplot(nrow, ncol, ifig+1)
                ax.imshow(np.abs(self.Et[:,:,ifig,frequency_index]),
                          extent=[xmin[frequency_index],xmax[frequency_index],
                                  ymin[frequency_index],ymax[frequency_index]])
                ax.set_xlabel(r'x [$\lambda_b$]')
                ax.set_ylabel(r'y [$\lambda_b$]')
                cbar = plt.colorbar()
                cbar.set_label(r'$|E_z^t|$ [V/m]')
                ax.set_title('Intern field, Source = %d' %(ifig+1) 
                             + ', Frequency = %.3f' %(self.f[frequency_index]
                                                      /1e9) 
                             + ' [GHz]')
        
        if ((frequency_index is not None and source_index is not None)
            or (source_index is not None and isinstance(self.f,float))):
            
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_path+file_name,format=file_format)
                plt.close()
        
        elif isinstance(fig,list):
            
            if file_name is None:
                for i in range(len(fig)):
                    fig[i].show()
            else:
                for i in range(len(fig)):
                    fig[i].savefig(file_path+file_name,format=file_format)
                plt.close()
        
        else:
            
            if file_name is None:
                plt.show()
            else:
                plt.savefig(file_path+file_name,format=file_format)
                plt.close()

     
def get_angles(n_samples):
    """ Compute angles [rad] in a circular array of points equaly spaced."""
    return np.arange(0,2*np.pi,2*np.pi/n_samples)

def get_coordinates(radius,n_samples):
    """ Compute coordinates of points in a circular array equaly spaced."""
    phi = get_angles(n_samples)
    return radius*np.cos(phi), radius*np.sin(phi)

def get_bounds(length):
    """ Compute the standard bound coordinates."""
    return -length/2, length/2

def get_domain_coordinates(dx,dy,xmin,xmax,ymin,ymax):
    """ Return the meshgrid of the image domain."""
    return np.meshgrid(np.arange(xmin+.5*dx,xmax+.5*dx,dx),
                       np.arange(ymin+.5*dy,ymax+.5*dy,dy))
    
def get_wavelength(frequencies,epsilon_r=1.,mu_r=1.):
    """ Compute wavelength [m]."""
    return 1/np.sqrt(epsilon_r*ct.epsilon_0*mu_r*ct.mu_0)/frequencies

def get_wavenumber(frequencies,epsilon_r=1.,mu_r=1.,sigma=.0):
    """ Compute the wavenumber."""
    return 2*np.pi*np.sqrt(epsilon_r*ct.epsilon_0*mu_r*ct.mu_0)*frequencies

def get_contrast_map(epsilon_r,sigma,epsilon_rb,sigma_b,omega):
    """ Compute the contrast function for a given image represented by the
    relative permittivity and conductivity."""
    
    if isinstance(omega,float):
        return ((epsilon_r - 1j*sigma/omega/ct.epsilon_0)
                /(epsilon_rb - 1j*sigma_b/omega/ct.epsilon_0) - 1)
    
    else:
        Xr = np.zeros((epsilon_r.shape[0],epsilon_r.shape[1],omega.size),
                      dtype=complex)
        for f in range(omega.size):
            Xr[:,:,f] = ((epsilon_r - 1j*sigma/omega[f]/ct.epsilon_0)
                         /(epsilon_rb - 1j*sigma_b/omega[f]/ct.epsilon_0) 
                         - 1)
        return Xr