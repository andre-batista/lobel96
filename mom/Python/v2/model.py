import sys
import numpy as np
import scipy.constants as ct
import domain as dm

class Model:
    
    domain = dm.Domain(.0,.0,.0,0,0)
    f = np.array([])
    E0 = float()
    
    def __init__(self,domain,frequencies=None,incident_field_magnitude=1.,
                 epsilon_r_background=1.,sigma_background=.0):
        
        self.domain = domain
        self.E0 = incident_field_magnitude
        self.epsilon_rb = epsilon_r_background
        self.sigma_b = sigma_background
        
        if frequencies is not None:
            self.f = frequencies
            
    def solve(self,epsilon_r=None,sigma=None,frequencies=None):
        
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
            print('SOLVE ERROR: Either a frequency or a set of frequencies must be given!')
            sys.exit()
        elif frequencies is not None:
            self.f = frequencies
            
        xm, ym = get_coordinates(self.domain.R_obs,self.domain.M)
        xmin, xmax = get_bounds(self.domain.Lx)
        ymin, ymax = get_bounds(self.domain.Ly)
        dx, dy = self.domain.Lx/Nx, self.domain.Ly/Ny
        x, y = get_domain_coordinates(dx,dy,xmin,xmax,ymin,ymax)
        lambda_b = get_wavelength(self.f,epsilon_r=self.epsilon_rb)
        kb = get_wavenumber(self.f,epsilon_r=self.epsilon_rb)
    
def get_coordinates(radius,n_samples):
    
    phi = np.arange(0,2*np.pi,2*np.pi/n_samples)
    return radius*np.cos(phi), radius*np.sin(phi)

def get_bounds(length):
    return -length/2, length/2

def get_domain_coordinates(dx,dy,xmin,xmax,ymin,ymax):
    return np.meshgrid(np.arange(xmin+.5*dx,xmax+.5*dx,dx),
                       np.arange(ymin+.5*dy,ymax+.5*dy,dy))
    
def get_wavelength(frequencies,epsilon_r=1.,mu_r=1.):
    return 1/np.sqrt(epsilon_r*ct.epsilon_0*mu_r*ct.mu_0)/frequencies

def get_wavenumber(frequencies,epsilon_r=1.,mu_r=1.,sigma=.0):
    return 2*np.pi*np.sqrt(epsilon_r*ct.epsilon_0*mu_r*ct.mu_0)*frequencies
    