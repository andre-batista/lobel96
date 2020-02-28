'''
  Program  : MoM_2D CG-FFT
  Author   : Jose Olger Vargas
  Date     : OCTOBER 2019, Universidade Federal de Minas Gerais  
  function : To compute the electric field for scattering
             by an arbitrary shape object with Method of Moments (MoM) 
            and Conjugate Gradient- Fast Fourier Trasnsform (CG-FFT).
                
===============================================================================
INPUTS: 
Note: the scattered field is measured in a circle of radius R_obs
* R_obs          % Far field observation radius  (m)
* E0             % Amplitude of the incident field (V/m)
* Ni             % Number of incidences
* Ns             % Number of receivers (scattered fields)
* f              % Frequency of the incident plane wave (Hz)
* N              % number of cells (square domain is NxN)
* eps_r          % Relative permittivity of the x,y components. (2D matrix NxN) 

OUTPUT: Scattered Field
Esc_z 
===============================================================================
'''

import numpy as np
import scipy.special as spc
import time
import CG_FFT as slv
import matplotlib.pyplot as plt

print('Computing Forward Problem')
## DEFINE PARAMETERS
R_obs = 3
E0 = 1
Ni = 1 # Points measured from phi=0
Ns = 16
f  = 300e6
N  = 60 # (DOI size is NxN)
eps_obj = 3.2 # Relative permittivity of the object
epsb    = 1 # Relative permittivity of background
sig_obj = 0 # Conductivity of the object
sigb    = 0 # Conductivity of background

## INCIDENT AND SCATTERED ANGLES
phi_i = np.arange(0,2*np.pi,2*np.pi/Ni) # angles of incidence
phi_s = np.arange(0,2*np.pi,2*np.pi/(Ns)) # scattered Field phi angles

# Observation Points from out of DOI  xs= x-axis, ys=y-axis
xs = R_obs*np.cos(phi_s)  
ys = R_obs*np.sin(phi_s) 

# case: coordinate inputs xs and ys
# xs = -1:0.025:1; 
# ys =  zeros(1,Ns); % y=0.

## DISCRETIZATION BY CELLS
# Setting 2D mesh size, define the limits of the DOI
xmin, xmax = -1, 1
ymin, ymax = -1, 1
# Length of each cell in x and y-axis
dx = (xmax-xmin)/N
dy = (ymax-ymin)/N

# Centers of each cell
[x,y] = np.meshgrid(np.arange(xmin+.5*dx,xmax+.5*dx,dx),
                    np.arange(ymin+.5*dy,ymax+.5*dy,dy)) # NxN

## Define constants
lambda_b = 299792458/f      #wavelength
kb = 2*np.pi/lambda_b       # wavenumber of background
deltasn = dx*dy             # area of the cell
an = np.sqrt(deltasn/np.pi) # radius of the equivalent circle

## DEFINE CONTRAST FUNCTION
omega = 2*np.pi*f                    # angular frequency
eps0  = 8.85418782e-12               # Permittivity of vacuum 
eps_r = epsb*np.ones((N,N))          # NxN
sig   = sigb*np.ones((N,N))          # NxN

# Assigning materials
# Defining a cylinder of radius 0.5 with eps_r and sigma
eps_r[(x**2+ y**2) <= 0.5**2] = eps_obj 
sig[(x**2+ y**2) <= 0.5**2]   = sig_obj

# Contrast function: \Chi(r)
Xr = (eps_r - 1j*sig/omega/eps0)/(epsb- 1j*sigb/omega/eps0) - 1

## Computing EFIE 
# Using circular convolution
[xe,ye] = np.meshgrid(np.arange(xmin-(N/2-1)*dx,xmax+N/2*dx,dx),
                      np.arange(ymin-(N/2-1)*dy,ymax+N/2*dy,dy)) # extended domain (2N-1)x(2N-1)

Rmn = np.sqrt(xe**2 + ye**2) # distance between the cells

# Matrix elements for off-diagonal entries
Zmn = ((1j*np.pi*kb*an)/2)*spc.jv(1,kb*an)*spc.hankel2(0,kb*Rmn) # m=/n
# Matrix elements for diagonal entries 
Zmn[N-1,N-1]= ((1j*np.pi*kb*an)/2)*spc.hankel2(1,kb*an)+1 # m==n

# Extended matrix (2N-1)x(2N-1) 
Z = np.zeros((2*N-1,2*N-1),dtype=complex)
Z[:N,:N] = Zmn[N-1:2*N-1,N-1:2*N-1]
Z[N:2*N-1,N:2*N-1] = Zmn[:N-1,:N-1]
Z[:N,N:2*N-1] = Zmn[N-1:2*N-1,:N-1]
Z[N:2*N-1,:N] = Zmn[:N-1,N-1:2*N-1]

## Incident Plane Wave [Ei]
Ei = E0*np.exp(-1j*kb*(x.reshape((-1,1),order='F')@np.cos(phi_i.reshape((1,-1),order='F')) 
                       + y.reshape((-1,1),order='F')@np.sin(phi_i.reshape((1,-1)))))
b = np.tile(Xr.reshape((-1,1),order='F'),(1,Ni))*Ei 

## Using Conjugate-gradient- Fast Fourier Transform Procedure
# Solving linear equation system Ax=b
max_it = 400        # number of iterations
TOL = 1e-6          # tolerance
tic = time.time()
J,niter,error = slv.CG_FFT(Z,b,N,Ni,Xr,max_it,TOL); 
time_cg_fft=time.time()-tic

## Observation Points from out of DOI
xg = np.tile(xs.reshape((-1,1),order='F'),(1,N*N))
yg = np.tile(ys.reshape((-1,1),order='F'),(1,N*N))

Rscat = np.sqrt((xg-np.tile(np.reshape(x,(N*N,1),order='F').T,(Ns,1)))**2
                + (yg-np.tile(np.reshape(y,(N*N,1),order='F').T,(Ns,1)))**2) # Ns x N*N

## Scattered Field
Zscat = -1j*kb*np.pi*an/2*spc.jv(1,kb*an)*spc.hankel2(0,kb*Rscat) # Ns x N^2
Esc_z = Zscat@J  # Ns x Ni

print(Esc_z.shape)
plt.plot(np.real(Esc_z),np.imag(Esc_z))
plt.show()

#save('Esc_z.mat');

# Plotting results
# figure(1)
# hold on
# plot(xs,abs(Esc_z),'b','LineWidth',1)
# title('Scattered electric field by a dielectric cylinder')
# xlabel('x (m)')
# ylabel(' abs(Ez) (V/m)' )
# 
# figure(2)
# hold on
# plot(xs,angle(Esc_z)*(180/pi),'b','LineWidth',1)
# title('Scattered electric field by a dielectric cylinder')
# xlabel('x (m)')
# ylabel('\angle Ez (�)' )

# figure(3)
# hold on
# semilogy(1:length(error),error,'ko','LineWidth',1.5),grid,
# title('2D MoM FFT-CG numerical convergence')
# xlabel('Iteration number')
# ylabel('Relative residual')
