'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                   The Conjugated Gradient Method                        %
%                                                                         %
%       This scripts implements the Conjugated Gradient for inverse       %
%   scattering 2D TMz electromagnetic problems (Lobel et al., 1996).      %
%   Given the measurements of the scattering field in specific points of  %
%   a domain denoted by S, the incident field on a investigation domain D %
%   and the Green Function for both domains, the method recovers the      %
%   dielectric distribution within the region D.                          %
%                                                                         %
%   Inputs:                                                               %
%   - es: a M by L matrix with the M measured scattered fields for the L  %
%       sources [V/m]                                                     %
%   - ei: a N by L matrix with the N computed incident fields for the L   %
%       sources [V/m]                                                     %
%   - gd: a N by N matrix with Green function computed for all of the N   %
%       points of the mesh in respect to each of them                     %
%   - gs: a M by N matrix with Green function computed for all of the N   %
%       points of the mesh in respect to the M scattered field            %
%       measurements                                                      %
%                                                                         %
%   Data struct:                                                          %
%   - dx, dy: cell sizes [m]                                              %
%   - epsr, sig: correct information of the dielectric distribution of    %
%       the experiment (relative permittivity and conductivity [S/m])     %
%   - epsrb, sigb: relative permittivity and conductivity [S/m] of the    %
%       background                                                        %
%   - lambdab: wavelength of the background [m]                           %
%   - f: linear frequency of measurements [Hz]                            %
%                                                                         %
%   Output variables:                                                     %
%   - epsr: retrieved relative permittivity                               %
%   - sig: retrieved conductivity [S/m]                                   %
%                                                                         %
%   Implemented by:                                                       %
%                                                                         %
%   Andre Costa Batista                                                   %
%   Universidade Federal de Minas Gerais                                  %
%                                                                         %
%   References                                                            %
%                                                                         %
%   Lobel, P., et al. "Conjugate gradient method for solving inverse      %
%   scattering with experimental data." IEEE Antennas and Propagation     %
%   Magazine 38.3 (1996): 48-51.                                          %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

import numpy as np
import copy as cp
import pickle
import time
from scipy import sparse as sps
from numpy import linalg as lag
import matplotlib.pyplot as plt
from numba import jit

def inner(v1,v2,d):
    return complex(d*v1.T@np.conj(v2))

def gsmethod(rho,v):

    gmean       = 0.618
    delta       = 1.0e-3
    [a, b]      = getinterval(rho,v)   
    xa          = b - gmean*(b-a)
    xb          = a + gmean*(b-a)
    fxa         = lag.norm(np.reshape(rho-xa*v,(-1,1)))**2
    fxb         = lag.norm(np.reshape(rho-xb*v,(-1,1)))**2
    
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
    
def getinterval(rho,v):
    
    step0           = 1.0e-03
    a               = 0
    Fa              = lag.norm(rho.reshape(-1))**2
    b               = step0
    Fb              = lag.norm(np.reshape(rho-step0*v,(-1,1)))**2
    stepsize        = step0
    acceleration    = 2
    
    while (Fa > Fb):
        stepsize = acceleration*stepsize
        b = b + stepsize
        Fb = lag.norm(np.reshape(rho-b*v,(-1,1)))**2
    return a,b

print('========== The Conjugated Gradient Method ==========')
expname = 'basic'

with open(expname,'rb') as datafile:
    data = pickle.load(datafile)

# Loading inputs
dx, dy = data['dx'], data['dy']
II, JJ = data['I'], data['J']
epsrb, sigb = data['epsrb'], data['sigb']
f = data['frequency']
kb, lambda_b = data['kb'], data['lambda_b']
ei, et, es = data['ei'], data['et'], data['es']
x, y = data['x'], data['y']
gs, gd = data['gs'], data['gd']
epsr, sig = data['epsr'], data['sig']

# General Parameters
maxit = 150             # Number of iterations
M, L = es.shape         # M measurements, L sources
N = ei.shape[0]         # N points within the mesh
dS = dx*dy              # Surface element [m^2]
eps0 = 8.85418782e-12   # Vaccum permittivity [F/m]
omega = 2*np.pi*f       # Angular frequency [rad/sec]

# How do you preffer the initial solution?
# 1 - Everything background
# 2 - Backpropagation method (Lobel et al., 1996)
# 3 - Exact solution
# 4 - Load last run
initopt = 2

if initopt is 1:
    C = sps.dia_matrix((N,N),dtype=complex)
    d = np.zeros((N,1),dtype=complex)
    g = np.ones((N,1),dtype=complex)

elif initopt is 2:
    gamma = lag.norm(np.reshape(gs.conj().T@es,(-1,1)))**2/lag.norm(np.reshape(gs@gs.conj().T@es,(-1,1)))**2
    w0 = gamma*gs.conj().T@es
    C = sps.dia_matrix(np.diag(1/L*np.sum(w0/ei,1)),dtype=complex)
    d = np.zeros((N,1),dtype=complex)
    g = np.ones((N,1),dtype=complex)

elif initopt is 3:
    C = sps.dia_matrix(np.diag(np.reshape((epsr-1j*sig/omega/eps0/epsrb)-(epsrb-1j*sigb/omega/eps0/epsrb),-1)),dtype=complex)
    d = np.zeros((N,1))
    g = np.ones((N,1))

else:
    pass
    # load ../../../../../../Documents/MATLAB/inverse-approximation/c.mat C g d   

# How do you preffer the choice of the alpha?
# 1 - (Lobel et al, 1996)
# 2 - Golden section method
alphaopt = 1

# Initializing variables
cnvg    = np.zeros((maxit+1,2))     # Convergence data
I       = sps.eye(N,dtype=complex)  # Identity matrix
LC      = lag.inv(I-gd@C)          # Initial inversion
rho     = es-gs@C@LC@ei            # Initial residual

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
        gsrho = gs.conj().T@rho[:,l]
        gradJ = gradJ - 2*np.conj(sps.spdiags(LC@ei[:,l],0,N,N)@LC)@gsrho
    
    g_last = np.copy(g)
    g = -gradJ
    
    # Computing the optimum direction
    d = g + inner(g,g-g_last,dx)/lag.norm(g_last)**2*d
    D = sps.spdiags(d.reshape(-1),0,N,N)

    # Computing v matrix
    v = gs@LC.T@D@LC@ei
    
    # Computing step
    if alphaopt is 1:
        alpha = 0
        for l in range(L):
            alpha = alpha + inner(rho[:,l],v[:,l],dx)
        alpha = alpha/lag.norm(v.reshape(-1))**2
    else:
        alpha = gsmethod(rho,v)
      
    # Computing next contrast
    C = C + alpha*D
    
    # Computing the inverse matriz
    LC = lag.inv(I-gd@C)
    
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
tau     = np.reshape(C.data,(II,JJ))             # Constrast fuction
epsr    = np.real(tau) + epsrb           # Retrieved relative permittivity
sig     = -omega*eps0*epsrb*np.imag(tau) # Relative conductivity [S/m]

plt.imshow(epsr, extent = [x[0], x[-1], y[0], y[-1]])
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title(r'Relative Permittivity  - $f = $ %.1e [Hz]' %f)
cbar = plt.colorbar()
cbar.set_label(r'$|\epsilon_r|$')
# plt.savefig(expname +'lobel96fig', format = 'jpeg')
plt.show()