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

def norm_grad(c):
    del_x, del_y = np.gradient(c)
    re = np.sqrt(np.real(del_x)**2+np.real(del_y)**2)
    im = np.sqrt(np.imag(del_x)**2+np.imag(del_y)**2)
    return re, im

def phip(t):
    return 2*t/(1+t**2)**2

@jit(nopython=True)
def weighted_laplacian(c,br,bi):
    I,J = c.shape
    re = np.zeros((I,J))
    im = np.zeros((I,J))
    for i in range(1,I-1):
        for j in range(1,J-1):
            re[i,j] = (-(2*br[i,j] + br[i-1,j] + br[i,j-1])*np.real(c[i,j]) 
                       + br[i,j]*np.real(c[i,j+1]) + br[i,j]*np.real(c[i+1,j]) 
                       + br[i,j-1]*np.real(c[i,j-1]) + br[i-1,j]*np.real(c[i-1,j]))
            im[i,j] = (-(2*bi[i,j] + bi[i-1,j] + bi[i,j-1])*np.imag(c[i,j]) 
                       + bi[i,j]*np.imag(c[i,j+1]) + bi[i,j]*np.imag(c[i+1,j]) 
                       + bi[i,j-1]*np.imag(c[i,j-1]) + bi[i-1,j]*np.imag(c[i-1,j]))
    re = re.reshape((I*J,1))
    im = im.reshape((I*J,1))
    return re, im

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
maxit = 50             # Number of iterations
M, L = es.shape         # M measurements, L sources
N = ei.shape[0]         # N points within the mesh
dS = dx*dy              # Surface element [m^2]
eps0 = 8.85418782e-12   # Vaccum permittivity [F/m]
omega = 2*np.pi*f       # Angular frequency [rad/sec]
delta_r, delta_i = 1.2, 1.2
lambda_r, lambda_i = 1., 1.

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
    
    re_grad, im_grad = norm_grad(np.reshape(C.data,(II,JJ)))
    b_r = phip(re_grad/delta_r)/(2*re_grad/delta_r)
    b_i = phip(im_grad/delta_i)/(2*im_grad/delta_i)
    re_wl, im_wl = weighted_laplacian(np.reshape(C.data,(II,JJ)),b_r,b_i)
    
    # Computing the gradient
    gradJ = np.zeros((N,1),dtype=complex)
    for l in range(L):
        gsrho = gs.conj().T@rho[:,l]
        gradJ = gradJ - 2*np.conj(sps.spdiags(LC@ei[:,l],0,N,N)@LC)@gsrho
    gradJ = gradJ - 2*lambda_r**2*re_wl/delta_r**2 - 2j*lambda_i**2*im_wl/delta_i**2
    
    g_last = np.copy(g)
    g = -gradJ
    
    # Computing the optimum direction
    d = g + inner(g,g-g_last,dS)/lag.norm(g_last)**2*d
    D = sps.spdiags(d.reshape(-1),0,N,N)

    # Computing v matrix
    v = gs@LC.T@D@LC@ei
    
    # Computing step
    if alphaopt is 1:
        # alpha = 0
        # for l in range(L):
        #     alpha = alpha + inner(rho[:,l],v[:,l],dx)
        # alpha = alpha/lag.norm(v.reshape(-1))**2
        betah = lambda_r*b_r+1j*lambda_i*b_i
        delta_x_d, delta_y_d = np.gradient(np.reshape(D.data,(II,JJ)))
        delta_x_d_b, delta_y_d_b = np.gradient(betah*np.reshape(D.data,(II,JJ)))
        delta_x_c, delta_y_c = np.gradient(betah*np.reshape(C.data,(II,JJ)))
        re_grad_d, im_grad_d = norm_grad(np.reshape(D.data,(II,JJ)))
        aux0 = np.sum(lag.norm(v,axis=0)**2)
        aux1 = np.sum(np.real(v*np.conj(rho)*dS))
        aux2 = np.sum(np.real(delta_x_d*delta_x_c+delta_y_d*delta_y_c))
        aux3 = np.sum(np.imag(delta_x_d_b*np.conj(delta_x_d)+np.conj(delta_y_d)*delta_y_d_b))
        num_alpha_r = (aux0*(aux1 -
                             aux2) +
                       np.sum(np.real(betah)*im_grad_d**4+np.imag(betah)*re_grad_d**4)*
                       (aux1-
                        aux2)+
                       (np.sum(np.imag(v*rho*dS))*
                        aux3)+
                       aux3*
                       np.sum(np.imag(delta_x_c*np.conj(delta_x_d)+np.conj(delta_y_d)*delta_y_c)))
        aux1 = np.sum(np.imag(v*np.conj(rho)*dS))
        aux2 = np.sum(np.imag(delta_x_d*delta_x_c+delta_y_d*delta_y_c))
        num_alpha_i = (-aux0*(aux1 -
                              aux2) -
                       np.sum(np.real(betah)*re_grad_d**4+np.imag(betah)*im_grad_d**4)*
                       (aux1+
                        aux2)-
                       (np.sum(np.real(v*rho*dS))*
                        np.sum(np.imag(delta_x_d_b*np.conj(delta_x_d)+np.conj(delta_y_d)*delta_y_d_b)))-
                       np.sum(np.real(delta_x_c*np.conj(delta_x_d)+np.conj(delta_y_d)*delta_y_c))*
                       np.sum(np.imag(delta_x_d*np.conj(delta_x_d_b)+np.conj(delta_y_d_b)*delta_y_d)))
        delta_d = delta_x_d**2+delta_y_d**2
        den_alpha = ((aux0)*(aux0+
                             np.sum((np.real(betah)+np.imag(betah))*delta_d))-
                     np.sum(np.imag(delta_x_d_b*np.conj(delta_x_d)+np.conj(delta_y_d)*delta_y_d_b))**2 +
                     np.sum(np.imag(betah)*np.real(delta_d)+np.real(betah)*np.imag(delta_d)))
        alpha = num_alpha_r/den_alpha + 1j*num_alpha_i/den_alpha
    else:
        alpha = gsmethod(rho,v)
      
    # Computing next contrast
    C = C + alpha*D
    
    # Computing the inverse matriz
    t1 = time.time()
    LC = lag.inv(I-gd@C)
    print('Inversion time: %f' %(time.time()-t1))
    
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

# % Plotting results
# figure
# load ./genfields/grid.mat


plt.imshow(epsr, extent = [x[0], x[-1], y[0], y[-1]])
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title(r'Relative Permittivity  - $f = $ %.1e [Hz]' %f)
cbar = plt.colorbar()
cbar.set_label(r'$|\epsilon_r|$')
plt.savefig(expname +'fig', format = 'jpeg')
# plt.show()
# plt.close()

# % Relative permittivity plot
# subplot(3,2,1)
# imagesc(y,x,epsr')
# set(gca,'YDir','normal')
# xlabel('x [m]')
# ylabel('y [m]')
# title('Relative permittivity')
# clb = colorbar;
# ylabel(clb, '\epsilon_r')

# % Conductivity plot
# subplot(3,2,2)
# imagesc(y,x,sig')
# set(gca,'YDir','normal')
# xlabel('x [m]')
# ylabel('y [m]')
# title('Conductivity')
# clb = colorbar;
# ylabel(clb, '\sigma [S/m]')

# % Gradient - Real
# subplot(3,2,3)
# imagesc(y,x,real(reshape(g,I,J))')
# set(gca,'YDir','normal')
# xlabel('x [m]')
# ylabel('y [m]')
# title('Gradient - Real')
# clb = colorbar;
# ylabel(clb, 'g')

# % Conductivity plot
# subplot(3,2,4)
# imagesc(y,x,imag(reshape(g,I,J))')
# set(gca,'YDir','normal')
# xlabel('x [m]')
# ylabel('y [m]')
# title('Gradient - Imaginary')
# clb = colorbar;
# ylabel(clb, 'g')

# % Convergence plot - Cost Function
# subplot(3,2,5)
# plot(0:maxit,cnvg(:,1),'linewidth',2)
# grid
# xlabel('Iterations')
# ylabel('J(C)')
# title('Cost Function')

# % Convergence plot - Gradient
# subplot(3,2,6)
# plot(0:maxit,cnvg(:,2),'linewidth',2)
# grid
# xlabel('Iterations')
# ylabel('|\nabla J(C)|')
# title('Gradient')

# savefig('lobel96fig.fig')

# % Saving solution
# save ../../../../../../Documents/MATLAB/inverse-approximation/lobel96.mat C cnvg totaltime -v7.3
