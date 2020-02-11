# import numpy as np
# import copy as cp
# import pickle
# import time
# from scipy import sparse as sps
# from numpy import linalg as lag
# import matplotlib.pyplot as plt
# from numba import jit

# def inner(v1,v2,d):
#     return complex(d*v1.T@np.conj(v2))

# def norm_grad(c):
#     del_x, del_y = np.gradient(c)
#     re = np.sqrt(np.real(del_x)**2+np.real(del_y)**2)
#     im = np.sqrt(np.imag(del_x)**2+np.imag(del_y)**2)
#     return re, im

# def phi(t):
#     return t**2/(1+t**2)

# def phip(t):
#     return 2*t/(1+t**2)**2

# @jit(nopython=True)
# def weighted_laplacian(c,br,bi):
#     I,J = c.shape
#     re = np.zeros((I,J))
#     im = np.zeros((I,J))
#     for i in range(1,I-1):
#         for j in range(1,J-1):
#             re[i,j] = (-(2*br[i,j] + br[i-1,j] + br[i,j-1])*np.real(c[i,j]) 
#                        + br[i,j]*np.real(c[i,j+1]) + br[i,j]*np.real(c[i+1,j]) 
#                        + br[i,j-1]*np.real(c[i,j-1]) + br[i-1,j]*np.real(c[i-1,j]))
#             im[i,j] = (-(2*bi[i,j] + bi[i-1,j] + bi[i,j-1])*np.imag(c[i,j]) 
#                        + bi[i,j]*np.imag(c[i,j+1]) + bi[i,j]*np.imag(c[i+1,j]) 
#                        + bi[i,j-1]*np.imag(c[i,j-1]) + bi[i-1,j]*np.imag(c[i-1,j]))
#     re = re.reshape((I*J,1))
#     im = im.reshape((I*J,1))
#     return re, im

# print('========== The Conjugated Gradient Method ==========')
# expname = 'basic'

# with open(expname,'rb') as datafile:
#     data = pickle.load(datafile)

# # Loading inputs
# dx, dy = data['dx'], data['dy']
# II, JJ = data['I'], data['J']
# epsrb, sigb = data['epsrb'], data['sigb']
# f = data['frequency']
# kb, lambda_b = data['kb'], data['lambda_b']
# ei, et, es = data['ei'], data['et'], data['es']
# x, y = data['x'], data['y']
# gs, gd = data['gs'], data['gd']
# epsr, sig = data['epsr'], data['sig']

# # General Parameters
# maxit = 0             # Number of iterations
# M, L = es.shape         # M measurements, L sources
# N = ei.shape[0]         # N points within the mesh
# dS = dx*dy              # Surface element [m^2]
# eps0 = 8.85418782e-12   # Vaccum permittivity [F/m]
# omega = 2*np.pi*f       # Angular frequency [rad/sec]
# delta_r, delta_i = 4.239523e-02, 4.239523e-02
# lambda_r, lambda_i = 1e-2, 1e-2

# # How do you preffer the initial solution?
# # 1 - Everything background
# # 2 - Backpropagation method (Lobel et al., 1996)
# # 3 - Exact solution
# # 4 - Load last run
# initopt = 2

# if initopt is 1:
#     C = sps.dia_matrix((N,N),dtype=complex)
#     d = np.zeros((N,1),dtype=complex)
#     g = np.ones((N,1),dtype=complex)

# elif initopt is 2:
#     gamma = lag.norm(np.reshape(gs.conj().T@es,(-1,1)))**2/lag.norm(np.reshape(gs@gs.conj().T@es,(-1,1)))**2
#     w0 = gamma*gs.conj().T@es
#     C = sps.dia_matrix(np.diag(1/L*np.sum(w0/ei,1)),dtype=complex)
#     d = np.zeros((N,1),dtype=complex)
#     g = np.ones((N,1),dtype=complex)

# elif initopt is 3:
#     C = sps.dia_matrix(np.diag(np.reshape((epsr-1j*sig/omega/eps0/epsrb)-(epsrb-1j*sigb/omega/eps0/epsrb),-1)),dtype=complex)
#     d = np.zeros((N,1))
#     g = np.ones((N,1))

# else:
#     pass
#     # load ../../../../../../Documents/MATLAB/inverse-approximation/c.mat C g d   

# # Initializing variables
# cnvg    = np.zeros((maxit+1,2))     # Convergence data
# I       = sps.eye(N,dtype=complex)  # Identity matrix
# LC      = lag.inv(I-gd@C)          # Initial inversion
# rho     = es-gs@C@LC@ei            # Initial residual

# # Printing first solution
# print('Iteration: 0 - Cost function: %.2e' %lag.norm(rho.reshape(-1))**2)

# if initopt is not 2:
#     cnvg[0,:] = np.array([lag.norm(rho.reshape(-1))**2,lag.norm(g)])
# else:
#     cnvg[0,:] = np.array([lag.norm(rho.reshape(-1))**2,.0])

# re_grad, im_grad = norm_grad(np.reshape(C.data,(II,JJ)))
# totaltime = time.time()

# # Iterations
# for it in range(maxit):
    
#     tic = time.time()
    
#     b_r = phip(re_grad/delta_r)/(2*re_grad/delta_r)
#     b_i = phip(im_grad/delta_i)/(2*im_grad/delta_i)
    
#     re_wl, im_wl = weighted_laplacian(np.reshape(C.data,(II,JJ)),b_r,b_i)
    
#     # Computing the gradient
#     gradJ = np.zeros((N,1),dtype=complex)
#     for l in range(L):
#         gsrho = gs.conj().T@rho[:,l]
#         gradJ = gradJ - 2*np.conj(sps.spdiags(LC@ei[:,l],0,N,N)@LC)@gsrho
#     gradJ = gradJ - 2*lambda_r**2*re_wl/delta_r**2 - 2j*lambda_i**2*im_wl/delta_i**2
        
#     g_last = np.copy(g)
#     g = -gradJ
    
#     # Computing the optimum direction
#     d = g + inner(g,g-g_last,dx)/lag.norm(g_last)**2*d
#     D = sps.spdiags(d.reshape(-1),0,N,N)
    
#     # Computing v matrix
#     v = gs@LC.T@D@LC@ei
    
#     # Computing step
#     bh = lambda_r*b_r+1j*lambda_i*b_i
#     delta_x_d, delta_y_d = np.gradient(np.reshape(D.data,(II,JJ)))
#     delta_x_d_b, delta_y_d_b = np.gradient(np.real(bh)*np.reshape(D.data,(II,JJ)) + 1j*np.imag(bh)*np.reshape(D.data,(II,JJ)))
#     delta_x_c, delta_y_c = np.gradient(np.real(bh)*np.reshape(C.data,(II,JJ)) + 1j*np.imag(bh)*np.reshape(C.data,(II,JJ)))
#     re_grad_d, im_grad_d = norm_grad(np.reshape(D.data,(II,JJ)))
#     aux0 = np.sum(lag.norm(v,axis=0)**2)
#     aux1 = np.sum(np.real(v*np.conj(rho)*dx))
#     aux2 = np.sum(np.real(delta_x_d*delta_x_c+delta_y_d*delta_y_c))
#     aux3 = np.sum(np.imag(delta_x_d_b*np.conj(delta_x_d)+np.conj(delta_y_d)*delta_y_d_b))
#     num_alpha_r = (aux0*(aux1 -
#                              aux2) +
#                        np.sum(np.real(bh)*im_grad_d**4+np.imag(bh)*re_grad_d**4)*
#                        (aux1-
#                         aux2)+
#                        (np.sum(np.imag(v*np.conj(rho)*dx))*
#                         aux3)+
#                        aux3*
#                        np.sum(np.imag(delta_x_c*np.conj(delta_x_d)+np.conj(delta_y_d)*delta_y_c)))
#     aux1 = np.sum(np.imag(v*np.conj(rho)*dx))
#     aux2 = np.sum(np.imag(np.conj(delta_x_d)*delta_x_c+np.conj(delta_y_d)*delta_y_c))
#     num_alpha_i = (-aux0*(aux1 -
#                               aux2) -
#                        np.sum(np.real(bh)*re_grad_d**4+np.imag(bh)*im_grad_d**4)*
#                        (aux1+
#                         aux2)-
#                        (np.sum(np.real(v*np.conj(rho)*dx))*
#                         np.sum(np.imag(delta_x_d_b*np.conj(delta_x_d)+np.conj(delta_y_d)*delta_y_d_b)))-
#                        np.sum(np.real(delta_x_c*delta_x_d+delta_y_d*delta_y_c))*
#                        np.sum(np.imag(delta_x_d*np.conj(delta_x_d_b)+np.conj(delta_y_d_b)*delta_y_d)))
#     delta_d = np.sqrt(delta_x_d*np.conj(delta_x_d)+delta_y_d*np.conj(delta_y_d))
#     den_alpha = ((aux0)*(aux0+
#                              np.sum((np.real(bh)+np.imag(bh))*delta_d**2))-
#                      np.sum(np.imag(delta_x_d_b*np.conj(delta_x_d)+np.conj(delta_y_d)*delta_y_d_b))**2 +
#                      np.sum(np.real(bh)*re_grad_d**4+np.imag(bh)*im_grad_d**4)*
#                      np.sum(np.imag(bh)*re_grad_d**4+np.real(bh)*im_grad_d**4))
#     alpha = num_alpha_r/den_alpha + 1j*num_alpha_i/den_alpha
      
#     # Computing next contrast
#     C = C + alpha*D
#     re_grad, im_grad = norm_grad(np.reshape(C.data,(II,JJ)))
#     # print('Mean of the norm of the gradient: %e' %np.mean(np.sqrt(re_grad**2+im_grad)))
    
#     plt.imshow(np.real(np.reshape(C.data,(II,JJ))))
#     cbar = plt.colorbar()
#     plt.savefig('testfig', format = 'jpeg')
    
#     # Computing the inverse matriz
#     # t1 = time.time()
#     LC = lag.inv(I-gd@C)
#     # print('Inversion time: %f' %(time.time()-t1))
    
#     # Computing the residual
#     # rho = es-gs@C@LC@ei
#     rho = rho-alpha*v
    
#     # Computing the objective function
#     J = lag.norm(rho.reshape(-1))**2 + np.sum(lambda_r**2*phi(re_grad/delta_r)+lambda_i**2*phi(im_grad/delta_i))
    
#     # Printing iteration
#     t = time.time()-tic
#     print('Iteration: %d' %(it+1)
#           + ' - Cost function: %.2e' %J
#           + ' - norm(g): %.2e' %lag.norm(g)
#           + ' - time: %.1f sec' %t)
    
#     # Saving objetive function and gradient magnitude
#     cnvg[it,:] = np.array([J,lag.norm(g)])


# # totaltime = time.time()-totaltime
# # print('Total time: %f' %totaltime + ' seconds')

# # # Recovering dielectric properties
# # tau     = np.reshape(C.data,(II,JJ))             # Constrast fuction
# # epsr    = np.real(tau) + epsrb           # Retrieved relative permittivity
# # sig     = -omega*eps0*epsrb*np.imag(tau) # Relative conductivity [S/m]

# # plt.imshow(epsr, extent = [x[0], x[-1], y[0], y[-1]])
# # plt.xlabel('x [m]')
# # plt.ylabel('y [m]')
# # plt.title(r'Relative Permittivity  - $f = $ %.1e [Hz]' %f)
# # cbar = plt.colorbar()
# # cbar.set_label(r'$|\epsilon_r|$')
# # plt.savefig(expname +'lobel97fig', format = 'jpeg')

# from joblib import Parallel, delayed
# import multiprocessing
     
# # what are your inputs, and what operation do you want to 
# # perform on each input. For example...
# inputs = range(10) 
# def processInput(i):
#     return i * i
 
# num_cores = multiprocessing.cpu_count()
     
# results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)

import model as md
import numpy as np

modelname = 'test'
I, J = 29, 29
dx, dy = 3.9445e-2, 3.9445e-2
epsrb, sigb = 1., 0.
sampled_frequencies = np.array([800e6])
wv_frequency = 800e6
dtheta = 12
Rs = 3.4317
magnitude = 1e2
time_window = 5e-8

mymodel = md.Model(modelname, I, J, dx, dy, epsrb, sigb, wv_frequency,
                   sampled_frequencies, dtheta, Rs, magnitude, time_window)

mymodel.compute_incident_field()