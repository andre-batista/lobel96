import numpy as np
from numpy import linalg as LA
from numpy import fft
import matplotlib.pyplot as plt

def CG_FFT(Z,b,N,Ni,Xr,max_it,TOL):
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

    Jo = np.zeros((N**2,Ni),dtype=complex) # initial guess
    ro = fft_A(Jo,Z,N,Ni,Xr)-b # ro = A.Jo - b;
    go = fft_AH(ro,Z,N,Ni,Xr) # Complex conjugate AH
    po = -go
    error_res = np.zeros(max_it)

    for n in range(max_it):
    
        alpha = -1*np.sum(np.conj(fft_A(po,Z,N,Ni,Xr))*(fft_A(Jo,Z,N,Ni,Xr)-b),
                         axis=0)/LA.norm(np.reshape(fft_A(po,Z,N,Ni,Xr),
                                                    (N**2*Ni,1),order='F'),
                                         ord='fro')**2 # 1 x Ni
                                                  
        J = Jo + np.tile(alpha,(N**2,1))*po 
        r = fft_A(J,Z,N,Ni,Xr)-b
        g = fft_AH(r,Z,N,Ni,Xr) 
    
        error = LA.norm(r)/LA.norm(b) # error tolerance
        print('%.4e' %error)
        error_res[n] = error
        if error < TOL: # stopping criteria
            break
    
        beta = np.sum(np.conj(g)*(g-go),axis=0)/np.sum(np.abs(go)**2,axis=0) 
        p    = -g + np.tile(beta,(N**2,1))*po 
        
        po = p 
        Jo = J 
        go = g 

    return J,n,error_res

def fft_A(J,Z,N,Ni,Xr):
# Compute Matrix-vector product by using two-dimensional FFT

    J = np.reshape(J,(N,N,Ni))
    Z = np.tile(Z[:,:,np.newaxis],(1,1,Ni))
    e = fft.ifft2(fft.fft2(Z,axes=(0,1))*fft.fft2(J,axes=(0,1),s=(2*N-1,2*N-1)),axes=(0,1))
    e = e[:N,:N,:]
    e = np.reshape(e,(N*N,Ni))
    e = np.reshape(J,(N*N,Ni)) + np.tile(Xr.reshape((-1,1)),(1,Ni))*e

    return e

def fft_AH(J,Z,N,Ni,Xr):
# Compute Matrix-vector product by using two-dimensional FFT*
# *complex conjugate operator

    J = np.reshape(J,(N,N,Ni))
    Z = np.tile(Z[:,:,np.newaxis],(1,1,Ni))
    e = fft.ifft2(fft.fft2(np.conj(Z),axes=(0,1))*fft.fft2(J,axes=(0,1),s=(2*N-1,2*N-1)),axes=(0,1))
    e = e[:N,:N,:]
    e = np.reshape(e,(N*N,Ni))
    e = np.reshape(J,(N*N,Ni)) + np.conj(np.tile(Xr.reshape((-1,1)),(1,Ni)))*e

    return e