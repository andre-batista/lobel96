import numpy as np
import domain as dm
import model as md

## DEFINE PARAMETERS
R_obs, Lx, Ly = 3., 2., 2.
E0 = 1
L = 7 # Points measured from phi=0
M = 16
# f = 300e6
f = np.array([300e6,400e6,500e6,600e6])
Nx, Ny = 60, 60 # (DOI size is NxxNy)
eps_obj = 3.2 # Relative permittivity of the object
epsb    = 1 # Relative permittivity of background
sig_obj = 0 # Conductivity of the object
sigb    = 0 # Conductivity of background

experiment = md.Model(dm.Domain(Lx,Ly,R_obs,L,M),frequencies=f,
                      incident_field_magnitude=E0,epsilon_r_background=epsb,
                      sigma_background=sigb)

xmin,xmax = md.get_bounds(Lx)
ymin, ymax = md.get_bounds(Ly)
x,y = md.get_domain_coordinates(Lx/Nx,Ly/Ny,xmin,xmax,ymin,ymax)

# Defining a cylinder of radius 0.5 with eps_r and sigma
eps_r, sig = epsb*np.ones((Nx,Ny)), sigb*np.ones((Nx,Ny))
eps_r[(x**2+ y**2) <= 0.5**2] = eps_obj 
sig[(x**2+ y**2) <= 0.5**2]   = sig_obj

# experiment.draw_setup(epsr=eps_r,sig=sig)
experiment.solve(epsilon_r=eps_r,sigma=sig)
experiment.plot_total_field(frequency_index=3)
