import model as md
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# Prevent strange menssages when using SSH
matplotlib.use('Agg')

class Experiment:
    
    setup_filename = ''
    setup_filepath = ''
    experiment_name = ''
    experiment_filepath = ''
    object_size = float()
    contrast = float()
    
    def __init__(self,setup_filename,setup_filepath,
                 experiment_name,experiment_filepath):
        
        self.setup_filename = setup_filename
        self.setup_filepath = setup_filepath
        self.experiment_name = experiment_name
        self.experiment_filepath = experiment_filepath
        
        self.synt_exp = md.Model(experiment_name,
                                 setup_filename,setup_filepath)
        
    def set_map(self,object_name,*args):
        
        if object_name is 'square':
            contrast = args[0]
            size = args[1]
            if len(args) is 3:
                center = args[2]
            else:
                center = np.array([0,0])
            
            self.epsilon_r, self.sigma = build_square(
                self.synt_exp.I, self.synt_exp.J, self.synt_exp.dx,
                self.synt_exp.dy, self.synt_exp.epsrb, 
                self.synt_exp.sigb, contrast*self.synt_exp.epsrb,
                contrast*self.synt_exp.sigb, size*self.synt_exp.lambda_b,
                center
            )
        
        elif object_name is 'triangle':
            contrast = args[0]
            size = args[1]
            
            self.epsilon_r, self.sigma = build_triangle(
                self.synt_exp.I, self.synt_exp.J, self.synt_exp.dx,
                self.synt_exp.dy, self.synt_exp.epsrb, 
                self.synt_exp.sigb, contrast*self.synt_exp.epsrb,
                contrast*self.synt_exp.sigb, size*self.synt_exp.lambda_b,
            )
        
        elif object_name is 'star':
            contrast = args[0]
            size = args[1]
            
            self.epsilon_r, self.sigma = build_star(
                self.synt_exp.I, self.synt_exp.J, self.synt_exp.dx,
                self.synt_exp.dy, self.synt_exp.epsrb, 
                self.synt_exp.sigb, contrast*self.synt_exp.epsrb,
                contrast*self.synt_exp.sigb, size*self.synt_exp.lambda_b,
            )
        
        elif object_name is 'ring':
            contrast = args[0]
            ra, rb, rc, delta = args[1], args[2], args[3], args[4]
            
            self.epsilon_r, self.sigma = build_ring(
                self.synt_exp.I, self.synt_exp.J, self.synt_exp.dx,
                self.synt_exp.dy, self.synt_exp.epsrb, 
                self.synt_exp.sigb, contrast*self.synt_exp.epsrb,
                contrast*self.synt_exp.sigb, ra*self.synt_exp.lambda_b,
                rb*self.synt_exp.lambda_b, rc*self.synt_exp.lambda_b,
                delta*self.synt_exp.lambda_b
            )
            
        elif object_name is 'ellipses':
            contrast = args[0]
            la, lb, delta = args[1], args[2], args[3]
            
            self.epsilon_r, self.sigma = build_ellipses(
                self.synt_exp.I, self.synt_exp.J, self.synt_exp.dx,
                self.synt_exp.dy, self.synt_exp.epsrb, 
                self.synt_exp.sigb, contrast*self.synt_exp.epsrb,
                contrast*self.synt_exp.sigb, la*self.synt_exp.lambda_b,
                lb*self.synt_exp.lambda_b, delta*self.synt_exp.lambda_b
            )
        
        elif object_name is '2circles':
            contrast = args[0]
            ra, delta = args[1], args[1]
            
            self.epsilon_r, self.sigma = build_2circles(
                self.synt_exp.I, self.synt_exp.J, self.synt_exp.dx,
                self.synt_exp.dy, self.synt_exp.epsrb, 
                self.synt_exp.sigb, contrast*self.synt_exp.epsrb,
                contrast*self.synt_exp.sigb, ra*self.synt_exp.lambda_b,
                delta*self.synt_exp.lambda_b
            )
        
        elif object_name is '3objects':
            contrast = args[0]
            ra, dela = args[1], args[2]
            lb, delb = args[3], args[4]
            lc, delc = args[5], args[6]
            
            self.epsilon_r, self.sigma = build_3objects(
                self.synt_exp.I, self.synt_exp.J, self.synt_exp.dx,
                self.synt_exp.dy, self.synt_exp.epsrb, 
                self.synt_exp.sigb, contrast*self.synt_exp.epsrb,
                contrast*self.synt_exp.sigb, ra*self.synt_exp.lambda_b,
                dela*self.synt_exp.lambda_b, lb*self.synt_exp.lambda_b,
                delb*self.synt_exp.lambda_b, lc*self.synt_exp.lambda_b,
                delc*self.synt_exp.lambda_b
            )
            
        elif object_name is 'filledring':
            contrast = args[0]
            ra, rb = args[1], args[2]
            
            self.epsilon_r, self.sigma = build_filledring(
                self.synt_exp.I, self.synt_exp.J, self.synt_exp.dx,
                self.synt_exp.dy, self.synt_exp.epsrb, 
                self.synt_exp.sigb, contrast*self.synt_exp.epsrb,
                contrast*self.synt_exp.sigb, ra*self.synt_exp.lambda_b,
                rb*self.synt_exp.lambda_b
            )
        
        else:
            print('ERROR - SET_MAP: the object name is wrong!')
            exit()
  
    def plot_map(self,fileformat='eps'):
        x, y = self.synt_exp.x/self.synt_exp.lambda_b, self.synt_exp.y/self.synt_exp.lambda_b
        plt.imshow(self.epsilon_r, extent = [x[0], x[-1], y[0], y[-1]])
        plt.xlabel(r'x [$\lambda$]')
        plt.ylabel(r'y [$\lambda$]')
        plt.title('Relative Permittivity Map')
        cbar = plt.colorbar()
        cbar.set_label(r'$\epsilon_r$')
        plt.savefig(self.experiment_filepath+self.experiment_name+'_map', format = fileformat)
        plt.close()
        
    def gen_data(self,*args):
        
        if len(args) is 0:
            self.synt_exp.gerenate_case_data(
                epsr=self.epsilon_r,sig=self.sigma,
                ei_data=self.setup_filepath+self.setup_filename,
                filepath=self.experiment_filepath
            )
            
        elif len(args) is 1:
            self.synt_exp.gerenate_case_data(
                epsr=args[0],ei_data=self.setup_filepath+self.setup_filename,
                filepath=self.experiment_filepath
            )
        
        elif len(args) is 2:
            self.synt_exp.gerenate_case_data(
                epsr=args[0],sig=args[1],
                ei_data=self.setup_filepath+self.setup_filename,
                filepath=self.experiment_filepath
            )
            
def build_square(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,
                 l,center=np.array([0,0])):
    
    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    Lx, Ly = I*dx, J*dy
    x = np.linspace(-Lx/2,Lx/2,I,endpoint=False)
    y = np.linspace(-Ly/2,Ly/2,J,endpoint=False)
    # x = np.arange(dx,dx*(I+1),dx)
    # y = np.arange(dy,dy*(J+1),dy)
    
    epsilon_r[np.ix_(np.logical_and(x >= center[0]-l/2, x <= center[0]+l/2),
                     np.logical_and(y >= center[1]-l/2, y <= center[1]+l/2))] = epsilon_robj
    
    sigma[np.ix_(np.logical_and(x >= center[0]-l/2, x <= center[0]+l/2),
                 np.logical_and(y >= center[1]-l/2, y <= center[1]+l/2))] = sigma_obj
    
    return epsilon_r,sigma

def build_triangle(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,l):

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(dx,dx*(I+1),dx)
    y = np.arange(dy,dy*(J+1),dy)
    Lx, Ly = I*dx, J*dy

    for i in range(I):
        for j in range(J):

            if x[i] >= Lx/2-l/2 and x[i] <= Lx/2+l/2:
                a = x[i]-.5*(Lx-l)
                FLAG = False
                if y[j] < Ly/2:
                    b = y[j]-.5*(Ly-l)
                    v = -.5*a+l/2
                    if b >= v:
                        FLAG = True
                else:
                    b = y[j]-Lx/2
                    v = .5*a
                    if b <= v:
                        FLAG = True
                if FLAG is True:
                    epsilon_r[i,j] = epsilon_robj
                    sigma[i,j] = sigma_obj
    return epsilon_r, sigma

def build_star(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,l):

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(dx,dx*(I+1),dx)
    y = np.arange(dy,dy*(J+1),dy)
    Lx, Ly = I*dx, J*dy
    xc = l/6

    for i in range(I):
        for j in range(J):

            if x[i]+xc >= Lx/2-l/2 and x[i]+xc <= Lx/2+l/2:
                a = x[i]+xc-.5*(Lx-l)
                FLAG = False
                if y[j] < Ly/2:
                    b = y[j]-.5*(Ly-l)
                    v = -.5*a+l/2
                    if b >= v:
                        FLAG = True
                else:
                    b = y[j]-Lx/2
                    v = .5*a
                    if b <= v:
                        FLAG = True
                if FLAG == True:
                    epsilon_r[i,j] = epsilon_robj
                    sigma[i,j] = sigma_obj

    for i in range(I):
        for j in range(J):

            if x[i]-xc >= Lx/2-l/2 and x[i]-xc <= Lx/2+l/2:
                a = x[i]-xc-.5*(Lx-l)
                FLAG = False
                if y[j] < Ly/2:
                    b = y(j)-.5*(Ly-l)
                    v = .5*a
                    if b >= v:
                        FLAG = True
                else:
                    b = y[j]-Lx/2
                    v = -.5*a+l/2
                    if b <= v:
                        FLAG = True
                if FLAG is True:
                    epsilon_r[i,j] = epsilon_robj
                    sigma[i,j] = sigma_obj
    return epsilon_r, sigma

def build_ring(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,
               ra,rb,rc,delta):

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    xc = I*dx/2+delta
    yc = J*dy/2+delta

    for i in range(I):
        for j in range(J):

            r = np.sqrt((x[i]-xc)**2+(y[j]-yc)**2)
            if r <= ra and r >= rb:
                epsilon_r[i,j] = epsilon_robj
                sigma[i,j] = sigma_obj
            elif r <= rc:
                epsilon_r[i,j] = epsilon_robj
                sigma[i,j] = sigma_obj
    return epsilon_r, sigma

def build_ellipses(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,
                   la,lb,delta):

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    xc = I*dx/2
    yc = J*dy/2

    for i in range(I):
        for j in range(J):

            if (x[i]-(xc-delta))**2/la**2 + (y[j]-yc)**2/lb**2 <= 1:
                epsilon_r[i,j] = epsilon_robj
                sigma[i,j] = sigma_obj
            elif (x[i]-(xc+delta))**2/la**2 + (y[j]-yc)**2/lb**2 <= 1:
                epsilon_r[i,j] = epsilon_robj
                sigma[i,j] = sigma_obj
    return epsilon_r, sigma

def build_2circles(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,ra,delta):

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    
    xc1 = I*dx/2+delta
    yc1 = J*dy/2+delta
    
    xc2 = I*dx/2-delta
    yc2 = J*dy/2+delta

    for i in range(I):
        for j in range(J):

            r1 = np.sqrt((x[i]-xc1)**2+(y[j]-yc1)**2)
            r2 = np.sqrt((x[i]-xc2)**2+(y[j]-yc2)**2)
        
            if r1 <= ra and r2 <= ra:
                epsilon_r[i,j] = epsilon_robj[2]
                sigma[i,j] = sigma_obj[2]
            elif r1 <= ra:
                epsilon_r[i,j] = epsilon_robj[0]
                sigma[i,j] = sigma_obj[0]
            elif r2 <= ra:
                epsilon_r[i,j] = epsilon_robj[1]
                sigma[i,j] = sigma_obj[1]
    return epsilon_r, sigma

def build_3objects(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,
                   ra,dela,lb,delb,lc,delc):

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    
    xca = I*dx/2+dela[0]
    yca = J*dy/2+dela[1]
    
    xcb = I*dx/2+delb[0]
    ycb = J*dy/2+delb[1]
    
    xcc = I*dx/2+delc[0]
    ycc = J*dy/2+delc[1]

    for i in range(I):
        for j in range(J):

            r = np.sqrt((x[i]-xca)**2+(y[j]-yca)**2)
           
            if r <= ra:
                epsilon_r[i,j] = epsilon_robj[0]
                sigma[i,j] = sigma_obj[0]
                
            elif (x[i] >= xcb-lb/2 and x[i] <= xcb+lb/2 
                  and y[j] >= ycb-lb/2 and y[j] <= ycb+lb/2):
                
                epsilon_r[i,j] = epsilon_robj[1]
                sigma[i,j] = sigma_obj[1]
                
            elif (x[i] >= xcc-lc/2 and x[i] <= xcc+lc/2 
                  and y[j] >= ycc-lc/2 and y[j] <= ycc+lc/2):
                
                FLAG = False
                v = lc/2*np.sqrt(3)
                if y[j] >= ycc:
                    a = lc/2/v
                    b = ycc-lc/2/v*(xcc-v/2)
                    if y[j] <= a*x[i]+b:
                        FLAG = True
                else:
                    yp = ycc + ycc-y[j]
                    a = lc/2/v
                    b = ycc-lc/2/v*(xcc-v/2)
                    if yp <= a*x[i]+b:
                        FLAG = True
                if FLAG is True:
                    epsilon_r[i,j] = epsilon_robj[2]
                    sigma[i,j] = sigma_obj[2]
    return epsilon_r, sigma

def build_filledring(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,ra,rb):

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    
    xc = I*dx/2
    yc = J*dy/2

    for i in range(I):
        for j in range(J):

            r = np.sqrt((x[i]-xc)**2+(y[j]-yc)**2)
           
            if r <= ra and r >= rb:
                epsilon_r[i,j] = epsilon_robj[0]
                sigma[i,j] = sigma_obj[0]
            elif r <= rb:
                epsilon_r[i,j] = epsilon_robj[1]
                sigma[i,j] = sigma_obj[1]
    return epsilon_r, sigma