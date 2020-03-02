class Domain:
    
    Lx, Ly = float(), float()
    R_obs = float()
    L, M = int(), int()
    
    def __init__(self,domain_size_x,domain_size_y,radius_observation,
                 n_sources, n_measurements):
        
        self.Lx, self.Ly = domain_size_x, domain_size_y
        self.R_obs = radius_observation
        self.L, self.M = n_sources, n_measurements
    