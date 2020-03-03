class Domain:
    """ DOMAIN:
        Base class to represent a domain.
    
        Keyword arguments:
        -- domain_size_x: length of x-axis [m]
        -- domain_size_y: length of y-axis [m]
        -- radius_observation: distance to measurement points [m]
        -- n_sources: number of sources
        -- n_measurements: number of measurements.
    """

    Lx, Ly = float(), float() # Domain size in x,y-axis [m], [m]
    R_obs = float() # Radius of observation [m]
    L, M = int(), int() # Number of sources and samples
    
    def __init__(self,domain_size_x,domain_size_y,radius_observation,
                 n_sources, n_measurements):
        
        self.Lx, self.Ly = domain_size_x, domain_size_y
        self.R_obs = radius_observation
        self.L, self.M = n_sources, n_measurements
    