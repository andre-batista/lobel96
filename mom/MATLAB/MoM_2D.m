%%  Program  : MoM_2D CG-FFT
%   Author   : Jose Olger Vargas
%   Date     : OCTOBER 2019, Universidade Federal de Minas Gerais  
%   function : To compute the electric field for scattering
%              by an arbitrary shape object with Method of Moments (MoM) 
%              and Conjugate Gradient- Fast Fourier Trasnsform (GG-FFT).    
%============================================================================================================
% INPUTS: 
% Note: the scattered field is measured in a circle of radius R_obs
% R_obs          % Far field observation radius  (m)
% E0             % Amplitude of the incident field (V/m)
% Ni             % Number of incidences
% Ns             % Number of receivers (scattered fields)
% f              % Frequency of the incident plane wave (Hz)
% N              % number of cells (square domain is NxN)
% eps_r          % Relative permittivity of the x,y components. (2D matrix NxN) 

% OUTPUT: Scattered Field
% Esc_z 
%============================================================================================================
disp('Computing Forward Problem');
%% DEFINE PARAMETERS
R_obs = 3;
E0 = 1;
Ni = 1;      % Points measured from phi=0
Ns = 16;
f  = 300e6;
N  = 60;     %(DOI size is NxN)
eps_obj = 3.2; % Relative permittivity of the object
epsb    = 1; % Relative permittivity of background
sig_obj = 0; % Conductivity of the object
sigb    = 0; % Conductivity of background

%% INCIDENT AND SCATTERED ANGLES
phi_i = 0:2*pi/(Ni):(2*pi-2*pi/(Ni)); % angles of incidence
phi_s = 0:2*pi/(Ns):(2*pi-2*pi/(Ns)); % scattered Field phi angles

% Observation Points from out of DOI  xs= x-axis, ys=y-axis
xs = R_obs*cos(phi_s);  
ys = R_obs*sin(phi_s); 

% case: coordinate inputs xs and ys
% xs = -1:0.025:1; 
% ys =  zeros(1,Ns); % y=0.

%% DISCRETIZATION BY CELLS
% Setting 2D mesh size, define the limits of the DOI
xmin= -1; xmax=1;
ymin= -1; ymax=1;
% Length of each cell in x and y-axis
dx = (xmax-xmin)/N; 
dy = (ymax-ymin)/N; 

% Centers of each cell
x_c = (xmin+ 0.5*dx:dx: xmax-0.5*dx);    % 1xN
y_c = (ymin+ 0.5*dy:dy: ymax-0.5*dy);    % 1xN
[x,y] = meshgrid(x_c,y_c);               % NxN
clear x_c y_c

%% Define constants
lambda = 3e8/f;         % wavelength
kb = 2*pi/lambda;       % wavenumber of background
deltasn = dx*dy;        % area of the cell
an = sqrt(deltasn/pi);  % radius of the equivalent circle

%% DEFINE CONTRAST FUNCTION
omega = 2*pi*f;                        % angular frequency
eps0  = 8.85418782e-12;                % Permittivity of vacuum 
eps_r = epsb *ones(N,N);               % NxN
sig   = sigb*ones(N,N);                % NxN

% Assigning materials
% Defining a cylinder of radius 0.5 with eps_r and sigma
eps_r((x.^2+ y.^2)<= 0.5^2) = eps_obj; 
sig((x.^2+ y.^2)<= 0.5^2)   = sig_obj; 
% Contrast function: \Chi(r)
Xr = (eps_r - 1i.*sig./omega./eps0)./(epsb- 1i.*sigb./omega./eps0) - 1;

%% Computing EFIE 
% Using circular convolution 
x_c = xmin-(N/2-1)*dx:dx: xmax+(N/2 -1)*dx;
y_c = ymin-(N/2-1)*dy:dy: ymax+(N/2 -1)*dy;
[xe,ye] = meshgrid(x_c,y_c);   % extended domain (2N-1)x(2N-1)
clear x_c y_c

Rmn = sqrt(xe.^2 + ye.^2);     % distance between the cells
% Matrix elements for off-diagonal entries
Zmn = ((1i*pi*kb*an)/2)*besselj(1,kb*an)*besselh(0,2,kb*Rmn); % m=/n
% Matrix elements for diagonal entries 
Zmn(N,N)= ((1i*pi*kb*an)/2)*besselh(1,2,kb*an) + 1;           % m==n

% Extended matrix (2N-1)x(2N-1) 
Z = zeros(2*N-1,2*N-1);
Z(1:N,1:N) = Zmn(N:2*N-1,N:2*N-1);
Z(N+1:2*N-1,N+1:2*N-1) = Zmn(1:N-1,1:N-1);
Z(1:N,N+1:2*N-1) = Zmn(N:2*N-1,1:N-1);
Z(N+1:2*N-1,1:N) = Zmn(1:N-1,N:2*N-1);

%% Incident Plane Wave [Ei]
Ei = E0*exp(-1i*kb*(x(:)*cos(phi_i(:).') + y(:)*sin(phi_i(:).')));
b = repmat(Xr(:),1,Ni).*Ei; 

%% Using Conjugate-gradient- Fast Fourier Transform Procedure
% Solving linear equation system Ax=b
max_it = 400;        % number of iterations
TOL = 1e-6;          % tolerance
tic, [J,iter,error] = CG_FFT(Z,b,N,Ni,Xr,max_it,TOL); time_cg_fft=toc;

%% Observation Points from out of DOI
xg = repmat(xs.',[1,N*N]);
yg = repmat(ys.',[1,N*N]);
Rscat = sqrt((xg-repmat(reshape(x,N*N,1).',[Ns,1])).^2+(yg-repmat(reshape(y,N*N,1).',[Ns,1])).^2); % Ns x N*N

%% Scattered Field
Zscat = -1i*kb*pi*an/2*besselj(1,kb*an)*besselh(0,2,kb*Rscat); % Ns x N^2
Esc_z = Zscat*J;  % Ns x Ni
% save('Esc_z.mat');

% Plotting results
% figure(1)
% hold on
% plot(xs,abs(Esc_z),'b','LineWidth',1)
% title('Scattered electric field by a dielectric cylinder')
% xlabel('x (m)')
% ylabel(' abs(Ez) (V/m)' )
% 
% figure(2)
% hold on
% plot(xs,angle(Esc_z)*(180/pi),'b','LineWidth',1)
% title('Scattered electric field by a dielectric cylinder')
% xlabel('x (m)')
% ylabel('\angle Ez (?)' )

% figure(3)
% hold on
% semilogy(1:length(error),error,'ko','LineWidth',1.5),grid,
% title('2D MoM FFT-CG numerical convergence')
% xlabel('Iteration number')
% ylabel('Relative residual')
