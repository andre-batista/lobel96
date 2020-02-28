function  [J,n,error_res] = CG_FFT(Z,b,N,Ni,Xr,max_it,TOL)

% Congugate-Gradient Method (CGM)
% inputs:

% Z      % extended matrix     (2N-1)x(2N-1)
% b      % excitation source    N^2 x Ni
% N      % DOI size             1x1
% Ni     % number of incidences 1x1
% Xr     % contrast function    NxN
% max_it % number of iterations (integer number)
% TOL    % error tolerance      

% output:  
% J      % current density N^2xNi

Jo = zeros(N^2,Ni);            % initial guess
ro = fft_A(Jo,Z,N,Ni,Xr)-b;    % ro = A.Jo - b;
go = fft_AH(ro,Z,N,Ni,Xr);     % Complex conjugate AH
po = -go;

for n = 1:max_it;
    alpha = -sum(conj(fft_A(po,Z,N,Ni,Xr)).*(fft_A(Jo,Z,N,Ni,Xr)-b),1)/norm(reshape(fft_A(po,Z,N,Ni,Xr),N^2*Ni,1),'fro')^2; % 1 x Ni
    J = Jo +repmat(alpha,N^2,1).*po; 
    r = fft_A(J,Z,N,Ni,Xr)-b;
    g = fft_AH(r,Z,N,Ni,Xr); 
    
    error = norm(r)/norm(b);   % error tolerance
    fprintf('%.4e\n',error)
    error_res(n)=error;
    if error < TOL             % stopping criteria 
       break
    end
    
    beta = sum(conj(g).*(g-go),1)./sum(abs(go).^2,1); 
    p    = -g +repmat(beta,N^2,1).*po; 
        
    po = p; 
    Jo = J; 
    go = g; 
end

function e = fft_A(J,Z,N,Ni,Xr)

% Compute Matrix-vector product by using two-dimensional FFT
J = reshape(J,N,N,Ni);
Z = repmat(Z,1,1,Ni);
e = ifft2(fft2(Z).*fft2(J,2*N-1,2*N-1));

e = e(1:N,1:N,:);
e = reshape(e,N*N,Ni);
e = reshape(J,N*N,Ni) + repmat(Xr(:),1,Ni).*e;

function  e = fft_AH(J,Z,N,Ni,Xr)

% Compute Matrix-vector product by using two-dimensional FFT*
% *complex conjugate operator
J = reshape(J,N,N,Ni);
Z = repmat(Z,1,1,Ni);
e = ifft2(fft2(conj(Z)).*fft2(J,2*N-1,2*N-1));
e = e(1:N,1:N,:);
e = reshape(e,N*N,Ni);
e = reshape(J,N*N,Ni) + conj(repmat(Xr(:),1,Ni)).*e;
