function [re,im] = weighted_laplacian(c,br,bi)
    [I,J] = size(c);
    re = zeros(I,J);
    im = zeros(I,J);

    ii = 2:I-1;
    jj = 2:J-1;
    re(ii,jj) = (-(2*br(ii,jj) + br(ii-1,jj) + br(ii,jj-1))*real(c(ii,jj)) + br(ii,jj)*real(c(ii,jj+1)) + br(ii,jj)*real(c(ii+1,jj)) + br(ii,jj-1)*real(c(ii,jj-1)) + br(ii-1,jj)*real(c(ii-1,jj)));
    im(ii,jj) = (-(2*bi(ii,jj) + bi(ii-1,jj) + bi(ii,jj-1))*imag(c(ii,jj)) + bi(ii,jj)*imag(c(ii,jj+1)) + bi(ii,jj)*imag(c(ii+1,jj)) + bi(ii,jj-1)*imag(c(ii,jj-1)) + bi(ii-1,jj)*imag(c(ii-1,jj)));
    
    i = 1;
    re(i,jj) = 2*re(i+1,jj) - re(i+2,jj);
    im(i,jj) = 2*im(i+1,jj) - im(i+2,jj);
    
    j = 1;
    re(ii,j) = 2*re(ii,j+1) - re(ii,j+2);
    im(ii,j) = 2*im(ii,j+1) - im(ii,j+2);
        
    i = I;
    re(i,jj) = 2*re(i-1,jj) - re(i-2,jj);
    im(i,jj) = 2*im(i-1,jj) - im(i-2,jj);
    
    j = J-1;
    re(ii,j) = 2*re(ii,j-1) - re(ii,j-2);
    im(ii,j) = 2*im(ii,j-1) - im(ii,j-2);
        
    re(1,1) = (re(1,2)+re(2,1))/2;
    im(1,1) = (im(1,2)+im(2,1))/2;
    
    re(1,J) = (re(1,J-1)+re(2,J))/2;
    im(1,J) = (im(1,J-1)+im(2,J))/2;
    
    re(I,1) = (re(I-1,1)+re(I,2))/2;
    im(I,1) = (im(I-1,1)+im(I,2))/2;
    
    re(I,J) = (re(I-1,J)+re(I,J-1))/2;
    im(I,J) = (im(I-1,J)+im(I,J-1))/2;
    
    re = reshape(re,[I*J,1]);
    im = reshape(im,[I*J,1]);