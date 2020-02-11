function [re,im] = normgrad(c)

    [del_x, del_y] = gradient(c);
    re = sqrt(real(del_x).^2+real(del_y).^2);
    im = sqrt(imag(del_x).^2+imag(del_y).^2);