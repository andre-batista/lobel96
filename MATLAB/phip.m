function [fx] = phip(t)
    fx = 2*t./(1+t.^2).^2;