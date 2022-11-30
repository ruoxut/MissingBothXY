function [ out ] = pi_11( X,Y,U,alpha)
% \pi_{1,1} evaluated at (X,Y,U) using the linear g_{r^x,r^y}'s with the
% coefficients alpha.

% Author: Ruoxu Tan; date: 2022/Nov/29; Matlab version: R2020a.

alpha_U = alpha(1:1+size(U,2));
alpha_XU = alpha(2+size(U,2):3+2*size(U,2));
alpha_YU = alpha(4+2*size(U,2):5+3*size(U,2));
XU = [X,U];
YU = [Y,U];

out = 1./(1+exp(g(alpha_U,U))+exp(g(alpha_XU,XU))+exp(g(alpha_YU,YU)));

end

