function [ a_hat ] = LogEE( X,Y )
% EE for estimating g_{r^x,r^y} using the linear g.
% Input:
% X: |I_{r^x,r^y}|*p arguments;
% Y: |I_{r^x,r^y}|*1 binary indicartor for the missing data pattern: 0 for
% (r^x,r^y) and 1 for (1,1).
% Outupt: 
% a_hat: 1*(p+1) estimated coefficients.

% Author: Ruoxu Tan; date: 2022/Nov/29; Matlab version: R2020a.

EEsol = @(a) [ sum(Y.*(1+exp(g(a,X)))-1), sum((Y.*(1+exp(g(a,X)))-1).*X)];
opt = optimset('Display','off');
a_ini = zeros(1,size(X,2)+1);
a_hat = fsolve(EEsol,a_ini,opt);

end

