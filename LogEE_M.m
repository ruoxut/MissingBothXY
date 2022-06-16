function [ a_hat ] = LogEE_M( X,Y )
% EE for estimating g_{r^x,r^y} using the quadratic g.
% Input:
% X: |I_{r^x,r^y}|*p arguments;
% Y: |I_{r^x,r^y}|*1 binary indicartor for the missing data pattern: 0 for
% (r^x,r^y) and 1 for (1,1).
% Outupt: 
% a_hat: 1*(p+1) estimated coefficients.

EEsol = @(a) [sum(Y.*(1+exp(g_M(a,X)))-1), sum((Y.*(1+exp(g_M(a,X)))-1).*X.^2)];
opt = optimset('Display','off');
a_ini = zeros(1,size(X,2)+1);
a_hat = fsolve(EEsol,a_ini,opt);


end

