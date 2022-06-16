function [ a_hat ] = LogMLE_M( X,Y )
% Standard MLE for estimating g_{r^x,r^y} using the quadratic g.
% Input:
% X: |I_{r^x,r^y}|*p arguments;
% Y: |I_{r^x,r^y}|*1 binary indicartor for the missing data pattern: 0 for
% (r^x,r^y) and 1 for (1,1).
% Outupt: 
% a_hat: 1*(p+1) estimated coefficients.

nelogL = @(a) sum(-(1-Y).*g_M(a,X)+log(1+exp(g_M(a,X))));
opt = optimset('Display','off');
a_ini = zeros(1,size(X,2)+1);
a_hat = fminsearch(nelogL,a_ini,opt);

end

