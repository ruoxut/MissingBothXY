function [ a_hat ] = LogCMLE( X,Y,xi )
% CMLE for estimating g_{r^x,r^y} using the linear g.
% Input:
% X: |I_{r^x,r^y}|*p arguments;
% Y: |I_{r^x,r^y}|*1 binary indicartor for the missing data pattern: 0 for
% (r^x,r^y) and 1 for (1,1).
% xi: tuning parameter for the constraint.
% Outupt: 
% a_hat: 1*(p+1) estimated coefficients.

% Author: Ruoxu Tan; date: 2022/Nov/29; Matlab version: R2020a.

function [c,ceq] = nlcon(a,X,Y,xi)
%Constriant
c = sum(Y.*(1+exp(g(a,X)))-1-xi);
ceq = [];
end

cons = @(a) nlcon(a,X,Y,xi);
              
nelogL = @(a) sum(-(1-Y).*g(a,X)+log(1+exp(g(a,X))));
opt = optimset('Display','off');
a_ini = zeros(1,size(X,2)+1);
a_hat = fmincon(nelogL,a_ini,[],[],[],[],[],[],cons,opt);

end

