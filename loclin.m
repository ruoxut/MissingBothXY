function [ t,f1,f2 ] = loclin( X,Y,a,b )
% The 1-d covariate local linear estimator for E(Y|X) and
% the local quadratic ridged estimator for E'(Y|X).  
% Input:
% X: n*1 covariate vector;
% Y: n*1 outcome vector;
% [a,b]: the interval for which we estimate E(Y|X), [q_0.025 and q_0.975] as default.
% Output:
% t: the vector where we evaluate the estimator;
% f1: the vector of estimated E(Y|X=x);
% f2: the vector of estimated E'(Y|X=x).

% Author: Ruoxu Tan; date: 2022/Nov/29; Matlab version: R2020a.

if nargin < 3
    a = quantile(X,0.025);
    b = quantile(X,0.975);
end

if isrow(X)
    X = X';
end

if isrow(Y)
    Y = Y';
end

if length(unique(X)) < 4
    error('Too few data points.')
end

t = linspace(a,b,200)'; %Where to evaluate regression

f1 = zeros(length(t),1);
f2 = zeros(length(t),1);

%Bandwidth selection
opt = optimset('Display','off','MaxIter',20);
Wei = X>=a & X<=b;%Weighting function
      
ht0 = @(h) CVloss(X,Y,h,a,b);
hCV = fminbnd(ht0,(b-a).*length(unique(X))^(-0.5),(b-a)./2,opt);
fpilot = zeros(length(X),1);
WV = zeros(length(X),length(X));

for i = 1:length(X)
    if X(i) >= a && X(i) <=b
    a0 = normpdf((X-X(i))./hCV)./hCV;
    a1 = (X-X(i)).*a0;
    a2 = (X-X(i)).*a1;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    T0 = sum(a0.*Y);
    T1 = sum(a1.*Y);    
        
    fpilot(i) = (T0*S2-T1*S1)/(S0*S2-S1^2);    
    WV(i,:) = (a0'.*S2-a1'.*S1)./(S0*S2-S1^2);    
    end
end

nu = length(X) - 2.*trace(WV)+sum(sum(WV.^2));   
Var = sum((Y-fpilot).^2.*Wei)./nu;
       
ht = @(h) CVloss3(X,Y,h,a,b);
hCV3 = fminbnd(ht,(b-a).*length(unique(X))^(-0.5),(b-a)./2,opt);
fCV3 = zeros(length(X),1); 
ffCV3 = zeros(length(X),1);

for i = 1:length(X)
    if X(i) >= a && X(i) <=b      
    a0 = normpdf((X-X(i))./hCV3)./hCV3;
    a1 = (X-X(i)).*a0;
    a2 = (X-X(i)).*a1;
    a3 = (X-X(i)).*a2;
    a4 = (X-X(i)).*a3;
    a5 = (X-X(i)).*a4;
    a6 = (X-X(i)).*a5;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    S3 = sum(a3);
    S4 = sum(a4);
    S5 = sum(a5);
    S6 = sum(a6);
    
    T0 = sum(a0.*Y);
    T1 = sum(a1.*Y); 
    T2 = sum(a2.*Y); 
    T3 = sum(a3.*Y);
    
    fCV3(i) = 2*(T3*(- S5*S1^2 + S4*S1*S2 + S1*S3^2 - S2^2*S3 + S0*S5*S2 - S0*S4*S3)...
                -T0*(- S6*S2^2 + S5*S2*S3 + S2*S4^2 - S3^2*S4 + S1*S6*S3 - S1*S5*S4)...
                -T1*(S3^3 - S0*S3*S6 + S0*S4*S5 + S1*S2*S6 - S1*S3*S5 - S2*S3*S4)...
                +T2*(S6*S1^2 - 2*S1*S3*S4 + S2*S3^2 + S0*S4^2 - S0*S2*S6))...
               /(S6*S1^2*S4 - S1^2*S5^2 - 2*S6*S1*S2*S3 + 2*S1*S2*S4*S5 + 2*S1*S3^2*S5 - 2*S1*S3*S4^2 + S6*S2^3 - 2*S2^2*S3*S5 - S2^2*S4^2 + 3*S2*S3^2*S4 - S0*S6*S2*S4 + S0*S2*S5^2 - S3^4 + S0*S6*S3^2 - 2*S0*S3*S4*S5 + S0*S4^3);            
    
    ffCV3(i) = 6*(T3*(S4*S1^2 - 2*S1*S2*S3 + S2^3 - S0*S4*S2 + S0*S3^2)...
        -T0*(S5*S2^2 - 2*S2*S3*S4 + S3^3 - S1*S5*S3 + S1*S4^2)...
        +T2*(- S5*S1^2 + S4*S1*S2 + S1*S3^2 - S2^2*S3 + S0*S5*S2 - S0*S4*S3)...
        +T1*(- S2^2*S4 + S2*S3^2 + S1*S5*S2 - S1*S3*S4 - S0*S5*S3 + S0*S4^2))...
    /(S6*S1^2*S4 - S1^2*S5^2 - 2*S6*S1*S2*S3 + 2*S1*S2*S4*S5 + 2*S1*S3^2*S5 - 2*S1*S3*S4^2 + S6*S2^3 - 2*S2^2*S3*S5 - S2^2*S4^2 + 3*S2*S3^2*S4 - S0*S6*S2*S4 + S0*S2*S5^2 - S3^4 + S0*S6*S3^2 - 2*S0*S3*S4*S5 + S0*S4^3);
    end
end

Theta1 = sum(fCV3.^2.*Wei)/length(X);
h1 = (Var./(2.*sqrt(pi).*Theta1.*length(X))).^(1/5);
if Theta1 == 0
    h1 = 2*(b-a);
end
  
Theta2 = sum(ffCV3.^2.*Wei)/length(X);
h2 = 0.8843.*(Var./(Theta2.*length(X))).^(1/7);
if Theta2 == 0
    h2 = 2*(b-a);
end

for i = 1:length(t)
    a0 = normpdf((X-t(i))./h1)./h1;
    a1 = (X-t(i)).*a0;
    a2 = (X-t(i)).*a1;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    T0 = sum(a0.*Y);
    T1 = sum(a1.*Y);    
    
    f1(i) = (T0*S2-T1*S1)/(S0*S2-S1^2);
    
    a0 = normpdf((X-t(i))./h2)./h2;
    a1 = (X-t(i)).*a0;
    a2 = (X-t(i)).*a1;
    a3 = (X-t(i)).*a2;
    a4 = (X-t(i)).*a3;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    S3 = sum(a3);
    S4 = sum(a4);  
    T0 = sum(a0.*Y);
    T1 = sum(a1.*Y);
    T2 = sum(a2.*Y);   
    
    f2(i) = (T2*(S0*S3 - S1*S2)+T0*(S1*S4 - S2*S3)-T1*(- S2^2 + S0*S4))/(S4*S1^2 - 2*S1*S2*S3 + S2^3 - S0*S4*S2 + S0*S3^2);
end

end

