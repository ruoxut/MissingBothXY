function [ t,f ] = loclin_IPW_M( a,b,X,Y,RX,RY,U,alpha )
% The inverse probability weighting local linear estimator where both X and
% Y are subject to missingness using the quadratic g_{r^x,r^y}'s.
% Input:
% [a,b]: the interval to evaluate the regression estimator;
% X: n*1 covariates;
% Y: n*1 outcomes;
% RX: n*1 missing indicator for X;
% RY: n*1 missing indicator for Y;
% U: n*p auxiliary variables;
% alpha: estimated coefficients for the linear g_{r^x,r^y}'s.
% Output:
% t: 200*1 the vector where to evaluate the regression estimator;
% f: 200*1 estimated regression values.

% Author: Ruoxu Tan; date: 2022/Nov/29; Matlab version: R2020a.

if isrow(X)
    X = X';
end

if isrow(Y)
    Y = Y';
end

if isrow(RX)
    RX = RX';
end

if isrow(RY)
    RY = RY';
end

t = linspace(a,b,200)'; % Where to evaluate regression

f = zeros(length(t),1);

% Remove incomplete cases
X = X(RX==1&RY==1);
Y = Y(RX==1&RY==1);
U = U(RX==1&RY==1,:);

% Bandwidth selection
opt = optimset('MaxIter',20,'Display','off');
hCV3 = @(h) CVloss3_m_M( h,X,Y,U,alpha,a,b);
h2 = fminbnd(hCV3,(b-a).*length(X)^(-0.5),(b-a)./2,opt);
fCV3 = zeros(length(X),1);% \hat{m}^{(2)}

for i = 1:length(X)
    if X(i) >= a && X(i) <=b
    a0 = normpdf((X-X(i))./h2)./h2./pi_11_M(X,Y,U,alpha);
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
    end
end
Theta_2 = sum(fCV3.^2./pi_11_M(X,Y,U,alpha).*(a<=X&X<=b))/length(RX);

hCV1 = @(h) CVloss1_m_M( h,X,Y,Y,U,alpha,a,b);
h3 = fminbnd(hCV1,(b-a).*length(X)^(-0.5),(b-a)./2,opt);
fCV1 = zeros(length(X),1);% \hat{m}_{lin}

for i = 1:length(X)
    if X(i) >= a && X(i) <=b
    a0 = normpdf((X-X(i))./h3)./h3./pi_11_M(X,Y,U,alpha);
    a1 = (X-X(i)).*a0;
    a2 = (X-X(i)).*a1;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    T0 = sum(a0.*Y);
    T1 = sum(a1.*Y);    
        
    fCV1(i) = (T0*S2-T1*S1)/(S0*S2-S1^2);  
    end
end

sigma2 = (Y-fCV1).^2./pi_11_M(X,Y,U,alpha);
hCVS1 = @(h) CVloss1_m_M( h,X,Y,sigma2,U,alpha,a,b);
h4 = fminbnd(hCVS1,(b-a).*length(X)^(-0.5),(b-a)./2,opt);
sigma_CV = zeros(length(t),1);

for i = 1:length(t)
    a0 = normpdf((X-t(i))./h4)./h4./pi_11_M(X,Y,U,alpha);
    a1 = (X-t(i)).*a0;
    a2 = (X-t(i)).*a1;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    T0 = sum(a0.*sigma2);
    T1 = sum(a1.*sigma2);    
    
    sigma_CV(i) = (T0*S2-T1*S1)/(S0*S2-S1^2);  
end
Sigma = trapz(t,sigma_CV);

if Sigma < 0
    Sigma = (b-a) * var(Y);
end

h_PI = (Sigma./(2.*sqrt(pi).*Theta_2.*length(RX))).^(1/5);
if Theta_2 == 0
    h_PI = 2*(b-a);
end

for i = 1:length(t)
    a0 = normpdf((X-t(i))./h_PI)./h_PI./pi_11_M(X,Y,U,alpha);
    a1 = (X-t(i)).*a0;
    a2 = (X-t(i)).*a1;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    T0 = sum(a0.*Y);
    T1 = sum(a1.*Y);    
    
    f(i) = (T0*S2-T1*S1)/(S0*S2-S1^2);  
end

end
