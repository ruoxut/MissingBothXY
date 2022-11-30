function [ S ] = CVloss3_m_M( h,X,Y,U,alpha,a,b ) 
% Cross-validation loss function of the local cubic estimator for the second derivative
% using the quadratic g_{r^x,r^y}'s.

% Author: Ruoxu Tan; date: 2022/Nov/29; Matlab version: R2020a.

S = 0;

for i = 1:length(X)
    if X(i) >= a && X(i) <=b
    Xi = X;
    Xi(i) = [];
    Yi = Y;
    Yi(i) = [];
    Ui = U;
    Ui(i,:) = [];
    
    a0 = normpdf((Xi-X(i))./h)./h./pi_11_M(Xi,Yi,Ui,alpha);
    a1 = (Xi-X(i)).*a0;
    a2 = (Xi-X(i)).*a1;
    a3 = (Xi-X(i)).*a2;
    a4 = (Xi-X(i)).*a3;
    a5 = (Xi-X(i)).*a4;
    a6 = (Xi-X(i)).*a5;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    S3 = sum(a3);
    S4 = sum(a4);
    S5 = sum(a5);
    S6 = sum(a6);
    
    T0 = sum(a0.*Yi);
    T1 = sum(a1.*Yi); 
    T2 = sum(a2.*Yi); 
    T3 = sum(a3.*Yi);
    
    Yi_hat = (T0*(S6*S3^2 - 2*S3*S4*S5 + S4^3 - S2*S6*S4 + S2*S5^2)...
              -T3*(S5*S2^2 - 2*S2*S3*S4 + S3^3 - S1*S5*S3 + S1*S4^2)...
              -T2*(- S6*S2^2 + S5*S2*S3 + S2*S4^2 - S3^2*S4 + S1*S6*S3 - S1*S5*S4)...
              -T1*(- S3^2*S5 + S3*S4^2 + S2*S6*S3 - S2*S4*S5 - S1*S6*S4 + S1*S5^2))...
             /(S6*S1^2*S4 - S1^2*S5^2 - 2*S6*S1*S2*S3 + 2*S1*S2*S4*S5 + 2*S1*S3^2*S5 - 2*S1*S3*S4^2 + S6*S2^3 - 2*S2^2*S3*S5 - S2^2*S4^2 + 3*S2*S3^2*S4 - S0*S6*S2*S4 + S0*S2*S5^2 - S3^4 + S0*S6*S3^2 - 2*S0*S3*S4*S5 + S0*S4^3);

    S = S + (Y(i)-Yi_hat).^2./pi_11_M(X(i),Y(i),U(i),alpha); 
    end
end
  
end
