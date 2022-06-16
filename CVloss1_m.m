function [ S ] = CVloss1_m( h,X,Y,YY,U,alpha,a,b )
% Cross-validation loss function for local linear estimator using the linear g_{r^x,r^y}'s.
S = 0;

for i = 1:length(X)
    if X(i) >= a && X(i) <=b
    Xi = X;
    Xi(i) = [];
    Yi = Y;
    Yi(i) = [];
    YYi = YY;
    YYi(i) = [];
    Ui = U;
    Ui(i,:) = [];
    
    a0 = normpdf((Xi-X(i))./h)./h./pi_11(Xi,Yi,Ui,alpha);
    a1 = (Xi-X(i)).*a0;
    a2 = (Xi-X(i)).*a1;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    T0 = sum(a0.*YYi);
    T1 = sum(a1.*YYi);    
    
    Yi_hat = (T0*S2-T1*S1)/(S0*S2-S1^2);
    S = S + (YY(i)-Yi_hat).^2./pi_11(X(i),Y(i),U(i),alpha); 
    end
end
  
end
