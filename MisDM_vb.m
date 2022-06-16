function [ RX,RY ] = MisDM_vb( X,Y,U )
% A variable missing data mechanism model used for model (iv) in the paper.

n = length(X);
RX = zeros(n,1);
RY = RX;
for i = 1:n 
    p00 = exp(-2+U(i))./(1+exp(-2+U(i))+exp(-3+(X(i)+U(i)).^2)+exp(1./log(abs(0.2+Y(i)))));
    p10 = exp(-3+(X(i)+U(i)).^2)./(1+exp(-2+U(i))+exp(-3+(X(i)+U(i)).^2)+exp(1./log(abs(0.2+Y(i)))));
    p01 = exp(1./log(abs(0.2+Y(i))))./(1+exp(-2+U(i))+exp(-3+(X(i)+U(i)).^2)+exp(1./log(abs(0.2+Y(i)))));
    p11 = 1./(1+exp(-2+U(i))+exp(-3+(X(i)+U(i)).^2)+exp(1./log(abs(0.2+Y(i)))));
    
    p = [p00,p10,p01,p11];
    r = mnrnd(1,p);
    if r(2)==1
        RX(i)=1;
    elseif r(3)==1
        RY(i)=1;
    elseif r(4)==1
        RX(i)=1;
        RY(i)=1;
    end    
end

end

