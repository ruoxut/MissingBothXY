function [ RX,RY ] = MisDM( X,Y,U )
% A stable/linear missing data mechanism model used for models (i), (ii) and (iii)
% in the paper.

n = length(X);
RX = zeros(n,1);
RY = RX;
for i = 1:n
    p00 = exp(-2+U(i))./(1+exp(-2+U(i))+exp(-1+X(i)+0.5.*U(i))+exp(-3+3.*Y(i)-0.5.*U(i)));
    p10 = exp(-1+X(i)+0.5.*U(i))./(1+exp(-2+U(i))+exp(-1+X(i)+0.5.*U(i))+exp(-3+3.*Y(i)-0.5.*U(i)));
    p01 = exp(-3+3.*Y(i)-0.5.*U(i))./(1+exp(-2+U(i))+exp(-1+X(i)+0.5.*U(i))+exp(-3+3.*Y(i)-0.5.*U(i)));
    p11 = 1./(1+exp(-2+U(i))+exp(-1+X(i)+0.5.*U(i))+exp(-3+3.*Y(i)-0.5.*U(i)));
     
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

