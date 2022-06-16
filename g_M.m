function [res] = g_M(a,X)
% The quadratic function g = a_0 + X^2*(a_1,...,a_{p})^T.
% Input: 
% a: 1*(p+1) coefficient vector (a_0,...,a_{p});
% X: n*p arguments.
% Output:
% res: n*1 values of a_0 + X^2*(a_1,...,a_{p})^T.

if iscolumn(a)
    a = a';
end

c = ones(size(X,1),1);
X = [c,X];

if length(a) ~= size(X,2)
    error('Matrix dimensions must agree.')
end

res = sum(a.*X.^2,2);

end
