function [ W ] = W_m_M( h,X,Y,U,alpha,t0 )
% Weighting matrix of X at t0 with bandwidth h under quadratic g_r's.

% Author: Ruoxu Tan; date: 2022/Nov/29; Matlab version: R2020a.

w = normpdf((X-t0)./h)./h./pi_11_M(X,Y,U,alpha);
W = diag(w);
end

