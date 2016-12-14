function [PSF,k] = myGaussFilter(A,scale,sigma)
% [PSF] = myGaussFilter(A,scale,sigma)
% Generate the two-dimensional Gaussian filter based on the amplitude A,
% scale and sigma,
% and the output are the filter function and the correspond k
% Version 1.0
% Author Zhixian Ma
% Date 2016/02/19

% Init
Row_half = fix(scale(1)/2);
Col_half = fix(scale(2)/2);
PSF = zeros(scale);

for i = 1 : scale(1)
    for j = 1 : scale(2)
        d = sqrt((i-Row_half)^2 + (j-Col_half)^2);
        PSF(i,j) = A * exp(-d^2/(2*sigma^2));
    end
end

k = 0.255/sigma;   % Transform sigma to scale k