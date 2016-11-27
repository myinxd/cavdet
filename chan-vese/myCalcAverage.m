function [c1,c2] = myCalcAverage(phi,img)
% [c1,c2] = myCalcAverage(phi)
% To calculate average densities in the segemented regions.
% Reference:
%   [1] Getreuer. P., "Chan-Vese Segmentation"
%       http://dx.doi.org/10.5201/ipol.2012.g-cv
%
% Input:
% phi: the level set matrix
% img: the raw image
% Output:
% c1,c2: average densities in the segmented regions
%
% Version: 1.0
% Date: 2016/11/26
% Author: Zhixian MA

% Init
if size(img,3) > 1
    img = double(rgb2gray(img));
end
% Calc averages
idx1 = find(phi >= 0);
c1 = sum(img(idx1))/length(idx1);
idx2 = find(phi < 0);
c2 = sum(img(idx2))/length(idx2);