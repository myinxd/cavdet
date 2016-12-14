function [ImgSub] = getUnsharpMask(ImgRaw,radius,sigma_low,sigma_high,mode)
% myUnshapMask(ImgRaw,radius,sigma_low,sigma_high)
% Detect centroid or peak max point in the raw image, and cut the subimage 
% according to provided radius.
% Then the smoothed image with different sigmas are generated, and
% the the final image by subtracting the sigma_high of sigma_low.
%
% Input
% ImgRaw: the raw image
% radius: the aim radius
% sigma_high,sigma_low: the two sigmas
% mode: mode of center point detection, could be 'cen' (centroid) or 'max'
%       (max peak).
% 
% Output
% ImgSub: the subtracted image.
% coeffs: coefficients of the beta 2d model
%
% Version: 1.0
% Date: 2016/12/01
% Author: Zhixian MA <zxma_sjtu@qq.com>

if nargin < 5
    mode = 'max';
end

% Init
[rows,cols] = size(ImgRaw);

% Center point
if strcmp(mode,'cen')
    cord_c = myCenAndPeak(ImgRaw);
else
    [~,cord_c] = myCenAndPeak(ImgRaw);
end

% Cut
% Edge limits
row_l = fix(cord_c(2)) - radius;
row_h = fix(cord_c(2)) + radius;
col_l = fix(cord_c(1)) - radius;
col_h = fix(cord_c(1)) + radius;
if row_l <= 0
    row_l = 1;
end
if row_h >= rows
    row_h = rows;
end
if col_l <= 0
    col_l = 1;
end
if col_h >= cols
    col_h = cols;
end
% cut
ImgCut = ImgRaw(row_l:row_h,col_l:col_h);

% Smooth
% filters
scale = size(ImgCut);
psf1 = myGaussFilter(1,scale,sigma_high);
psf2 = myGaussFilter(1,scale,sigma_low);

% Calc difference
ImgFilted1 = imfilter(ImgCut,abs(fftshift(fft2(psf1))),'conv','circular');
ImgFilted2 = imfilter(ImgCut,abs(fftshift(fft2(psf2))),'conv','circular');
ImgSub = ImgFilted1 - ImgFilted2;