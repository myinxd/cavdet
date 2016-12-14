function [ImgSub,coeffs] = getBetaSub(ImgRaw,radius,mode)
% ImgSub = getBetaSub(ImgRaw,radius)
% Detect centroid or peak max point in the raw image, and cut the subimage 
% according to provided radius.
% Then the beta2d curve fitting will be performed on the cutted image, and
% the the final image by subtracting the fitted model in the cutted image
% is output.
%
% Input
% ImgRaw: the raw image
% radius: the aim radius
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

if nargin < 3
    mode = 'cen';
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
cen_new = [cord_c(1)-col_l+1,cord_c(2)-row_l+1];

% curve fit
[myfit,coeffs] = myBeta2dFit(ImgCut,cen_new);

% subtract
[row_cut,col_cut] = size(ImgCut);
ImgSub = zeros(row_cut,col_cut);
for i = 1 : col_cut
    for j = 1 : row_cut
        ImgSub(j,i) = ImgCut(j,i) - myfit((i-cen_new(1)),(j-cen_new(2)));
    end
end

% Draw
myBetaDraw(ImgSub,cen_new,myfit)