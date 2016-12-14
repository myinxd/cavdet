function [cord_c,cord_m] = myCenAndPeak(mat)
% [cord_c,cord_m] = myCenAndPeak(mat)
% Calculate centroid and maximum peak in the mat
% 
% Input
% mat: the surface matrix
%
% Output
% cord_c: [centroid_x,centroid_y]
% cord_m: [peak_x, peak_y]
%
% Version: 1.0
% Date: 2016/12/01
% Author:Zhixian MA <zxma_sjtu@qq.com>

% Peak
p_max = max(mat(:));
[y,x] = find(mat == p_max);
peak_x = mean(x);
peak_y = mean(y);
cord_m = [peak_x,peak_y];

% Centroid
[rows,cols] = size(mat);
[X,Y] = meshgrid(1:cols,1:rows);
sum_x = X.*mat;
sum_y = Y.*mat;
cen_x = sum(sum_x(:))/sum(mat(:));
cen_y = sum(sum_y(:))/sum(mat(:));
cord_c = [cen_x,cen_y];
