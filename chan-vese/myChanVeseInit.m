function phi_init = myChanVeseInit(img)
% phi_init = myChanVeseInit(img)
% To initialize the phi mat in Chan-Vese segementation method
% phi = sin(pi/5*x)sin(pi/5*y)
% Reference:
%   [1] Getreuer. P., "Chan-Vese Segmentation"
%       http://dx.doi.org/10.5201/ipol.2012.g-cv
%
% Input:
% img: the img to be segmented.
% Output:
% phi_init: the initialized phi mat
%
% Version: 1.0
% Date: 2016/11/26
% Author: Zhixian MA

% Init
if size(img,3) > 1
    img = double(rgb2gray(img));
end
[rows,cols] = size(img);

% phi_init
[X,Y] = meshgrid(1:cols,1:rows);
phi_init = sin(X*pi/5) .* sin(Y*pi/5);

