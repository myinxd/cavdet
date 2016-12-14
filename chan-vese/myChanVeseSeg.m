function [phi] = myChanVeseSeg(img,mu,nu,lamda1,lamda2,numiter,phi_total,dt,phi_init)
% [phi] = myChanVeseSeg(img,mu,nu,lamda1,lamda2,numiter,phiotal,dt,phi_init)
% The Chan-Vese segmentation main method
% Reference:
%   [1] Getreuer. P., "Chan-Vese Segmentation"
%       http://dx.doi.org/10.5201/ipol.2012.g-cv
%
% Input:
% img: the raw image
% mu: coefficient of the boundary
% nu: coefficient of the segmented region
% lamda1: coefficient of internal term
% lamda2: coefficient of external term
% numiter: iteration times
% dt: time step
% phi_init: initialized matrix of the level set funtion
% Output:
% phi: the final level set function
% 
% Example:
%   phi = myChanVeseSeg(img,0.2,0,1,1,100,1e-3,0.5)
% Version: 1.0
% Date: 2016/11/26
% Author: Zhixian MA

if nargin < 9
    if size(img,3) > 1
        img = double(rgb2gray(img));
    end
    phi_init = myChanVeseInit(img);
else
    if size(img,3) > 1
        img = double(rgb2gray(img));
    end
end

% Init
% Norm
img = (img - min(img(:)))/(max(img(:)) - min(img(:)));
yita = 1e-8; % divided parameter
[rows,cols] = size(img);
% Init average densities
[c1,c2] = myCalcAverage(phi_init,img);
% Iteration
phi = phi_init;
for iter = 1 : numiter
    phidiffnorm = 0;
    for j = 1 : rows
        if j == 1
            iu = 0;
        else
            iu = -1;
        end
        if j == rows
            id = 0;
        else
            id = 1;
        end
        for i = 1 : cols
            if i == 1
                il = 0;
            else
                il = -1;
            end
            if i == cols
                ir = 0;
            else
                ir = 1;
            end
            
            Delta = dt/(pi * (1 + phi(j,i)*phi(j,i)));
            phi_x = phi(j,i+ir) - phi(j,i);
            phi_y = (phi(j+id,i) - phi(j+iu,i)) / 2;
            IDivR = 1/sqrt(yita + phi_x^2 + phi_y^2);
            phi_x = phi(j,i) - phi(j,i+il);
            IDivL = 1/sqrt(yita + phi_x^2 + phi_y^2);
            phi_x = (phi(j,i+ir) - phi(j,i+il))/2;
            phi_y = phi(j+id,i) - phi(j,i);
            IDivD = 1/sqrt(yita + phi_x^2 + phi_y^2);
            phi_y = phi(j,i) - phi(j+iu,i);
            IDivU = 1/sqrt(yita + phi_x^2 + phi_y^2);
            
            % Other two terms
            dist1 = (img(j,i) - c1)^2;
            dist2 = (img(j,i) - c2)^2;
            
            % Semi-implicit update of phi at the current point
            phi_last = phi(j,i);
            phi(j,i) = (phi(j,i) + Delta*(mu*(phi(j,i+ir)*IDivR + phi(j,i+il)*IDivL + ...
                        phi(j+id,i)*IDivD + phi(j+iu,i)*IDivU) - nu - lamda1 * dist1 + ...
                        lamda2 * dist2)) / ...
                        (1 + Delta*mu*(IDivR + IDivL + IDivD + IDivU));
            phidiff = phi(j,i) - phi_last;
            phidiffnorm = phidiffnorm + phidiff^2;
        end
    end
    if phidiffnorm <= phi_total && iter >= 2
        break;
    end
    % Renew c1 and c2
    [c1,c2] = myCalcAverage(phi,img);
end    
