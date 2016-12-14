function [myfit,coeffs] = myBeta2dFit(mat,cord_c)
% f = myBeta2dFit(mat,cord_c)
% The 2d beta curve fitting code
% Beta 2d model
% ----------------------------------
% f(x,y) = f(r) = A * (1+(r/r0)^2)^(-beta) + C0
% rotation situation
% x_r = x * cos(theta) + y * sin(theta)
% y_r = x * sin(-theta) + y * sin(theta)
% r = x_r^2 + y_r^2*(a/b)^2, a and b are the semi-major and -minor axes.
% Denote e = a/b
% ---------------------------------
%
% Input
% mat: the mat to be fitted
% cord_c: the center of the peak
%
% Output
% coeffs: the fitted coefficients
% 
% Version: 1.0
% Date: 2016/12/01
% Author: Zhixian MA <zxma_sjtu@qq.com>

% Init
[rows,cols] = size(mat);
[X,Y] = meshgrid(1:cols,1:rows);

% Build beta function
X_d = X - cord_c(1);
Y_d = Y - cord_c(2);
myfittype = fittype( @(A,theta,e,r0,beta,C0,x,y) A*(1+((x*cos(theta)+y*sin(theta)).^2 + ...
                                     (x*sin(-theta)+y*cos(theta)).^2*e^2)/r0^2).^-beta+C0, ...
            'independent', {'x', 'y'}, ...
            'dependent', 'z' );
[x,y,z] = prepareSurfaceData(X_d,Y_d,mat);
myfit = fit([x,y],z,myfittype);
coeffs = coeffvalues(myfit);

  