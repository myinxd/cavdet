function [fphi,u_p,u_n] = myCVsmooth(imgraw,u_p,u_n,fphi,params,numiter)
% levelSet = myCVsmooth(imgraw,levelSet,params)
% This script realizes the basic iterations of Chan-Vese based multiphase
% segmentation algorithm, the piecewise smooth case, and four phases
% Reference
% Vese and Chan, Image segmentation using the Mumford and Shah model IJCV
%
% Input
% imgraw: the raw image, i.e., the matrix u0
% u_p,u_n: the initialzed positive and negative u matrices
% fphi: the initialized level set
% params: parameters for the Mumford and Shah model
%         params = [h,dt,mu,nu]
%         h : space step
%         dt: time step
%         mu: coefficient for the area penalty
%         nu: coefficient for the boardline penalty
% numiter: number of iterations, default as 100
%
% Output:
% fphi:the updated level set.
%
% Version 1.2
% Date: 2017/03/28
% Author: Zhixian MA <zxma_sjtu@qq.com>

if nargin < 4
    disp('Missing required inputs.')
    return
elseif nargin < 5
    numiter = 100;
    params = [1,1,1,1];    
elseif nargin < 6
   numiter = 100;
end

%% Init
[rows, cols] = size(imgraw);
% u_p = imgraw; u_p(fphi<0) = 0; 
% u_n = imgraw; u_n(fphi>=0) = 0;
grad_p = zeros(rows,cols);
grad_n = zeros(rows,cols);
% Coefficient of the Heaviside equation and Dirac equation
h = params(1);
dt = params(2);
yp = h;
yita = 1e-8;
% Other coefficients
mu = params(3);
nu = params(4);

% Define functions
fh = @(x)(1/2*(1+2/pi*atan(x/yp)));
fd = @(x)(1/pi*(yp/(yp^2+x^2)));

%% Begin iteration
t = 1;
while t <= numiter
    t = t + 1;
    for i = 1 : cols
        if i == 1
            il = 1;ir = i + 1;
        elseif i == cols
            il = i - 1; ir = cols;
        else
            il = i - 1; ir = i + 1;
        end
        for j = 1 : rows
            if j == 1
                ju = 1;jd = j + 1;
            elseif j == rows
                ju = j - 1; jd = rows;
            else
                ju = j - 1; jd = j + 1;
            end
                        % init elements
            phi_cc = fphi(j,i);
            phi_lc = fphi(j,il);phi_rc = fphi(j,ir);
            phi_cu = fphi(ju,i);phi_cd = fphi(jd,i);
            phi_lu = fphi(ju,il);phi_ld = fphi(jd,il);
            phi_ru = fphi(ju,ir);
            % Calc differences scheme
            D1 = 1/sqrt((phi_rc-phi_cc)^2/h^2 + (phi_cd-phi_cu)^2/(2*h)^2 + yita^2);
            D2 = 1/sqrt((phi_cc-phi_lc)^2/h^2 + (phi_ld-phi_lu)^2/(2*h)^2 + yita^2);
            D3 = 1/sqrt((phi_rc-phi_lc)^2/(2*h)^2 + (phi_cd-phi_cc)^2/h^2 + yita^2);
            D4 = 1/sqrt((phi_ru-phi_lu)^2/(2*h)^2 + (phi_cc-phi_cu)^2/h^2 + yita^2);
            % update fphi
            m = dt/h^2*fd(phi_cc)*nu;
            D = 1 + m * (D1 + D2 + D3 + D4);
            % gradiends
            grad_p(j,i) = (u_p(j,ir)-u_p(j,il))^2/(2*h)^2 + (u_p(jd,i)-u_p(ju,i))^2/(2*h)^2;
            grad_n(j,i) = (u_n(j,ir)-u_n(j,il))^2/(2*h)^2 + (u_n(jd,i)-u_n(ju,i))^2/(2*h)^2;
            % phi
            fphi(j,i) = phi_cc + m * (D1*phi_rc+D2*phi_lc+D3*phi_cd+D4*phi_cu) + ...
                        dt * fd(phi_cc)*(-(imgraw(j,i)-u_p(j,i))^2 + (imgraw(j,i)-u_n(j,i))^2 - ...
                        mu*grad_p(j,i) + mu*grad_n(j,i));
            fphi(j,i) = fphi(j,i) / D;
            % Update of u
            %u_p(i,j) = 0.25*(u_p(il,j)+u_p(ir,j)+u_p(i,ju)+u_p(i,jd));
            c_p = fh(phi_cc) + mu/h^2*(2*fh(phi_cc) + fh(phi_lc) + fh(phi_cu));
            % update u_p
            u_p(j,i) = mu/h^2*(fh(phi_cc)*u_p(j,ir) + fh(phi_lc)*u_p(j,il)+ ...
                                 fh(phi_cc)*u_p(jd,i) + fh(phi_cu)*u_p(ju,i)) + fh(phi_cc)*imgraw(j,i);
            u_p(j,i) = u_p(j,i) / (c_p+yita);
            % Extention of u_n
            %u_n(i,j) = 0.25*(u_n(il,j)+u_n(ir,j)+u_n(i,ju)+u_n(i,jd));
            c_n = (1-fh(phi_cc)) + mu/h^2*(2*(1-fh(phi_cc)) + (1-fh(phi_lc)) + (1-fh(phi_cu)));
            % update u_n
            u_n(j,i) = mu/h^2*((1-fh(phi_cc))*u_n(j,ir) + (1-fh(phi_lc))*u_n(j,il)+ ...
                                 (1-fh(phi_cc))*u_n(jd,i) + (1-fh(phi_cu))*u_n(ju,i)) + ...
                                 (1-fh(phi_cc))*imgraw(j,i);
            u_n(j,i) = u_n(j,i) / (c_n+yita);
        end
    end
end
            
            
            
            
            
            
            
            
            
            
