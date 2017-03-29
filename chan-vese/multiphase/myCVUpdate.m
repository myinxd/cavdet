function levelSet = myCVUpdate(imgraw,levelSet,params)
% levelSet = myCVUpdate(imgraw,levelSet,params)
% This script realizes the basic iterations of Chan-Vese based multiphase
% segmentation algorithm, the piecewise constant case, and four phases
% Reference
% Vese and Chan, Image segmentation using the Mumford and Shah model IJCV
%
% Input
% imgraw: the raw image, i.e., the matrix u0
% levelSet: the initialized level set group, or a middle one
%           levelSet.phi1 = level set of the first phase
%           levelSet.phi2 = level set of the second phase
%           levelSet.target = 1 or 2, w.r.t the level set.
% params: parameters for the Mumford and Shah model
%         params = [h,dt,mu,nu]
%         h : space step
%         dt: time step
%         mu: coefficient for the area penalty
%         nu: coefficient for the boardline penalty
%
% Output:
% levelSet:the updated level sets
%
% Version 1.2
% Date: 2017/03/28
% Author: Zhixian MA <zxma_sjtu@qq.com>

if nargin < 3
    params = [1,1,1,1];
elseif nargin < 2
    disp('Missing required inputs.')
    return
end

%% Init
[rows,cols] = size(imgraw);
% Coefficient of the Heaviside equation and Dirac equation
h = params(1);
dt = params(2);
yp = h;
% Other coefficients
mu = params(3);
nu = params(4);

% Define functions
fh = @(x)(1/2*(1+2/pi*atan(x/yp)));
fd = @(x)(1/pi*(yp/(yp^2+x^2)));

%% Begin iteration
if levelSet.target == 1
    fphi = levelSet.phi1;
    fphi_other = levelSet.phi2;
    % calc averages of each region
    c11 = myCalcAvg(imgraw,levelSet,'11');
    c10 = myCalcAvg(imgraw,levelSet,'10');
    c01 = myCalcAvg(imgraw,levelSet,'01');
    c00 = myCalcAvg(imgraw,levelSet,'00');
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
            phi_ru = fphi(ju,ir);phi_rd = fphi(jd,ir);
            % Calc differences scheme
            D1 = 1/sqrt((phi_rc-phi_cc)^2/h^2 + (phi_cd-phi_cu)^2/(2*h)^2);
            D2 = 1/sqrt((phi_cc-phi_lc)^2/h^2 + (phi_ld-phi_lu)^2/(2*h)^2);
            D3 = 1/sqrt((phi_rc-phi_lc)^2/(2*h)^2 + (phi_cd-phi_cc)^2/h^2);
            D4 = 1/sqrt((phi_ru-phi_lu)^2/(2*h)^2 + (phi_cc-phi_cu)^2/h^2);
            % update fphi
            m = dt/h^2*fd(phi_cc)*nu;
            D = 1 + m*(D1+D2+D3+D4);
            fphi(j,i) = phi_cc + m * (D1 * phi_rc + D2 * phi_lc + D3 * phi_cd + D4 * phi_cu) + ...
                        dt*fd(phi_cc) * ( ...
                        - (imgraw(j,i)-c11)^2*fh(fphi_other(j,i)) ...
                        - (imgraw(j,i)-c10)^2*(1-fh(fphi_other(j,i))) ...
                        + (imgraw(j,i)-c01)^2*fh(fphi_other(j,i)) ...
                        + (imgraw(j,i)-c00)^2*(1-fh(fphi_other(j,i))));
            fphi(j,i) = fphi(j,i) / D;
        end
    end
    levelSet.target = 2;
    levelSet.phi1 = fphi;
elseif levelSet.target == 2
    fphi = levelSet.phi2;
    fphi_other = levelSet.phi1;
    % calc averages of each region
    c11 = myCalcAvg(imgraw,levelSet,'11');
    c10 = myCalcAvg(imgraw,levelSet,'10');
    c01 = myCalcAvg(imgraw,levelSet,'01');
    c00 = myCalcAvg(imgraw,levelSet,'00');
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
            phi_ru = fphi(ju,ir);%phi_rd = fphi(jd,ir);
            % Calc differences scheme
            D1 = 1/sqrt((phi_rc-phi_cc)^2/h^2 + (phi_cd-phi_cu)^2/(2*h)^2);
            D2 = 1/sqrt((phi_cc-phi_lc)^2/h^2 + (phi_ld-phi_lu)^2/(2*h)^2);
            D3 = 1/sqrt((phi_rc-phi_lc)^2/(2*h)^2 + (phi_cd-phi_cc)^2/h^2);
            D4 = 1/sqrt((phi_ru-phi_lu)^2/(2*h)^2 + (phi_cc-phi_cu)^2/h^2);
            % update fphi
            m = dt/h^2*fd(phi_cc)*nu;
            D = 1 + m*(D1+D2+D3+D4);
            fphi(j,i) = phi_cc + m * (D1 * phi_rc + D2 * phi_lc + D3 * phi_cd + D4 * phi_cu) + ...
                        dt*fd(phi_cc) * ( ...
                        - (imgraw(j,i)-c11)^2*fh(fphi_other(j,i)) ...
                        - (imgraw(j,i)-c10)^2*(1-fh(fphi_other(j,i))) ...
                        + (imgraw(j,i)-c01)^2*fh(fphi_other(j,i)) ...
                        + (imgraw(j,i)-c00)^2*(1-fh(fphi_other(j,i))));
            fphi(j,i) = fphi(j,i) / D;
        end
    end
    levelSet.target = 1;
    levelSet.phi2 = fphi;
else
    disp('Wrong target index.')
    return
end







