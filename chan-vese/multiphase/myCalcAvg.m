function avg = myCalcAvg(img,levelSet,regcode,param)
% avg = myCalcAvg(img,fphi,regcode)
% Calculate the average of the region
% Input
% img: the image matrix
% levelSet: the level set group
%           levelSet.phi1 = level set of the first phase
%           levelSet.phi2 = level set of the second phase
%           levelSet.target = 1 or 2, w.r.t the level set.
% regcode: code w.r.t. the target region, could be '00','01','10','11'
% param: the coefficient for the Heaviside function.
%
% Output
% avg: the corresponding average of the target region
%
% Version 1.0
% Date: 2017/03/25
% Author: Zhixian MA <zxma_sjtu@qq.com>

if nargin < 4
    param = 1;
end
% Define functions
fh = @(x)(1/2*(1+2/pi*atan(x/param)));

phi1 = levelSet.phi1;
phi2 = levelSet.phi2;
% calc average
if strcmp(regcode,'11')
    H = fh(phi1) .* fh(phi2);
    H_w = img .* H;
    avg = sum(H_w(:)) / sum(H(:));
elseif strcmp(regcode,'10')
    H = fh(phi1) .* (1 - fh(phi2));
    H_w = img .* H;
    avg = sum(H_w(:)) / sum(H(:));
elseif strcmp(regcode,'01')
    H = (1 - fh(phi1)) .* fh(phi2);
    H_w = img .* H;
    avg = sum(H_w(:)) / sum(H(:)); 
elseif strcmp(regcode,'00')
    H = (1 - fh(phi1)) .* (1 - fh(phi2));
    H_w = img .* H;
    avg = sum(H_w(:)) / sum(H(:));
else
    disp('Wrong regcode')
    avg = inf;
end