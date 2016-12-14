function myBetaDraw(mat,cen,fitfun)
% myBetaDraw(mat,cen,coeffs)
% Draw beta model

[rows,cols] = size(mat);
mat_draw = zeros(rows,cols);

for i = 1 : cols
    for j = 1: rows
        mat_draw(j,i) = fitfun(i-cen(1),j-cen(2));
    end
end

figure(1)
surf(mat_draw);
figure(2)
imshow(mat_draw,[])
