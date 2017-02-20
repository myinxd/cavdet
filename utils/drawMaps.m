% draw the feature maps
% load data
map_path = '../result/Train_500_Acc_66.79_Kernel_12_box_20/params.mat';
load(map_path)

% scale
scale = 40;
gap = 5;
% first layer
rows = 3; cols = 4;
map1 = ones(rows*scale+(rows-1)*gap,cols*scale+(cols-1)*gap);
for i = 1 : rows
    for j = 1 : cols
        img = w1((i-1)*rows+j,:,:,:);
        img = sum(img,2);
        img = squeeze(img);
        img_norm = (img - min(img(:))) / (max(img(:))-min(img(:)));
        map_single = imresize(img_norm,[scale,scale]);
        map1((i-1)*(scale+gap)+1 : (i-1)*(scale+gap)+scale, ...
             (j-1)*(scale+gap)+1 : (j-1)*(scale+gap)+scale) = map_single;
    end
end

figure(1)
imshow(map1,[])
    
% second layer
rows = 3; cols = 4;
map2 = ones(rows*scale+(rows-1)*gap,cols*scale+(cols-1)*gap);
for i = 1 : rows
    for j = 1 : cols
        img = w2((i-1)*rows+j,:,:,:);
        img = sum(img,2);
        img = squeeze(img);
        img_norm = (img - min(img(:))) / (max(img(:))-min(img(:)));
        map_single = imresize(img_norm,[scale,scale]);
        map2((i-1)*(scale+gap)+1 : (i-1)*(scale+gap)+scale, ...
             (j-1)*(scale+gap)+1 : (j-1)*(scale+gap)+scale) = map_single;
    end
end

figure(2)
imshow(map2,[])

% third layer
rows = 3; cols = 4;
map3 = ones(rows*scale+(rows-1)*gap,cols*scale+(cols-1)*gap);
for i = 1 : rows
    for j = 1 : cols
        img = w3((i-1)*rows+j,:,:,:);
        img = sum(img,2);
        img = squeeze(img);
        img_norm = (img - min(img(:))) / (max(img(:))-min(img(:)));
        map_single = imresize(img_norm,[scale,scale]);
        map3((i-1)*(scale+gap)+1 : (i-1)*(scale+gap)+scale, ...
             (j-1)*(scale+gap)+1 : (j-1)*(scale+gap)+scale) = map_single;
    end
end

figure(3)
imshow(map3,[])