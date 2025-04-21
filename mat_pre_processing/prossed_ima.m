clc;clear
%用于调试脚本
img = imread('1-1.bmp');

% img = img(1124-600:1124+600,1124-600:1124+600,:);
% imshow(img);
% subplot(2,3,1);
% imshow(img);
% title('原图');

hsv = rgb2hsv(img);
V_eq = histeq(hsv(:,:,3)); 
hsv_eq = hsv;
hsv_eq(:,:,3) = V_eq;  

img_eq = hsv2rgb(hsv_eq);
% subplot(2,3,2);
% imshow(img_eq);
% title('亮度直方图均衡化');
% subplot(1,3,1);
% imshow(img_eq(:,:,1));
% 
% subplot(1,3,2);
% imshow(img_eq(:,:,2));
% 
% subplot(1,3,3);
% imshow(img_eq(:,:,3));

mask = (img_eq(:,:,1) > 0.7);
% subplot(2,3,3);
% imshow(mask)
% title('R通道阈值分割');

se = strel('disk', 5); 
% 开操作
mask = imerode(mask, se);
mask = imdilate(mask, se);
% subplot(2,3,4);
% imshow(mask);
% title('开操作去除离散点');
%选取最大连通区域
mask = imerode(mask, se);

stats = regionprops(mask,'Area','BoundingBox','PixelIdxList');
[max_area,idx] = max([stats.Area]);
if max_area > 30000
    mask(:) = 0;
    mask(stats(idx).PixelIdxList) = 1;
end
mask = imdilate(mask, se);
imshow(mask);
% subplot(1,2,5);
% imshow(mask);
% title('种子外形掩膜');
% img = img.*uint8(mask);
% subplot(2,3,6);
% imshow(img);
% title('提取种子外形图案');

%裁切种子外形
bbox = stats(idx).BoundingBox;
box(1,1) = (2*bbox(1,1)+bbox(1,3))/2 - (1176/2+5);
box(1,2) = (2*bbox(1,2)+bbox(1,4))/2 - (1193/2+5);
box(1,3) = 1176+10;
box(1,4) = 1193+10;
img = imcrop(img,box);
figure;
imshow(img);