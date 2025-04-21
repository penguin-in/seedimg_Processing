%此脚本用于确定每厘米像素点个数
img = imread('biaochi.png');


bw = (img(:,:,1)>100)&(img(:,:,2)<100)&(img(:,:,3)<100);
imshow(bw);
stats = regionprops(bw, 'Area', 'PixelIdxList');
[~, maxIdx] = max([stats.Area]);
largest_bw = false(size(bw)); 
largest_bw(stats(maxIdx).PixelIdxList) = true; 

[B, ~] = bwboundaries(largest_bw, 'noholes');

max_distance = 0;
boundary = B{1};
p1 = [0, 0]; 
p2 = [0, 0]; 
for i = 1:size(boundary, 1)
    for j = i+1:size(boundary, 1)
        dist = norm(boundary(i,:) - boundary(j,:));
        if dist > max_distance
            max_distance = dist;
            p1 = boundary(i, :);
            p2 = boundary(j, :);
        end
    end
end

figure, imshow(img);
hold on;


plot([p1(2), p2(2)], [p1(1), p2(1)], 'r-', 'LineWidth', 2);

text(mean([p1(2), p2(2)]), mean([p1(1), p2(1)]), ...
    sprintf('%.2f pixels', max_distance), ...
    'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold');

fprintf('最大白色矩形的长度为: %.2f 像素\n', max_distance);
hold off;