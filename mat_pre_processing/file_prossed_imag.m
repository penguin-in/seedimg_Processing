clc;clear;close all
%用于处理种子图片
datapath = 'D:\论文\种子形状识别\数据\original_imag';
output1path = 'D:\论文\种子形状识别\数据\prosessing_imag';
outputpath = 'D:\论文\种子形状识别\数据\prosessed_imag';
% imds = imageDatastore(datapath,"IncludeSubfolders",true,"FileExtensions",{'.bmp'});
filename = 'D:\code\seed classification\Seed regression\sorted_output.xlsx';

%读取表格
[~, sheets] = xlsfinfo(filename);
entire_data = [];

alldata = struct();
for i = 2:11
    sheetname = sheets{i};
    filedname = matlab.lang.makeValidName(sheetname);
    T = readtable(filename, 'Sheet', sheetname, 'ReadVariableNames', false);
    T(:,9) = T(:,12);
    entire_data = [entire_data;T(:,2:9)];
end
if exist(output1path, 'dir')
    rmdir(output1path, 's');  
end
mkdir(output1path);

if exist(outputpath, 'dir')
    rmdir(outputpath, 's');  
end
mkdir(outputpath);

stats_all = struct();
subFolders = dir(datapath);

subFolders = subFolders([subFolders.isdir]);
foldernums = [];
for j = 1:length(subFolders)
    fold_name = subFolders(j).name;
    foldernum = str2double(regexp(fold_name,'\d+','match','once'));
    foldernums = [foldernums;foldernum];
end

[~,sortidx] = sort(foldernums);
subFolders = subFolders(sortidx);
%去除背景
img_num = 0;
for j = 1:length(subFolders)
    if subFolders(j).isdir && ~strcmp(subFolders(j).name, '.') && ~strcmp(subFolders(j).name, '..')
        currentFolder = fullfile(datapath,subFolders(j).name);
        imds = imageDatastore(currentFolder,"IncludeSubfolders",true,"FileExtensions",{'.bmp','.png','.jpg'});
        A = zeros(numel(imds.Files),2);
        for u = 1:numel(imds.Files)
            [~,fname] = fileparts(imds.Files{u});
            parts = split(fname,'-');
            A(u,1) = str2double(parts{1});
            A(u,2) = str2double(parts{2});
        end
        [~, sort_Idx] = sortrows(A, [1 2]);
        imds.Files = imds.Files(sort_Idx);

        for i = 1:numel(imds.Files)
            img = imread(imds.Files{i});
            hsv = rgb2hsv(img);

            % 仅对亮度通道(V通道)进行均衡
            V_eq = histeq(hsv(:,:,3));

            % 重建HSV图像
            hsv_eq = hsv;
            hsv_eq(:,:,3) = V_eq;  % 保持色调和饱和度不变

            % 转换回RGB空间
            img_eq = hsv2rgb(hsv_eq);

            mask = (img_eq(:,:,1) > 0.7);
            %开操作去除白色孔洞边缘
            se = strel('disk', 5); % 结构元素，2 代表处理程度
            % 开操作
            mask = imerode(mask, se);
            mask = imdilate(mask, se);
            %选取最大连通区域
            mask = imerode(mask, se);
            stats = regionprops(mask,'Area','BoundingBox','PixelIdxList','Perimeter');
            [max_area,idx] = max([stats.Area]);
            img_num = img_num+1;
            if max_area > 30000
                mask(:) = 0;
                mask(stats(idx).PixelIdxList) = 1;
                stats_all(img_num).Area = max_area;
                stats_all(img_num).BoundingBox = stats(idx).BoundingBox;
                stats_all(img_num).PixelIdxList = stats(idx).PixelIdxList;
                stats_all(img_num).Perimeter = stats(idx).Perimeter;
            end
            mask = imdilate(mask, se);
            img = img.*uint8(mask);       
            file_name = sprintf('%d',img_num);
            imwrite(img, fullfile(output1path,[file_name,'.png']));
            % % 图片预处理
            % hsv = rgb2hsv(img);
            % V_eq = histeq(hsv(:,:,3));
            % hsv_eq = hsv;
            % img_eq = hsv2rgb(hsv_eq);
            % mask = (img_eq(:,:,1) > 0.7);
            % se = strel('disk', 5);
            % mask = imerode(mask, se);
            % mask = imdilate(mask, se);
            % %选取最大连通区域
            % mask = imerode(mask, se);
            % stats = regionprops(mask,'Area','BoundingBox','PixelIdxList');
            % [max_area,idx] = max([stats.Area]);
            % if max_area > 30000
            %     mask(:) = 0;
            %     mask(stats(idx).PixelIdxList) = 1;
            %      stats_all(i).Area = max_area;
            %      stats_all(i).BoundingBox = stats(idx).BoundingBox;
            %      stats_all(i).PixelIdxList = stats(idx).PixelIdxList;
            % end
            % mask = imdilate(mask, se);
            % img = img.*uint8(mask);
            % imshow(img)
        end
    end
end
area_table = table(vertcat(stats_all.Area), 'VariableNames', {'Area'});
area_table_num = area_table.Area;
area_table_num = (area_table_num - min(area_table_num))/(max(area_table_num) - min(area_table_num));
perim_table = table(vertcat(stats_all.Perimeter), 'VariableNames', {'Perimeter'});
perim_table_num = perim_table.Perimeter;
perim_table_num = (perim_table_num - min(perim_table_num))/(max(perim_table_num) - min(perim_table_num));

entire_data = [entire_data, table(area_table_num), table(perim_table_num)];
writetable(entire_data,'entire_data.xlsx');
%等面积裁切图片
imag_length = [stats_all.BoundingBox];
imag_length = (reshape(imag_length, 4,img_num))';
max_lenth = 1270;
max_higth = 1270;

imds = imageDatastore(output1path,"IncludeSubfolders",true,"FileExtensions",{'.bmp','.png','.jpg'});
filesort = [];
for i = 1:numel(imds.Files)
    [~,fname] = fileparts(imds.Files{i});
    fnum = str2double(regexp(fname,'\d+','match','once'));
    filesort = [filesort;fnum];
end
[~,sidx] = sort(filesort);
newFiles = imds.Files(sidx);

for i = 1:numel(newFiles)
    img = imread(newFiles{i});
    bbox = imag_length(i,:);
    box(1,1) = (2*bbox(1,1)+bbox(1,3))/2 - (max_lenth/2+5);
    box(1,2) = (2*bbox(1,2)+bbox(1,4))/2 - (max_higth/2+5);
    box(1,3) = max_lenth+9;
    box(1,4) = max_higth+9;
    img = imcrop(img,box);
    [~, filename] = fileparts(newFiles{i});
    imwrite(img, fullfile(outputpath, [filename '.png']));    
end