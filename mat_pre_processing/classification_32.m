%此脚本用于32分类活力种子比例数据分布直方图
clear;clc;

filename = 'D:\论文\种子形状识别\数据\datasheet.xlsx';
[~, sheets] = xlsfinfo(filename);
p1 = [];
pp1 = [];
pp2 = [];
% p2 = [];
% p3 = [];
% p4 = [];
% p5 = [];
alldata = struct();
for i = 2:11
    sheetname = sheets{i};
    filedname = matlab.lang.makeValidName(sheetname);
    alldata.(filedname) = readtable(filename,'Sheet',sheetname);
    % [~,colomn] = size(alldata.(filedname));
    % initialweigh = alldata.(filedname)(:,4);
    % seedingweigh = alldata.(filedname)(:,7);
    alldata.(filedname){:,2} = (alldata.(filedname){:,7} - alldata.(filedname){:,4})./alldata.(filedname){:,4};
    alldata.(filedname){:,2}(alldata.(filedname){:,2}>1.8) = 1.79;
    alldata.(filedname){:,2}(alldata.(filedname){:,2}<0.2) = 0.21;
    alldata.(filedname){:,3} = ceil((alldata.(filedname){:,2} - 0.2)/0.05);
    p1 = [p1;alldata.(filedname){:,2}];
    pp1 = [pp1;alldata.(filedname){:,3}];
    pp2 = [pp2;alldata.(filedname){:,12}];
    % p2 = [p2;alldata.(filedname){:,4}];
    % p3 = [p3;alldata.(filedname){:,5}];
    % p4 = [p4;alldata.(filedname){:,6}];
    % p5 = [p5;alldata.(filedname){:,7}];
    % writetable(alldata.(filedname), 'D:\论文\种子形状识别\数据\datasheet.xlsx', 'Sheet',sheetname);%将数据写入excel表格中
end
pp3 = pp2.*pp1;
m = zeros(2,32);
m(1,:) = 0.225:(1.8-0.2)/32:1.775;
%重量增长率
h1 = histogram(p1,32,'FaceColor','b');
hold on; 
binEdges = h1.BinEdges; % 获取区间边界
binCounts = h1.Values;   % 获取每个柱子的频数
for i = 1:10
    m(2,i) = binCounts(i)-sum(pp3 == i);
end

for i = 11:32
    m(2,i) = sum(pp3 == i);
end
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2; % 计算柱体中心位置
bar(m(1,:), m(2,:), 'BarWidth', 1, 'FaceColor', 'r','FaceAlpha',0.5);
% bar(m(1,9:10), m(2,9:10), 'BarWidth', 1, 'FaceColor', 'c','FaceAlpha',0.5);
% bar(m(1,11:32), m(2,11:32), 'BarWidth', 1, 'FaceColor', 'g','FaceAlpha',0.5);
totalCount = sum(binCounts); % 计算总样本数
for i = 1:length(binCounts)
    percentage = ( m(2,i)/ binCounts(i)) * 100;
    text(binCenters(i), binCounts(i), sprintf('%.1f%%', percentage), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');
end
xline(0.7, ':k', 'LineWidth', 2.0);
xline(0.6, ':k', 'LineWidth', 2.0);
text(0.65, 130, sprintf('模糊区'), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 16, 'FontWeight', 'bold', 'Color', 'c');
text(0.3, 130, sprintf('低活力区'), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 16, 'FontWeight', 'bold', 'Color', 'b');
text(1.2, 130, sprintf('高活力区'), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 16, 'FontWeight', 'bold', 'Color', 'r');
legend('区间种子数量','活力种子数量')
xlabel('种子重量生长率');
ylabel('数量');
title('32分类活力种子比例数据分布直方图');
grid on;
% 
% %初始重量分布
% figure;
% h2 = histogram(p2,32);
% hold on; 
% h2.FaceColor = 'y';
% binEdges = h2.BinEdges; % 获取区间边界
% binCounts = h2.Values;   % 获取每个柱子的频数
% binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2; % 计算柱体中心位置
% 
% totalCount = sum(binCounts); % 计算总样本数
% for i = 1:length(binCounts)
%     percentage = (binCounts(i) / totalCount) * 100;
%     text(binCenters(i), binCounts(i), sprintf('%.1f%%', percentage), ...
%          'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
%          'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');
% end
% xlabel('种子初始重量');
% ylabel('数量');
% title('初始重量数据分布直方图');
% grid on;
% 
% %第一天重量分布
% figure;
% h3 = histogram(p3,32);
% hold on; 
% h3.FaceColor = 'g';
% binEdges = h3.BinEdges; % 获取区间边界
% binCounts = h3.Values;   % 获取每个柱子的频数
% binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2; % 计算柱体中心位置
% 
% totalCount = sum(binCounts); % 计算总样本数
% for i = 1:length(binCounts)
%     percentage = (binCounts(i) / totalCount) * 100;
%     text(binCenters(i), binCounts(i), sprintf('%.1f%%', percentage), ...
%          'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
%          'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');
% end
% xlabel('种子初始重量');
% ylabel('数量');
% title('第一天重量数据分布直方图');
% grid on;
% 
% %第二天重量分布
% figure;
% h4 = histogram(p4,32);
% hold on; 
% h4.FaceColor = 'm';
% binEdges = h4.BinEdges; % 获取区间边界
% binCounts = h4.Values;   % 获取每个柱子的频数
% binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2; % 计算柱体中心位置
% 
% totalCount = sum(binCounts); % 计算总样本数
% for i = 1:length(binCounts)
%     percentage = (binCounts(i) / totalCount) * 100;
%     text(binCenters(i), binCounts(i), sprintf('%.1f%%', percentage), ...
%          'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
%          'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');
% end
% xlabel('种子初始重量');
% ylabel('数量');
% title('第二天重量数据分布直方图');
% grid on;
% 
% %第三天重量分布
% figure;
% h5 = histogram(p5,32);
% hold on; 
% h5.FaceColor = 'c';
% binEdges = h5.BinEdges; % 获取区间边界
% binCounts = h5.Values;   % 获取每个柱子的频数
% binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2; % 计算柱体中心位置
% 
% totalCount = sum(binCounts); % 计算总样本数
% for i = 1:length(binCounts)
%     percentage = (binCounts(i) / totalCount) * 100;
%     text(binCenters(i), binCounts(i), sprintf('%.1f%%', percentage), ...
%          'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
%          'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');
% end
% xlabel('种子初始重量');
% ylabel('数量');
% title('第三天重量数据分布直方图');
% grid on;


