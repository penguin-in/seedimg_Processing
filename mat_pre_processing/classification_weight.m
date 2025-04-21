%此脚本用于绘制活力种子重量比例数据分布直方图
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
    alldata.(filedname){:,2}(alldata.(filedname){:,2}>1.8) = 1.7999;
    alldata.(filedname){:,2}(alldata.(filedname){:,2}<0.2) = 0.2001;
    alldata.(filedname){:,3} = ceil((alldata.(filedname){:,2}-0.2)/0.025);
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
m = zeros(2,64);
m(1,:) = (0.2+0.225)/2:0.025:(1.775+1.8)/2;
%重量增长率
h1 = histogram(p1,64,'FaceColor','b');
hold on; 
binEdges = h1.BinEdges; % 获取区间边界
binCounts = h1.Values;   % 获取每个柱子的频数
for i = 1:19
    m(2,i) = binCounts(i)-sum(pp3 == i);
end

for i = 20:64
    m(2,i) = sum(pp3 == i);
end
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2; % 计算柱体中心位置
h11 = bar(m(1,:), m(2,:), 'BarWidth', 1, 'FaceColor', 'r','FaceAlpha',0.5);
totalCount = sum(binCounts); % 计算总样本数
for i = 1:length(binCounts)
    percentage = ( m(2,i)/ binCounts(i)) * 100;
    text(binCenters(i), binCounts(i), sprintf('%.1f%%', percentage), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');
end
xline(0.675, ':k', 'LineWidth', 2.0);
text(0.3, 70, sprintf('低活力区:%.3f%',sum(m(2,1:19))/sum(binCounts(1:19))), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 16, 'FontWeight', 'bold', 'Color', 'b');
text(1.3, 70, sprintf('高活力区：%.3f%',sum(m(2,20:64))/sum(binCounts(20:64))), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 16, 'FontWeight', 'bold', 'Color', 'r');
text(0.675, 70, sprintf('分割点：0.675'), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 16, 'FontWeight', 'bold', 'Color', 'c');
legend('区间种子数量','活力种子数量')
xlabel('种子重量生长率');
ylabel('数量');
title('活力种子重量比例数据分布直方图');
grid on;