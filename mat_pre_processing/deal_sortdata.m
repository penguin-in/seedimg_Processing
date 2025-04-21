% 用于处理乱序excel表格数据，按照A-B顺序排布
inputFile = 'D:\论文\种子形状识别\数据\datasheet.xlsx';   
outputFile = 'sorted_output.xlsx';

[~, sheetNames] = xlsfinfo(inputFile);


ab_col_index = 1; 


for s = 1:length(sheetNames)
    sheet = sheetNames{s};
    T = readtable(inputFile, 'Sheet', sheet, 'ReadVariableNames', true);
    ab_col = T{:, ab_col_index};
    if iscell(ab_col)
        ab_col = string(ab_col);
    end
    n = length(ab_col);
    pairs = zeros(n, 2);
    for i = 1:n
        parts = split(ab_col(i), '-');
        pairs(i, 1) = str2double(parts{1});
        pairs(i, 2) = str2double(parts{2});
    end

    [~, sortIdx] = sortrows(pairs, [1 2]);

    T_sorted = T(sortIdx, :);

    writetable(T_sorted, outputFile, 'Sheet', sheet);
end

fprintf('finisied');