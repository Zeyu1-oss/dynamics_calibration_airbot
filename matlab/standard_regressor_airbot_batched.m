function Y_std_total = standard_regressor_airbot_batched(q_matrix, qd_matrix, qdd_matrix)
% STANDARD_REGRESSOR_AIRBOT_BATCHED - 批量计算标准回归矩阵
%
% 输入:
%   q_matrix   - 关节位置矩阵 (N x 6)
%   qd_matrix  - 关节速度矩阵 (N x 6)
%   qdd_matrix - 关节加速度矩阵 (N x 6)
%
% 输出:
%   Y_std_total - 堆叠的回归矩阵 (6N x 60)
%
% 这是一个加速版本，批量调用单个样本的回归函数

    N = size(q_matrix, 1);  % 样本数量
    
    % 预分配输出矩阵
    Y_std_total = zeros(6*N, 60);
    
    % 逐个样本计算
    for i = 1:N
        % 提取第i个样本
        q_i = q_matrix(i, :)';     % 列向量 (6 x 1)
        qd_i = qd_matrix(i, :)';
        qdd_i = qdd_matrix(i, :)';
        
        % 调用单样本回归函数
        Y_i = standard_regressor_airbot(q_i, qd_i, qdd_i);  % (6 x 60)
        
        % 存储到总矩阵
        Y_std_total((i-1)*6+1:i*6, :) = Y_i;
    end
end