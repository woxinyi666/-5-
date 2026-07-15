%% =========================================================================
%% 任意编号无人机协同去噪全连接神经网络 & 5次多项式时空平滑模型 (升级版)
%% 说明：运行后可对全编队50架无人机进行5次多项式拟合，并手动选择目标机进行多轨对比可视化
%% =========================================================================
clear; clc; close all;

%% 1. 手动选择无人机编号
fprintf('======================================================\n');
target_drone = input('请输入你想要可视化的无人机编号 (1-50): ');

% 验证输入合法性
if isempty(target_drone) || target_drone < 1 || target_drone > 50
    error('错误：输入的编号不合法！请输入 1 到 50 之间的整数。');
end
fprintf('已选择 %d 号无人机进行 3D 轨迹重构、5次多项式拟合与协同去噪对比...\n', target_drone);
fprintf('======================================================\n\n');

%% 2. 文件加载与环境配置
file_clean = 'wrj50pf.xlsx';
file_noise = 'wrj50pf_noise.xlsx';

fprintf('Step 1: 正在从Excel加载全编队运动数据...\n');
data_clean = readmatrix(file_clean);
data_noise = readmatrix(file_noise);

t = data_clean(:, 1);                % 提取时间轴
num_timesteps = length(t);          % 总时间步数
num_drones = 50;                    % 总无人机数量
window_size = 5;                    % 时间滑动窗口大小 (过去5帧 + 当前1帧)

%% 3. 构建全编队所有 50 架无人机的空间残差矩阵 (中心化)
fprintf('Step 2: 正在提取50架无人机的空间协同残差...\n');
all_drones_errors = zeros(num_timesteps, num_drones * 3); 

for d = 1:num_drones
    % 计算当前无人机在矩阵中对应的起始列索引
    col_idx = (d - 1) * 3 + 2; 
    
    % 提取第 d 架飞机的真实预设值与测量值
    r_t  = data_clean(:, col_idx);   el_t  = data_clean(:, col_idx+1);   az_t  = data_clean(:, col_idx+2);
    r_n  = data_noise(:, col_idx);   el_n  = data_noise(:, col_idx+1);   az_n  = data_noise(:, col_idx+2);
    
    % 计算并存入该机的残差 (测量值 - 预设真实值)
    all_drones_errors(:, (d-1)*3 + 1) = r_n - r_t;
    all_drones_errors(:, (d-1)*3 + 2) = el_n - el_t;
    all_drones_errors(:, (d-1)*3 + 3) = az_n - az_t;
end

%% 4. 构造空时融合数据集 (基于你选择的目标无人机标签)
fprintf('Step 3: 正在切片滑动窗口，构建空时联合输入特征...\n');
X_data = [];
Y_data = [];

% 【核心动态提取】：根据选择的 target_drone 提取其对应的残差作为网络训练的客观标签
target_col_start = (target_drone - 1) * 3 + 1;
raw_error_target = all_drones_errors(:, target_col_start : target_col_start+2); 

for i = (window_size + 1) : num_timesteps
    % 提取从 (i-window_size) 到 i 时刻全编队所有50架机的残差片段
    global_window_input = all_drones_errors(i - window_size : i, :); 
    
    % 将该时空片段拉平为一行一维向量: 6帧 * 150特征 = 900维
    X_data = [X_data; global_window_input(:)']; 
    
    % 输出标签保持为 目标无人机 当前时刻 i 的 3 维真实残差
    Y_data = [Y_data; raw_error_target(i, :)]; 
end

% 划分数据集: 前 80% 帧用于网络协同演练，后 20% 帧作为未知的外推测试集
split_idx = round(size(X_data, 1) * 0.8);
X_train = X_data(1:split_idx, :);
Y_train = Y_data(1:split_idx, :);

X_test = X_data(split_idx+1:end, :);
Y_test = Y_data(split_idx+1:end, :);

%% 5. 配置并精细训练全连接协同神经网络 (MLP)
fprintf('Step 4: 正在配置具有正则化约束的深度网络...\n');

% 对高维空时特征和标签进行双向 mapminmax 归一化限制
[X_train_norm, settings_in]   = mapminmax(X_train', -1, 1);
[Y_train_norm, settings_out]  = mapminmax(Y_train', -1, 1);

% 拓扑升级：两层隐藏层 [128, 64] 支撑 900 维的高维特征映射
hidden_layers = [128, 64];
net = feedforwardnet(hidden_layers, 'trainscg'); 

% 开启 L2 正则化，惩罚过大的权重，保持去噪平滑度
net.performParam.regularization = 0.15;  
net.trainParam.epochs = 300;             % 最大迭代次数
net.trainParam.max_fail = 8;             % 提前终止触发门槛
net.trainParam.showWindow = true;        % 开启MATLAB官方训练控制面板窗口

% 启动网络训练
fprintf('--> 神经网络正在进行群体动力学与噪声解耦学习，请稍候...\n');
net = train(net, X_train_norm, Y_train_norm);

%% 6. 协同滤波预测与残差修复
fprintf('Step 5: 正在将训练好的协同网络用于未知序列测试...\n');

% 归一化未知测试段的 900 维输入
X_test_norm = mapminmax('apply', X_test', settings_in);

% 网络自动结合全编队趋势，给出目标机当前剥离高频噪声后的最优系统误差估计
Y_pred_norm = net(X_test_norm);

% 反归一化，恢复物理单位 (米 / 弧度)
Y_pred = mapminmax('reverse', Y_pred_norm, settings_out)';

%% 7. 反中心化：将预测残差从目标机原始噪声测量中剔除
% 定位测试段对应的时间索引
test_time_indices = (split_idx + window_size + 1) : num_timesteps;
t_test = t(test_time_indices);

% 获取目标机对应的Excel数据原始列
target_raw_col = (target_drone - 1) * 3 + 2;

r_noise_target  = data_noise(test_time_indices, target_raw_col);
el_noise_target = data_noise(test_time_indices, target_raw_col+1);
az_noise_target = data_noise(test_time_indices, target_raw_col+2);

% 从原始带噪球坐标中减去神经网络预测的纯净系统偏置项
r_filtered  = r_noise_target  - Y_pred(:, 1);
el_filtered = el_noise_target - Y_pred(:, 2);
az_filtered = az_noise_target - Y_pred(:, 3);

%% 8. 全球坐标系转换 (球面坐标 -> 直角坐标)
fprintf('Step 6: 正在将球坐标系解算至空间直角坐标系...\n');

% 8.1 目标机预设真实 3D 航线
r_true_target  = data_clean(test_time_indices, target_raw_col);
el_true_target = data_clean(test_time_indices, target_raw_col+1);
az_true_target = data_clean(test_time_indices, target_raw_col+2);

x_true = r_true_target .* cos(el_true_target) .* cos(az_true_target);
y_true = r_true_target .* cos(el_true_target) .* sin(az_true_target);
z_true = r_true_target .* sin(el_true_target);

% 8.2 目标机原始带噪声 3D 测量轨迹
x_noise = r_noise_target .* cos(el_noise_target) .* cos(az_noise_target);
y_noise = r_noise_target .* cos(el_noise_target) .* sin(az_noise_target);
z_noise = r_noise_target .* sin(el_noise_target);

% 8.3 目标机经【全编队协同神经网络】滤波后的 3D 纯净轨迹
x_filtered = r_filtered .* cos(el_filtered) .* cos(az_filtered);
y_filtered = r_filtered .* cos(el_filtered) .* sin(az_filtered);
z_filtered = r_filtered .* sin(el_filtered);

%% 9. 核心新增：【5次多项式拟合】处理
%% 9. 核心新增：【5次多项式拟合】处理（已加入中心化与缩放 mu 以消除警告）
fprintf('Step 7: 正在对全部50架无人机的带噪轨迹进行空间5次多项式拟合...\n');

% 预先定义三维拟合轨迹存储矩阵（测试集区间）
all_x_poly_noise = zeros(length(test_time_indices), num_drones);
all_y_poly_noise = zeros(length(test_time_indices), num_drones);
all_z_poly_noise = zeros(length(test_time_indices), num_drones);

for d = 1:num_drones
    col_d = (d - 1) * 3 + 2;
    % 获取第 d 架机在测试区间的带噪原始球坐标
    r_n_d  = data_noise(test_time_indices, col_d);
    el_n_d = data_noise(test_time_indices, col_d+1);
    az_n_d = data_noise(test_time_indices, col_d+2);
    
    % 解算为直角坐标
    x_n_d = r_n_d .* cos(el_n_d) .* cos(az_n_d);
    y_n_d = r_n_d .* cos(el_n_d) .* sin(az_n_d);
    z_n_d = r_n_d .* sin(el_n_d);
    
    % 分别对 X, Y, Z 三轴进行关于时间轴 t 的5次多项式拟合（使用 3 个输出参数进行中心化缩放）
    [px_noise, ~, mu_x] = polyfit(t_test, x_n_d, 5);
    [py_noise, ~, mu_y] = polyfit(t_test, y_n_d, 5);
    [pz_noise, ~, mu_z] = polyfit(t_test, z_n_d, 5);
    
    % 存储拟合后的平滑轨迹
    all_x_poly_noise(:, d) = polyval(px_noise, t_test, [], mu_x);
    all_y_poly_noise(:, d) = polyval(py_noise, t_test, [], mu_y);
    all_z_poly_noise(:, d) = polyval(pz_noise, t_test, [], mu_z);
end

% 提取当前选择的目标无人机的直接多项式拟合轨迹
x_noise_poly = all_x_poly_noise(:, target_drone);
y_noise_poly = all_y_poly_noise(:, target_drone);
z_noise_poly = all_z_poly_noise(:, target_drone);

% 8.4 对【神经网络去噪后】的数据再次进行5次多项式平滑处理（同样使用中心化缩放）
fprintf('Step 8: 正在对神经网络去噪后的轨迹进行5次多项式平滑二次过滤...\n');
[px_filt, ~, mu_fx] = polyfit(t_test, x_filtered, 5);
[py_filt, ~, mu_fy] = polyfit(t_test, y_filtered, 5);
[pz_filt, ~, mu_fz] = polyfit(t_test, z_filtered, 5);

x_filtered_smoothed = polyval(px_filt, t_test, [], mu_fx);
y_filtered_smoothed = polyval(py_filt, t_test, [], mu_fy);
z_filtered_smoothed = polyval(pz_filt, t_test, [], mu_fz);
%% 10. 五轨联合 3D 高级可视化与定量评估
figure('Color', [1 1 1], 'Position', [150 100 1150 780]);

% 1. 原始带噪离散点云
scatter3(x_noise, y_noise, z_noise, 8, [0.9 0.4 0.4], 'filled', 'MarkerFaceAlpha', 0.25);
hold on;

% 2. 原始带噪轨迹直接进行5次多项式拟合后的轨迹 (橙色点划线)
plot3(x_noise_poly, y_noise_poly, z_noise_poly, 'Color', [0.9 0.5 0], 'LineStyle', '-.', 'LineWidth', 2);

% 3. 神经网络去噪后的纯净轨迹 (蓝色虚线)
plot3(x_filtered, y_filtered, z_filtered, 'b--', 'LineWidth', 1.8);

% 4. 神经网络去噪 + 5次多项式平滑融合轨迹 (绿色极其显眼实线)
plot3(x_filtered_smoothed, y_filtered_smoothed, z_filtered_smoothed, 'g-', 'LineWidth', 3);

% 5. 程序预设的标准真实轨迹基准 (黑色实线)
plot3(x_true, y_true, z_true, 'k-', 'LineWidth', 2);

% 图表深度美化
grid on; box on;
ax = gca;
ax.GridLineStyle = ':';
ax.LineWidth = 1.1;
view(42, 24); % 调整至最符合空气动力学轨迹观察的三维空间立体透视视角
rotate3d on;  % 默认开启 3D 旋转交互，用鼠标即可任意拖动

xlabel('空间 X 轴位置 (米)', 'FontWeight', 'bold');
ylabel('空间 Y 轴位置 (米)', 'FontWeight', 'bold');
zlabel('空间 Z 轴位置 (米)', 'FontWeight', 'bold');
title(sprintf('%d号无人机 3D航线对比 (基于50机协同去噪 & 5次多项式时空平滑)', target_drone), 'FontSize', 13, 'FontWeight', 'bold');

legend(sprintf('%d号机 原始带高频噪声轨迹 ', target_drone), ...
       sprintf('%d号机 噪声轨迹直接5次多项式拟合', target_drone), ...
       '全编队协同神经网络滤波轨迹 ', ...
       '神经网络协同去噪 + 5次多项式平滑滤波', ...
       sprintf('%d号机 程序预设真实航线 ', target_drone), ...
       'Location', 'best', 'FontSize', 10.5);

% 误差定量评估与精密计算
rmse_noise       = sqrt(mean((x_noise - x_true).^2 + (y_noise - y_true).^2 + (z_noise - z_true).^2));
rmse_noise_poly  = sqrt(mean((x_noise_poly - x_true).^2 + (y_noise_poly - y_true).^2 + (z_noise_poly - z_true).^2));
rmse_nn_only     = sqrt(mean((x_filtered - x_true).^2 + (y_filtered - y_true).^2 + (z_filtered - z_true).^2));
rmse_nn_smoothed = sqrt(mean((x_filtered_smoothed - x_true).^2 + (y_filtered_smoothed - y_true).^2 + (z_filtered_smoothed - z_true).^2));

fprintf('\n======================= %d号机协同滤波综合成效报告 =======================\n', target_drone);
fprintf('1.【原始传感器】 引入高频随机白噪声空间均方根误差 (RMSE):   %.4f 米\n', rmse_noise);
fprintf('2.【多项式拟合】 带噪轨迹直接进行5次多项式拟合后的误差 (RMSE): %.4f 米\n', rmse_noise_poly);
fprintf('3.【协同神经网络】仅通过群动力学神经网络滤波后的误差 (RMSE):   %.4f 米\n', rmse_nn_only);
fprintf('4.【NN + 多项式】 协同网络去噪再经5次多项式平滑后的误差 (RMSE):  %.4f 米\n', rmse_nn_smoothed);
fprintf('-------------------------------------------------------------------------\n');
fprintf('👉 相比原始带噪轨迹，单用【直接多项式拟合】的误差改善率:       %.2f%%\n', (rmse_noise - rmse_noise_poly)/rmse_noise * 100);
fprintf('👉 相比原始带噪轨迹，级联【NN + 5次多项式平滑】的综合去噪改善率: %.2f%%\n', (rmse_noise - rmse_nn_smoothed)/rmse_noise * 100);
fprintf('=========================================================================\n');