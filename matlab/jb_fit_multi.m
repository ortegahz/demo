%% 
close all;

%%
% 定义二维输入数据
X1 = [1, 2, 3, 4, 5];
X2 = [2, 4, 6, 8, 10];

% 定义二维输出数据
Y1 = [1.5, 3.5, 5.5, 7.5, 9.5];
Y2 = [2, 4, 6, 8, 10];

% 将输入数据组合成一个矩阵
X = [X1', X2'];

% 对第一个输出变量 Y1 进行拟合
model1 = fitlm(X, Y1);

% 对第二个输出变量 Y2 进行拟合
model2 = fitlm(X, Y2);

% 显示两个模型的参数
disp(model1.Coefficients);
disp(model2.Coefficients);

% 使用两个模型进行预测
YPred1 = predict(model1, X);
YPred2 = predict(model2, X);

% 绘制原始数据和拟合结果
figure;
subplot(1, 2, 1);
plot(Y1, 'ro-'); hold on;
plot(YPred1, 'b*-');
title('Y1');

subplot(1, 2, 2);
plot(Y2, 'ro-'); hold on;
plot(YPred2, 'b*-');
title('Y2');

legend('Actual', 'Predicted');

