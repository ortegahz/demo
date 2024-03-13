%% 
close all;

%%
% 假设我们有一组数据点（x和y）
x = CO;
y = CO1;

% 使用polyfit函数进行多项式拟合，这里我们选择一个一阶多项式（线性拟合）
p = polyfit(x, y, 3);

% 使用polyval函数计算拟合的多项式在x点上的值
y_fit = polyval(p, x);

% 绘制原始数据点和拟合曲线
plot(x, y, 'o', x, y_fit, '-');
xlabel('x');
ylabel('y');
title('Curve Fitting Example');
legend('Data Points', 'Fitted Curve');
