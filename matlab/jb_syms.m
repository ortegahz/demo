%% 
close all;

%%
syms x

f1 = 3.5 * x ^ 2 + 1.5 * x ^ 1 + 1;
f2 = (3.5 * x ^ 2 + 1.5 * x ^ 1 + 1) * 0.5;

hold on;
fplot(f1, [0, 10], 'b')
fplot(f2, [0, 10], 'r')
hold off;

xlabel('x')
ylabel('y')
