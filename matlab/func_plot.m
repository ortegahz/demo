%%
clear; close all;

%
figure;
hold on;
fplot(@(u) 12.92 * u, [0 0.0031308]);
fplot(@(u) ((1.055 * power(u, 1.0 / 2.4)) - 0.055), [0.0031308 1]);
fplot(@(u) u, [0 1], 'g')
fplot(@(u) power(u, 1.5), [0 1], 'r')
fplot(@(u) power(u, 2), [0 1], 'y')
fplot(@(u) -u * log(u), [0 1], 'c')
fplot(@(x) 0.9999 * (1 - exp(-x / 2000)), [0 1000], 'c')
hold off;

r = 0;
g = 1;
b = 0;
cb = r * -0.100644 + g * -0.338572 + b * 0.439216 + 0.501961;
cr = r * 0.439216 + g * -0.398942 + b * -0.040274 + 0.501961;

%%
