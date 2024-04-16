%% same calc results between params weighting and value weighting

%%
close all;

%%
temp = 20; humid = 20;
X1 = [792, 1095, 1386, 1540, 1639, 1709, 1906, 1921, 2031, 2111, 2189, 2254];
X2 = [716, 986, 1247, 1399, 1485, 1545, 1721, 1731, 1827, 1881, 1934, 1973];
Y = [35, 68, 120, 163, 194, 224, 297, 330, 392, 446, 489, 522];
[p1_low, p2_low] = fit_sensor(X1, X2, Y, temp, humid);

temp = 50; humid = 20;
X1 = [983, 1470, 1860, 1920, 2083, 2113, 2163, 2294, 2393, 2294, 2544, 2631, 2796];
X2 = [867, 1321, 1644, 1693, 1796, 1857, 1902, 2029, 2124, 2173, 2257, 2314, 2417];
Y = [33, 65, 148, 161, 185, 208, 223, 269, 318, 359, 404, 448, 500];
[p1_high, p2_high] = fit_sensor(X1, X2, Y, temp, humid);

temp = 40; humid = 20;
X1 = [1083, 1445, 1653, 1721, 1917, 1983, 2071, 2182, 2249, 2318, 2413, 2527, 2981];
X2 = [924, 1293, 1473, 1592, 1709, 1746, 1821, 1925, 1980, 2040, 2120, 2205, 2590];
Y = [45, 82, 115, 150, 195, 224, 254, 302, 334, 377, 419, 467, 632];
hold on;
plot(X1, Y, 'b+', X2, Y, 'g+');
hold off;

l = 20; h = 50; p = 40;

yfit1_low = polyval(p1_low, X1);
yfit1_high = polyval(p1_high, X1);
alpha = (p - l) / (h - l);
yfit1_calc_temp_s1 = yfit1_low * (1 - alpha) + yfit1_high * alpha;
p1 = p1_low * (1 - alpha) + p1_high * alpha;
hold on;
fplot(@(u) polyval(p1, u), [0, 3300], 'r');
plot(X1, yfit1_calc_temp_s1, 'r*', 'DisplayName', sprintf('s1 t40 calc'));
hold off;

yfit_low = polyval(p2_low, X2);
yfit_high = polyval(p2_high, X2);
alpha = (p - l) / (h - l);
yfit1_calc_temp_s2 = yfit_low * (1 - alpha) + yfit_high * alpha;
p2 = p2_low * (1 - alpha) + p2_high * alpha;
hold on;
fplot(@(u) polyval(p2, u), [0, 3300], 'y');
plot(X2, yfit1_calc_temp_s2, 'y*', 'DisplayName', sprintf('s2 t40 calc'));
hold off;

ylim([0, 700]);
xlabel('raw');
ylabel('ppm');

%%
function [p1, p2] = fit_sensor(X1, X2, Y, temp, humid)
    order = 2;
    p1 = polyfit(X1, Y, order);
    p2 = polyfit(X2, Y, order);
    hold on;
    fplot(@(u) polyval(p1, u), [0, 3300], 'b');
    fplot(@(u) polyval(p2, u), [0, 3300], 'g');
    plot(X1, Y, 'bo', 'DisplayName', sprintf('s1 gt pred t%d h%d', temp, humid));
    plot(X2, Y, 'go', 'DisplayName', sprintf('s2 gy pred t%d h%d', temp, humid));
    hold off;
end

%%
