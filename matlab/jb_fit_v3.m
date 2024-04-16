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
hold on;
plot(X1, yfit1_calc_temp_s1, 'r*', 'DisplayName', sprintf('s1 t40 calc'));
hold off;

yfit_low = polyval(p2_low, X2);
yfit_high = polyval(p2_high, X2);
alpha = (p - l) / (h - l);
yfit1_calc_temp_s2 = yfit_low * (1 - alpha) + yfit_high * alpha;
hold on;
plot(X2, yfit1_calc_temp_s2, 'y*', 'DisplayName', sprintf('s2 t40 calc'));
hold off;

%%
% temp = 55; humid = 20;
% X1 = [744, 1575, 1867, 1989, 2123, 2104, 2142, 2208, 2311, 2355, 2474, 2548, 2651, 2752, 2913];
% X2 = [868, 1358, 1630, 1747, 1874, 1856, 1891, 1949, 2045, 2085, 2202, 2269, 2341, 2428, 2540];
% Y = [20, 72, 117, 150, 189, 198, 213, 235, 271, 293, 337, 380, 418, 459, 507];
% [p1_low, p2_low] = fit_sensor(X1, X2, Y, temp, humid);
% 
% temp = 55; humid = 90;
% X1 = [638, 1986, 2368, 2523, 2546, 2574, 2646, 2775, 2867, 2920, 2978, 3041, 3141, 3263];
% X2 = [584, 1784, 2132, 2275, 2208, 2302, 2369, 2495, 2581, 2635, 2678, 2728, 2805, 2895];
% Y = [0, 61, 104, 146, 163, 181, 207, 259, 315, 355, 392, 425, 461, 504];
% [p1_high, p2_high] = fit_sensor(X1, X2, Y, temp, humid);
% 
% temp = 55; humid = 60;
% X1 = [688, 1325, 1927, 2138, 2347, 2359, 2403, 2431, 2544, 2671, 2703, 2827, 2923, 3022, 3133];
% X2 = [573, 1253, 1637, 1869, 2062, 2067, 2114, 2148, 2253, 2378, 2419, 2526, 2623, 2703, 2779];
% Y = [4, 47, 82, 124, 173, 183, 199, 215, 248, 294, 323, 381, 426, 464, 503];
% hold on;
% plot(X1, Y, 'b+', X2, Y, 'g+');
% hold off;
% 
% l = 20; h = 90; p = 60;
% 
% yfit1_low = polyval(p1_low, X1);
% yfit1_high = polyval(p1_high, X1);
% alpha = (p - l) / (h - l);
% yfit1_calc_humid_s1 = yfit1_low * (1 - alpha) + yfit1_high * alpha;
% hold on;
% plot(X1, yfit1_calc_humid_s1, 'r+', 'DisplayName', sprintf('s1 t40 calc'));
% hold off;
% 
% yfit_low = polyval(p2_low, X2);
% yfit_high = polyval(p2_high, X2);
% alpha = (p - l) / (h - l);
% yfit1_calc_humid_s2 = yfit_low * (1 - alpha) + yfit_high * alpha;
% hold on;
% plot(X2, yfit1_calc_humid_s2, 'y+', 'DisplayName', sprintf('s2 t40 calc'));
% hold off;
% 
% % title(sprintf('h%d l%d p%d', h, l, p));
% % legend show;

%%
% alpha = 0.5;
% yfit1_calc_s2 = yfit1_calc_humid_s2 * (1 - alpha) + yfit1_calc_temp_s2 * alpha;
% hold on;
% plot(X2, yfit1_calc_s2, 'r*');
% hold off;

%%

function [p1, p2] = fit_sensor(X1, X2, Y, temp, humid)
    order = 2;
    p1 = polyfit(X1, Y, order);
    p2 = polyfit(X2, Y, order);
    yfit1 = polyval(p1, X1);
    yfit2 = polyval(p2, X2);
    hold on;
    plot(X1, Y, 'bo', 'DisplayName', sprintf('s1 gt pred t%d h%d', temp, humid));
    plot(X2, Y, 'go', 'DisplayName', sprintf('s2 gy pred t%d h%d', temp, humid));
    plot(X1, yfit1, 'b-', 'DisplayName', sprintf('s1 pred t%d h%d', temp, humid));
    plot(X2, yfit2, 'g-', 'DisplayName', sprintf('s2 pred t%d h%d', temp, humid));
    hold off;
    xlim([0, 3300]); ylim([0, 700]);
    xlabel('raw');
    ylabel('ppm');
end

%%
