%%
close all;

%%
temp = 20; humid = 20;
X1 = [792, 1095, 1386, 1540, 1639, 1709, 1906, 1921, 2031, 2111, 2189, 2254];
X2 = [716, 986, 1247, 1399, 1485, 1545, 1721, 1731, 1827, 1881, 1934, 1973];
Y = [35, 68, 120, 163, 194, 224, 297, 330, 392, 446, 489, 522];
[p1_low, p2_low] = fit_sensor(X1, X2, Y, temp, humid);

temp = 55; humid = 90;
X1 = [638, 1986, 2368, 2523, 2546, 2574, 2646, 2775, 2867, 2920, 2978, 3041, 3141, 3263];
X2 = [584, 1784, 2132, 2275, 2208, 2302, 2369, 2495, 2581, 2635, 2678, 2728, 2805, 2895];
Y = [0, 61, 104, 146, 163, 181, 207, 259, 315, 355, 392, 425, 461, 504];
[p1_high, p2_high] = fit_sensor(X1, X2, Y, temp, humid);

temp = 55; humid = 60;
X1 = [688, 1325, 1927, 2138, 2347, 2359, 2403, 2431, 2544, 2671, 2703, 2827, 2923, 3022, 3133];
X2 = [573, 1253, 1637, 1869, 2062, 2067, 2114, 2148, 2253, 2378, 2419, 2526, 2623, 2703, 2779];
Y = [4, 47, 82, 124, 173, 183, 199, 215, 248, 294, 323, 381, 426, 464, 503];
hold on;
plot(X1, Y, 'b+', X2, Y, 'g+');
hold off;

l = 20; h = 90; p = 60;

yfit1_low = polyval(p1_low, X1);
yfit1_high = polyval(p1_high, X1);
alpha = (p - l) / (h - l);
yfit1_calc = yfit1_low * (1 - alpha) + yfit1_high * alpha;
hold on;
plot(X1, yfit1_calc, 'r+', 'DisplayName', sprintf('s1 40 calc'));
hold off;

yfit_low = polyval(p2_low, X2);
yfit_high = polyval(p2_high, X2);
alpha = (p - l) / (h - l);
yfit_calc = yfit_low * (1 - alpha) + yfit_high * alpha;
hold on;
plot(X2, yfit_calc, 'y+', 'DisplayName', sprintf('s2 40 calc'));
hold off;

title(sprintf('h%d l%d p%d', h, l, p));
% legend show;

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
