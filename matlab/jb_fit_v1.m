%% 
close all;

%%
save_root = '/home/manu/tmp/matlab_plot';
rmdir(save_root, 's');
mkdir(save_root);

%%
temp = 20; humid = 20;
X1 = [792, 1095, 1386, 1540, 1639, 1709, 1906, 1921, 2031, 2111, 2189, 2254];
X2 = [716, 986, 1247, 1399, 1485, 1545, 1721, 1731, 1827, 1881, 1934, 1973];
Y = [35, 68, 120, 163, 194, 224, 297, 330, 392, 446, 489, 522];
fit_sensor(X1, X2, Y, temp, humid);

temp = 30; humid = 20;
X1 = [661, 1456, 1663, 1782, 1843, 1957, 2084, 2146, 2209, 2257, 2301];
X2 = [739, 1306, 1489, 1595, 1641, 1743, 1854, 1879, 1949, 1978, 2010];
Y = [45, 131, 180, 225, 256, 317, 394, 433, 471, 498, 521];
fit_sensor(X1, X2, Y, temp, humid);

temp = 40; humid = 20;
X1 = [1083, 1445, 1653, 1721, 1917, 1983, 2071, 2182, 2249, 2318, 2413, 2527, 2981];
X2 = [924, 1293, 1473, 1592, 1709, 1746, 1821, 1925, 1980, 2040, 2120, 2205, 2590];
Y = [45, 82, 115, 150, 195, 224, 254, 302, 334, 377, 419, 467, 632];
fit_sensor(X1, X2, Y, temp, humid);

temp = 50; humid = 20;
X1 = [983, 1470, 1860, 1920, 2083, 2113, 2163, 2294, 2393, 2294, 2544, 2631, 2796];
X2 = [867, 1321, 1644, 1693, 1796, 1857, 1902, 2029, 2124, 2173, 2257, 2314, 2417];
Y = [33, 65, 148, 161, 185, 208, 223, 269, 318, 359, 404, 448, 500];
fit_sensor(X1, X2, Y, temp, humid);

temp = 55; humid = 20;
X1 = [744, 1575, 1867, 1989, 2123, 2104, 2142, 2208, 2311, 2355, 2474, 2548, 2651, 2752, 2913];
X2 = [868, 1358, 1630, 1747, 1874, 1856, 1891, 1949, 2045, 2085, 2202, 2269, 2341, 2428, 2540];
Y = [20, 72, 117, 150, 189, 198, 213, 235, 271, 293, 337, 380, 418, 459, 507];
fit_sensor(X1, X2, Y, temp, humid);

temp = 55; humid = 60;
X1 = [688, 1325, 1927, 2138, 2347, 2359, 2403, 2431, 2544, 2671, 2703, 2827, 2923, 3022, 3133];
X2 = [573, 1253, 1637, 1869, 2062, 2067, 2114, 2148, 2253, 2378, 2419, 2526, 2623, 2703, 2779];
Y = [4, 47, 82, 124, 173, 183, 199, 215, 248, 294, 323, 381, 426, 464, 503];
fit_sensor(X1, X2, Y, temp, humid);

temp = 55; humid = 90;
X1 = [638, 1986, 2368, 2523, 2546, 2574, 2646, 2775, 2867, 2920, 2978, 3041, 3141, 3263];
X2 = [584, 1784, 2132, 2275, 2208, 2302, 2369, 2495, 2581, 2635, 2678, 2728, 2805, 2895];
Y = [0, 61, 104, 146, 163, 181, 207, 259, 315, 355, 392, 425, 461, 504];
fit_sensor(X1, X2, Y, temp, humid);


%%
% function fit_sensor(X1, X2, Y, temp, humid)
%     X = [X1, X2]; Y = [Y, Y];
%     p = polyfit(X, Y, 1);
%     Y_fit = polyval(p, X);
%     plot(X, Y, 'ro', X, Y_fit, 'g-');
%     fprintf('%.4f %.4f --> %.4f, %.4f\n', temp, humid, p(1), p(2));
%     figureHandle = gcf;
%     save_name = sprintf('/home/manu/tmp/matlab_plot/%f-%f-%f-%f.png', temp, humid, p(1), p(2));
%     title(save_name);
%     saveas(figureHandle, save_name);
% end

function fit_sensor(X1, X2, Y, temp, humid)
    order = 2;
    p1 = polyfit(X1, Y, order);
    p2 = polyfit(X2, Y, order);
    yfit1 = polyval(p1, X1);
    yfit2 = polyval(p2, X2);
    hold on;
    plot(X1, Y, 'bo', X1, yfit1, 'b-');
    plot(X2, Y, 'go', X2, yfit2, 'g-');
    hold off;
    xlim([0, 3300]); ylim([0, 700]);
    fprintf('p1 --> %.32f\n', p1);
%     fprintf('p2 --> %.32f\n', p2);
%     disp(p1); disp(p2);
    figureHandle = gcf;
    save_name = sprintf('/home/manu/tmp/matlab_plot/%f-%f.png', temp, humid);
    title(save_name);
    saveas(figureHandle, save_name);
end

%%