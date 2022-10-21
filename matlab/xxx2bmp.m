%%
clear; close all;

path_in = '/home/manu/nfs/glmark2/data/textures/effect-2d.png';

%
img = imread(path_in);
imwrite(img, strrep(path_in, '.png', '.bmp'));

%%
