%%
clear; close all;

path_in = '/home/manu/nfs/data/gray.bmp';

%
img = imread(path_in);
% imwrite(img, strrep(path_in, '.png', '.bmp'));
imshow(img);

%%
