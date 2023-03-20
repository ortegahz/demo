%%
clear; close all;

%%
path_in = '/home/manu/tmp/summer_fun-wallpaper-1920x1080.jpg';

path_out = strrep(path_in, '.jpg', '.bmp');

%%
I = imread(path_in);
imwrite(I, path_out);

imshow(I);

%%
