%%
clear; close all;

%%
w = 1280;
h = 1280;

path_in = '/media/manu/samsung/pics/4.jpg';

path_out = strrep(path_in, '.jpg', '_rs.bmp');

%%
I = imread(path_in);
I = imresize(I, [h, w]);
imwrite(I, path_out);

imshow(I);

%%
