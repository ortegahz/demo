%%
clear; close all;

%%
path_in = '/media/manu/data/data/2016-ITS-BrnoCompSpeed/dataset/session6_center/video_mask.png';

path_out = strrep(path_in, '.png', '.bmp');

%%
I = imread(path_in);
imwrite(I, path_out);

imshow(I);

%%
