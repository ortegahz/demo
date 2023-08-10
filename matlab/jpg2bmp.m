%%
clear; close all;

%%
path_in = '/home/manu/图片/smplayer_screenshots/cap_20230605-10.20.164.67_00:10:33_01.jpg';

path_out = strrep(path_in, '.jpg', '.bmp');

%%
I = imread(path_in);
imwrite(I, path_out);

imshow(I);

%%
