%%
clear; close all;

%%
w = 960;
h = 540;

path_in = '/home/manu/nfs/tmp/mediapipe/vlcsnap-2023-02-24-11h30m17s065.png';

path_out = strrep(path_in, '.png', '_rs.bmp');

%%
I = imread(path_in);
I = imresize(I, [h, w]);
imwrite(I, path_out);

imshow(I);

%%
