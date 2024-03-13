%%
clear; close all;

%%
path_in = '/home/manu/nfs/rknn-toolkit2/examples/onnx/yolov5/bus.jpg';

path_out = strrep(path_in, '.jpg', '.bmp');

%%
I = imread(path_in);
imwrite(I, path_out);

imshow(I);

%%
