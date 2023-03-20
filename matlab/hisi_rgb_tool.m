%%
clear; close all;

%%
w = 3840;
h = 2160;

s = w * h * 3;

fileID = fopen('/home/manu/nfs/data/Q4Terminal/3840x2160_spliter_3.rgb');
A = fread(fileID, s, 'uint8')';
fclose(fileID);

B = uint8(reshape(A(1:3:end), w, h))';
G = uint8(reshape(A(2:3:end), w, h))';
R = uint8(reshape(A(3:3:end), w, h))';

RGB = cat(3, R, G, B);

imshow(RGB);

%%
