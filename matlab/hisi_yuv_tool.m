%%
clear; close all;

%%
scale = 2;
w = 3840 / scale;
h = 2160 / scale;

s = w * h * 3 / 2;

fileID = fopen('/home/manu/nfs/tmp/0.yuv');
A = fread(fileID, s, 'uint8')';
fclose(fileID);

Y = uint8(reshape(A(1:w*h), w, h))';
U = uint8(reshape(A(w*h+1:2:end), w/2, h/2))';
V = uint8(reshape(A(w*h+2:2:end), w/2, h/2))';

U = imresize(U, 2);
V = imresize(V, 2);

R = 1.1644 * (Y - 16) + 1.7928 * (V - 128);
G = 1.1644 * (Y - 16) - 0.2133 * (U - 128) - 0.533 * (V - 128);
B = 1.1644 * (Y - 16) + 2.1124 * (U - 128);

RGB = cat(3, R, G, B);

imshow(RGB);

%%
