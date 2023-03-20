%%
clear; close all;

%%
path = '/media/manu/kingstop/itx-3588j/Linux_SDK/rk3588/external/linux-rga/samples/sample_file/in0w1280-h720-rgba8888.bin';

w = 1280;
h = 720;

c = 4;
s = w * h * c;

fileID = fopen(path);
A = fread(fileID, s, 'uint8')';
fclose(fileID);

R = uint8(reshape(A(1:c:end), w, h))';
G = uint8(reshape(A(2:c:end), w, h))';
B = uint8(reshape(A(3:c:end), w, h))';
A = uint8(reshape(A(4:c:end), w, h))';

RGB = cat(3, R, G, B);

imshow(R);

%%
