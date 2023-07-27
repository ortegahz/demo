%%
clear; close all;

%%
fp = '/home/manu/nfs/mpp/sample/svp/nnie/data/nnie_image/yuv/out360.nv12';
wi = 640; hi = 360;

fid = fopen(fp);
A = fread(fid, wi*hi*3/2, 'uint8')';
fclose(fid);

Y = reshape(A(1:wi*hi), wi, hi)';
UV = A(wi*hi+1:wi*hi*3/2);
U = UV(1:2:end); V = UV(2:2:end);
U = reshape(U, wi/2, hi/2)';
V = reshape(V, wi/2, hi/2)';
UR = imresize(U, 2); VR = imresize(V, 2);
R = Y + 1.4075 * (VR - 128);
G = Y - 0.3455 * (UR - 128) - 0.7169 * (VR - 128);
B = Y + 1.779 * (UR - 128);
RGB = cat(3, R, G, B);
imshow(uint8(RGB));

%%
