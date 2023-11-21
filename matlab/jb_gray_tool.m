%%
clear; close all;

%%
path = '/home/manu/nfs/data/out.txt';

w = 960;
h = 540;

I = load(path);
I = uint8(reshape(I, w, h))';

imshow(I);

%%
