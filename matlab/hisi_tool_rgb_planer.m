%%
clear; close all;

%%
w = 640;
h = 640;

s = w * h * 3;

fileID = fopen('/home/manu/tmp/acfree_640_640_rgb_planner');
A = fread(fileID, s, 'uint8')';
fclose(fileID);

R = uint8(reshape(A(1:w*h), w, h))';
G = uint8(reshape(A(w*h+1:2*w*h), w, h))';
B = uint8(reshape(A(2*w*h+1:3*w*h), w, h))';

RGB = cat(3, R, G, B);

imshow(RGB);
% imwrite(RGB,'/home/manu/tmp/horse_dog_car_person_224x224.jpg')

%%
