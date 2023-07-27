%%
clear; close all;

%%
w = 640;
h = w;

s = w * h * 3;

fileID = fopen('/home/manu/nfs/mpp/sample/svp/nnie/data/nnie_image/rgb_planar/students_lt_640x640.bgr');
A = fread(fileID, s, 'uint8')';
fclose(fileID);

B = uint8(reshape(A(1:w*h), w, h))';
G = uint8(reshape(A(w*h+1:2*w*h), w, h))';
R = uint8(reshape(A(2*w*h+1:3*w*h), w, h))';

RGB = cat(3, R, G, B);

imshow(RGB);
% imwrite(RGB,'/home/manu/nfs/mpp/sample/svp/nnie/data/nnie_image/rgb_planar/horse_dog_car_person_224x224.jpg')

%%
