clear; close all;
h = 18; w = 34;

% out_ref = load('/home/manu/nfs/data/python_img_.txt');
% out_ref = load('/home/manu/nfs/data/python_img_gray.txt');
% out_ref = load('/home/manu/nfs/data/python_img_now.txt');
% out_ref = load('/home/manu/nfs/data/python_labels.txt');
% out_ref = load('/home/manu/nfs/data/python_stats.txt');
% out_ref = load('/home/manu/nfs/data/python_centroids.txt');
% out_ref = load('/home/manu/nfs/data/python_mat200.txt');
% out_ref = load('/home/manu/nfs/data/python_gaussian_one.txt');
% out_ref = load('/home/manu/nfs/data/python_img_crop_fire.txt');
% out_ref = load('/home/manu/nfs/data/python_x.txt');
out_ref = load('/home/manu/nfs/data/python_y.txt');

% figure;
% imshow(reshape(out_ref, w, h)');

% out_q = load('/home/manu/nfs/data/cpp_srcNormDebug.txt');
% out_q = load('/home/manu/nfs/data/cpp_srcGray.txt');
% out_q = load('/home/manu/nfs/data/cpp_bin.txt');
% out_q = load('/home/manu/nfs/data/cpp_labels.txt');
% out_q = load('/home/manu/nfs/data/cpp_stats.txt');
% out_q = load('/home/manu/nfs/data/cpp_centroids.txt');
% out_q = load('/home/manu/nfs/data/cpp_binSecond.txt');
% out_q = load('/home/manu/nfs/data/cpp_gaussianOne.txt');
% out_q = load('/home/manu/nfs/data/cpp_srcGray(expandRectOne).txt');
% out_q = load('/home/manu/nfs/data/cpp_SrcSobelRoi.txt');
out_q = load('/home/manu/nfs/data/cpp_SrcSobelRoiMixed.txt');

% figure;
% imshow(reshape(out_q, w, h)');

% error = sum((out_ref - out_q) .^ 2) / length(out_ref);
sim_cos = dot(out_ref, out_q) / (norm(out_ref) * norm(out_q));  

figure;
subplot(3,1,1);
plot(1:length(out_ref), out_ref, 'g', 1:length(out_q), out_q, 'y'); title(sprintf('cos sim %f', sim_cos));
subplot(3,1,2); plot(1:length(out_ref), out_ref, 'g'); title('ref');
subplot(3,1,3); plot(1:length(out_q), out_q, 'y'); title('q');

