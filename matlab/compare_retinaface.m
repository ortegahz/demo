clear all; close all;

th = 0.8;

stride = 8;
w = 1920 / stride;
h = 1088 / stride;

img = imread('/home/manu/tmp/9.bmp');
figure; imshow(img);

out_mxnet = load('mxnet.txt');
out_nnie = load('output6-240x136x4.txt');

% reverse order of python numpy
feature_mxnet = reshape(out_mxnet, w, h, 4, 1);
feature_show_mxnet = squeeze(feature_mxnet(:, :, 3));
feature_show_mxnet = feature_show_mxnet';
[r, c, ~] = find(feature_show_mxnet > th);
hold on; plot(c*stride, r*stride, '.g', 'MarkerSize', 10); hold off;

% reverse order of python numpy
feature_nnie = reshape(out_nnie, w, h, 4, 1);
feature_show_nnie = squeeze(feature_nnie(:, :, 3));
feature_show_nnie = feature_show_nnie';
[r, c, ~] = find(feature_show_nnie > 0.6);
hold on; plot(c*stride, r*stride, '.r', 'MarkerSize', 10); hold off;

figure;
% idx_start = 1; idx_end = 4000;
% plot(idx_start:idx_end, out_mxnet(idx_start:idx_end), 'g', ...
%     idx_start:idx_end, out_nnie(idx_start:idx_end), 'r')
plot(1:length(out_mxnet), out_mxnet, 1:length(out_nnie), out_nnie)