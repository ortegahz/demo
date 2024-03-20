clear; close all;
h = 17; w = 30; c = 24;
% h = 34; w = 60; c = 24;
% h = 68; w = 120; c = 24;
% h = 270; w = 480; c = 32;
% h = 540; w = 1024;

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
% out_ref = load('/home/manu/nfs/data/python_y.txt');
out_ref = load('/home/manu/tmp/rknn_output_2.txt');  % c h w [w h c]

% out_ref = reshape(out_ref, w, h, c);
% out_ref = permute(out_ref, [2 1 3]);
% figure;
% imshow(out_ref(:, :, 1));

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
% out_q = load('/home/manu/nfs/data/cpp_SrcSobelRoiMixed.txt');
% out_q = load('/home/manu/workspace/sca/yolo/output/yolo/quant_ori_result/image/lQDPKGkwiWEfP8nNBBDNB4Cw1DnkovXlHIwF5XdcYaNdAQ_960_540.bmp/_model.0_conv_Conv.txt');
out_q = load('/home/manu/tmp/output2.txt');
out_q = reshape(out_q, c, w, h);  % h w c [c w h] --> [w h c]
out_q = permute(out_q, [2 3 1]);
out_q = out_q(:);

% out_q = permute(out_q, [2 1 3]);
% figure;
% imshow(out_q(:, :, 1));

% error = sum((out_ref - out_q) .^ 2) / length(out_ref);
sim_cos = dot(out_ref, out_q) / (norm(out_ref) * norm(out_q));  

figure;
subplot(3,1,1);
plot(1:length(out_ref), out_ref, 'g', 1:length(out_q), out_q, 'y'); title(sprintf('cos sim %f', sim_cos));
subplot(3,1,2); plot(1:length(out_ref), out_ref, 'g'); title('ref');
subplot(3,1,3); plot(1:length(out_q), out_q, 'y'); title('q');

