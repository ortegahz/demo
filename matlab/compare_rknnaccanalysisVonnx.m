%%
clear; close all;

%%
% h = 256; w = 192; c = 3;
% h = 128; w = 96; c = 64;
% h = 64; w = 48; c = 64;
h = 64; w = 48; c = 17;

%%
out_ref = load('/home/manu/tmp/onnx_outputs_rsn_res.txt');  % onnx --> c x h x w

% out_q = load('/media/manu/kingstop/workspace/rknn-toolkit/examples/onnx/rsn/snapshot/fp32/images_2_out0_nhwc_1_256_192_3.tensor');  % rknn --> h x w x c
% out_q = load('/media/manu/kingstop/workspace/rknn-toolkit/examples/onnx/rsn/snapshot/fp32/Conv_Conv_1_42_out0_nhwc_1_128_96_64.tensor');  % rknn --> h x w x c
% out_q = load('/media/manu/kingstop/workspace/rknn-toolkit/examples/onnx/rsn/snapshot/fp32/MaxPool_MaxPool_3_24_out0_nhwc_1_64_48_64.tensor');  % rknn --> h x w x c
out_q = load('/media/manu/kingstop/workspace/rknn-toolkit/examples/onnx/rsn/snapshot/fp32/Conv_Conv_314_1_out0_nhwc_1_64_48_17.tensor');  % rknn --> h x w x c

out_ref_rs = reshape(out_ref, w, h, c);
out_ref_rs_perm = permute(out_ref_rs, [2, 1, 3]);
out_ref_rs_perm_flat = reshape(out_ref_rs_perm, 1, []);
out_ref_rs_perm_pick = out_ref_rs_perm(:, :, 3);
figure; imshow(out_ref_rs_perm_pick);

out_q_rs = reshape(out_q, c, w, h);
out_q_rs_perm = permute(out_q_rs, [3, 2, 1]);
out_q_rs_perm_flat = reshape(out_q_rs_perm, 1, []);
out_q_rs_perm_pick = out_q_rs_perm(:, :, 3);
figure; imshow(out_q_rs_perm_pick);

%%
sim_cos = dot(out_ref_rs_perm_flat, out_q_rs_perm_flat) / (norm(out_ref_rs_perm_flat) * norm(out_q_rs_perm_flat));  

%%
figure;
subplot(3,1,1);
plot(1:length(out_ref_rs_perm_flat), out_ref_rs_perm_flat, 'g', 1:length(out_q_rs_perm_flat), out_q_rs_perm_flat, 'y'); title(sprintf('cos sim %f', sim_cos));
subplot(3,1,2); plot(1:length(out_q_rs_perm_flat), out_q_rs_perm_flat, 'y'); title('q');
subplot(3,1,3); plot(1:length(out_ref_rs_perm_flat), out_ref_rs_perm_flat, 'g'); title('ref');

