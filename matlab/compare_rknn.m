%%
clear; close all;

%%
h = 79; w = 79; c = 32;

%%
out_ref = load('/home/manu/nfs/rv1126/install/rknn_yolov5_demo/NodeID_66_DeConvolutionLayer_w_79_h_79_d_32_batchID_0_out_0.txt');
out_q = load('/media/manu/kingstop/workspace/rknn-toolkit/examples/onnx/acfree/snapshot/fp32/ConvTranspose_ConvTranspose_60_59_plane_131_out0_nhwc_1_79_79_32.tensor');

out_q_rs = reshape(out_q, c, h, w);
out_q_rs_perm = permute(out_q_rs, [3, 2, 1]);
out_q_rs_perm_pick = out_q_rs_perm(:, :, 3);
out_q_rs_perm_flat = reshape(out_q_rs_perm, 1, []);
figure; imshow(out_q_rs_perm_pick);

out_ref_rs = reshape(out_ref, h, w, c);
out_ref_rs_perm = permute(out_ref_rs, [2, 1, 3]);
out_ref_rs_perm_pick = out_ref_rs_perm(:, :, 3);
out_ref_rs_perm_flat = reshape(out_ref_rs_perm, 1, []);
figure; imshow(out_ref_rs_perm_pick);

sim_cos = dot(out_ref_rs_perm_flat, out_q_rs_perm_flat) / (norm(out_ref_rs_perm_flat) * norm(out_q_rs_perm_flat));  

%%
figure;
subplot(3,1,1);
plot(1:length(out_ref_rs_perm_flat), out_ref_rs_perm_flat, 'g', 1:length(out_q_rs_perm_flat), out_q_rs_perm_flat, 'y'); title(sprintf('cos sim %f', sim_cos));
subplot(3,1,2); plot(1:length(out_q_rs_perm_flat), out_q_rs_perm_flat, 'y'); title('q');
subplot(3,1,3); plot(1:length(out_ref_rs_perm_flat), out_ref_rs_perm_flat, 'g'); title('ref');

