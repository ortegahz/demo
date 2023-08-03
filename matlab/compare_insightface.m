clear; close all;

out_ref = load('/media/manu/kingstop/workspace/rknn-toolkit/examples/onnx/acfree/snapshot/fp32/ConvTranspose_ConvTranspose_60_59_plane_131_out0_nhwc_1_79_79_32.tensor');
out_q = load('/media/manu/kingstop/workspace/rknn-toolkit/examples/onnx/acfree/snapshot/entire_qnt/ConvTranspose_ConvTranspose_60_59_plane_131_out0_nhwc_1_79_79_32.tensor');

error = sum((out_ref - out_q) .^ 2) / length(out_ref);
sim_cos = dot(out_ref, out_q) / (norm(out_ref) * norm(out_q));  

figure;
subplot(3,1,1);
plot(1:length(out_ref), out_ref, 'g', 1:length(out_q), out_q, 'y'); title(sprintf('cos sim %f', sim_cos));
subplot(3,1,2); plot(1:length(out_q), out_q, 'y'); title('q');
subplot(3,1,3); plot(1:length(out_ref), out_ref, 'g'); title('ref');

