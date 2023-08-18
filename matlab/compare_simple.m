clear; close all;

out_ref = load('/home/manu/tmp/pytorch_outputs_ys_4.txt');
% out_q = load('/home/manu/tmp/rknn_output_4_nq.txt');
out_q = load('/home/manu/nfs/rv1126/install/rknn_yolov5_demo/rknn_output_real_nq_4.txt');

error = sum((out_ref - out_q) .^ 2) / length(out_ref);
sim_cos = dot(out_ref, out_q) / (norm(out_ref) * norm(out_q));  

figure;
subplot(3,1,1);
plot(1:length(out_ref), out_ref, 'g', 1:length(out_q), out_q, 'y'); title(sprintf('cos sim %f', sim_cos));
subplot(3,1,2); plot(1:length(out_q), out_q, 'y'); title('q');
subplot(3,1,3); plot(1:length(out_ref), out_ref, 'g'); title('ref');

