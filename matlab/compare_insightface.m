clear; close all;

out_q = load('/home/manu/tmp/rknn_output_1.txt');
out_ref = load('/home/manu/tmp/pytorch_outputs_cls_output_1.txt');

error = sum((out_ref - out_q) .^ 2) / length(out_ref);

figure;
subplot(3,1,1);
plot(1:length(out_ref), out_ref, 'g', 1:length(out_q), out_q, 'y'); title(num2str(error));

subplot(3,1,2); plot(1:length(out_q), out_q, 'y'); title('q');
subplot(3,1,3); plot(1:length(out_ref), out_ref, 'g'); title('ref');

