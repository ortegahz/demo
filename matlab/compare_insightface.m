clear; close all;

out_ref = load('/media/manu/kingstop/workspace/ONNXToCaffe/output/dump/layers/input_640x640.txt');
out_q = load('/home/manu/nfs/mpp/sample/svp/nnie/seg0_layer2_output0_inst.linear.txt');

error = sum((out_ref - out_q) .^ 2) / length(out_ref);

figure;
subplot(3,1,1);
plot(1:length(out_ref), out_ref, 'g', 1:length(out_q), out_q, 'y'); title(num2str(error));
subplot(3,1,2); plot(1:length(out_q), out_q, 'y'); title('q');
subplot(3,1,3); plot(1:length(out_ref), out_ref, 'g'); title('ref');

