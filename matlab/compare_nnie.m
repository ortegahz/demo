clear; close all;

out_ref = load('/home/manu/tmp/rknn_outputs_2.txt');
out_q = load('/home/manu/nfs/mpp/sample/svp/nnie/seg0_layer80_output0_inst.linear.float');

% out_ref_rs = reshape(out_ref, 13, 13, 85 * 3);
% out_ref_perm = permute(out_ref_rs,[3 1 2]);
% out_ref = reshape(out_ref_perm,[],1);

error = sum((out_ref - out_q) .^ 2) / length(out_ref);

figure;
subplot(3,1,1);
plot(1:length(out_ref), out_ref, 'g', 1:length(out_q), out_q, 'y'); title(num2str(error));
subplot(3,1,2); plot(1:length(out_q), out_q, 'y'); title('q');
subplot(3,1,3); plot(1:length(out_ref), out_ref, 'g'); title('ref');

