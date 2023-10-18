clear; close all;

out_ref = load('/home/manu/tmp/pytorch_outputs_0.txt');  % caffe c h w -> matlab w h c
out_q = load('/home/manu/nfs/mpp/sample/svp/nnie/seg0_layer73_output0_inst.linear.float');  % caffe w h c -> matlab c w h

out_ref_rs = reshape(out_ref, 80, 80, 8);  % caffe c h w -> matlab w h c
out_ref_perm = permute(out_ref_rs,[3 1 2]);  % matlab w h c -> matlab c w h
out_ref = reshape(out_ref_perm,[],1);

% error = sum((out_ref - out_q) .^ 2) / length(out_ref);
sim_cos = dot(out_ref, out_q) / (norm(out_ref) * norm(out_q));  

figure;
subplot(3,1,1);
plot(1:length(out_ref), out_ref, 'g', 1:length(out_q), out_q, 'y'); title(num2str(sim_cos));
subplot(3,1,2); plot(1:length(out_ref), out_ref, 'g'); title('ref');
subplot(3,1,3); plot(1:length(out_q), out_q, 'y'); title('q');

