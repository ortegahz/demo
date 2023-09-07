%%
clear; close all;

%%
% TODO: different list order for ref and rk / npu

% out_ref = load('/home/manu/tmp/pytorch_results.txt');
% out_ref = load('/home/manu/tmp/results_pytorch_rsn_mid.txt');
% out_ref = load('/home/manu/tmp/results_pytorch_rsn.txt');
% out_ref = load('/home/manu/tmp/pytorch_results.txt');
% out_ref = load('/home/manu/tmp/results_rknn_sim.txt');
% out_ref = load('/home/manu/tmp/pytorch_parser_final_results.txt');
% out_ref = load('/media/manu/kingstop/workspace/YOLOv6/runs/inference/head/labels/sylgd_rp.txt');
out_ref = load('/media/manu/kingstop/workspace/YOLOv6/runs/inference/yolov6n/labels/students_lt.txt');

% out_rk = load('/home/manu/tmp/results_rknn.txt');
% out_rk = load('/home/manu/tmp/results_rknn_sim_mid.txt');
% out_rk = load('/home/manu/tmp/results_rknn_sim.txt');
% out_rk = load('/home/manu/nfs/rv1126/install/rknn_yolov5_demo/npu_parser_results_kps.txt');
% out_rk = load('/home/manu/nfs/rv1126/install/rknn_yolov5_demo/npu_parser_final_results.txt');
% out_rk = load('/home/manu/tmp/results_rknn.txt');
% out_rk = load('/home/manu/nfs/rv1126/install/rknn_yolov5_demo/npu_parser_results.txt');
out_rk = load('/home/manu/nfs/mpp/sample/svp/nnie/results_ruyi.txt');

out_ref = out_ref'; out_rk = out_rk';

% out_ref = out_ref(:, 6); out_rk = out_rk(:, 6);
% out_ref = horzcat(out_ref(:, 1:5), out_ref(:, end));

out_ref = reshape(out_ref, 1, []); out_rk = reshape(out_rk, 1, []);

sim_cos = dot(out_ref, out_rk) / (norm(out_ref) * norm(out_rk));  

%%
figure;
subplot(3,1,1);
plot(1:length(out_ref), out_ref, 'g', 1:length(out_rk), out_rk, 'y');
title(sprintf('cos sim %f', sim_cos));
subplot(3,1,2); plot(1:length(out_ref), out_ref, 'g'); title('ref');
subplot(3,1,3); plot(1:length(out_rk), out_rk, 'y'); title('q');
