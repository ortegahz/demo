clear; close all;

% out_ref = load('/home/manu/tmp/pytorch_results.txt');
out_ref = load('/home/manu/tmp/pytorch_outputs_rsn.txt');
% out_ref = load('/home/manu/tmp/onnx_output_img_wa.txt');
% out_ref = load('/home/manu/tmp/pytorch_output_dr_0.txt');
% out_ref = load('/home/manu/tmp/rknn_output_dr_0.txt');
% out_ref = load('/home/manu/tmp/pytorch_output_img_wa.txt');
% out_ref = load('/home/manu/tmp/pytorch_outputs_reg_output_3.txt');
% out_ref = load('/home/manu/tmp/pytorch_outputs_0.txt');
% out_ref = load('/media/manu/kingstop/workspace/ONNXToCaffe/output/dump/layers/257_onnx.txt');

% out_q = load('/home/manu/tmp/results_rknn.txt');
% out_q = load('/home/manu/tmp/onnx_outputs_rsn_outputs.txt');
% out_q = load('/media/manu/kingstop/workspace/rknn-toolkit/examples/onnx/rsn/snapshot/fp32/images_2_out0_nhwc_1_256_192_3.tensor');
% out_q = load('/home/manu/tmp/rknn_output_rsn_0_nq.txt');
% out_q = load('/home/manu/tmp/rknn_output_rsn_0.txt');
% out_q = load('/home/manu/tmp/rknn_output_dr_0.txt');
% out_q = load('/home/manu/tmp/rknn_output_img_wa.txt');
% out_q = load('/home/manu/tmp/rknn_output_0_nq.txt');
% out_q = load('/home/manu/nfs/rv1126/install/rknn_yolov5_demo/rknn_output_real_kps_nq_0.txt');
out_q = load('/home/manu/nfs/rv1126/install/rknn_yolov5_demo/rknn_output_real_kps_0.txt');
% out_q = load('/home/manu/nfs/rv1126/install/rknn_yolov5_demo/rknn_output_featureMap_0.txt');
% out_q = load('/home/manu/nfs/rv1126/install/rknn_yolov5_demo/rknn_output_input_data.txt');
% out_q = load('/home/manu/tmp/rknn_output_7_nq.txt');
% out_q = load('/home/manu/nfs/rv1126/install/rknn_yolov5_demo/rknn_output_real_nq_7.txt');
% out_q = load('/home/manu/nfs/rv1126/install/rknn_yolov5_demo/rknn_output_real_7.txt');
% out_q = load('/home/manu/tmp/onnx_outputs_2.txt');
% out_q = load('/home/manu/nfs/mpp/sample/svp/nnie/seg0_layer127_output0_inst.linear.float');
% out_q = load('/home/manu/nfs/mpp/sample/svp/nnie/seg0_layer124_output0_inst.linear.float');
% out_q = load('/home/manu/nfs/mpp/sample/svp/nnie/seg0_layer72_output0_inst.linear.float');

% error = sum((out_ref - out_q) .^ 2) / length(out_ref);
sim_cos = dot(out_ref, out_q) / (norm(out_ref) * norm(out_q));  

figure;
subplot(3,1,1);
plot(1:length(out_ref), out_ref, 'g', 1:length(out_q), out_q, 'y'); title(sprintf('cos sim %f', sim_cos));
subplot(3,1,2); plot(1:length(out_ref), out_ref, 'g'); title('ref');
subplot(3,1,3); plot(1:length(out_q), out_q, 'y'); title('q');

