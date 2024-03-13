import argparse
import copy
import logging
import os
import sys
from multiprocessing import Process, Queue

import cv2
import numpy as np
import torch
from sklearn import preprocessing

from utils.decoder import process_decoder
from utils.ious import iogs_calc
from utils.utils_funcs import set_logging

# using branch manu_demo_api
os.environ['RSN_HOME'] = '/media/manu/kingstop/workspace/RSN'
sys.path.append(os.environ['RSN_HOME'])
sys.path.append(os.path.join(os.environ['RSN_HOME'], 'exps', 'RSN18.coco'))
from infer import Inferer as RSNInferer

sys.path.append('/home/manu/workspace/YOLOv6')
from yolov6.core.inferer import Inferer


def run(args):
    inferer_phone = Inferer(args.path_in_mp4, False, 0, args.weights_phone, 0, args.yaml_phone, args.img_size, False)
    inferer_play = Inferer(args.path_in_mp4, False, 0, args.weights_play, 0, args.yaml_play, args.img_size, False)
    inferer_people = Inferer(args.path_in_mp4, False, 0, args.weights_people, 0, args.yaml_people, args.img_size, False)
    inferer_kps = RSNInferer(args.weights_kps)

    q_decoder = Queue()
    p_decoder = Process(target=process_decoder, args=(args.path_in, q_decoder), daemon=True)
    p_decoder.start()

    save_path = '/home/manu/tmp/results.mp4'  # force *.mp4 suffix on results videos
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (1920, 1080))

    while True:
        item_frame = q_decoder.get()
        idx_frame, frame, fc = item_frame

        det_phone, _, pred_phone = inferer_phone.infer_custom(frame, 0.4, 0.45, None, False, 1000)
        det_play, det_play_unrs, _ = inferer_play.infer_custom(frame, 0.2, 0.45, None, False, 1000)
        det_people, _, _ = inferer_people.infer_custom(frame, 0.4, 0.45, None, False, 1000)

        bboxes_phone, bboxes_play, bboxes_people = \
            det_phone[:, :4].cpu().detach().numpy(), \
            det_play[:, :4].cpu().detach().numpy(), \
            det_people[:, :4].cpu().detach().numpy()

        strides = [8, 16, 32, 64]
        model_in_sz = [768, 1280]
        ratio = min(model_in_sz[0] / frame.shape[0], model_in_sz[1] / frame.shape[1])
        padding = (model_in_sz[1] - frame.shape[1] * ratio) / 2, (model_in_sz[0] - frame.shape[0] * ratio) / 2
        pred_phone_cls_lst = []
        pred_phone[:, :, 5:] *= pred_phone[:, :, 4:5]  # conf = obj_conf * cls_conf
        pred_phone_cls = pred_phone[:, :, 5]
        idx_s = 0
        for s in strides:
            h, w = int(model_in_sz[0] / s), int(model_in_sz[1] / s)
            idx_e = idx_s + int(h * w)
            pred_phone_cls_lst.append(pred_phone_cls[:, idx_s: idx_e].reshape(h, w))
            idx_s = idx_e

        iogs_phone = iogs_calc(bboxes_phone, bboxes_play)
        bboxes_people[:, 3] -= (bboxes_people[:, 3] - bboxes_people[:, 1]) / 2
        iogs_pepole = iogs_calc(bboxes_play, bboxes_people)

        results_kps = None
        if len(det_people):
            idets_kps = det_people[:, :5].cpu().detach().numpy()
            idets_kps[:, 2:4] = idets_kps[:, 2:4] - idets_kps[:, :2]  # xyxy to xywh
            results_kps = inferer_kps.inference(frame, idets_kps)

        if args.ext_info:
            if len(det_phone):
                for _, (*xyxy, conf, cls) in enumerate(det_phone):
                    label = f'{conf:.2f}'
                    inferer_phone.plot_box_and_label(frame, max(round(sum(frame.shape) / 2 * 0.003), 2), xyxy, label,
                                                     color=(0, 255, 0))

            if len(det_play):
                for idx, (*xyxy, conf, cls) in enumerate(det_play):
                    label = f'{conf:.2f}'
                    inferer_play.plot_box_and_label(frame, max(round(sum(frame.shape) / 2 * 0.003), 2), xyxy, label,
                                                    color=(0, 255, 255))

            if len(det_people):
                for *xyxy, conf, cls in reversed(det_people):
                    label = f'{conf:.2f}'
                    inferer_play.plot_box_and_label(frame, max(round(sum(frame.shape) / 2 * 0.003), 2), xyxy, label,
                                                    color=(255, 0, 0))

            if results_kps is not None:
                frame = inferer_kps.draw_results(frame, results_kps)

        for idx, (*xyxy, conf_play, _) in enumerate(det_play):
            joints = np.ones((17, 3)) * sys.maxsize
            dir_r_norm = np.ones((1, 2)) * sys.maxsize
            dir_l_norm = np.ones((1, 2)) * sys.maxsize
            dist_l, dist_r = sys.maxsize, sys.maxsize
            sz_pick = 1.
            conf_kps = 0.

            xyxy_unrs = det_play_unrs[idx][:4]
            max_phone_ac = [-1, -1, -1, 0.]
            values = []
            for idx_s, s in enumerate(strides):
                x1, y1, x2, y2 = (xyxy_unrs / s).int()
                feat_psc = copy.deepcopy(pred_phone_cls_lst[idx_s])
                mask = torch.zeros_like(feat_psc)
                mask[y1:y2, x1:x2] = 1.
                feat_psc *= mask
                value = torch.max(feat_psc.flatten())
                pos = torch.argmax(feat_psc.flatten())
                # if value > 0:
                #     logging.info((s, value, len(feat_psc.flatten()[feat_psc.flatten() == value])))
                xp, yp = pos % feat_psc.shape[1], pos // feat_psc.shape[1]
                max_phone_ac = [xp, yp, s, value] if value > max_phone_ac[-1] else max_phone_ac
                if value > max_phone_ac[-1]:
                    values.append(value)
                if len(values) > 1:
                    logging.info(len(values))
            poi_x, poi_y, poi_conf = \
                max_phone_ac[0] * max_phone_ac[2], max_phone_ac[1] * max_phone_ac[2], max_phone_ac[-1]
            poi_x, poi_y = poi_x - padding[0], poi_y - padding[1]
            poi_x, poi_y = poi_x / ratio, poi_y / ratio
            cv2.circle(frame, (int(poi_x), int(poi_y)), 2, (0, 255, 0), 2)
            cv2.putText(frame, f'{max_phone_ac[-1]:.2f}', (int(poi_x), int(poi_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            ppc_x, ppc_y = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
            cv2.line(frame, (int(ppc_x), int(ppc_y)), (int(poi_x), int(poi_y)), (0, 255, 0), 2)

            max_match_iog_people_th = 0.6
            max_match_iog_people = max(iogs_pepole[idx, :]) if len(iogs_pepole) > 0 else 0.
            det_people_pick = det_people[
                np.argmax(iogs_pepole[idx, :])] if max_match_iog_people > max_match_iog_people_th else None
            det_people_pick_idx = np.argmax(
                iogs_pepole[idx, :]) if max_match_iog_people > max_match_iog_people_th else -1
            if det_people_pick is not None:
                sz_pick = torch.sqrt(
                    (det_people_pick[2] - det_people_pick[0]) * (det_people_pick[3] - det_people_pick[1]))
                *xyxy_p, conf_p, _ = det_people_pick
                label_p = f'{conf_play:.2f}'
                # inferer_play.plot_box_and_label(frame, max(round(sum(frame.shape) / 2 * 0.003), 2), xyxy_p, label_p,
                #                                 color=(0, 255, 255))
                results_kps_pick = results_kps[det_people_pick_idx]
                joints = np.array(results_kps_pick['keypoints']).reshape((17, 3))
                # for i in [9, 10]:
                #     color = (0, 255, 255)
                #     if joints[i, 0] > 0 and joints[i, 1] > 0:
                #         cv2.circle(frame, tuple(joints[i, :2].astype(int)), 2, color, 2)
                dir_r = joints[10, :2] - joints[8, :2]
                dir_r_norm = preprocessing.normalize(dir_r[None, :])
                dir_l = joints[9, :2] - joints[7, :2]
                dir_l_norm = preprocessing.normalize(dir_l[None, :])
                # cv2.line(frame, tuple(joints[10, :2].astype(int)), tuple(joints[8, :2].astype(int)), (0, 255, 255), 2)
                # cv2.line(frame, tuple(joints[9, :2].astype(int)), tuple(joints[7, :2].astype(int)), (0, 255, 255), 2)

            max_match_iog_th = 0.9
            max_match_iog = max(iogs_phone[:, idx]) if len(iogs_phone) > 0 else 0.
            # conf_phone = det_phone[np.argmax(iogs_phone[:, idx]), -2] if len(
            #     iogs_phone) > 0 and max_match_iog > max_match_iog_th else 0.
            det_phone_pick = det_phone[np.argmax(iogs_phone[:, idx])] if len(
                iogs_phone) and max_match_iog > max_match_iog_th else None
            # if det_phone_pick is not None:
            if det_phone_pick is not None or args.poi:
                rescale = 10.
                # cxcy_phone = (int((det_phone_pick[0] + det_phone_pick[2]) / 2.),
                #               int((det_phone_pick[1] + det_phone_pick[3]) / 2.))
                # cv2.circle(frame, cxcy_phone, 2, (0, 255, 0), 2)
                cxcy_phone = (int(poi_x), int(poi_y))
                dir_r_p = cxcy_phone - joints[10, :2]
                dir_r_p_norm = preprocessing.normalize(dir_r_p[None, :])
                cos_r = np.dot(dir_r_p_norm, dir_r_norm.T)[0][0]
                w_r = 2 - cos_r
                dist_r = np.linalg.norm(dir_r_p) * w_r / sz_pick * rescale
                dir_l_p = cxcy_phone - joints[9, :2]
                dir_l_p_norm = preprocessing.normalize(dir_l_p[None, :])
                cos_l = np.dot(dir_l_p_norm, dir_l_norm.T)[0][0]
                w_l = 2 - cos_l
                dist_l = np.linalg.norm(dir_l_p) * w_l / sz_pick * rescale
                if joints[10, 2] < sys.maxsize and joints[9, 2] < sys.maxsize:
                    color = (0, 0, 128)
                    cv2.line(frame, tuple(joints[10, :2].astype(int)), cxcy_phone, color, 2)
                    cv2.line(frame, tuple(joints[9, :2].astype(int)), cxcy_phone, color, 2)
                    cv2.putText(frame, f'{dist_r:.2f}', tuple(joints[10, :2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 255), 2)
                    cv2.putText(frame, f'{dist_l:.2f}', tuple(joints[9, :2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (255, 255, 0), 2)
                # conf_kps = (max(1 / dist_l * joints[9, 2], 1 / dist_r * joints[10, 2]) + det_phone_pick[-2]) / 2.
                conf_kps = (1 / (dist_l + 1.) * joints[9, 2] * poi_conf +
                            1 / (dist_r + 1.) * joints[10, 2] * poi_conf) / 2.

            conf = args.alpha * conf_play + (1 - args.alpha) * conf_kps
            p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            if conf > args.th_esb:
                cv2.rectangle(frame, p1, p2, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
                cv2.putText(frame, f'{conf:.2f}', (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                # cv2.putText(frame, f'{conf_kps:.2f}', (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            cv2.putText(frame, f'{conf_kps:.2f}', (p1[0], int((p1[1] + p2[1]) / 2.)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        cv2.putText(frame, f'{idx_frame} / {fc}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow('results', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or (idx_frame > fc - 10 and fc > 0):
            break

        vid_writer.write(frame)

    cv2.destroyAllWindows()
    vid_writer.release()


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in_mp4', default='/media/manu/kingstoo/tmp/20230605-10.20.164.67.mp4', type=str)  # TODO
    parser.add_argument('--path_in', default='/media/manu/kingstoo/tmp/20230605-10.20.164.67.mp4', type=str)
    # parser.add_argument('--path_in', default='/media/manu/kingstoo/tmp/20230605-10.20.166.45.mp4', type=str)
    # parser.add_argument('--path_in', default='/media/manu/kingstoo/tmp/20230605-10.20.164.49.mp4', type=str)
    # parser.add_argument('--path_in', default='rtsp://192.168.1.40:554/live/av0', type=str)
    # parser.add_argument('--path_in', default='rtsp://192.168.3.200:554/ch0_1', type=str)
    parser.add_argument('--yaml_phone', default='/home/manu/workspace/YOLOv6/data/phone.yaml', type=str)
    parser.add_argument('--weights_phone', default='/home/manu/tmp/nn6_ft_b64_nab_s1280_dpc/weights/best_ckpt.pt',
                        type=str)
    # parser.add_argument('--weights_phone', default='/home/manu/tmp/aux.pt', type=str)
    parser.add_argument('--yaml_play', default='/home/manu/workspace/YOLOv6/data/play.yaml', type=str)
    parser.add_argument('--weights_play', default='/home/manu/tmp/nn6_ft_b64_nab_s1280_dl/weights/best_ckpt.pt',
                        type=str)
    # parser.add_argument('--weights_play', default='/home/manu/tmp/main.pt', type=str)
    parser.add_argument('--yaml_people', default='/media/manu/kingstop/workspace/YOLOv6/data/people.yaml', type=str)
    parser.add_argument('--weights_people', default='/home/manu/tmp/nn6_ft_b32_nab_s1280/weights/best_ckpt.pt',
                        type=str)
    parser.add_argument('--weights_kps', default='/home/manu/tmp/iter-96000.pth', type=str)
    parser.add_argument('--img_size', nargs='+', type=int, default=[1280, 1280])
    parser.add_argument('--hide_labels', default=True, action='store_true', help='hide labels.')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences.')
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--th_esb', default=0.5, type=float)
    parser.add_argument('--ext_info', default=True, action='store_true')
    parser.add_argument('--poi', default=True, type=bool)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
