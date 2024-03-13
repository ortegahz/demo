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
from utils.utils_funcs import set_logging

# using branch manu_demo_api
os.environ['RSN_HOME'] = '/media/manu/kingstop/workspace/RSN'
sys.path.append(os.environ['RSN_HOME'])
sys.path.append(os.path.join(os.environ['RSN_HOME'], 'exps', 'RSN18.coco'))
from infer import Inferer as RSNInferer

sys.path.append('/media/manu/kingstop/workspace/YOLOv6')
from yolov6.core.inferer import Inferer


class PlayerDetector:
    def __init__(self, args):
        self.poi_x, self.poi_y, self.poi_conf = None, None, None
        self.ratio, self.padding, self.pred_phone_cls_lst = None, None, None
        self.inferer_esb = Inferer(args.path_in_mp4, False, 0, args.weights_esb, 0, args.yaml_esb, args.img_size, False)
        self.inferer_kps = RSNInferer(args.weights_kps)

    def conf_calc(self, frame, args, idx, det_people, det_play, results_kps):
        *xyxy, conf_play, _ = det_play[idx]
        det_people_pick = det_people[idx]
        det_people_pick_idx = idx
        sz_pick = torch.sqrt(
            (det_people_pick[2] - det_people_pick[0]) * (det_people_pick[3] - det_people_pick[1]))
        *xyxy_p, conf_p, _ = det_people_pick
        results_kps_pick = results_kps[det_people_pick_idx]
        joints = np.array(results_kps_pick['keypoints']).reshape((17, 3))
        print(det_people_pick)
        print(sz_pick)
        print((self.poi_x, self.poi_y, self.poi_conf))
        print(joints[7:11, :])
        print(conf_play)
        dir_r = joints[10, :2] - joints[8, :2]
        dir_r_norm = preprocessing.normalize(dir_r[None, :])
        dir_l = joints[9, :2] - joints[7, :2]
        dir_l_norm = preprocessing.normalize(dir_l[None, :])
        print(dir_l_norm)
        print(dir_r_norm)

        rescale = 10.
        cxcy_phone = np.array((int(self.poi_x), int(self.poi_y)))
        dir_r_p = cxcy_phone - joints[10, :2]
        dir_r_p_norm = preprocessing.normalize(dir_r_p[None, :])
        print(dir_r_p_norm)
        cos_r = np.dot(dir_r_p_norm, dir_r_norm.T)[0][0]
        w_r = 2 - cos_r
        dist_r = np.linalg.norm(dir_r_p) * w_r / sz_pick * rescale
        dir_l_p = cxcy_phone - joints[9, :2]
        dir_l_p_norm = preprocessing.normalize(dir_l_p[None, :])
        print(dir_l_p_norm)
        cos_l = np.dot(dir_l_p_norm, dir_l_norm.T)[0][0]
        w_l = 2 - cos_l
        dist_l = np.linalg.norm(dir_l_p) * w_l / sz_pick * rescale
        print((dist_l, dist_r))
        color = (0, 0, 128)
        cv2.line(frame, tuple(joints[10, :2].astype(int)), cxcy_phone, color, 2)
        cv2.line(frame, tuple(joints[9, :2].astype(int)), cxcy_phone, color, 2)
        cv2.putText(frame, f'{dist_r:.2f}', tuple(joints[10, :2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 255), 2)
        cv2.putText(frame, f'{dist_l:.2f}', tuple(joints[9, :2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 0), 2)
        # conf_kps = (max(1 / dist_l * joints[9, 2], 1 / dist_r * joints[10, 2]) + det_phone_pick[-2]) / 2.
        conf_kps = (1 / (dist_l + 1.) * joints[9, 2] * self.poi_conf +
                    1 / (dist_r + 1.) * joints[10, 2] * self.poi_conf) / 2.

        print(conf_kps)

        conf = args.alpha * conf_play + (1 - args.alpha) * conf_kps
        print(conf)
        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        if conf > args.th_esb:
            cv2.rectangle(frame, p1, p2, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
            cv2.putText(frame, f'{conf:.2f}', (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            # cv2.putText(frame, f'{conf_kps:.2f}', (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(frame, f'{conf_kps:.2f}', (p1[0], int((p1[1] + p2[1]) / 2.)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    def poi_calc(self, frame, idx, xyxy, det_play_unrs):
        xyxy_unrs = det_play_unrs[idx][:4]
        max_phone_ac = [-1, -1, -1, 0.]  # xp, yp, s, conf
        for idx_s, s in enumerate(self.inferer_esb.model_rknn.stride):
            x1, y1, x2, y2 = (xyxy_unrs / s).int()
            feat_psc = copy.deepcopy(self.pred_phone_cls_lst[idx_s])
            mask = torch.zeros_like(feat_psc)
            mask[y1:y2, x1:x2] = 1.
            feat_psc *= mask
            value = torch.max(feat_psc.flatten())
            pos = torch.argmax(feat_psc.flatten())
            xp, yp = pos % feat_psc.shape[1], pos // feat_psc.shape[1]
            max_phone_ac = [xp, yp, s, value] if value > max_phone_ac[-1] else max_phone_ac
        poi_x, poi_y, poi_conf = \
            max_phone_ac[0] * max_phone_ac[2], max_phone_ac[1] * max_phone_ac[2], max_phone_ac[-1]
        poi_x, poi_y = poi_x - self.padding[0], poi_y - self.padding[1]
        poi_x, poi_y = poi_x / self.ratio, poi_y / self.ratio
        cv2.circle(frame, (int(poi_x), int(poi_y)), 2, (0, 255, 0), 2)
        cv2.putText(frame, f'{max_phone_ac[-1]:.2f}', (int(poi_x), int(poi_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        ppc_x, ppc_y = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
        cv2.line(frame, (int(ppc_x), int(ppc_y)), (int(poi_x), int(poi_y)), (0, 255, 0), 2)
        self.poi_x, self.poi_y, self.poi_conf = poi_x, poi_y, poi_conf

    def gen_pred_phone_cls_lst(self, frame, pred_results_phone):
        if frame.shape[0] == 1080 and frame.shape[1] == 1920:
            model_in_sz = [768, 1280]
        elif frame.shape[0] == 1280 and frame.shape[1] == 1280:
            model_in_sz = [1280, 1280]
        else:
            raise ValueError('un-support frame shape')
        strides = self.inferer_esb.model_rknn.stride
        self.ratio = min(model_in_sz[0] / frame.shape[0], model_in_sz[1] / frame.shape[1])  # pick 1280 / 1920
        self.padding = (model_in_sz[1] - frame.shape[1] * self.ratio) / 2, (
                model_in_sz[0] - frame.shape[0] * self.ratio) / 2
        logging.info(f'padding --> {self.padding}')
        pred_phone_cls_lst = list()
        pred_results_phone[:, :, 0] *= pred_results_phone[:, :, 1]  # conf = obj_conf * cls_conf
        pred_phone_cls = pred_results_phone[:, :, 0]
        logging.info(f'pred_phone_cls.shape --> {pred_phone_cls.shape}')
        idx_s = 0
        for s in strides:
            h, w = int(model_in_sz[0] / s), int(model_in_sz[1] / s)
            idx_e = idx_s + int(h * w)
            pred_phone_cls_lst.append(pred_phone_cls[:, idx_s: idx_e].reshape(h, w))
            idx_s = idx_e
        self.pred_phone_cls_lst = pred_phone_cls_lst

    def write_final_results(self, frame, xyxy, conf_play, results_kps,
                            path_save='/home/manu/tmp/pytorch_parser_final_results.txt'):
        joints = np.array(results_kps['keypoints']).reshape(-1)
        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
        xywh = (self.inferer_esb.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(
            -1).tolist()  # normalized xywh
        line = (0, *xywh, self.poi_x, self.poi_y, self.poi_conf, conf_play, *joints)
        with open(path_save, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def write_esb_results(self, frame, xyxy, conf_play, path_save):
        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
        xywh = (self.inferer_esb.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(
            -1).tolist()  # normalized xywh
        line = (0, *xywh, self.poi_x, self.poi_y, self.poi_conf, conf_play)
        with open(path_save, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def draw_debug_info_s0(self, frame, det_people, det_play, results_kps):
        if det_people is not None:
            for *xyxy, conf, cls in reversed(det_people):
                label = f'{conf:.2f}'
                self.inferer_esb.plot_box_and_label(frame, max(round(sum(frame.shape) / 2 * 0.003), 2), xyxy, label,
                                                    color=(255, 0, 0))
        if len(det_play):
            for idx, (*xyxy, conf, cls) in enumerate(det_play):
                label = f'{conf:.2f}'
                self.inferer_esb.plot_box_and_label(frame, max(round(sum(frame.shape) / 2 * 0.003), 2), xyxy, label,
                                                    color=(0, 255, 255))
        if results_kps is not None:
            frame = self.inferer_kps.draw_results(frame, results_kps)

    def inference(self, frame, args):
        # esb inference
        det_play, det_play_unrs, pred_results_phone = self.inferer_esb.infer_rknn(frame, 0.4, 0.45, None, False, 1000)

        # prepare pred_phone_cls_lst
        self.gen_pred_phone_cls_lst(frame, pred_results_phone)

        for i, pred_phone_cls_lst_s in enumerate(self.pred_phone_cls_lst):
            np.savetxt('/home/manu/tmp/pytorch_outputs_pred_phone_cls_lst_s_%s.txt' % i,
                       pred_phone_cls_lst_s.detach().cpu().numpy().flatten(),
                       fmt="%f", delimiter="\n")

        # kps detection
        results_kps, det_people = None, None
        if len(det_play):
            idets_kps = det_play[:, :5].cpu().detach().numpy()
            logging.info(det_play)
            idets_kps[:, 2:4] = idets_kps[:, 2:4] - idets_kps[:, :2]  # xyxy to xywh
            idets_kps[:, 2] *= 1.1
            idets_kps[:, 3] *= 2.2
            logging.info(idets_kps)
            results_kps = self.inferer_kps.inference(frame, idets_kps)
            det_people = copy.deepcopy(det_play)
            det_people[:, 2:4] = torch.tensor(idets_kps[:, :2] + idets_kps[:, 2:4])  # xywh to xyxy

        # draw debug info
        self.draw_debug_info_s0(frame, det_people, det_play, results_kps)

        # confs calculation
        for idx, (*xyxy, conf_play, _) in enumerate(det_play):
            # calc poi
            self.poi_calc(frame, idx, xyxy, det_play_unrs)

            # calc conf
            self.conf_calc(frame, args, idx, det_people, det_play, results_kps)

            # write esb results
            self.write_esb_results(frame, xyxy, conf_play, args.path_save_txt)

            # write final results
            self.write_final_results(frame, xyxy, conf_play, results_kps[idx])


def run(args):
    player_detector = PlayerDetector(args)

    q_decoder = Queue()
    p_decoder = Process(target=process_decoder, args=(args.path_in, q_decoder), daemon=True)
    p_decoder.start()

    vid_writer = cv2.VideoWriter(args.path_save_video, cv2.VideoWriter_fourcc(*'mp4v'), 5, (1920, 1080))

    if os.path.exists(args.path_save_txt):
        os.remove(args.path_save_txt)

    while True:
        item_frame = q_decoder.get()
        idx_frame, frame, fc = item_frame

        player_detector.inference(frame, args)

        cv2.putText(frame, f'{idx_frame} / {fc}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow('results', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or (idx_frame > fc - 10 and fc > 0):
            break

        vid_writer.write(frame)
        break

    cv2.destroyAllWindows()
    vid_writer.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_save_txt', default='/home/manu/tmp/pytorch_results.txt', type=str)
    parser.add_argument('--path_save_video', default='/home/manu/tmp/results.mp4', type=str)
    parser.add_argument('--path_in_mp4', default='/media/manu/kingstoo/tmp/20230605-10.20.164.67.mp4', type=str)  # TODO
    parser.add_argument('--path_in', default='/media/manu/kingstoo/tmp/20230605-10.20.164.67.mp4', type=str)
    parser.add_argument('--yaml_esb', default='/home/manu/workspace/YOLOv6/data/phone.yaml', type=str)
    parser.add_argument('--weights_esb', default='/home/manu/tmp/model_phone/phone.pt', type=str)
    parser.add_argument('--weights_kps', default='/home/manu/tmp/iter-96000.pth', type=str)
    parser.add_argument('--img_size', nargs='+', type=int, default=[1280, 1280])
    parser.add_argument('--hide_labels', default=True, action='store_true', help='hide labels.')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences.')
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--th_esb', default=0.4, type=float)
    parser.add_argument('--poi', default=True, type=bool)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
