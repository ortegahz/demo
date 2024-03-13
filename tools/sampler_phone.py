import argparse
import logging
import shutil
import copy
import time
import cv2
import sys
import os

import numpy as np

from multiprocessing import Process, Queue

sys.path.append('./')
from utils.decoder import process_decoder
from utils.utils_funcs import set_logging
from utils.ious import iogs_calc

sys.path.append('/home/manu/workspace/YOLOv6')
from yolov6.core.inferer import Inferer


def make_dirs(args):
    if os.path.exists(args.save_dir_root) and args.reset:
        shutil.rmtree(args.save_dir_root)
    subset = 'train'  # lazy people
    os.makedirs(os.path.join(args.save_dir_root, 'labels', subset), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir_root, 'images', subset), exist_ok=True)


def run(args):
    inferer_phone = Inferer(args.path_in_mp4, False, 0, args.weights_phone, 0, args.yaml_phone, args.img_size, False)
    inferer_play = Inferer(args.path_in_mp4, False, 0, args.weights_play, 0, args.yaml_play, args.img_size, False)

    q_decoder = Queue()
    p_decoder = Process(target=process_decoder, args=(args.path_in, q_decoder), daemon=True)
    p_decoder.start()

    make_dirs(args)

    vid_writer = None
    if args.save_path_video:
        vid_writer = cv2.VideoWriter(args.save_path_video, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1920, 1080))

    t_last = time.time()
    while True:
        item_frame = q_decoder.get()
        idx_frame, frame, fc = item_frame
        frame_org = copy.deepcopy(frame)

        det_phone = inferer_phone.infer_custom(frame, 0.4, 0.45, None, False, 1000)
        det_play = inferer_play.infer_custom(frame, 0.4, 0.45, None, False, 1000)

        bboxes_phone, bboxes_play = det_phone[:, :4].cpu().numpy(), det_play[:, :4].cpu().numpy()
        iogs = iogs_calc(bboxes_phone, bboxes_play)

        if args.ext_info:
            if len(det_phone):
                for *xyxy, conf, cls in reversed(det_phone):
                    label = f'{conf:.2f}'
                    inferer_phone.plot_box_and_label(frame, max(round(sum(frame.shape) / 2 * 0.003), 2), xyxy, label,
                                                     color=(0, 255, 0))

            if len(det_play):
                for *xyxy, conf, cls in reversed(det_play):
                    label = f'{conf:.2f}'

                    inferer_play.plot_box_and_label(frame, max(round(sum(frame.shape) / 2 * 0.003), 2), xyxy, label,
                                                    color=(255, 255, 0))

        det_play_pick = []
        if len(det_play):
            for idx, (*xyxy, conf_play, _) in enumerate(det_play):
                max_match_iog = max(iogs[:, idx]) if iogs.shape[0] > 0 else 0.
                conf_phone = det_phone[np.argmax(iogs[:, idx]), -2] if iogs.shape[0] > 0 and max_match_iog > 0.9 else 0.
                conf = args.alpha * conf_play + (1 - args.alpha) * conf_phone
                if conf > args.th_esb:
                    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(frame, p1, p2, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
                    cv2.putText(frame, f'{conf:.2f}', (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                if args.th_esb < conf < args.th_pick_ub:
                    det_play_pick.append(det_play[idx])

        cv2.putText(frame, f'{idx_frame} / {fc}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow('results', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or (idx_frame > fc - 10 and fc > 0):
            break

        if args.save_path_video:
            vid_writer.write(frame)

        if len(det_play_pick) > 0 and time.time() - t_last > args.sample_interval_s:
            subset = 'train'  # lazy people
            t_last = time.time()
            str_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path_img_out = os.path.join(args.save_dir_root, 'images', subset, f'{str_time}.jpg')
            cv2.imwrite(path_img_out, frame_org)
            if args.ext_info:
                cv2.imwrite(path_img_out[:-4] + '_draw' + path_img_out[-4:], frame)

            h_img, w_img, _ = frame_org.shape

            path_label_out = path_img_out.replace('.jpg', '.txt')
            path_label_out = path_label_out.replace('images', 'labels')
            with open(path_label_out, 'w') as f:
                for *xyxy, conf_play, cls in det_play_pick:
                    cx = (xyxy[0] + xyxy[2]) / 2. / w_img
                    cy = (xyxy[1] + xyxy[3]) / 2. / h_img
                    w = (xyxy[2] - xyxy[0]) / w_img
                    h = (xyxy[3] - xyxy[1]) / h_img
                    f.writelines(f'0 {cx} {cy} {w} {h}\n')
                for *xyxy, conf_play, cls in det_phone:
                    cx = (xyxy[0] + xyxy[2]) / 2. / w_img
                    cy = (xyxy[1] + xyxy[3]) / 2. / h_img
                    w = (xyxy[2] - xyxy[0]) / w_img
                    h = (xyxy[3] - xyxy[1]) / h_img
                    f.writelines(f'1 {cx} {cy} {w} {h}\n')

    cv2.destroyAllWindows()
    if args.save_path_video:
        vid_writer.release()


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in_mp4', default='/media/manu/kingstoo/tmp/20230605-10.20.164.67.mp4', type=str)  # TODO
    # parser.add_argument('--path_in', default='/media/manu/kingstoo/tmp/20230605-10.20.166.45.mp4', type=str)
    # parser.add_argument('--path_in', default='/media/manu/kingstoo/tmp/20230605-10.20.164.49.mp4', type=str)
    # parser.add_argument('--path_in', default='/media/manu/kingstoo/tmp/20230605-10.20.164.67.mp4', type=str)
    # parser.add_argument('--path_in', default='rtsp://192.168.1.40:554/live/av0', type=str)
    parser.add_argument('--path_in', default='rtsp://192.168.3.200:554/ch0_1', type=str)
    parser.add_argument('--yaml_phone', default='/home/manu/workspace/YOLOv6/data/phone.yaml', type=str)
    parser.add_argument('--weights_phone', default='/home/manu/tmp/exp12/weights/best_ckpt.pt', type=str)
    parser.add_argument('--yaml_play', default='/home/manu/workspace/YOLOv6/data/play.yaml', type=str)
    parser.add_argument('--weights_play', default='/home/manu/tmp/exp16/weights/best_ckpt.pt', type=str)
    parser.add_argument('--img_size', nargs='+', type=int, default=[1280, 1280])
    parser.add_argument('--hide_labels', default=True, action='store_true', help='hide labels.')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences.')
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--th_esb', default=0.4, type=float)
    parser.add_argument('--ext_info', default=False, action='store_true')
    parser.add_argument('--save_path_video', default=None)
    parser.add_argument('--save_dir_root', default='/media/manu/kingstoo/tmp/pics_phone_sample', type=str)
    parser.add_argument('--th_pick_ub', default=0.6, type=float)
    parser.add_argument('--sample_interval_s', default=60 * 20, type=float)
    parser.add_argument('--reset', default=False, type=bool)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
