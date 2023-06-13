import argparse
import logging
import cv2
import sys

import numpy as np

from multiprocessing import Process, Queue

from utils.decoder import process_decoder
from utils.logging import set_logging
from utils.ious import iogs_calc

sys.path.append('/home/manu/workspace/YOLOv6')
from yolov6.core.inferer import Inferer


def run(args):
    inferer_phone = Inferer(args.path_in_mp4, False, 0, args.weights_phone, 0, args.yaml_phone, args.img_size, False)
    inferer_play = Inferer(args.path_in_mp4, False, 0, args.weights_play, 0, args.yaml_play, args.img_size, False)

    q_decoder = Queue()
    p_decoder = Process(target=process_decoder, args=(args.path_in, q_decoder), daemon=True)
    p_decoder.start()

    save_path = '/home/manu/tmp/results.mp4'  # force *.mp4 suffix on results videos
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1920, 1080))

    while True:
        item_frame = q_decoder.get()
        idx_frame, frame, fc = item_frame

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

        if len(det_play):
            for idx, (*xyxy, conf_play, _) in enumerate(det_play):
                max_match_iog = max(iogs[:, idx]) if iogs.shape[0] > 0 else 0.
                conf_phone = det_phone[np.argmax(iogs[:, idx]), -2] if iogs.shape[0] > 0 and max_match_iog > 0.9 else 0.
                conf = args.alpha * conf_play + (1 - args.alpha) * conf_phone
                if conf > args.th_esb:
                    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(frame, p1, p2, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
                    cv2.putText(frame, f'{conf:.2f}', (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

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
    # parser.add_argument('--path_in', default='/media/manu/kingstoo/tmp/20230605-10.20.166.45.mp4', type=str)
    # parser.add_argument('--path_in', default='/media/manu/kingstoo/tmp/20230605-10.20.164.49.mp4', type=str)
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
    parser.add_argument('--ext_info', default=True, action='store_true')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
