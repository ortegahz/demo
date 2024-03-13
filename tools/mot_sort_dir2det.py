import argparse
import glob
import logging
import os
import sys

import cv2
from tqdm import tqdm

sys.path.append('./')
from utils.utils_funcs import set_logging

sys.path.append('/media/manu/kingstop/workspace/YOLOv6')
from yolov6.core.inferer import Inferer


def run(args):
    inferer_head = Inferer(args.path_in_mp4, False, 0, args.weights_head, 0, args.yaml_head, args.img_size, False)

    paths_img = glob.glob(os.path.join(args.dir_in_root, '*'))
    paths_img.sort()

    if os.path.exists(args.path_det):
        os.remove(args.path_det)

    for path_img in tqdm(paths_img):
        frame = cv2.imread(path_img)
        name, _ = os.path.basename(path_img).split('.')

        det_head = inferer_head.infer_custom(frame, 0.4, 0.45, None, False, 1000)

        if args.show:
            for *xyxy, conf, cls in reversed(det_head):
                label = f'{conf:.2f}'
                inferer_head.plot_box_and_label(frame, max(round(sum(frame.shape) / 2 * 0.003), 2), xyxy, label,
                                                color=(0, 255, 0))
            cv2.imshow('results', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if len(det_head) > 0:
            with open(args.path_det, 'a') as f:
                for *xyxy, conf, cls in det_head:
                    x = xyxy[0]
                    y = xyxy[1]
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    f.writelines(f'{name}, -1, {x}, {y}, {w}, {h}, {conf}, -1, -1, -1\n')

    if args.show:
        cv2.destroyAllWindows()


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in_mp4', default='/media/manu/kingstoo/tmp/20230605-10.20.164.67.mp4', type=str)  # TODO
    parser.add_argument('--dir_in_root', default='/media/manu/intem/sort/2DMOT2015/train/ETH-Pedcross2/img1_pg',
                        type=str)
    parser.add_argument('--path_det', default='/home/manu/tmp/det.txt', type=str)
    parser.add_argument('--yaml_head', default='/media/manu/kingstop/workspace/YOLOv6/data/head.yaml', type=str)
    parser.add_argument('--weights_head', default='/home/manu/tmp/nn6_ft_b32_nab_s1280_dv1r/weights/best_ckpt.pt',
                        type=str)
    parser.add_argument('--img_size', nargs='+', type=int, default=[1280, 1280])
    parser.add_argument('--hide_labels', default=True, action='store_true', help='hide labels.')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences.')
    parser.add_argument('--show', default=False, type=bool)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
