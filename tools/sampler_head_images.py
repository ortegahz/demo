import argparse
import logging
import shutil
import copy
import time
import glob
import cv2
import sys
import os

from tqdm import tqdm

sys.path.append('./')
from utils.logging import set_logging

sys.path.append('/media/manu/kingstop/workspace/YOLOv6')
from yolov6.core.inferer import Inferer


def make_dirs(args):
    if os.path.exists(args.save_dir_root) and args.reset:
        shutil.rmtree(args.save_dir_root)
    subset = 'train'  # lazy people
    os.makedirs(os.path.join(args.save_dir_root, 'labels', subset), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir_root, 'images', subset), exist_ok=True)


def run(args):
    inferer_head = Inferer(args.path_in_mp4, False, 0, args.weights_head, 0, args.yaml_head, args.img_size, False)

    make_dirs(args)

    paths_img = glob.glob(os.path.join(args.dir_in_root, '*'))

    for path_img in tqdm(paths_img):
        frame = cv2.imread(path_img)
        frame_org = copy.deepcopy(frame)

        det_head = inferer_head.infer_custom(frame, 0.4, 0.45, None, False, 1000)

        if len(det_head):
            for *xyxy, conf, cls in reversed(det_head):
                label = f'{conf:.2f}'
                inferer_head.plot_box_and_label(frame, max(round(sum(frame.shape) / 2 * 0.003), 2), xyxy, label,
                                                color=(0, 255, 0))

        cv2.imshow('results', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(det_head) > 0:
            subset = 'train'  # lazy people
            name, _ = os.path.basename(path_img).split('.')
            path_img_out = os.path.join(args.save_dir_root, 'images', subset, f'{name}.jpg')
            cv2.imwrite(path_img_out, frame_org)

            h_img, w_img, _ = frame_org.shape

            path_label_out = path_img_out.replace('.jpg', '.txt')
            path_label_out = path_label_out.replace('images', 'labels')
            with open(path_label_out, 'w') as f:
                for *xyxy, conf_play, cls in det_head:
                    cx = (xyxy[0] + xyxy[2]) / 2. / w_img
                    cy = (xyxy[1] + xyxy[3]) / 2. / h_img
                    w = (xyxy[2] - xyxy[0]) / w_img
                    h = (xyxy[3] - xyxy[1]) / h_img
                    f.writelines(f'0 {cx} {cy} {w} {h}\n')

    cv2.destroyAllWindows()


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in_mp4', default='/media/manu/kingstoo/tmp/20230605-10.20.164.67.mp4', type=str)  # TODO
    parser.add_argument('--dir_in_root', default='/media/manu/kingstoo/tmp/sucai', type=str)
    parser.add_argument('--save_dir_root', default='/media/manu/kingstoo/tmp/pics_head_sample', type=str)
    parser.add_argument('--yaml_head', default='/media/manu/kingstop/workspace/YOLOv6/data/head.yaml', type=str)
    parser.add_argument('--weights_head', default='/home/manu/tmp/nn6_ft_b32_nab_s1280/weights/best_ckpt.pt', type=str)
    parser.add_argument('--img_size', nargs='+', type=int, default=[1280, 1280])
    parser.add_argument('--hide_labels', default=True, action='store_true', help='hide labels.')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences.')
    parser.add_argument('--reset', default=True, action='store_true')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
