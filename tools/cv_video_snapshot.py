import argparse
import logging

import cv2

from tools.cv_im_resize_pad import letterbox


def set_logging():
    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser('video player')
    parser.add_argument('--path_in_video', default='/home/manu/tmp/fire (219).mp4')
    parser.add_argument('--img_size', nargs='+', type=int, default=[540, 960], help='new shape h x w')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)

    cap = cv2.VideoCapture(args.path_in_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sizes = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    logging.info(f'fps: {fps} size: {sizes} n_frames: {n_frames}')

    cnt, idx = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.error(f"Can't receive frame (stream end?). Exiting ...")
            break
        cnt += 1

        if not cnt % 10 == 0:
            continue

        if len(args.img_size) > 0:
            frame, _, _ = letterbox(frame, new_shape=(args.img_size[0], args.img_size[1]))

        cv2.imwrite(f"/home/manu/tmp/snapshot/{idx}.bmp", frame)
        idx += 1

    cap.release()


if __name__ == '__main__':
    main()
