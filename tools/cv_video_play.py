import cv2
import logging
import argparse


def set_logging():
    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser('video player')
    parser.add_argument('--path_in_video', default='/home/manu/视频/vlc-record-2023-11-29-15h24m48s-rtsp___172.20.20.181_ir_stream-.mp4')
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
    if not cap.isOpened():
        logging.error("error opening video steam")
    cap.release()


if __name__ == '__main__':
    main()
