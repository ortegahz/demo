import argparse
import logging
import cv2

from utils.logging import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_img_in',
        default='/media/manu/samsung/pics/2.jpg',
        type=str)
    parser.add_argument(
        '--path_img_out',
        default='/media/manu/samsung/pics/2_416x416.bmp',
        type=str)
    parser.add_argument('--img_size', nargs='+', type=int, default=[416, 416])
    return parser.parse_args()


def main():
    args = parse_args()
    set_logging()
    logging.info(args)

    img = cv2.imread(args.path_img_in)
    if len(args.img_size) > 0:
        img = cv2.resize(img, (args.img_size[0], args.img_size[1]))
    cv2.imwrite(args.path_img_out, img)


if __name__ == '__main__':
    main()
