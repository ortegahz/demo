import argparse
import logging
import cv2

from utils.logging import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_img_in',
        default='/media/manu/kingstoo/AffectNet/full_res/val/Happiness/0e03a26950a7b2166217c082d841aa446a37b3e6a078f1428c11082c.jpg',
        type=str)
    parser.add_argument(
        '--path_img_out',
        default='/home/manu/tmp/0e03a26950a7b2166217c082d841aa446a37b3e6a078f1428c11082c.bmp',
        type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    set_logging(name='tool')
    logging.info(args)

    img = cv2.imread(args.path_img_in)
    cv2.imwrite(args.path_img_out, img)


if __name__ == '__main__':
    main()
