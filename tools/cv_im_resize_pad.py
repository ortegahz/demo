import argparse
import logging

import cv2

from utils.logging import set_logging


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_img_in',
        default='/media/manu/samsung/pics/player.bmp',
        type=str)
    parser.add_argument(
        '--path_img_out',
        default='/media/manu/samsung/pics/player_1280.bmp',
        type=str)
    parser.add_argument('--img_size', nargs='+', type=int, default=[1280, 1280], help='new shape h x w')
    return parser.parse_args()


def main():
    args = parse_args()
    set_logging()
    logging.info(args)

    img = cv2.imread(args.path_img_in)
    if len(args.img_size) > 0:
        # img = cv2.resize(img, (args.img_size[0], args.img_size[1]))
        img, _, _ = letterbox(img, new_shape=(args.img_size[0], args.img_size[1]))
    cv2.imwrite(args.path_img_out, img)


if __name__ == '__main__':
    main()
