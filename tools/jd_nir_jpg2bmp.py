import argparse
import glob
import logging
import os

import cv2

from utils.utils_funcs import set_logging, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', default='/home/manu/workspace/sca/yolo/model/quan')
    parser.add_argument('--dir_out', default='/home/manu/workspace/sca/yolo/model/quan_bmp')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    make_dirs(args.dir_out)
    paths_in = glob.glob(os.path.join(args.dir_in, '*'))
    for path_in in paths_in:
        file_name = os.path.basename(path_in).split('_')[-1]
        file_idx = int(file_name.split('.')[0])
        logging.info(file_idx)
        path_out = os.path.join(args.dir_out, str(file_idx) + '.bmp')
        logging.info(path_out)
        img = cv2.imread(path_in)
        cv2.imwrite(path_out, img)


if __name__ == '__main__':
    main()
