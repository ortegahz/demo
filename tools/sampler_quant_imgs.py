import argparse
import glob
import logging
import os
import random
import shutil

from tqdm import tqdm
from utils.utils_funcs import set_logging


def make_dirs(args):
    if os.path.exists(args.save_dir_root):
        shutil.rmtree(args.save_dir_root)
    os.makedirs(args.save_dir_root)


def run(args):
    paths_img_in = glob.glob(os.path.join(args.dir_in_root, '*.jpg'))
    logging.info(paths_img_in[:10])
    random.shuffle(paths_img_in)
    logging.info(paths_img_in[:10])
    cnt = 0
    for i, path_img_in in enumerate(tqdm(paths_img_in)):
        if cnt >= args.sample_number:
            break
        path_img_out = os.path.join(args.save_dir_root, os.path.split(path_img_in)[-1])
        shutil.copyfile(path_img_in, path_img_out)
        cnt += 1



def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in_root', default='/media/sdb/data/custom_head_v2r/images/train', type=str)
    parser.add_argument('--save_dir_root', default='/media/sdb/data/imgs_quat', type=str)
    parser.add_argument('--sample_number', default=512, type=int)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    make_dirs(args)
    run(args)


if __name__ == '__main__':
    main()
