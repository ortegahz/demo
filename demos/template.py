import argparse
import logging

from utils.utils_funcs import set_logging


def run(**args):
    logging.info(args)


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(**vars(args))


if __name__ == '__main__':
    main()
