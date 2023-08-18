import argparse
import logging

from utils.logging import set_logging


def run(args):
    with open(args.path_results_in, 'r') as f:
        lines = f.readlines()
    is_pass = True
    for line in lines:
        value = float(line.split()[-1].split('=')[-1])
        if value < args.threshold:
            logging.info(f'abnormal value --> {value} in {line}')
            is_pass = False
    if is_pass:
        logging.info(f'test pass !!!')
    else:
        logging.info(f'test fail !!!')


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_results_in',
                        default='/media/manu/kingstop/workspace/rknn-toolkit/examples/onnx/acfree/snapshot/individual_qnt_error_analysis.txt',
                        type=str)
    parser.add_argument('--threshold', default=0.99, type=float)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
