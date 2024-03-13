import argparse
import logging
import torch
import sys

from utils.utils_funcs import set_logging

sys.path.append('/home/manu/workspace/YOLOv6')


def run(args):
    ckpt = torch.load(args.weights_phone, map_location='cpu')  # load
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    torch.save(model.state_dict(), args.weights_phone_out)

    ckpt = torch.load(args.weights_play, map_location='cpu')  # load
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    torch.save(model.state_dict(), args.weights_play_out)


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_phone', default='/home/manu/tmp/exp12/weights/best_ckpt.pt', type=str)
    parser.add_argument('--weights_phone_out', default='/home/manu/tmp/aux.pt', type=str)
    parser.add_argument('--weights_play', default='/home/manu/tmp/nn6_ft_b64_nab_s1280_dl/weights/best_ckpt.pt',
                        type=str)
    parser.add_argument('--weights_play_out', default='/home/manu/tmp/main.pt', type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
