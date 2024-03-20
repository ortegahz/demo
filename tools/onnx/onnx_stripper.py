import argparse
import logging

import onnx


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_model_in',
                        default='/home/manu/workspace/sca/yolo/model/yolov5s_fire_detection_AIcamera_20240315.onnx')
    parser.add_argument('--path_model_out',
                        default='/home/manu/workspace/sca/yolo/model/yolov5s_fire_detection_AIcamera_20240315_strip.onnx')
    parser.add_argument('--node_pick', default='/model.0/conv/Conv_output_0')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)

    onnx.utils.extract_model(args.path_model_in, args.path_model_out, ['image'], [args.node_pick])


if __name__ == '__main__':
    main()
