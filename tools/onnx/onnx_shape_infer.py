import argparse
import logging

import onnx


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_model_in', default='/home/manu/tmp/model_bn.onnx')
    parser.add_argument('--path_model_out', default='/home/manu/tmp/model_bn_shape.onnx')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)

    onnx_graph = onnx.load(args.path_model_in)
    estimated_graph = onnx.shape_inference.infer_shapes(onnx_graph)
    onnx.save(estimated_graph, args.path_model_out)


if __name__ == '__main__':
    main()
