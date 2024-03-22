import argparse
import logging

import numpy as np
import onnxruntime as rt
from PIL import Image
import cv2


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_model_in',
                        default='/home/manu/workspace/sca/yolo/model/yolov5s_fire_detection_AIcamera_20240315.onnx')
    parser.add_argument('--path_img_in', default='/home/manu/tmp/snapshot/1.bmp')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)

    sess = rt.InferenceSession(args.path_model_in)
    image_array = cv2.imread(args.path_img_in).astype('float32')
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_array /= 255.0
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    input_data = np.expand_dims(image_array, axis=0)
    # input_data = np.resize(input_data, input_shape)
    # permute input data
    input_data = np.transpose(input_data, (0, 3, 1, 2))

    result = sess.run(None, {input_name: input_data})
    for save_i in range(len(result)):
        save_output = result[save_i].flatten()
        np.savetxt('/home/manu/tmp/onnx_output_%s.txt' % save_i, save_output,
                   fmt="%f", delimiter="\n")


if __name__ == '__main__':
    main()
