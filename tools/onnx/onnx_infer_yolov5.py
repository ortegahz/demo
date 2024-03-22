import argparse
import glob
import logging
import os

import onnxruntime as rt

from tools.general import set_logging, make_dirs
from tools.onnx.yolov5_parser import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_model_in',
                        default='/home/manu/workspace/sca/yolo/model/yolov5s_fire_detection_AIcamera_20240315.onnx')
    parser.add_argument('--dir_img_in', default='/home/manu/tmp/undistorted_image_single')
    parser.add_argument('--dir_out_tp', default='/home/manu/tmp/undistorted_image_single_tp')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    make_dirs(args.dir_out_tp, reset=True)
    sess = rt.InferenceSession(args.path_model_in)
    paths_in = glob.glob(os.path.join(args.dir_img_in, '*'))
    for path_in in paths_in:
        img_name = os.path.basename(path_in)
        image_array = cv2.imread(path_in).astype('float32')
        image_array, _, _ = letterbox(image_array, new_shape=(IMG_SIZE_H, IMG_SIZE_W))
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_array /= 255.0
        input_name = sess.get_inputs()[0].name
        input_data = np.expand_dims(image_array, axis=0)
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        result = sess.run(None, {input_name: input_data})
        # for save_i in range(len(result)):
        #     save_output = result[save_i].flatten()
        #     np.savetxt('/home/manu/tmp/onnx_output_%s.txt' % save_i, save_output,
        #                fmt="%f", delimiter="\n")

        input0_data = result[0]
        input1_data = result[1]
        input2_data = result[2]

        input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3, -1] + list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3, -1] + list(input2_data.shape[-2:]))

        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

        boxes, classes, scores = yolov5_post_process(input_data)

        img_1 = cv2.imread(path_in)
        if boxes is not None:
            draw(img_1, boxes, scores, classes)
            cv2.imwrite(os.path.join(args.dir_out_tp, img_name), img_1)


if __name__ == '__main__':
    main()
