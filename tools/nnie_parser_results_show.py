import argparse
import logging

import cv2

from utils.logging import set_logging


def run(args):
    img = cv2.imread(args.path_in_img)
    with open(args.path_in_txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        logging.info(line)
        line_lst = line.split()
        _, bbox, kps, conf = line_lst[0], line_lst[1:5], line_lst[5:-1], line_lst[-1]

        left = int((float(bbox[0]) - float(bbox[2]) / 2) * args.model_sz)
        top = int((float(bbox[1]) - float(bbox[3]) / 2) * args.model_sz)
        right = int((float(bbox[0]) + float(bbox[2]) / 2) * args.model_sz)
        bottom = int((float(bbox[1]) + float(bbox[3]) / 2) * args.model_sz)

        color = (0, 255, 0)
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(img, conf,
                    (left, top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        for i in range(5):
            cv2.circle(img, (int(kps[2 * i]), int(kps[2 * i + 1])), 1, colors[i], -1)

    # show output
    cv2.imshow("parser result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('/home/manu/tmp/parser_result.jpg', img)


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in_img',
                        default='/home/manu/nfs/mpp/sample/svp/nnie/data/nnie_image/rgb_planar/students_lt.bmp',
                        type=str)
    # parser.add_argument('--path_in_txt',
    #                     default='/media/manu/kingstop/workspace/YOLOv6/runs/inference/yolov6n/labels/students_lt.txt',
    #                     type=str)
    parser.add_argument('--path_in_txt',
                        default='/home/manu/nfs/mpp/sample/svp/nnie/results_ruyi.txt',
                        type=str)
    parser.add_argument('--model_sz', default=640)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
