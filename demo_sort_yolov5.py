# author: zerg

# libs
from multiprocessing import Process, Queue
from sklearn import preprocessing
import numpy as np
import cv2
import process
import torch
import copy
import glob
import os
import sys

import sort

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


# params
num_skip = 1  # for speed reason
name_window = 'frame'
# path_video = '/media/manu/samsung/videos/at2021/mp4/Video1.mp4'
# path_video = '/media/manu/samsung/videos/at2021/mp4/Video1年级.mp4'
# path_video = 'rtsp://192.168.3.233:554/live/ch2'
path_video = 'rtsp://192.168.3.34:554/live/ch2'

weights = ['/home/manu/tmp/yolov5s_e300_ceil_relua_rfocus_synbn_weightse300/weights/best.pt', ]
device = torch.device('cuda:0')
conf_thres = 0.5
iou_thres = 0.5
classes = None
agnostic_nms = False
half = True
imgsz = 416

sort_max_age = 1
sort_min_hits = 3
sort_iou_threshold = 0.3
sort_max_track_num = 32


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


if __name__ == '__main__':
    print('tracker init start ...')
    sort_colours = np.random.rand(sort_max_track_num, 3) * 255
    mot_tracker = sort.Sort(max_age=sort_max_age,
                            min_hits=sort_min_hits,
                            iou_threshold=sort_iou_threshold)  # create instance of the SORT tracker
    print('tracker init done')

    print('detect init start ...')
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    print('detect init done')

    q_decoder = Queue()
    p_decoder = Process(target=process.process_decoder, args=(q_decoder, path_video, num_skip), daemon=True)
    p_decoder.start()

    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 540)
    face_recog_aligned_save_idx = 0
    while True:
        item_frame = q_decoder.get()
        frame = item_frame[0]
        img = letterbox(frame, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

        det = det[:, 0:5].detach().cpu().numpy()
        mot_tracker.update(det)

        if mot_tracker.trackers is not None:

            # plot
            for track in mot_tracker.trackers:
                d = track.get_state()[0]
                bbox = np.concatenate((d, [track.id+1])).reshape(1, -1)
                bbox = np.squeeze(bbox)
                box = bbox.astype(int)
                # print('score', faces[i][4])
                # track = mot_tracker.trackers[len(faces) - i - 1]  # reversed order
                # box = faces[i].astype(int)
                sort_id = box[4].astype(np.int32)
                color_id = sort_colours[sort_id % sort_max_track_num, :]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color_id, 2)
                info = 'tid' + ' %d' % sort_id
                fontScale = 1.2
                cv2.putText(frame, info,
                            (box[0], box[1]+int(fontScale * 25)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color_id, 2)
                info = 'age' + ' %d' % track.age
                fontScale = 1.2
                cv2.putText(frame, info,
                            (box[0], box[1]+int(fontScale * 25 * 2)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color_id, 2)

        cv2.imshow(name_window, frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
