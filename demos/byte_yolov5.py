# author: zerg

# libs
import multiprocessing as mp
from multiprocessing import Process, Manager
from sklearn import preprocessing
import numpy as np
import cv2
import process
import torch
import copy

import glob
import os
import sys
import pickle

import sort

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.decoder import process_decoder
from utils.displayer import process_displayer
from utils.recognizer import process_recognizer
from utils.recognizer import letterbox

sys.path.append('/media/manu/kingstop/workspace/ByteTrack')
from yolox.tracker.byte_tracker import BYTETracker

# params
num_skip = 1  # for speed reason
# name_window = 'frame'
# path_video = '/media/manu/samsung/videos/siamrpn/20200701.mp4'
path_video = '/media/manu/samsung/videos/siamrpn/20200707.mp4'
# path_video = '/media/manu/samsung/videos/at2021/mp4/Video1.mp4'
# path_video = '/media/manu/samsung/videos/at2021/mp4/Video1年级.mp4'
# path_video = 'rtsp://192.168.3.233:554/live/ch2'
# path_video = 'rtsp://192.168.3.51:554/ch2'

max_track_num = 30
buff_len = 30

# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
# v_writer = cv2.VideoWriter('/home/manu/tmp/test.mp4', fourcc, 20.0, (1280, 720))


if __name__ == '__main__':
    weights = ['/home/manu/tmp/yolov5s_e300_ceil_relua_rfocus_synbn_weightse300/weights/best.pt', ]
    device = torch.device('cuda:0')
    conf_thres = 0.5
    iou_thres = 0.5
    classes = None
    agnostic_nms = False
    half = True
    imgsz = 416

    print('tracker init start ...')
    sort_colours = np.random.rand(max_track_num, 3) * 255

    with open('/home/manu/tmp/args_tracker.pickle', 'rb') as f:
        args = pickle.load(f)

    tracker = BYTETracker(args, frame_rate=30)
    print('tracker init done')

    print('detect init start ...')
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    print('detect init done')

    # cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(name_window, 960, 540)
    face_recog_aligned_save_idx = 0

    mp.set_start_method('forkserver')

    with Manager() as manager:

        arr_frames = manager.list()
        dict_tracker_res = manager.dict()
        dict_recog_res = manager.dict()

        p_decoder = Process(target=process_decoder, args=(path_video, buff_len, arr_frames), daemon=True)
        p_decoder.start()

        p_displayer = Process(target=process_displayer,
                              args=(max_track_num, buff_len, arr_frames, dict_tracker_res, dict_recog_res),
                              daemon=True)
        p_displayer.start()

        p_displayer = Process(target=process_recognizer, args=(buff_len, arr_frames, dict_recog_res), daemon=True)
        p_displayer.start()

        item_frame = []
        idx_frame_last = 0
        while True:
            if len(arr_frames) < 1:
                continue
            item_frame = arr_frames[-1]
            idx_frame, frame = item_frame
            if idx_frame_last == idx_frame:
                continue
            idx_frame_last = idx_frame
            print('processing frame -> %d' % idx_frame)
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

            if det is not None:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                det = det[:, 0:5].detach().cpu().numpy()
            else:
                det = np.empty((0, 5))
            online_targets = tracker.update(det, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])

            if len(online_targets) > 0:
                dict_tracker_res[idx_frame] = online_targets
