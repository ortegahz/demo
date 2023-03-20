# author: zerg

# libs
from PIL import Image, ImageDraw, ImageFont
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
import subprocess

import track

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

# params
# box_act = [0, 0, 1920, 1080]  # x1, y1, x2, y2
box_act = [0 + 10, 10, 1920 - 10, 500]  # x1, y1, x2, y2
# box_act = [0 + 10, 200, 1152 - 10, 350]  # x1, y1, x2, y2
num_skip = 1  # for speed reason
name_window = 'frame'
# path_video = '/media/manu/samsung/videos/siamrpn/20200701.mp4'
# path_video = 'rtsp://192.168.3.25:554/ch0_1'
path_video = '/media/manu/samsung/videos/at2021/mp4/Video1.mp4'
# path_video = '/media/manu/samsung/videos/at2021/mp4/Video1年级.mp4'
# path_video = 'rtsp://192.168.3.233:554/live/ch2'

path_out_rtmp = 'rtmp://localhost:1935/live/ch0'

weights = ['/home/manu/tmp/yolov5s_e300_ceil_relua_rfocus_synbn_weightse300/weights/best.pt', ]
# weights_bhv = ['/home/manu/tmp/yolov5s_e300_ceil_relua_rfocus_synbn_weights-e300-behavior/weights/best.pt', ]
# weights = ['/home/manu/tmp/yolov5s_default-_coco/weights/best.pt', ]
device = torch.device('cuda:0')
conf_thres = 0.5
iou_thres = 0.5
classes = None
agnostic_nms = False
half = True
imgsz = 640

weights_bhv = ['/home/manu/tmp/yolov5s_e300_ceil_relua_rfocus_synbn_weights-e300-behavior/weights/best.pt', ]
conf_thres_bhv = 0.5
imgsz_bhv = 640
is_bhv = False
num_skip_bhv = 10

track_max_age = 1
track_min_hits = 3
track_iou_threshold = 0.3
track_max_track_num = 32


def show_english(img, text, pos):
    cv2.putText(img, text, (pos[0], pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)


def show_chinese(img, text, pos):
    """
    :param img: opencv 图片
    :param text: 显示的中文字体
    :param pos: 显示位置
    :return:    带有字体的显示图片（包含中文）
    """
    # img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_pil = Image.fromarray(img)
    font = ImageFont.truetype(font='NotoSansCJK-Black.ttc', size=30)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=(255, 0, 0))
    img = np.array(img_pil)  # PIL图片转换为numpy
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def overlap_master(bm, bs):
    # box -> x1, y1, x2, y2
    bm = bm.astype('float32')
    bs = bs.astype('float32')
    sm = (bm[3] - bm[1]) * (bm[2] - bm[0])
    x1 = max(bm[0], bs[0])
    y1 = max(bm[1], bs[1])
    x2 = min(bm[2], bs[2])
    y2 = min(bm[3], bs[3])
    h = (y2 - y1) if (y2 - y1) > 0 else 0
    w = (x2 - x1) if (x2 - x1) > 0 else 0
    so = h * w
    score = so / sm
    return score


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
    track_colours = np.random.rand(track_max_track_num, 3) * 255
    mot_tracker = track.Track(max_age=track_max_age,
                              min_hits=track_min_hits,
                              iou_threshold=track_iou_threshold)  # create instance of the track tracker
    print('tracker init done')

    print('detect init start ...')
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    print('detect init done')

    print('detect bhv init start ...')
    # Load bhv model
    model_bhv = attempt_load(weights_bhv, map_location=device)  # load FP32 model
    imgsz_bhv = check_img_size(imgsz_bhv, s=model.stride.max())  # check img_size
    if half:
        model_bhv.half()  # to FP16
    print('detect init done')

    q_decoder = Queue()
    p_decoder = Process(target=process.process_decoder, args=(q_decoder, path_video, num_skip), daemon=True)
    p_decoder.start()

    # cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(name_window, 960, 540)
    cv2.namedWindow(name_window, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(name_window, 0, 0)
    cv2.setWindowProperty(name_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    face_recog_aligned_save_idx = 0
    while True:
        item_frame = q_decoder.get()
        frame, idx_frame = item_frame

        # sizeStr = str(frame.shape[1]) + 'x' + str(frame.shape[0])
        # command = ['ffmpeg',
        #            '-y', '-an',
        #            '-f', 'rawvideo',
        #            '-vcodec', 'rawvideo',
        #            '-pix_fmt', 'bgr24',
        #            '-s', sizeStr,
        #            '-r', '25',
        #            '-i', '-',
        #            '-c:v', 'libx264',
        #            '-pix_fmt', 'yuv420p',
        #            '-preset', 'ultrafast',
        #            '-f', 'flv',
        #            path_out_rtmp]
        # pipe = subprocess.Popen(command
        #                         , shell=False
        #                         , stdin=subprocess.PIPE
        #                         )

        img = letterbox(frame, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        det = pred[0]

        if det is not None:
            det = det[det[:, -1] == 0]  # pick certain class
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            det = det[:, 0:5].detach().cpu().numpy()

            det = det[
                (det[:, 0] > box_act[0]) &
                (det[:, 1] > box_act[1]) &
                (det[:, 2] < box_act[2]) &
                (det[:, 3] < box_act[3])
                ]

        if det is None:
            det = np.empty((0, 5))

        # run bhv model
        is_bhv = False
        if idx_frame % num_skip_bhv == 0:
            pred_bhv = model_bhv(img, augment=False)[0]
            pred_bhv = non_max_suppression(pred_bhv, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            det_bhv = pred_bhv[0]

            if det_bhv is not None:
                det_bhv[:, :4] = scale_coords(img.shape[2:], det_bhv[:, :4], frame.shape).round()
                det_bhv = det_bhv.detach().cpu().numpy()

                # det_bhv = det_bhv[
                #     (det_bhv[:, 0] > box_act[0]) &
                #     (det_bhv[:, 1] > box_act[1]) &
                #     (det_bhv[:, 2] < box_act[2]) &
                #     (det_bhv[:, 3] < box_act[3])
                #     ]

                if det_bhv is not None:
                    is_bhv = True

                    # # plot
                    # for d in det_bhv:
                    #     bbox = d[0:4]
                    #     box = bbox.astype(int)
                    #     color_id = track_colours[d[5].astype(int), :]
                    #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color_id, 2)

        cv2.rectangle(frame, (box_act[0], box_act[1]), (box_act[2], box_act[3]), (0, 255, 0), thickness=5)

        mot_tracker.update(det)

        # print('num of trackers --> %d' % len(mot_tracker.trackers))

        # if len(mot_tracker.trackers) > 0:
        #     # plot
        #     for track in mot_tracker.trackers:
        #         d = track.get_state()[0]
        #         bbox = np.concatenate((d, [track.id + 1])).reshape(1, -1)
        #         bbox = np.squeeze(bbox)
        #         box = bbox.astype(int)
        #         # print('score', faces[i][4])
        #         # track = mot_tracker.trackers[len(faces) - i - 1]  # reversed order
        #         # box = faces[i].astype(int)
        #         track_id = box[4].astype(np.int32)
        #         color_id = track_colours[track_id % track_max_track_num, :]
        #         cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color_id, 2)
        #         info = 'tid' + ' %d' % track_id
        #         fontScale = 1.2
        #         cv2.putText(frame, info,
        #                     (box[0], box[1] + int(fontScale * 25)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color_id, 2)
        #         info = 'age' + ' %d' % track.age
        #         fontScale = 1.2
        #         cv2.putText(frame, info,
        #                     (box[0], box[1] + int(fontScale * 25 * 2)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color_id,
        #                     2)
        #         info = 'debug' + ' %d' % track.traces.qsize()
        #         fontScale = 1.2
        #         cv2.putText(frame, info,
        #                     (box[0], box[1] + int(fontScale * 25 * 3)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color_id,
        #                     2)
        #         if track.trace_c is not None and track.trace_l is not None:
        #             cv2.line(frame, (int(track.trace_c[0][0]), int(track.trace_c[0][1])),
        #                      (int(track.trace_l[0][0]), int(track.trace_l[0][1])), color_id, 2)


        # analyse
        if len(mot_tracker.trackers) < 1:
            info = 'XunShi'
            cv2.putText(frame, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            # frame = show_chinese(frame, info, (0, 0))
        elif len(mot_tracker.trackers) > 1:
            info = 'ShiShengHuDong'
            cv2.putText(frame, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            # frame = show_chinese(frame, info, (0, 0))
        else:
            track = mot_tracker.trackers[0]
            bbox = track.get_state()[0]
            bbox = bbox.astype(int)
            if track.trace_c is not None and track.trace_l is not None and \
                    abs(track.trace_c[0][0] - track.trace_l[0][0]) > (bbox[2] - bbox[0]) * 2:
                if track.trace_c[0][0] - track.trace_l[0][0] < 0:
                    info = 'XiangYouZouDong'
                    cv2.putText(frame, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    # frame = show_chinese(frame, info, (0, 0))
                else:
                    info = 'XiangZuoZouDong'
                    cv2.putText(frame, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    # frame = show_chinese(frame, info, (0, 0))
            elif is_bhv:
                # ['stand', 'lookback', 'handsup', 'overdesk']
                for bbox_bhv in det_bhv:
                    ol_s = overlap_master(bbox, bbox_bhv[:4])
                    if ol_s > 0.7:
                        # info = 'debug' + ' %s' % ol_s
                        # fontScale = 1.2
                        # cv2.putText(frame, info, (bbox[0], bbox[1] + int(fontScale * 25 * 3)),
                        #             cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 0, 0), 2)
                        if bbox_bhv[5] == 2:
                            info = 'TiWen'
                            cv2.putText(frame, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                            # frame = show_chinese(frame, info, (0, 0))
                        if bbox_bhv[5] == 1:
                            info = 'BanShu'
                            cv2.putText(frame, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                            # frame = show_chinese(frame, info, (0, 0))

        cv2.imshow(name_window, frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # pipe.stdin.write(frame.tostring())

    cv2.destroyAllWindows()
