# author: zerg

import pickle
import sys
# libs
from multiprocessing import Process, Queue

import cv2
import numpy as np
import torch

import process
from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import time_synchronized

sys.path.append('/media/manu/kingstop/workspace/ByteTrack')
from yolox.tracker.byte_tracker import BYTETracker

# params
num_skip = 1  # for speed reason
name_window = 'frame'
path_video = '/media/manu/samsung/videos/counter/9a903c802240a8ce281bb2f5fa82a4d4.mp4'
# path_video = '/media/manu/samsung/videos/siamrpn/20200701.mp4'
# path_video = '/media/manu/samsung/videos/siamrpn/20200707.mp4'
# path_video = '/media/manu/samsung/videos/at2021/mp4/Video1.mp4'
# path_video = '/media/manu/samsung/videos/at2021/mp4/Video1年级.mp4'
# path_video = 'rtsp://192.168.3.233:554/live/ch2'
# path_video = 'rtsp://192.168.3.51:554/ch2'

weights = ['/home/manu/tmp/yolov5s_e300_ceil_relua_rfocus_synbn_weightse300/weights/best.pt', ]
device = torch.device('cuda:0')
conf_thres = 0.5
iou_thres = 0.5
classes = None
agnostic_nms = False
half = True
imgsz = 416

max_track_num = 30

line_mark = [1000, 100]

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
v_writer = cv2.VideoWriter('/home/manu/tmp/test.mp4', fourcc, 20.0, (1920, 1080))


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

    q_decoder = Queue()
    p_decoder = Process(target=process.process_decoder, args=(q_decoder, path_video, num_skip), daemon=True)
    p_decoder.start()

    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 540)
    face_recog_aligned_save_idx = 0
    tids_in = list()
    tids_out = list()
    cnt = 0
    while True:
        item_frame = q_decoder.get()
        frame = item_frame[0]
        h, w, _ = frame.shape
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

        cv2.line(frame, (line_mark[0], h), (w, line_mark[1]), (255, 0, 255), 5)

        info = f'CNT {cnt}'
        cv2.putText(frame, info, (1500, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        if len(online_targets) > 0:
            for track in online_targets:
                tlwh = track.tlwh
                tid = track.track_id
                tlbr = [tlwh[0], tlwh[1], tlwh[2] + tlwh[0], tlwh[3] + tlwh[1]]
                bbox = np.concatenate((tlbr, [tid])).reshape(1, -1)
                bbox = np.squeeze(bbox)
                box = bbox.astype(int)
                age = track.end_frame - track.start_frame

                # print(track._tlwh[:2])
                # print(tlwh[:2])

                org_up_down = (track._tlwh[0] - line_mark[1]) / (h - line_mark[1]) - \
                              (track._tlwh[1] - w) / (line_mark[0] - w)
                cur_up_down = (tlwh[0] - line_mark[1]) / (h - line_mark[1]) - (tlwh[1] - w) / (line_mark[0] - w)

                # if tid == 1:
                #     print(f'cur_up_down --> {cur_up_down}')

                # print(tids_in)
                # print(tids_out)

                if age > 30 and org_up_down > 0 and cur_up_down < 0 and tid not in tids_in:
                    track._tlwh[0] = tlwh[0]
                    track._tlwh[1] = tlwh[1]
                    cnt += 1
                    tids_in.append(tid)

                if age > 30 and org_up_down < 0 and cur_up_down > 0 and tid not in tids_out:
                    track._tlwh[0] = tlwh[0]
                    track._tlwh[1] = tlwh[1]
                    cnt -= 1
                    tids_out.append(tid)

                # print(f'cnt --> {cnt}')

                # print(tois.keys())

                # print(age)
                # print('score', faces[i][4])
                # track = mot_tracker.trackers[len(faces) - i - 1]  # reversed order
                # box = faces[i].astype(int)
                sort_id = box[4].astype(np.int32)
                color_id = sort_colours[sort_id % max_track_num, :]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color_id, 2)
                info = 't' + '%d' % sort_id
                fontScale = 1.2
                cv2.putText(frame, info,
                            (box[0], box[1] + int(fontScale * 25)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color_id, 2)
                info = 'a' + '%d' % age
                fontScale = 1.2
                cv2.putText(frame, info,
                            (box[0], box[1] + int(fontScale * 25 * 2)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color_id,
                            2)
                cv2.line(frame, (int(track._tlwh[0]), int(track._tlwh[1])),
                         (int(tlwh[0]), int(tlwh[1])), (255, 0, 0), 5)

        cv2.imshow(name_window, frame)
        v_writer.write(frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # print(f'cnt --> {cnt}')

    cv2.destroyAllWindows()
    # v_writer.release()
