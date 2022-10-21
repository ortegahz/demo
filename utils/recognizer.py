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
import time
import pickle
import math

import sort

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

# import hpe.lib.models.pose_resnet
sys.path.append('/media/manu/kingstop/workspace/human-pose-estimation.pytorch/lib/models')
import pose_resnet

sys.path.append('/media/manu/kingstop/workspace/demo/hpe/lib/utils')
import transforms as transforms_hpe


def box2cs(box, aspect_ratio, pixel_std):
    def xywh2cs(x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array(
            [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    x, y, w, h = box[:4]
    return xywh2cs(x, y, w, h)


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                     hm[py + 1][px] - hm[py - 1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transforms_hpe.transform_preds(coords[i], center[i], scale[i],
                                                  [heatmap_width, heatmap_height])

    return preds, maxvals


def get_final_preds_local(config, batch_heatmaps):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                     hm[py + 1][px] - hm[py - 1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    preds = preds[:] * 4

    return preds, maxvals


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


def process_recognizer(buff_len, arr_frames, dict_tracker_res):
    weights = ['/home/manu/tmp/yolov5s_e300_ceil_relua_rfocus_synbn/weights/best.pt', ]
    device = torch.device('cuda:1')
    conf_thres = 0.5
    iou_thres = 0.5
    classes = None
    agnostic_nms = False
    half = True
    imgsz = 416

    print('[RECOGNIZER] detect init start ...')
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    print('[RECOGNIZER] detect init done')

    print('[RECOGNIZER] hpe init start ...')
    with open('/home/manu/tmp/args.pickle', 'rb') as f:
        args = pickle.load(f)
    with open('/home/manu/tmp/config.pickle', 'rb') as f:
        config = pickle.load(f)

    pixel_std = args.pixel_std
    th_pt = args.pt_th

    image_width, image_height = config.MODEL.IMAGE_SIZE
    aspect_ratio = image_width * 1.0 / image_height
    num_joints = config.MODEL.NUM_JOINTS

    colours = np.random.rand(num_joints, 3) * 255

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model_hpe = eval('pose_resnet.get_pose_net')(
        config, is_train=False
    )

    model_hpe.load_state_dict(torch.load(config.TEST.MODEL_FILE))

    # gpus = [1]
    # model_hpe = torch.nn.DataParallel(model_hpe, device_ids=gpus).cuda()
    model_hpe = model_hpe.cuda(device)
    print('[RECOGNIZER] hpe init done ...')

    while True:
        if len(arr_frames) < 1:
            continue

        item_frame = arr_frames[-1]
        # item_frame = copy.deepcopy(arr_frames[-1])
        idx_frame, frame = item_frame
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
            det = det[det[:, -1] == 0]  # pick certain class
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            det = det[:, 0:5].detach().cpu().numpy()

            # dict_tracker_res[idx_frame] = det

            hpe_res = []
            hpe_det = copy.deepcopy(det)
            for sdet in hpe_det:
                image_file = args.image_file
                box = sdet[:4]
                box[2:] = box[2:] - box[:2]
                score = sdet[-1]

                kpt_db = []
                center, scale = box2cs(box, aspect_ratio, pixel_std)
                joints_3d = np.zeros((num_joints, 3), dtype=np.float)
                joints_3d_vis = np.ones(
                    (num_joints, 3), dtype=np.float)
                kpt_db.append({
                    'image': image_file,
                    'center': center,
                    'scale': scale,
                    'score': score,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                })

                data_numpy = frame

                db_rec = kpt_db[0]
                joints = db_rec['joints_3d']
                joints_vis = db_rec['joints_3d_vis']

                c = db_rec['center']
                s = db_rec['scale']
                score = db_rec['score'] if 'score' in db_rec else 1
                r = 0

                trans = transforms_hpe.get_affine_transform(c, s, r, [image_width, image_height])
                input = cv2.warpAffine(
                    data_numpy,
                    trans,
                    (int(image_width), int(image_height)),
                    flags=cv2.INTER_LINEAR)

                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])

                input = transform(input)
                input = input.unsqueeze(0)
                input = input.to(device)

                model_hpe.eval()
                output = model_hpe(input)
                # output_save = output.detach().cpu().numpy()
                # np.savetxt(os.path.join('/home/manu/tmp', 'output_save_infer.txt'), output_save.flatten(), fmt="%f", delimiter="\n")
                # pred, _ = get_max_preds(output)

                c = np.expand_dims(c, 0)
                s = np.expand_dims(s, 0)
                preds, maxvals = get_final_preds(
                    config, output.detach().clone().cpu().numpy(), c, s)

                hpe_res.append((preds, maxvals))
            dict_tracker_res[idx_frame] = (det, hpe_res)

        # time.sleep(0.5)
