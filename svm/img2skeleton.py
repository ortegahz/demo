import os
import sys
import glob
import cv2
import pickle
import copy
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from utils.recognizer import attempt_load, check_img_size, box2cs, letterbox, get_final_preds, get_final_preds_local
from utils.displayer import plot_hpe

from utils.general import non_max_suppression, scale_coords

from demo_track_yolov5 import overlap_master

# import hpe.lib.models.pose_resnet
sys.path.append('/media/manu/kingstop/workspace/human-pose-estimation.pytorch/lib/models')
import pose_resnet

sys.path.append('/media/manu/kingstop/workspace/demo/hpe/lib/utils')
import transforms as transforms_hpe


def main():
    # general params
    dir_imgs = '/media/manu/kingstoo/yolov5/custom_behavior/images/train2017'
    img_suffix = 'jpg'
    dir_pos = '/media/manu/kingstoo/svm/pos'
    dir_neg = '/media/manu/kingstoo/svm/neg'

    os.system(f'rm {dir_pos} -rvf')
    os.system(f'mkdir {dir_pos}')
    os.system(f'rm {dir_neg} -rvf')
    os.system(f'mkdir {dir_neg}')

    # yolov5 params
    weights = ['/home/manu/tmp/yolov5s_e300_ceil_relua_rfocus_synbn/weights/best.pt', ]
    device = torch.device('cuda:1')
    conf_thres = 0.5
    iou_thres = 0.5
    classes = None
    agnostic_nms = False
    half = True
    imgsz = 1024

    print('detect init start ...')
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    print('detect init done')

    # hpe params
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
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('[RECOGNIZER] hpe init start ...')
    model_hpe = eval('pose_resnet.get_pose_net')(
        config, is_train=False
    )
    model_hpe.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    model_hpe = model_hpe.cuda(device)
    print('[RECOGNIZER] hpe init done ...')

    name_window = 'show'
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 540)

    total_n = len(glob.glob(os.path.join(dir_imgs, '*.%s' % img_suffix)))
    for paths_idx, path_img in enumerate(glob.glob(os.path.join(dir_imgs, '*.%s' % img_suffix))):
        print(f'processing {paths_idx}th img {path_img} [total {total_n}]')

        path_label = path_img.replace('images', 'labels')
        path_label = path_label.replace('.%s' % img_suffix, '.txt')

        img_org_uc = cv2.imread(path_img)
        img_org = copy.deepcopy(img_org_uc)
        h_img = img_org.shape[0]
        w_img = img_org.shape[1]

        bhv_dets = []
        if os.path.exists(path_label):
            with open(path_label, 'r') as f:
                lines = f.readlines()
            for line in lines:
                label, xc, yc, w, h = line.split(' ')
                label = int(label)
                xc = float(xc) * w_img
                yc = float(yc) * h_img
                w = float(w) * w_img
                h = float(h) * h_img

                x1 = xc - w / 2
                y1 = yc - h / 2
                x2 = xc + w / 2
                y2 = yc + h / 2

                if label == 0:
                    bhv_dets.append([x1, y1, x2, y2, 1])
                    cv2.rectangle(img_org, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        img = letterbox(img_org, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        dets = pred[0]

        if dets is not None:
            dets = dets[dets[:, -1] == 0]  # pick certain class
            dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], img_org.shape).round()
            dets = dets[:, 0:5].detach().cpu().numpy()

            for det in dets:
                bbox = det[0:4]
                box = bbox.astype(int)
                cv2.rectangle(img_org, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)

        dets = np.zeros([0, 5], dtype=float) if dets is None else dets
        bhv_dets = np.array(bhv_dets) if len(bhv_dets) > 0 else np.zeros([0, 5], dtype=float)

        # if path_img == '/media/manu/kingstoo/yolov5/custom_behavior/images/train2017/24168_r1-14.jpg':
        #     print(path_img)

        for det in dets:
            for bhv_det in bhv_dets:
                ol_s = overlap_master(bhv_det[:4], det[:4])
                if ol_s > 0.7:
                    det[4] = 2
                    cv2.putText(img_org, 'match', (int(det[0]), int(det[1] + int(1.2 * 25))),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    break

        # if len(dets) + len(bhv_dets) > 0:
        #     dets = np.vstack((dets, bhv_dets))
        if len(dets) > 0:

            hpe_dets = copy.deepcopy(dets)
            for s_idx, sdet in enumerate(hpe_dets):
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

                data_numpy = copy.deepcopy(img_org_uc)

                db_rec = kpt_db[0]
                joints = db_rec['joints_3d']
                joints_vis = db_rec['joints_3d_vis']

                c = db_rec['center']
                s = db_rec['scale']
                score = db_rec['score'] if 'score' in db_rec else 1
                r = 0

                trans = transforms_hpe.get_affine_transform(c, s, r, [image_width, image_height])
                input_numpy = cv2.warpAffine(
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

                input = transform(input_numpy)
                input = input.unsqueeze(0)
                input = input.to(device)

                model_hpe.eval()
                output = model_hpe(input)

                c = np.expand_dims(c, 0)
                s = np.expand_dims(s, 0)
                preds, maxvals = get_final_preds(
                    config, output.detach().clone().cpu().numpy(), c, s)

                preds_local, maxvals_local = get_final_preds_local(
                    config, output.detach().clone().cpu().numpy())

                plot_hpe(input_numpy, preds_local, maxvals_local)
                name_img = os.path.basename(path_img)[:-4]
                if sdet[-1] == 2:
                    cv2.imwrite(os.path.join(dir_pos, name_img + f'_{s_idx}.{img_suffix}'), input_numpy)
                    with open(os.path.join(dir_pos, name_img + f'_{s_idx}.txt'), 'w') as f:
                        line = ''
                        for pred, maxval in zip(preds_local[0], maxvals_local[0]):
                            x, y, z = pred[0], pred[1], maxval[0]
                            line += f'{x / image_width}, {y / image_height}, {z}, '
                        f.writelines(line[:-2])
                else:
                    cv2.imwrite(os.path.join(dir_neg, name_img + f'_{s_idx}.{img_suffix}'), input_numpy)
                    with open(os.path.join(dir_neg, name_img + f'_{s_idx}.txt'), 'w') as f:
                        line = ''
                        for pred, maxval in zip(preds_local[0], maxvals_local[0]):
                            x, y, z = pred[0], pred[1], maxval[0]
                            line += f'{x / image_width}, {y / image_height}, {z}, '
                        f.writelines(line[:-2])

                plot_hpe(img_org, preds, maxvals)

        cv2.imshow(name_window, img_org)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
