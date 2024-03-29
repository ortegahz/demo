# -*-coding:utf-8-*-

from multiprocessing import Process, Queue
from sklearn import preprocessing
import numpy as np
import cv2
import process
import copy
import glob
import os
import sys
import torch

from retinaface import RetinaFace

sys.path.append('/media/manu/kingstop/workspace/insightface/recognition/arcface_torch')
from backbones import get_model

sys.path.append('/media/manu/kingstop/workspace/demo/common')
import face_preprocess


def main():
    # params
    name_window = 'frame'
    # path_video = 'rtsp://192.168.3.34:554/live/ch4'
    # path_video = 'rtsp://192.168.3.233:554/live/ch4'
    # path_video = '/media/manu/samsung/videos/at2021/mp4/Video1.mp4'
    path_video = '/media/manu/kingstoo/tmp/B201-2.mp4'
    out_dir_reset = False

    model_face_detect_path = '/home/manu/tmp/mobilenet_v1_0_25/retina'
    warmup_img_path = '/media/manu/samsung/pics/material3000_1920x1080.jpg'  # image size should be same as actual input
    gpuid = 0
    thresh = 0.3
    scales = [1.0]
    flip = False

    face_recog_debug_dir = '/home/manu/tmp/demo_snapshot/'
    face_dataset_dir = '/media/manu/kingstoo/tmp/201-照片'
    network_name = 'r100'
    path_weight = '/home/manu/tmp/wf42m_pfc02_r100_8gpus_bs512/model.pt'
    local_rank = 'cuda:0'
    epsilon = 1e-10
    face_recog_sim_th = 0.35
    db_suffix = 'JPG'

    print('face detect init start ...')
    detector = RetinaFace(model_face_detect_path, 0, gpuid, 'net3')
    img = cv2.imread(warmup_img_path)
    print('face detect init done')

    print('face recog init start ...')
    face_recog_dataset = []
    face_recog_net = get_model(network_name, fp16=False).to(local_rank)
    face_recog_net.load_state_dict(torch.load(path_weight, map_location=torch.device(local_rank)))
    face_recog_net.eval()
    out_dir = face_recog_debug_dir
    if out_dir_reset:
        os.system('rm %s -rvf' % out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for path_img in glob.glob(os.path.join(face_dataset_dir, f'*.{db_suffix}')):
        _, img_name = os.path.split(path_img)
        img_name = img_name.replace(f'.{db_suffix}', '')
        # stu_id, stu_name = img_name.split('_')
        stu_id, stu_name = img_name, img_name
        img = cv2.imread(path_img)
        h, w, c = img.shape
        max_l = max(h, w)
        scales_reg = [1.0]
        if max_l > 1920:  # can not detect faces on some large input images
            scales_reg = [0.5]
        faces, landmarks = detector.detect(img, 0.8, scales=scales_reg, do_flip=flip)  # using high detect th for reg
        assert len(faces) == 1  # TODO
        # face align and feature extract
        bbox = faces
        points = np.squeeze(landmarks).transpose().reshape(1, 10)
        bbox = bbox[0, 0:4]
        points = points[0, :].reshape((2, 5)).T
        img_aligned = face_preprocess.preprocess(img, bbox, points, image_size='112,112')
        out_path = os.path.join(out_dir, stu_name + '.jpg')
        # cv2.imwrite(out_path, img_aligned)
        img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
        img_aligned = np.transpose(img_aligned, (2, 0, 1))
        img_aligned = torch.from_numpy(img_aligned).unsqueeze(0).float()
        img_aligned.div_(255).sub_(0.5).div_(0.5)
        img_aligned = img_aligned.cuda()
        feat = face_recog_net(img_aligned).cpu().detach().numpy().astype('float')
        feat = preprocessing.normalize(feat).flatten()
        feat_norm = np.linalg.norm(feat)
        assert 1.0 - epsilon <= feat_norm <= 1.0 + epsilon
        # img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
        # img_aligned = np.transpose(img_aligned, (2, 0, 1))
        # feat = model.get_feature(img_aligned)
        item = [stu_id, stu_name, feat, -1.0]
        face_recog_dataset.append(item)
        print('record student %s with id %s' % (stu_name, stu_id))
    print('face recog init done')

    print('warm up start ...')
    # warm up
    for _ in range(10):
        faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
    print('warm up done')

    q_decoder = Queue()
    p_decoder = Process(target=process.process_decoder, args=(q_decoder, path_video), daemon=True)
    p_decoder.start()

    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 540)
    while True:
        item_frame = q_decoder.get()
        frame_org = item_frame[0]
        frame = copy.deepcopy(frame_org)
        h, w, c = frame_org.shape
        if w > 1920:
            scales = [0.5]
        faces, landmarks = detector.detect(frame, thresh, scales=scales, do_flip=flip)
        if faces is not None:
            print('find', faces.shape[0], 'faces')

            # recognition
            for i in range(faces.shape[0]):
                box = faces[i].astype(int)
                bbox = faces[i]
                landmarks_recog = landmarks[i]
                points = landmarks_recog.transpose().reshape(1, 10)
                bbox = bbox[0:4]
                points = points[0, :].reshape((2, 5)).T
                img_aligned = face_preprocess.preprocess(frame_org, bbox, points, image_size='112,112')
                img_aligned_write = img_aligned
                # cv2.imwrite('/home/manu/tmp/snap.jpg', img_aligned_write)
                img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
                img_aligned = np.transpose(img_aligned, (2, 0, 1))
                img_aligned = torch.from_numpy(img_aligned).unsqueeze(0).float()
                img_aligned.div_(255).sub_(0.5).div_(0.5)
                img_aligned = img_aligned.cuda()
                feat = face_recog_net(img_aligned).cpu().detach().numpy().astype('float')
                feat = preprocessing.normalize(feat).flatten()
                feat_norm = np.linalg.norm(feat)
                assert 1.0 - epsilon <= feat_norm <= 1.0 + epsilon
                # img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
                # img_aligned = np.transpose(img_aligned, (2, 0, 1))
                # feat = model.get_feature(img_aligned)
                [sim_highest, stu_name_highest, idx_db_highest] = [-1., '', -1]
                for idx_db, (stu_id, stu_name, feat_ref, rhs) in enumerate(face_recog_dataset):
                    sim = np.dot(feat_ref, feat.T)  # sim is wired
                    if sim > sim_highest:
                        sim_highest = sim
                        stu_name_highest = stu_name
                        idx_db_highest = idx_db
                if sim_highest > face_recog_sim_th:
                    info = stu_name_highest + ' ' + '%f' % sim_highest
                    cv2.putText(frame, info, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    out_dir = face_recog_debug_dir
                    if sim_highest > face_recog_dataset[idx_db_highest][3]:
                        out_path = os.path.join(out_dir, '%s' % stu_name_highest + '.jpg')
                        cv2.imwrite(out_path, img_aligned_write)
                        face_recog_dataset[idx_db_highest][3] = sim_highest
                else:
                    info = stu_name_highest + ' ' + '%f' % sim_highest
                    cv2.putText(frame, info, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # plot
            for i in range(faces.shape[0]):
                # print('score', faces[i][4])
                box = faces[i].astype(int)
                # color = (255,0,0)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                if landmarks is not None:
                    landmark5 = landmarks[i].astype(int)
                    # print(landmark.shape)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(frame, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

        cv2.imshow(name_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
