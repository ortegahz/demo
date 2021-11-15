# author: zerg

# libs
from multiprocessing import Process, Queue
import numpy as np
import cv2
import process
import glob
import os
import sys

from retinaface import RetinaFace
import face_model

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_preprocess

# params
num_skip = 6  # for speed reason
name_window = 'frame'
path_video = 'rtsp://192.168.3.34:554/live/ch4'
# path_video = 'rtsp://192.168.3.233:554/live/ch4'

model_face_detect_path = './models/mobilenet_v1_0_25/retina'
warmup_img_path = '/media/manu/samsung/pics/material3000_1920x1080.jpg'  # image size should be same as actual input
gpuid = 0
thresh = 0.3
scales = [1.0]
flip = False

face_recog_debug_dir = '/home/manu/tmp/demo_snapshot/'
face_dataset_dir = '/media/manu/samsung/pics/人脸底图'
model_face_recog_path = '/media/manu/intel/workspace/insightface_manu_subcenter/models/model,0'
face_recog_sim_th = 0.35
face_recog_dist_th = 2.0

if __name__ == '__main__':
    print('face detect init start ...')
    detector = RetinaFace(model_face_detect_path, 0, gpuid, 'net3')
    img = cv2.imread(warmup_img_path)
    print('face detect init done')

    print('face recog init start ...')
    face_recog_dataset = []
    model = face_model.FaceModel(gpuid, model_face_recog_path)
    out_dir = face_recog_debug_dir
    for path_img in glob.glob(os.path.join(face_dataset_dir, '*.jpg')):
        _, img_name = os.path.split(path_img)
        img_name = img_name.replace('.jpg', '')
        stu_id, stu_name = img_name.split('_')
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
        cv2.imwrite(out_path, img_aligned)
        img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
        img_aligned = np.transpose(img_aligned, (2, 0, 1))
        feat = model.get_feature(img_aligned)
        item = (stu_id, stu_name, feat)
        face_recog_dataset.append(item)
        print('record student %s with id %s' % (stu_name, stu_id))
    print('face recog init done')

    print('warm up start ...')
    # warm up
    for _ in range(10):
        faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
    print('warm up done')

    q_decoder = Queue()
    p_decoder = Process(target=process.process_decoder, args=(q_decoder, path_video, num_skip), daemon=True)
    p_decoder.start()

    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 540)
    face_recog_aligned_save_idx = 0
    while True:
        item_frame = q_decoder.get()
        frame = item_frame[0]
        h, w, c = frame.shape
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
                img_aligned = face_preprocess.preprocess(frame, bbox, points, image_size='112,112')
                img_aligned_write = img_aligned
                img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
                img_aligned = np.transpose(img_aligned, (2, 0, 1))
                feat = model.get_feature(img_aligned)
                [sim_highest, stu_name_highest, isfind] = [0, None, False]
                for stu_id, stu_name, feat_ref in face_recog_dataset:
                    sim = np.dot(feat_ref, feat.T)  # sim is wired
                    if sim > sim_highest:
                        sim_highest = sim
                        stu_name_highest = stu_name
                    # dist = np.sum(np.square(feat_ref - feat))
                    # if dist < face_recog_dist_th:
                    if sim > face_recog_sim_th:
                        # info = stu_name + ' with dist ' + '%f' % dist
                        info = stu_name + ' with sim ' + '%f' % sim
                        img = cv2.putText(frame, info, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                        # save aligned image for debug reason
                        out_dir = face_recog_debug_dir
                        # out_path = os.path.join(out_dir,
                        #                         '%s_%d_%f' % (stu_name, face_recog_aligned_save_idx, dist) + '.jpg')
                        out_path = os.path.join(out_dir,
                                                '%s_%d_%f' % (stu_name, face_recog_aligned_save_idx, sim) + '.jpg')
                        cv2.imwrite(out_path, img_aligned_write)
                        face_recog_aligned_save_idx += 1
                        isfind = True
                if not isfind:
                    info = stu_name_highest + ' with sim ' + '%f' % sim_highest
                    img = cv2.putText(frame, info, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

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
