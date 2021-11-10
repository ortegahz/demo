# author: zerg

# libs
from multiprocessing import Process, Queue
import cv2
import process
import numpy as np
import os

from retinaface import RetinaFace

# params
num_skip = 5  # for speed reason
name_window = 'frame'
path_video = 'rtsp://192.168.3.34:554/live/ch4'

model_face_detect_path = './models/mobilenet_v1_0_25/retina'
warmup_img_path = '/media/manu/samsung/pics/material3000_1920x1080.jpg'  # image size should be same as actual input
gpuid = 0
thresh = 0.8
scales = [1.0]
flip = False

if __name__ == '__main__':
    detector = RetinaFace(model_face_detect_path, 0, gpuid, 'net3')
    img = cv2.imread(warmup_img_path)
    # warm up
    for _ in range(10):
        faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)

    q_decoder = Queue()
    p_decoder = Process(target=process.process_decoder, args=(q_decoder, path_video, num_skip), daemon=True)
    p_decoder.start()

    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 540)
    while True:
        item_frame = q_decoder.get()
        frame_4k = item_frame[0]
        frame_2k = cv2.resize(frame_4k, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        faces, landmarks = detector.detect(frame_2k, thresh, scales=scales, do_flip=flip)
        if faces is not None:
            print('find', faces.shape[0], 'faces')
            for i in range(faces.shape[0]):
                # print('score', faces[i][4])
                box = faces[i].astype(np.int)
                # color = (255,0,0)
                color = (0, 0, 255)
                cv2.rectangle(frame_2k, (box[0], box[1]), (box[2], box[3]), color, 2)
                if landmarks is not None:
                    landmark5 = landmarks[i].astype(np.int)
                    # print(landmark.shape)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(frame_2k, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
        cv2.imshow(name_window, frame_2k)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
