import cv2
import time
import logging


def process_decoder(path_video, queue, buff_len=5):
    idx_frame = 0
    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        logging.error("Error opening video steam")

    t_last = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        idx_frame += 1
        if queue.qsize() > buff_len:
            queue.get()
            # logging.info('dropping frame !!!')
        queue.put([idx_frame, frame, fc])

        while time.time() - t_last < 1. / fps:
            time.sleep(0.001)
        t_last = time.time()

    cap.release()
