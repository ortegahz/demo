import cv2
import time


# from PIL import Image


def process_decoder(q, path_video, len_buffer=1):
    idx_frame = 0
    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if cap.isOpened() is False:
        print("Error opening video steam")

    t_last = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        # frame = cv2.imread('/home/manu/tmp/1/3.jpg')
        # frame = cv2.imread('/home/manu/tmp/唐可/15-56-46.jpg')
        # frame = cv2.imread('/home/manu/tmp/周朝进/08-44-24.jpg')
        # frame = cv2.imread('/home/manu/tmp/0e03a26950a7b2166217c082d841aa446a37b3e6a078f1428c11082c.bmp')
        # frame = Image.open('/home/manu/tmp/0e03a26950a7b2166217c082d841aa446a37b3e6a078f1428c11082c.bmp')
        if ret is True:
            idx_frame += 1
            while time.time() - t_last < 1. / fps:
                time.sleep(0.001)
            q.put([frame, idx_frame])
            t_last = time.time()
            if q.qsize() > len_buffer:
                q.get()
            print('[process_decoder] number of frames in the queue: %d' % q.qsize())
        else:
            break

    cap.release()
