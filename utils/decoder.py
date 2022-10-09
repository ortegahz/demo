import cv2
import time


def process_decoder(path_video, buff_len, arr_frames):
    idx_frame = 0
    cap = cv2.VideoCapture(path_video)
    if cap.isOpened() is False:
        print("Error opening video steam")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            idx_frame += 1
            if len(arr_frames) == buff_len * 2:
                arr_frames.pop(0)
                print('[process_decoder] dropping frame !!!')
            arr_frames.append([idx_frame, frame])
            print('[process_decoder] number of frames in the arr_frames: %d' % len(arr_frames))
        else:
            break

        time.sleep(0.015)

    cap.release()
