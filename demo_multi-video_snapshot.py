import cv2
import os
import time
import shutil

if __name__ == '__main__':
    # params
    PATHS_VID = [
        '/home/manu/tmp/videos/vlc-record-2022-01-15-14h36m02s-rtsp___192.168.3.93_554_ch0_1-.mp4',
        '/home/manu/tmp/videos/vlc-record-2022-01-15-14h56m30s-rtsp___192.168.3.93_554_ch0_1-.mp4',
        '/home/manu/tmp/videos/vlc-record-2022-01-15-16h58m01s-rtsp___192.168.3.93_554_ch0_1-.mp4',
    ]
    DIR_OUT = '/home/manu/tmp/snapshots'
    B_RESET = True
    NUM_SKIP = 25 * 10  # [NUM_SKIP s]

    if B_RESET:
        shutil.rmtree(DIR_OUT)
    if not os.path.exists(DIR_OUT):
        os.mkdir(DIR_OUT)

    for path in PATHS_VID:
        cnt_skip = 0
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            cnt_skip += 1
            str_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path_out = os.path.join(DIR_OUT, str_time + '.jpg')
            if cnt_skip == NUM_SKIP:
                print(f'saving {path_out}')
                cv2.imwrite(path_out, frame)
                cnt_skip = 0

        cap.release()
