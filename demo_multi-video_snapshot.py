import cv2
import os
import time
import shutil

if __name__ == '__main__':
    # params
    PATHS_VID = [
        '/media/manu/samsung/videos/modnet/举手.mp4',
    ]
    DIR_OUT = '/home/manu/tmp/snapshots'
    B_RESET = True
    NUM_SKIP = 1  # [NUM_SKIP s]

    if B_RESET:
        shutil.rmtree(DIR_OUT)
    if not os.path.exists(DIR_OUT):
        os.mkdir(DIR_OUT)

    idx_frame = 0
    for path in PATHS_VID:
        cnt_skip = 0
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            cnt_skip += 1
            str_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path_out = os.path.join(DIR_OUT, f'manu_{idx_frame}' + '.jpg')
            if cnt_skip == NUM_SKIP:
                print(f'saving {path_out}')
                cv2.imwrite(path_out, frame)
                cnt_skip = 0
                idx_frame += 1

        cap.release()
