import os
import shutil
import time

import cv2


def read_paths_from_file(file_path):
    with open(file_path, 'r') as file:
        paths = [line.strip() for line in file if line.strip()]
    return paths


if __name__ == '__main__':
    # params
    FILE_PATHS_VID = 'video_paths.txt'
    DIR_OUT = 'snapshots'
    B_RESET = True
    NUM_SKIP = 25 * 10  # [NUM_SKIP s]

    PATHS_VID = read_paths_from_file(FILE_PATHS_VID)

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
