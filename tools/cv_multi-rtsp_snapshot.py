import cv2
import os
import time
import shutil
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('manu')


def main():
    # params
    rtsp_addrs = [
        'rtsp://192.168.1.97:554/ch1',
        'rtsp://192.168.1.62:554/ch0_1',
    ]
    dir_save = '/home/manu/tmp/snapshots'
    b_del_dir_save = True
    time_delay_s = 10

    if b_del_dir_save and os.path.exists(dir_save):
        shutil.rmtree(dir_save)
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)

    while True:
        for rtsp_addr in rtsp_addrs:
            cap = cv2.VideoCapture(rtsp_addr)
            if cap.isOpened() is False:
                logger.error('Error opening video steam')

            ret, frame = cap.read()
            str_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path_save = os.path.join(dir_save, str_time + '.jpg')
            if ret is True:
                logger.info(f'saving {path_save}')
                cv2.imwrite(path_save, frame)
            else:
                logger.info(f'skipping {path_save}')

            cap.release()
            time.sleep(time_delay_s)


if __name__ == '__main__':
    main()
