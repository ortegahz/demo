import cv2
import os
import time
import shutil

if __name__ == '__main__':
    # params
    rtsp_addrs = [
        'rtsp://192.168.3.34:554/live/ch1',
        'rtsp://192.168.3.51:554/ch2',
        'rtsp://192.168.3.222:554/live/ch4',
    ]
    dir_save = '/home/manu/tmp/snapshots'
    b_del_dir_save = True
    time_delay_s = 10

    if b_del_dir_save:
        shutil.rmtree(dir_save)
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)

    while True:
        for rtsp_addr in rtsp_addrs:
            cap = cv2.VideoCapture(rtsp_addr)
            if cap.isOpened() is False:
                print("Error opening video steam")

            ret, frame = cap.read()
            str_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path_save = os.path.join(dir_save, str_time + '.jpg')
            if ret is True:
                print(f'saving {path_save}')
                cv2.imwrite(path_save, frame)
            else:
                break

            cap.release()
            time.sleep(time_delay_s)
