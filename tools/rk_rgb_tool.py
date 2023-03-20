import numpy as np
import cv2

if __name__ == "__main__":
    path = \
        '/media/manu/kingstop/itx-3588j/Linux_SDK/rk3588/external/linux-rga/samples/sample_file/in0w1280-h720-rgba8888.bin'

    w, h, c = 1280, 720, 4

    data = np.fromfile(path, np.uint8)

    rgba = data.reshape(h, w, c)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    cv2.imshow('img', bgr)
    cv2.waitKey(0)
