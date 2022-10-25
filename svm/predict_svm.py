import glob
import os

import cv2 as cv
import numpy as np


def load_feat(path):
    data = []
    with open(path) as f:
        line = f.readline()
        feat = line.split(',')
        feat = [float(x) for x in feat]
        data.append(feat)
    data = np.array(data)
    return np.float32(data)


if __name__ == '__main__':

    path_feat = '/media/manu/kingstoo/svm/test/pos/10687_r1-14_2.txt'
    path_model = '/home/manu/tmp/svm_data.dat'

    path_img = path_feat.replace('.txt', '.jpg')

    data = load_feat(path_feat)

    svm = cv.ml.SVM_load(path_model)

    result = svm.predict(data)[1]
    print(result[0][0])

    name_window = 'result'
    cv.namedWindow(name_window, cv.WINDOW_NORMAL)
    img = cv.imread(path_img)
    cv.putText(img, f'{result[0][0]}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv.imshow(name_window, img)
    cv.waitKey(3000)
