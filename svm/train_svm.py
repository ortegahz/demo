import glob
import os

import cv2 as cv
import numpy as np


def load_data(dir, label=1):
    sample_files = glob.glob(os.path.join(dir, '*.txt'))
    total_n = len(sample_files)
    data = []
    for i, sample_file in enumerate(sample_files):
        print(f'processing {i}th img {sample_file} [total {total_n}]')
        with open(sample_file) as f:
            line = f.readline()
            feat = line.split(',')
            feat = [float(x) for x in feat]
            data.append(feat)
    data = np.array(data)
    labels = np.ones((len(data), 1)) * label
    return np.float32(data), np.int64(labels)


path_save = '/home/manu/tmp/svm_data.dat'

dir_pos = '/media/manu/kingstoo/svm/train/pos'
dir_neg = '/media/manu/kingstoo/svm/train/neg'

data_pos, resp_pos = load_data(dir_pos, 1)
data_neg, resp_neg = load_data(dir_neg, -1)

trainData = np.vstack((data_pos, data_neg))
trainRsp = np.vstack((resp_pos, resp_neg))

dir_pos = '/media/manu/kingstoo/svm/test/pos'
dir_neg = '/media/manu/kingstoo/svm/test/neg'

data_pos, resp_pos = load_data(dir_pos, 1)
data_neg, resp_neg = load_data(dir_neg, -1)

testData = np.vstack((data_pos, data_neg))
testRsp = np.vstack((resp_pos, resp_neg))

svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.train(trainData, cv.ml.ROW_SAMPLE, trainRsp)
svm.save(path_save)
# svm = cv.ml.SVM_load(path_save)

result = svm.predict(trainData)[1]
mask = result == trainRsp
correct = np.count_nonzero(mask)
print(f'acc train {correct * 100.0 / result.size}')

result = svm.predict(testData)[1]
mask = result == testRsp
correct = np.count_nonzero(mask)
print(f'acc test {correct * 100.0 / result.size}')
