import glob
import os

import cv2 as cv
import numpy as np


def load_data(dir, label=1, max=-1):
    sample_files = glob.glob(os.path.join(dir, '*.txt'))
    total_n = len(sample_files)
    data = []
    for i, sample_file in enumerate(sample_files):
        if i == max:
            break
        print(f'processing {i}th img {sample_file} [total {total_n}]')
        with open(sample_file) as f:
            line = f.readline()
            feat = line.split(',')
            feat = [float(x) for x in feat]
            data.append(feat)
    data = np.array(data)
    labels = np.ones((len(data), 1)) * label
    return np.float32(data), np.int64(labels)


if __name__ == '__main__':
    path_save = '/home/manu/tmp/svm_data.dat'

    dir_pos_train = '/media/manu/kingstoo/svm/train/pos'
    dir_neg_train = '/media/manu/kingstoo/svm/train/neg'
    dir_pos_test = '/media/manu/kingstoo/svm/test/pos'
    dir_neg_test = '/media/manu/kingstoo/svm/test/neg'
    path_out_libsvm_train = '/home/manu/tmp/hpe_train'
    path_out_libsvm_test = '/home/manu/tmp/hpe_test'

    data_pos, resp_pos = load_data(dir_pos_train, 1)
    data_neg, resp_neg = load_data(dir_neg_train, -1)
    trainData = np.vstack((data_pos, data_neg))
    trainRsp = np.vstack((resp_pos, resp_neg))

    with open(path_out_libsvm_train, 'w') as f:
        line = ''
        for i in range(len(trainData)):
            line = '+1' if trainRsp[i] == 1 else '-1'
            for j in range(len(trainData[0])):
                line += f' {j + 1}:{trainData[i][j]}'
            line += '\n'
            f.write(line)

    data_pos, resp_pos = load_data(dir_pos_test, 1)
    data_neg, resp_neg = load_data(dir_neg_test, -1)
    testData = np.vstack((data_pos, data_neg))
    testRsp = np.vstack((resp_pos, resp_neg))

    with open(path_out_libsvm_test, 'w') as f:
        line = ''
        for i in range(len(testData)):
            line = '+1' if testRsp[i] == 1 else '-1'
            for j in range(len(testData[0])):
                line += f' {j + 1}:{testData[i][j]}'
            line += '\n'
            f.write(line)
