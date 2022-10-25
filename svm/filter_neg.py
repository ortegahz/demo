import glob
import os
import cv2 as cv

from predict_svm import load_feat

path_model = '/home/manu/tmp/svm_data.dat'
dir_neg = '/media/manu/kingstoo/svm/train/neg'
dir_neg_out = '/media/manu/kingstoo/svm/train/neg_filtered'
dir_neg_rm = '/media/manu/kingstoo/svm/train/neg_pick'

os.system(f'rm {dir_neg_out} -rvf')
os.system(f'mkdir {dir_neg_out}')

os.system(f'rm {dir_neg_rm} -rvf')
os.system(f'mkdir {dir_neg_rm}')

label_files = glob.glob(os.path.join(dir_neg, '*.txt'))
total_n = len(label_files)

svm = cv.ml.SVM_load(path_model)

for i, path_feat in enumerate(label_files):
    print(f'processing {i}th img {path_feat} [total {total_n}]')

    path_img = path_feat.replace('.txt', '.jpg')

    data = load_feat(path_feat)
    res_raw = svm.predict(data)
    res = res_raw[1][0][0]

    if res == -1:
        os.system(f'cp {path_feat} {path_img} {dir_neg_out} -rvf')
    else:
        os.system(f'cp {path_feat} {path_img} {dir_neg_rm} -rvf')
