import glob
import os
from subprocess import *

from predict_svm import load_feat

path_model = '/media/manu/kingstop/workspace/libsvm/tools/hpe_train_pick.model'
dir_neg = '/media/manu/kingstoo/svm/train/neg'
dir_neg_out = '/media/manu/kingstoo/svm/train/neg_filtered'
dir_neg_rm = '/media/manu/kingstoo/svm/train/neg_pick'

os.system(f'rm {dir_neg_out} -rvf')
os.system(f'mkdir {dir_neg_out}')

os.system(f'rm {dir_neg_rm} -rvf')
os.system(f'mkdir {dir_neg_rm}')

label_files = glob.glob(os.path.join(dir_neg, '*.txt'))
total_n = len(label_files)

# svm = cv.ml.SVM_load(path_model)

for i, path_feat in enumerate(label_files):
    path_img = path_feat.replace('.txt', '.jpg')

    print(f'processing {i}th img {path_img} [total {total_n}]')

    data = load_feat(path_feat)

    cmd = f'/media/manu/kingstop/workspace/libsvm/svm-predict-python {path_model}'
    for feat in data[0]:
        cmd += f' {feat}'

    f = Popen(cmd, shell=True, stdout=PIPE).stdout

    line = f.readline()
    label, pos_p, neg_p = map(float, line.split())
    print((label, pos_p, neg_p))

    if label == -1:
        os.system(f'cp {path_feat} {path_img} {dir_neg_out} -rvf')
    else:
        os.system(f'cp {path_feat} {path_img} {dir_neg_rm} -rvf')
