import cv2
import glob
import os

face_dataset_dir = '/media/manu/samsung/pics/人脸底图'
face_dataset_dir_out = '/home/manu/tmp/face_dataset_dir_out'
rs_scale = 1. / 4.

os.system('rm %s -rvf' % face_dataset_dir_out)
if not os.path.exists(face_dataset_dir_out):
    os.makedirs(face_dataset_dir_out)

for path_img in glob.glob(os.path.join(face_dataset_dir, '*.jpg')):
    _, img_name = os.path.split(path_img)
    img = cv2.imread(path_img)
    img_rs = cv2.resize(img, None, fx=rs_scale, fy=rs_scale)
    out_path = os.path.join(face_dataset_dir_out, img_name)
    cv2.imwrite(out_path, img_rs)


