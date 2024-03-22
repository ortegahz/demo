import itertools

import cv2
import numpy as np

distorted_image = cv2.imread('/home/manu/tmp/vlcsnap-2024-03-22-16h28m42s336.png')

camera_matrix = np.array([[1028, 0, 960],
                          [0, 771, 540],
                          [0, 0, 1]])

k1 = -0.01  # 径向畸变系数1
k2 = -0.0005  # 径向畸变系数2
p1 = -0.01  # 切向畸变系数1
p2 = -0.01  # 切向畸变系数2
k3 = -0.02  # 径向畸变系数3（可选，用于校正更高程度的畸变）

num = 10
# k1_values = np.linspace(-0.01, 0, num=num)
# k2_values = np.linspace(-0.0001, 0, num=num)
# p1_values = np.linspace(-0.01, 0.01, num=num)
# p2_values = np.linspace(-0.01, 0.01, num=num)
k3_values = np.linspace(-0.03, 0, num=num)

k1_values = [k1]
k2_values = [k2]
p1_values = [p1]
p2_values = [p2]
# k3_values = [k3]

for distortion_coefficients in itertools.product(k1_values, k2_values, p1_values, p2_values, k3_values):
    distortion_coefficients = np.array(distortion_coefficients)
    print(distortion_coefficients)
    name_img_out = f'{distortion_coefficients}.png'
    h, w = distorted_image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(distorted_image, camera_matrix, distortion_coefficients, None, new_camera_matrix)
    cv2.imwrite(f'/home/manu/tmp/undistorted_image_single/{name_img_out}', undistorted_image)
