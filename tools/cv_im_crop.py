import cv2


def main():
    PATH_IMG_IN = '/home/manu/图片/vlcsnap-2023-03-20-17h42m14s180.png'
    WIN_NAME = 'MediaPipe Hands'
    WS, HS = 960, 540
    # X0, Y0, X1, Y1 = 858+50, 533, 1351+50, 1080
    X0, Y0, X1, Y1 = 1076, 375, 1223, 621

    path_img_out = PATH_IMG_IN.replace('.png', '-crop.bmp')

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WS, HS)

    img = cv2.imread(PATH_IMG_IN)
    # h, w, c = img.shape

    img_crop = img[Y0:Y1, X0:X1, :]
    cv2.imwrite(path_img_out, img_crop)

    cv2.imshow(WIN_NAME, img_crop)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
