import cv2

name_window = 'frame'
path_video = 'rtsp://192.168.3.222:554/live/ch4'

cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(path_video)
if cap.isOpened() is False:
    print("Error opening video steam")

while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        cv2.imshow(name_window, frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
