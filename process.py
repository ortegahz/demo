import cv2


def process_decoder(q, path_video):
    idx_frame = 0
    cap = cv2.VideoCapture(path_video)
    if cap.isOpened() is False:
        print("Error opening video steam")

    while cap.isOpened():
        ret, frame = cap.read()
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        # frame = cv2.imread('/home/manu/tmp/1/3.jpg')
        if ret is True:
            idx_frame += 1
            q.put([frame, idx_frame])
            if q.qsize() > 30:
                q.get()
            print('[process_decoder] number of frames in the queue: %d' % q.qsize())
        else:
            break

    cap.release()
