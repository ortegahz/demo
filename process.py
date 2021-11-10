import cv2


def process_decoder(q, path_video, num_skip):
    idx_frame = 0
    cap = cv2.VideoCapture(path_video)
    if cap.isOpened() is False:
        print("Error opening video steam")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            idx_frame += 1
            if idx_frame % num_skip == 0:
                q.put([frame, idx_frame])
            print('[process_decoder] number of frames in the queue: %d' % q.qsize())
        else:
            break

    cap.release()
