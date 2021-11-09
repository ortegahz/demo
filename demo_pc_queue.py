# author: zerg

# libs
from multiprocessing import Process, Queue
import cv2

# params
num_skip = 2  # for 4k process speed problem
name_window = 'frame'
path_video = 'rtsp://192.168.3.34:554/live/ch4'


def process_decoder(process_decoder_q, process_decoder_path_video, process_decoder_num_skip):
    idx_frame = 0
    cap = cv2.VideoCapture(process_decoder_path_video)
    if cap.isOpened() is False:
        print("Error opening video steam")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            idx_frame += 1
            if idx_frame % process_decoder_num_skip == 0:
                process_decoder_q.put([frame, idx_frame])
            print('[process_decoder] number of frames in the queue: %d' % process_decoder_q.qsize())
        else:
            break

    cap.release()


if __name__ == '__main__':
    q_decoder = Queue()

    p_decoder = Process(target=process_decoder, args=(q_decoder, path_video, num_skip), daemon=True)
    p_decoder.start()

    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    while True:
        item_frame = q_decoder.get()
        frame = item_frame[0]
        cv2.imshow(name_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
