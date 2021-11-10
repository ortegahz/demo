# author: zerg

# libs
from multiprocessing import Process, Queue
import cv2
import process

# params
num_skip = 2  # for 4k process speed problem
name_window = 'frame'
path_video = 'rtsp://192.168.3.34:554/live/ch4'


if __name__ == '__main__':
    q_decoder = Queue()
    p_decoder = Process(target=process.process_decoder, args=(q_decoder, path_video, num_skip), daemon=True)
    p_decoder.start()

    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    while True:
        item_frame = q_decoder.get()
        frame = item_frame[0]
        cv2.imshow(name_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
