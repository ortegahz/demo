# author: zerg

# libs
from multiprocessing import Process, Manager
import cv2

# params
max_buff_size = 25
num_skip = 2  # for 4k process speed problem
name_window = 'frame'
path_video = 'rtsp://192.168.3.34:554/live/ch4'


def process_decoder(buff_frame, path_video, num_skip, max_buff_size):
    idx_frame = 0
    cap = cv2.VideoCapture(path_video)
    if cap.isOpened() is False:
        print("Error opening video steam")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            if len(buff_frame) >= max_buff_size:
                frame_idxs = buff_frame.keys()
                frame_idxs = list(frame_idxs)
                frame_idxs = list(map(int, frame_idxs))
                frame_idxs.sort()
                idx_frame_del = frame_idxs[0]
                buff_frame.pop('%d' % idx_frame_del)
            idx_frame += 1
            # idx_frame_del = idx_frame - len(buff_frame) + 1
            if idx_frame % num_skip == 0:
                buff_frame['%d' % idx_frame] = frame
            print('number of frames in the buffer: %d' % len(buff_frame))
        else:
            break

    cap.release()


if __name__ == '__main__':
    with Manager() as manager:
        buff_frame = manager.dict()

        p = Process(target=process_decoder, args=(buff_frame, path_video, num_skip, max_buff_size), daemon=True)

        p.start()

        cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
        while True:
            if len(buff_frame) > 0:
                frame_idxs = buff_frame.keys()
                frame_idxs = list(frame_idxs)
                frame_idxs = list(map(int, frame_idxs))
                frame_idxs.sort()
                current_frame_idx = frame_idxs[-1]
                frame = buff_frame['%d' % current_frame_idx]  # get the current frame
                cv2.imshow(name_window, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
