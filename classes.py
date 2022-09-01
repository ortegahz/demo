from multiprocessing import Process, Queue
import time
import cv2


class Decoder:
    def __init__(self, path, qs=10):
        self.queue = Queue(qs)
        self.process = Process(target=self.__update, args=(path, self.queue))
        self.process.daemon = True
        self.process.start()

    @staticmethod
    def __update(path, q):
        cap = cv2.VideoCapture(path)

        while cap.isOpened():
            ret, frame = cap.read()
            # time.sleep(0.02)
            if q.full():
                q.get()
            if ret is True:
                q.put(frame)
                print('[Decoder] number of frames in the queue: %d' % q.qsize())
            else:
                break

        print('[Decoder] exit !')
        cap.release()

    def read(self):
        frame = self.queue.get()
        return frame


class FPSTracker:
    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio

    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        fps_sample = 1.0 / (t_new - self._last_tick)
        self._avg_fps = self.ratio * fps_sample + (
                1 - self.ratio) * self._avg_fps if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()

    def get(self):
        return self._avg_fps


class Displayer:
    def __init__(self, title, width=None, height=None, show_info=True):
        self.title, self.width, self.height = title, width, height
        self.show_info = show_info
        self.fps_tracker = FPSTracker()
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if width is not None and height is not None:
            cv2.resizeWindow(self.title, width, height)

    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
            cv2.putText(image, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        cv2.imshow(self.title, image)
        return cv2.waitKey(1) & 0xFF
