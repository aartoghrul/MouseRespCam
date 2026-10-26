"""Frame sources. Recorded video is replayed frame by frame; timestamps come from the frame index,
so analysis is identical whether it is paced at real time or run as fast as possible."""
import cv2


class VideoFileSource:
    def __init__(self, path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f'cannot open {path}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.n_frames / self.fps
        self.shape = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.idx = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        t = self.idx / self.fps
        self.idx += 1
        return frame, t

    def release(self):
        self.cap.release()
