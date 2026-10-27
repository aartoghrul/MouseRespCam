"""Frame sources.

VideoFileSource replays a recording; timestamps come from the frame index, so analysis is identical
whether it is paced at real time or run as fast as possible.

CameraSource reads a live camera. The monitor assumes evenly spaced samples at a known rate (fps
turns a spectral frequency into breaths/min), and webcams do not provide that: the frame rate the
driver reports is often wrong (a MacBook camera reports 15 fps and delivers 30), the real rate drops
in dim light, and frames arrive with jitter. So the rate is measured, and a capture thread resamples
the camera onto a uniform time grid (nearest frame to each tick). Timestamps are exact multiples of
1/fps whatever the camera does, so the rate stays correct.
"""
import ctypes
import ctypes.util
import math
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np


class VideoFileSource:
    live = False

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
        self.ended = False

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            self.ended = True
            return None, None
        t = self.idx / self.fps
        self.idx += 1
        return frame, t

    def describe(self):
        return {'source': 'recording', 'path': self.path, 'fps': self.fps, 'size': self.shape[::-1]}

    def release(self):
        self.cap.release()


# ---------------------------------------------------------------- opening cameras
COMMON_FPS = (5, 7.5, 10, 12, 15, 20, 24, 25, 30, 50, 60)

MAC_DENIED = ('macOS is blocking camera access. Open System Settings > Privacy & Security > Camera, allow the '
              'app this was started from (Terminal, iTerm, VS Code, ...), then quit and reopen that app.')
MAC_PENDING = ('No answer to the macOS camera-permission prompt. Allow camera access for the app this was '
               'started from (System Settings > Privacy & Security > Camera), then try again.')


def _backends():
    """Capture APIs to try, most reliable first. On Windows DirectShow opens quickly and honours the
    requested size and MJPG format; Media Foundation is the fallback for cameras only it supports."""
    if sys.platform == 'darwin':
        return [cv2.CAP_AVFOUNDATION]
    if sys.platform == 'win32':
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    if sys.platform.startswith('linux'):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def _objc_send(restype, *argtypes):
    """objc_msgSend with a fixed prototype (required on Apple silicon)."""
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
    objc.objc_getClass.restype = objc.sel_registerName.restype = ctypes.c_void_p
    send = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(('objc_msgSend', objc))
    return lambda obj, sel, *a: send(obj, objc.sel_registerName(sel), *a), objc.objc_getClass


def _avfoundation():
    avf = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/AVFoundation.framework/AVFoundation')
    return lambda name: ctypes.c_void_p.in_dll(avf, name)


def mac_camera_status():
    """macOS camera authorisation of this process: 0 not asked yet, 1 restricted, 2 denied,
    3 allowed. None if not on macOS or it cannot be determined."""
    if sys.platform != 'darwin':
        return None
    try:
        send, cls = _objc_send(ctypes.c_long, ctypes.c_void_p)
        return int(send(cls(b'AVCaptureDevice'), b'authorizationStatusForMediaType:', _avfoundation()('AVMediaTypeVideo')))
    except Exception:
        return None


def mac_camera_names():
    """Camera names in OpenCV's AVFoundation index order (video + muxed devices sorted by
    uniqueID, as in OpenCV's cap_avfoundation_mac.mm). None if it cannot be determined."""
    if sys.platform != 'darwin':
        return None
    try:
        send_arg, cls = _objc_send(ctypes.c_void_p, ctypes.c_void_p)
        send_idx, _ = _objc_send(ctypes.c_void_p, ctypes.c_ulong)
        send_obj, _ = _objc_send(ctypes.c_void_p)
        send_count, _ = _objc_send(ctypes.c_ulong)
        send_str, _ = _objc_send(ctypes.c_char_p)
        text = lambda obj, prop: send_str(send_obj(obj, prop), b'UTF8String').decode()
        sym = _avfoundation()
        devices = []
        for media in ('AVMediaTypeVideo', 'AVMediaTypeMuxed'):
            arr = send_arg(cls(b'AVCaptureDevice'), b'devicesWithMediaType:', sym(media))
            for i in range(send_count(arr, b'count')):
                d = send_idx(arr, b'objectAtIndex:', i)
                devices.append((text(d, b'uniqueID'), text(d, b'localizedName')))
        return [name for _, name in sorted(devices)]
    except Exception:
        return None


def ensure_camera_permission(wait_s=60.0):
    """On macOS, ask for camera access if it has not been decided yet and wait for the answer.
    Raises PermissionError if access is denied (or the prompt is not answered in time)."""
    st = mac_camera_status()
    if st == 0:
        cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION).release()     # OpenCV shows the system prompt
        t0 = time.monotonic()
        while st == 0 and time.monotonic() - t0 < wait_s:
            time.sleep(0.25)
            st = mac_camera_status()
        if st == 0:
            raise PermissionError(MAC_PENDING)
    if st in (1, 2):
        raise PermissionError(MAC_DENIED)


def _configure(cap, size, api):
    w, h = size
    if api != cv2.CAP_AVFOUNDATION:
        # compressed frames: full HD at 30 fps fits through USB 2 (raw YUYV gets ~5 fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


def _first_frame(cap, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return frame
    return None


def open_camera(index=0, size=(1920, 1080), wait_permission_s=60.0):
    """Open camera `index` with the first capture API that delivers frames. The size is a request;
    the camera picks its nearest supported mode. Returns (VideoCapture, backend name)."""
    ensure_camera_permission(wait_permission_s)
    for api in _backends():
        for configure in (True, False):      # some cameras stop delivering when asked for MJPG/size
            cap = cv2.VideoCapture(index, api)
            if cap.isOpened():
                if configure:
                    _configure(cap, size, api)
                if _first_frame(cap) is not None:
                    return cap, cap.getBackendName()
            cap.release()
            if not configure:
                break
    raise IOError(f'Camera {index} could not be opened or delivers no frames. Check that it is connected '
                  f'and not in use by another program (video call, camera app).')


def list_cameras(max_index=8, skip=()):
    """[(index, label)] of available cameras. On macOS the names come from AVFoundation (no camera
    is switched on); elsewhere each index is opened, and probing stops after two misses in a row.
    Indices in `skip` (cameras in use) are listed without probing."""
    if mac_camera_status() not in (None, 3):
        ensure_camera_permission(wait_s=0.0)        # triggers the prompt; raises if not allowed yet
    names = mac_camera_names()
    if names is not None:
        return [(i, name) for i, name in enumerate(names)]
    found, misses = [], 0
    for i in range(max_index):
        if i in skip:
            found.append((i, f'Camera {i} (in use)'))
            misses = 0
            continue
        cap = None
        for api in _backends():
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                break
            cap.release()
            cap = None
        if cap is None:
            misses += 1
            if misses >= 2:
                break
            continue
        misses = 0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        found.append((i, f'Camera {i} ({w}x{h})'))
        cap.release()
    return found


# ---------------------------------------------------------------- live camera
def live_session_name(index, name=None):
    """Results folder name for a live session: `name` (e.g. an animal ID) or camera<N>, plus the time."""
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in (name or '').strip())
    return f'{safe or f"camera{index}"}_{time.strftime("%Y%m%d-%H%M%S")}'


class CameraSource:
    """Live camera resampled to a uniform time grid.

    After construction the camera is running and `cam_fps` holds its measured frame rate. Call
    `start(fps)` to begin delivering frames; `read()` then returns (frame, t) with t = k / fps.
    If the consumer falls more than ~1 s behind, the backlog is dropped and `gaps` is incremented
    (t then jumps forward; the monitor restarts its signal buffers on such jumps)."""
    live = True
    duration = None

    def __init__(self, index=0, size=(1920, 1080), wait_permission_s=60.0):
        self.index = index
        self.cap, self.backend = open_camera(index, size, wait_permission_s)
        names = mac_camera_names()
        self.name = names[index] if names and index < len(names) else f'Camera {index}'
        self.ended = False
        self.error = None
        self.gaps = 0
        self.fps = None
        self.shape = None
        self._cond = threading.Condition()
        self._latest = None            # (arrival time, frame)
        self._arrivals = deque(maxlen=256)
        self._grid = None              # (t0, dt) once started
        self._k = 0
        self._prev = None
        self._out = deque()
        self._max_queue = 8
        self._running = True
        self._thread = threading.Thread(target=self._capture, name='camera', daemon=True)
        self._thread.start()
        try:
            self.cam_fps = self._measure_fps()
        except Exception:
            self.release()
            raise

    # ---------- capture thread
    def _normalise(self, frame):
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if self.shape is None:
            self.shape = frame.shape[:2]
        elif frame.shape[:2] != self.shape:     # some drivers change mode on the fly
            frame = cv2.resize(frame, self.shape[::-1], interpolation=cv2.INTER_AREA)
        return frame

    def _capture(self):
        last_ok = time.perf_counter()
        while self._running:
            ok, frame = self.cap.read()
            ts = time.perf_counter()
            if not ok or frame is None or not frame.size:
                if ts - last_ok > 5.0:
                    self.error = 'The camera stopped delivering frames (disconnected?).'
                    break
                time.sleep(0.01)
                continue
            last_ok = ts
            frame = self._normalise(frame)
            with self._cond:
                self._arrivals.append(ts)
                if self._grid is not None:
                    self._resample(ts, frame)
                self._latest = (ts, frame)
                self._cond.notify_all()
        with self._cond:
            self.ended = True
            self._cond.notify_all()

    def _resample(self, ts, frame):
        """Emit every grid tick up to `ts`, each with whichever of the previous and the new frame
        was captured closer to it. Missing camera frames become repeats; extra ones are skipped."""
        t0, dt = self._grid
        while t0 + self._k * dt <= ts:
            tick = t0 + self._k * dt
            prev_t, prev_f = self._latest
            self._out.append((prev_f if tick - prev_t < ts - tick else frame, self._k / self.fps))
            self._k += 1
        if len(self._out) > self._max_queue:
            # processing fell too far behind to catch up: drop the backlog (a jump in the timeline)
            self._out.clear()
            self.gaps += 1

    def _measure_fps(self, settle_s=0.5, span_s=2.0):
        time.sleep(settle_s + span_s)
        with self._cond:
            ts = np.array([a for a in self._arrivals if a > self._arrivals[-1] - span_s]) if self._arrivals else []
        if self.error or len(ts) < 3:
            raise IOError(self.error or f'Camera {self.index} opened but delivers no frames.')
        fps = (len(ts) - 1) / (ts[-1] - ts[0])
        near = min(COMMON_FPS, key=lambda c: abs(c - fps))
        return float(near) if abs(near - fps) < 0.03 * near else round(fps, 2)

    # ---------- consumer side
    def camera_fps(self, span_s=2.0):
        """Frame rate the camera is delivering right now (it can drop in dim light)."""
        with self._cond:
            if len(self._arrivals) < 2:
                return 0.0
            ts = [a for a in self._arrivals if a > self._arrivals[-1] - span_s]
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0

    def latest(self):
        with self._cond:
            return None if self._latest is None else self._latest[1]

    def start(self, fps):
        """Begin delivering frames at `fps` (at most the camera rate)."""
        with self._cond:
            self.fps = min(float(fps), self.cam_fps)
            self._max_queue = max(8, int(round(self.fps)))
            self._grid = (self._latest[0], 1.0 / self.fps)
            self._k = 0
            self._out.clear()

    def describe(self):
        return {'source': 'camera', 'camera': self.name, 'index': self.index, 'backend': self.backend,
                'camera_fps': self.cam_fps, 'fps': self.fps, 'size': self.shape[::-1], 'gaps': self.gaps}

    @property
    def lag_s(self):
        return len(self._out) / self.fps if self.fps else 0.0

    def read(self, timeout=0.5):
        """Next grid frame as (frame, t). (None, None) on timeout or once the camera has ended
        (check `ended`)."""
        with self._cond:
            self._cond.wait_for(lambda: self._out or self.ended, timeout)
            if self._out:
                return self._out.popleft()
        return None, None

    def release(self):
        self._running = False
        self._thread.join(2.0)
        if not self._thread.is_alive():   # never release while a read is still blocking on the device
            self.cap.release()


def choose_processing_fps(src, max_fps=30.0, min_fps=15.0, budget=0.5, params=None, n=40):
    """Processing rate for this machine: the camera rate divided by the smallest integer that keeps
    the monitor's measured per-frame cost below `budget` of real time and the rate at most `max_fps`.
    Returns (fps, seconds per frame)."""
    from .monitor import RespirationMonitor
    mon = RespirationMonitor(src.shape, src.cam_fps, params)
    h, w = src.shape
    mon.process(src.latest(), 0.0)
    mon.set_manual_roi((w * 0.4, h * 0.4, w * 0.6, h * 0.6), 0.0)   # locked: tracking plus the background whole-frame flow
    t0 = time.perf_counter()
    for i in range(1, n + 1):
        mon.process(src.latest(), i / src.cam_fps)
    mon.loc.reset()                    # waits for the whole-frame flow still running on its worker thread
    cost = (time.perf_counter() - t0) / n
    mon.loc.pool.shutdown()
    stride = max(1, math.ceil(src.cam_fps / max_fps - 1e-6), math.ceil(src.cam_fps * cost / budget - 1e-6))
    if src.cam_fps / stride < min_fps:
        stride = max(1, math.floor(src.cam_fps / min_fps))
    return src.cam_fps / stride, cost


def open_camera_source(index=0, size=(1920, 1080), max_fps=30.0, params=None, log=print):
    """Open a camera, measure its real frame rate, pick a processing rate and start it."""
    src = CameraSource(index, size)
    try:
        fps, cost = choose_processing_fps(src, max_fps, params=params)
        src.start(fps)
    except Exception:
        src.release()
        raise
    h, w = src.shape
    reported = src.cap.get(cv2.CAP_PROP_FPS)
    log(f'{src.name} ({src.backend}): {w}x{h}, delivering {src.cam_fps:g} fps'
        + (f' (driver reports {reported:g})' if reported and abs(reported - src.cam_fps) > 0.5 else '')
        + f'; processing {fps:g} fps ({cost * 1000:.0f} ms/frame)')
    if src.cam_fps < 15:
        log(f'Warning: the camera delivers only {src.cam_fps:g} fps (too dark? auto-exposure lowers the frame '
            f'rate). Rates above {src.cam_fps * 0.4 * 60:.0f}/min cannot be measured.')
    return src
