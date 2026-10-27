"""Accumulates monitor output and writes it to disk (for later comparison with ground truth), and
writes live camera footage."""
import csv
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SaveOptions:
    """Which outputs a session writes. See README, 'Outputs'."""
    rate: bool = True            # rate.csv
    breaths: bool = True         # breaths.csv
    motion: bool = True          # motion.csv
    roi: bool = True             # roi.csv
    summary: bool = True         # summary.png
    footage: bool = True         # footage.mp4 (live sessions)
    footage_scale: float = 0.5   # 0.5: about 1.3 GB/hour at 1080p; 1.0: about 5 GB/hour


def _num(v, fmt='.4g'):
    return '' if v is None or not np.isfinite(v) else format(v, fmt)


class ResultRecorder:
    def __init__(self):
        self.t, self.y, self.art = [], [], []
        self.disp, self.flow, self.state = [], [], []
        self.rates = []       # RateEstimate
        self.breaths = []     # (t, y)
        self.rois = []        # (t, x0, y0, x1, y1, source)

    def add(self, res, bbox=None):
        self.t.append(res.t)
        self.y.append(res.y)
        self.art.append(res.artifact)
        self.disp.append(res.disp)
        self.flow.append(res.flow)
        self.state.append(res.state)
        self.breaths.extend(res.breaths)
        if res.rate is not None:
            self.rates.append(res.rate)
            if bbox is not None:
                self.rois.append((res.t, *bbox, res.roi_source))

    def save(self, out_dir, meta=None, opts=None, title=None):
        """Write the outputs selected in `opts` (all by default) plus session.json."""
        opts = opts or SaveOptions()
        meta = dict(meta or {})
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = []
        if opts.rate:
            files.append('rate.csv')
            with open(out / 'rate.csv', 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['t_s', 'rate_bpm', 'confidence', 'sqi', 'peak_bpm', 'tracker_bpm', 'state'])
                for r in self.rates:
                    w.writerow([f'{r.t:.2f}', _num(r.bpm, '.1f'), r.confidence, f'{r.sqi:.3f}',
                                _num(r.peak_bpm, '.1f'), _num(r.raw_bpm, '.1f'), r.state])
        if opts.breaths:
            files.append('breaths.csv')
            with open(out / 'breaths.csv', 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['t_s'])
                w.writerows([[f'{b:.3f}'] for b, _ in self.breaths])
        if opts.roi:
            files.append('roi.csv')
            with open(out / 'roi.csv', 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['t_s', 'x0', 'y0', 'x1', 'y1', 'source'])
                w.writerows([[f'{r[0]:.2f}', *r[1:]] for r in self.rois])
        if opts.motion:
            # per frame: displayed waveform (band-passed), raw integrated motion along the breathing
            # axis, raw mean optical flow of the region (full-resolution px/frame), artifact flag
            files.append('motion.csv')
            with open(out / 'motion.csv', 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['t_s', 'waveform', 'displacement', 'flow_x_px', 'flow_y_px', 'artifact', 'state'])
                for t, y, d, (fx, fy), a, st in zip(self.t, self.y, self.disp, self.flow, self.art, self.state):
                    w.writerow([f'{t:.4f}', _num(y), _num(d), _num(fx), _num(fy), int(a), st])
        if opts.summary and self.t:
            from .report import summary_plot
            files.append('summary.png')
            summary_plot(self, meta.get('fps', 30.0), title or out.name, out / 'summary.png')
        if (out / 'footage.mp4').exists() or (out / 'footage.avi').exists():
            files.append('footage.mp4' if (out / 'footage.mp4').exists() else 'footage.avi')
        meta.update(saved=time.strftime('%Y-%m-%dT%H:%M:%S'), duration_s=round(self.t[-1], 2) if self.t else 0,
                    breaths=len(self.breaths), files=files)
        with open(out / 'session.json', 'w') as fh:
            json.dump(meta, fh, indent=2, default=str)
        return out


class FootageWriter:
    """Writes frames to a video file on a background thread. The frames are the evenly resampled
    ones the monitor analyses, so replaying the file reproduces the live analysis (checked at half
    resolution: rates within 0.5 /min of full resolution)."""

    def __init__(self, out_dir, fps, frame_shape, scale=0.5):
        h, w = frame_shape[:2]
        self.size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)      # even sizes for the codecs
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(out_dir) / 'footage.mp4'
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'), fps, self.size)
        if not self.writer.isOpened():          # no MP4 encoder in this OpenCV build: MJPG in AVI
            self.path = self.path.with_suffix('.avi')
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'MJPG'), fps, self.size)
        if not self.writer.isOpened():
            raise IOError(f'cannot write footage to {self.path}')
        self.q = queue.Queue(maxsize=max(8, int(fps)))
        self.thread = threading.Thread(target=self._run, name='footage', daemon=True)
        self.thread.start()

    def _run(self):
        while (frame := self.q.get()) is not None:
            if frame.shape[1::-1] != self.size:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            self.writer.write(frame)
        self.writer.release()

    def write(self, frame):
        self.q.put(frame)       # blocks if the encoder falls behind, so no frame is lost

    def close(self):
        self.q.put(None)
        self.thread.join()
