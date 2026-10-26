"""Accumulates monitor output and writes it to disk (for later comparison with ground truth)."""
import csv
from pathlib import Path

import numpy as np


class ResultRecorder:
    def __init__(self):
        self.t, self.y, self.art = [], [], []
        self.rates = []       # RateEstimate
        self.breaths = []     # (t, y)
        self.rois = []        # (t, x0, y0, x1, y1, source)

    def add(self, res, bbox=None):
        self.t.append(res.t)
        self.y.append(res.y)
        self.art.append(res.artifact)
        self.breaths.extend(res.breaths)
        if res.rate is not None:
            self.rates.append(res.rate)
            if bbox is not None:
                self.rois.append((res.t, *bbox, res.roi_source))

    def save(self, out_dir, meta=None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'rate.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['t_s', 'rate_bpm', 'confidence', 'sqi', 'peak_bpm', 'tracker_bpm', 'state'])
            for r in self.rates:
                w.writerow([f'{r.t:.2f}', '' if np.isnan(r.bpm) else f'{r.bpm:.1f}', r.confidence, f'{r.sqi:.3f}',
                            '' if np.isnan(r.peak_bpm) else f'{r.peak_bpm:.1f}',
                            '' if np.isnan(r.raw_bpm) else f'{r.raw_bpm:.1f}', r.state])
        with open(out / 'breaths.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['t_s'])
            w.writerows([[f'{b:.3f}'] for b, _ in self.breaths])
        with open(out / 'roi.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['t_s', 'x0', 'y0', 'x1', 'y1', 'source'])
            w.writerows([[f'{r[0]:.2f}', *r[1:]] for r in self.rois])
        np.savez_compressed(out / 'signal.npz', t=np.array(self.t), y=np.array(self.y, float),
                            artifact=np.array(self.art), **(meta or {}))
        return out
