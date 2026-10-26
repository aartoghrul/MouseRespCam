"""Automatic localisation of the breathing region.

Whole-frame dense optical flow at low resolution, averaged in small blocks. Each block's
motion over the last ~20 s is scored for periodicity in the breathing band; the best
compact group of blocks sharing one rate is proposed as the region of interest.

Flow: Farneback (2003). Rate per block: harmonic sum (Noll 1970; Hermes 1988). Score: spectral
concentration around the fundamental and harmonics, in the spirit of the rPPG SNR of de Haan &
Jeanne (2013). Finding the respiratory region from intrinsic properties of respiratory motion
follows the idea of Janssen et al. (2016). See docs/ALGORITHM.md, section 3.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import signal

from .dsp import harmonic_sum


@dataclass
class Proposal:
    box: tuple          # (x0, y0, x1, y1) in full-resolution pixels
    bpm: float
    score: float        # mean periodicity score of the region (0..1)
    n_blocks: int


class BreathLocalizer:
    def __init__(self, frame_shape, fps, p):
        H, W = frame_shape[:2]
        B = p.loc_block
        self.p = p
        # resize to loc_width (fast when it divides the frame width), then crop rows to a block multiple
        self.w = p.loc_width - p.loc_width % B
        self.rh = int(round(H * self.w / W))
        self.h = self.rh - self.rh % B
        self.sx = self.sy = W / self.w
        self.by, self.bx = self.h // B, self.w // B
        if p.loc_flow == 'dis':
            self.dis = cv2.DISOpticalFlow_create(getattr(cv2, f'DISOPTICAL_FLOW_PRESET_{p.loc_dis_preset.upper()}'))
        self.fs = fps / p.loc_stride
        self.buf = deque(maxlen=int(p.loc_window_s * self.fs))
        self.prev = None
        self.pending = deque()  # flow jobs running on a worker thread (OpenCV releases the GIL)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.score_map = None   # last periodicity map (by, bx), for display
        self.sos = signal.butter(2, [0.1, min(5.0, 0.45 * self.fs)], 'band', fs=self.fs, output='sos')

    def reset(self):
        self._drain(wait=True)
        self.buf.clear()
        self.prev = None

    def ready(self):
        # count queued jobs too, so readiness does not depend on worker-thread timing
        return len(self.buf) + len(self.pending) >= self.p.loc_min_s * self.fs

    def push(self, gray_full):
        g = cv2.resize(gray_full, (self.w, self.rh), interpolation=cv2.INTER_AREA)[:self.h]
        if self.prev is not None:
            self.pending.append(self.pool.submit(self._block_flow, self.prev, g))
        self.prev = g
        self._drain()

    def _block_flow(self, a, b):
        if self.p.loc_flow == 'dis':
            fl = self.dis.calc(a, b, None)
        else:
            fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 9, 3, 5, 1.1, 0)
        B = self.p.loc_block
        return fl.reshape(self.by, B, self.bx, B, 2).mean((1, 3))

    def _drain(self, wait=False):
        while self.pending and (wait or self.pending[0].done()):
            self.buf.append(self.pending.popleft().result())

    def localize(self):
        self._drain(wait=True)
        if not self.ready():
            return None
        p = self.p
        D = np.cumsum(np.asarray(self.buf, np.float32), axis=0)       # (T, by, bx, 2) displacement
        D = signal.sosfiltfilt(self.sos, D, axis=0)
        D -= D.mean(0)
        # project each block on its dominant motion direction
        cxx, cyy, cxy = (D[..., 0] ** 2).mean(0), (D[..., 1] ** 2).mean(0), (D[..., 0] * D[..., 1]).mean(0)
        th = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
        v = D[..., 0] * np.cos(th) + D[..., 1] * np.sin(th)            # (T, by, bx)
        amp = v.std(0)
        T = v.shape[0]
        nfft = 2048
        P = np.abs(np.fft.rfft(v * np.hanning(T)[:, None, None], n=nfft, axis=0)) ** 2
        f = np.fft.rfftfreq(nfft, 1 / self.fs)
        Hs = harmonic_sum(P)
        band = np.nonzero((f > p.search_bpm[0] / 60) & (f < p.search_bpm[1] / 60))[0]
        i0 = band[0] + np.argmax(Hs[band], axis=0)                      # (by, bx) fundamental bin
        bpm = f[i0] * 60
        # periodicity score: power at fundamental + harmonics over total (slow surgeon motion counts against)
        cs = np.concatenate([np.zeros((1,) + P.shape[1:]), np.cumsum(P, axis=0)])
        df = f[1]
        bw = int(round(0.08 / df))
        num = np.zeros(i0.shape)
        for k in (1, 2, 3):
            c = np.clip(k * i0, 0, len(f) - 1)
            lo, hi = np.clip(c - bw, 0, len(f)), np.clip(c + bw + 1, 0, len(f))
            num += np.take_along_axis(cs, hi[None], 0)[0] - np.take_along_axis(cs, lo[None], 0)[0]
        tot = cs[np.searchsorted(f, min(6.0, f[-1]))] - cs[np.searchsorted(f, 0.1)]
        score = num / np.maximum(tot, 1e-12)
        # absolute noise floor (static background), not relative to the moving surgeon
        act = score * (amp > 3 * np.percentile(amp, 10))
        self.score_map = act
        thr = min(0.5, max(0.3, np.percentile(act, 98) * 0.8))
        best = None
        for r0 in np.unique(np.round(bpm[act > thr])):
            cand = ((act > thr) & (np.abs(bpm - r0) < 4)).astype(np.uint8)
            n, lab = cv2.connectedComponents(cand, connectivity=8)
            for k in range(1, n):
                reg = lab == k
                if reg.sum() < 3:
                    continue
                val = act[reg].mean() * np.sqrt(reg.sum())
                if best is None or val > best[0]:
                    best = (val, reg)
        if best is None:
            return None
        reg = best[1]
        ys, xs = np.nonzero(reg)
        B = p.loc_block
        box = (int(xs.min() * B * self.sx), int(ys.min() * B * self.sy),
               int((xs.max() + 1) * B * self.sx), int((ys.max() + 1) * B * self.sy))
        return Proposal(box, float(np.median(bpm[reg])), float(act[reg].mean()), int(reg.sum()))
