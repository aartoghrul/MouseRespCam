"""Frame-by-frame respiration monitor: localise -> track -> motion signal -> rate.

Feed frames in order with `process(frame_bgr, t)`; every call returns a FrameResult. A new
RateEstimate is attached once per `rate_every_s`. Nothing here looks into the future, so the
same code serves recorded video (simulated real time) and a live camera.

A rate is reported only when the spectral estimate and the breath-by-breath estimate agree
(agreement-based fusion, Karlen et al. 2013). See docs/ALGORITHM.md, sections 5-7.
"""
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import signal

from .config import Params
from .dsp import CausalBandpass, RateTracker, harmonic_concentration, pitch_salience, power_spectrum
from .localizer import BreathLocalizer
from .tracker import AnchorTracker

SEARCHING, LOCKED, ARTIFACT, LOST = 'searching', 'locked', 'artifact', 'lost'


@dataclass
class RateEstimate:
    t: float
    bpm: float              # reported rate (nan unless confidence is good/fair)
    confidence: str         # 'good' | 'fair' | 'poor' | 'none'
    sqi: float              # harmonic concentration of the window (0..1)
    peak_bpm: float         # from breath-to-breath intervals in the same window
    raw_bpm: float          # tracker output regardless of confidence
    state: str


@dataclass
class FrameResult:
    t: float
    state: str
    quad: np.ndarray | None          # ROI corners in full-res pixels
    y: float = np.nan                # display waveform sample (breaths = upward peaks)
    artifact: bool = False
    breaths: list = field(default_factory=list)   # newly confirmed breaths [(t, y), ...]
    rate: RateEstimate | None = None
    map_updated: bool = False
    roi_source: str | None = None    # 'auto' | 'manual'


def _longest_run(mask):
    """slice of the longest run of True values (or None)."""
    if not mask.any():
        return None
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts, stops = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    k = np.argmax(stops - starts)
    return slice(starts[k], stops[k])


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class RespirationMonitor:
    def __init__(self, frame_shape, fps, params: Params | None = None):
        self.p = p = params or Params()
        self.fps = fps
        self.loc = BreathLocalizer(frame_shape, fps, p)
        self.trk = AnchorTracker(p)
        self.hmm = RateTracker(*p.rate_bpm, sigma=p.hmm_sigma_bpm)
        self.bp = CausalBandpass(*p.display_band, fs=fps)
        self.state = SEARCHING
        self.roi_source = None
        self.recovering = False
        self.idx = 0
        self.last_loc_t = -np.inf
        self.next_rate_t = 0.0
        self.poor_since = None
        self.last_breath_t = -np.inf
        self.breath_log = []     # all confirmed breaths [(t, y)] since start
        self.prev_gray = None
        self.last_rekey = -np.inf
        self._reset_signal()

    # ------------------------------------------------------------------ control
    def _reset_signal(self):
        n = int(self.p.rate_window_s * self.fps)
        self.prev_roi = None
        self.s = 0.0
        self.axis = np.array([0.0, 1.0])
        self.cov = None
        self.conf_bpm = None      # (t, bpm) of the last good/fair estimate in this region
        self.reg_bpm = None       # (t, bpm) from regular breath-to-breath intervals
        self.polarity = 1.0
        self.polarity_set = False
        self.flip_votes = 0
        self.inc, self.sbuf, self.abuf, self.tbuf, self.ybuf = (deque(maxlen=n) for _ in range(5))
        self.sec_max = deque(maxlen=60)   # per-second peak ROI motion, ALL frames (breaths set the scale)
        self.cur_max, self.cur_sec = 0.0, None
        self.art_until = -np.inf
        self.bp.reset()

    def lock(self, gray, box, source, t):
        H, W = gray.shape
        x0, y0, x1, y1 = box
        box = (max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1)))
        self.trk.init(gray, box)
        self.last_rekey = t
        self.roi_source = source
        self.state = LOCKED
        self.recovering = False
        self.poor_since = None
        self._reset_signal()

    def set_manual_roi(self, box, t):
        if self.prev_gray is not None and box[2] - box[0] >= 16 and box[3] - box[1] >= 16:
            self.lock(self.prev_gray, box, 'manual', t)

    def redetect(self):
        self.state = SEARCHING
        self.recovering = False
        self.trk.ref_des = None
        self.roi_source = None
        self.last_loc_t = -np.inf
        self._reset_signal()

    def bbox(self):
        if not self.trk.initialised:
            return None
        q = self.trk.quad()
        return (*q.min(0).astype(int), *q.max(0).astype(int))

    # ------------------------------------------------------------------ main step
    def process(self, frame_bgr, t):
        p, fps = self.p, self.fps
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        res = FrameResult(t=t, state=self.state, quad=None)

        if self.idx % p.loc_stride == 0:
            self.loc.push(gray)

        # --- anchor tracking
        if self.trk.initialised and self.idx % p.reg_every == 0 and self.idx > 0:
            ok = self.trk.register(gray)
            if self.state in (LOCKED, ARTIFACT) and self.trk.fail_frames / fps >= p.lost_after_s:
                self.state, self.recovering = LOST, True
                self.loc.reset()
                self._reset_signal()
            elif self.recovering and ok:
                self.state, self.recovering = LOCKED, False   # same scene came back: re-use the region
                self._reset_signal()
            elif ok and self.prev_gray is not None and self.state in (LOCKED, ARTIFACT):
                self.prev_roi = self._roi(self.prev_gray)       # re-sample previous frame with new transform
                if self.state == LOCKED and t - self.last_rekey >= p.rekey_every_s:
                    self.trk.rekey(gray)
                    self.last_rekey = t

        # --- automatic (re)localisation
        period = p.loc_search_every_s if self.state in (SEARCHING, LOST) else p.loc_every_s
        if self.loc.ready() and t - self.last_loc_t >= period:
            self.last_loc_t = t
            prop = self.loc.localize()
            res.map_updated = True
            if prop is not None and prop.score >= p.loc_min_score:
                if self.state in (SEARCHING, LOST):
                    self.lock(gray, prop.box, 'auto', t)
                elif (self.roi_source == 'auto' and self.poor_since is not None
                      and t - self.poor_since > p.relocate_after_s and _iou(prop.box, self.bbox()) < 0.3):
                    self.lock(gray, prop.box, 'auto', t)

        # --- motion signal inside the tracked region
        if self.state in (LOCKED, ARTIFACT):
            roi = self._roi(gray)
            y, art = np.nan, False
            if self.prev_roi is not None and self.prev_roi.shape == roi.shape:
                fl = cv2.calcOpticalFlowFarneback(self.prev_roi, roi, None, 0.5, 2, 9, 3, 5, 1.1, 0)
                v = fl.reshape(-1, 2).mean(0)
                mag = float(np.sqrt((fl ** 2).sum(2)).mean())
                # artifact = motion several times larger than a typical breath. The scale comes from the
                # median per-second peak over the last minute, including flagged frames, so the
                # threshold cannot ratchet down and start rejecting breaths.
                sec = int(t)
                if sec != self.cur_sec:
                    if self.cur_sec is not None:
                        self.sec_max.append(self.cur_max)
                    self.cur_sec, self.cur_max = sec, 0.0
                self.cur_max = max(self.cur_max, mag)
                if len(self.sec_max) >= 5:
                    thr = float(np.clip(p.artifact_k * np.median(self.sec_max), p.artifact_floor, p.artifact_ceiling))
                else:
                    thr = p.artifact_ceiling
                if mag > thr or self.trk.fail_frames > 0:
                    self.art_until = t + p.artifact_hold_s
                art = t < self.art_until
                inc = np.zeros(2) if art else v
                self.inc.append(inc)
                self.s += float(inc @ (self.axis * self.polarity))
                y = self.bp(self.s)
                self.sbuf.append(self.s)
                self.abuf.append(art)
                self.tbuf.append(t)
                self.ybuf.append(y)
                if self.idx % 3 == 0:
                    res.breaths = self._detect_breaths(t)
            self.prev_roi = roi
            self.state = ARTIFACT if art else LOCKED
            res.y, res.artifact = y, art

        self.prev_gray = gray
        if t >= self.next_rate_t:
            self.next_rate_t = t + p.rate_every_s
            res.rate = self._estimate_rate(t)

        res.state = self.state
        res.roi_source = self.roi_source
        if self.trk.initialised and self.state != SEARCHING:
            res.quad = self.trk.quad()
        self.idx += 1
        return res

    # ------------------------------------------------------------------ helpers
    def _roi(self, gray):
        """Region resampled to a small fixed-size patch (area-averaged, then warped)."""
        q = self.trk.quad()
        x0, y0, x1, y1 = self.trk.box
        s = max(0.25, self.p.roi_target_px / min(x1 - x0, y1 - y0))
        w, h = max(8, int((x1 - x0) * s)), max(8, int((y1 - y0) * s))
        H, W = gray.shape
        bx0, by0 = np.clip(np.floor(q.min(0)).astype(int) - 2, 0, [W - 1, H - 1])
        bx1, by1 = np.clip(np.ceil(q.max(0)).astype(int) + 2, 1, [W, H])
        small = cv2.resize(gray[by0:by1, bx0:bx1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        qs = ((q - [bx0, by0]) * s).astype(np.float32)
        M = cv2.getAffineTransform(qs[:3], np.float32([[0, 0], [w, 0], [w, h]]))
        return cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _update_axis(self, t):
        """Principal motion direction of the region; polarity chosen so breaths are sharp upward peaks."""
        fps = self.fps
        if len(self.inc) < 3 * fps:
            return
        D = signal.detrend(np.cumsum(np.asarray(self.inc), axis=0), axis=0)
        f0_bpm = None
        if self.conf_bpm is not None and t - self.conf_bpm[0] < 30:
            f0_bpm = self.conf_bpm[1]
        elif self.reg_bpm is not None and t - self.reg_bpm[0] < 5:
            f0_bpm = self.reg_bpm[1]          # regular breath intervals, even if the spectrum is unsure
        if f0_bpm is not None:
            # direction carrying the most power at the breathing rate and its harmonics, so slow
            # pushes or fast hand jitter cannot rotate the axis away from breathing
            X = np.fft.rfft(D * np.hanning(len(D))[:, None], n=8192, axis=0)
            f = np.fft.rfftfreq(8192, 1 / fps)
            f0 = f0_bpm / 60
            sel = np.zeros(len(f), bool)
            for k in (1, 2, 3):
                sel |= np.abs(f - k * f0) < 0.08
            C = np.real(X[sel].conj().T @ X[sel])
        else:                         # no trusted rate yet: broadband breathing band
            C = np.cov(signal.sosfiltfilt(signal.butter(2, [0.2, 5], 'band', fs=fps, output='sos'), D, axis=0).T)
        C = C / (np.trace(C) + 1e-12)
        self.cov = C if self.cov is None else 0.8 * self.cov + 0.2 * C
        w, V = np.linalg.eigh(self.cov)
        a = V[:, -1]
        if a @ self.axis < 0:
            a = -a
        self.axis = a
        if len(self.inc) < 6 * fps:
            return
        # asymmetry: sharp breath spikes reach much further from the median than the slow phase
        z = signal.sosfiltfilt(signal.butter(2, [0.2, 5], 'band', fs=fps, output='sos'), D, axis=0) @ a
        lo, med, hi = np.percentile(z, [1, 50, 99])
        asym = ((hi - med) - (med - lo)) / (hi - lo + 1e-12)     # >0: spikes point up
        if not self.polarity_set:
            if abs(asym) > 0.1:
                self.polarity, self.polarity_set = float(np.sign(asym)), True
                self.last_breath_t = max(self.last_breath_t, t)   # older samples had unknown polarity
            return
        if asym * self.polarity < -0.15:
            self.flip_votes += 1
            if self.flip_votes >= 3:
                self.polarity, self.flip_votes = -self.polarity, 0
                self.last_breath_t = max(self.last_breath_t, t)   # don't detect on pre-flip samples
        else:
            self.flip_votes = 0

    def _detect_breaths(self, t):
        fps = self.fps
        if not self.polarity_set:
            return []
        y = np.asarray(self.ybuf)
        art = np.asarray(self.abuf)
        tt = np.asarray(self.tbuf)
        ok = ~art & np.isfinite(y)
        if ok.sum() < 3 * fps:
            return []
        lo, hi = np.percentile(y[ok], [5, 95])
        # fixed limits, deliberately NOT derived from the tracked rate: otherwise a wrong rate
        # (e.g. a sub-harmonic) makes the detector skip breaths and "confirm" itself
        dist = int(0.3 * fps)                      # <= 200 breaths/min
        pk, _ = signal.find_peaks(np.nan_to_num(y), prominence=0.3 * (hi - lo), distance=dist,
                                  wlen=int(2.5 * fps) | 1)
        out = []
        for i in pk:
            ti = tt[i]
            if t - 3.0 < ti < t - 0.35 and not art[i] and ti > self.last_breath_t + dist / fps:
                out.append((float(ti), float(y[i])))
                self.last_breath_t = ti
        self.breath_log.extend(out)
        return out


    def _estimate_rate(self, t):
        p, fps = self.p, self.fps
        none = RateEstimate(t, np.nan, 'none', 0.0, np.nan, np.nan, self.state)
        if self.state in (LOCKED, ARTIFACT):
            self._update_axis(t)
        if self.state not in (LOCKED, ARTIFACT) or len(self.sbuf) < 8 * fps:
            self.hmm.predict()
            self._mark_poor(t, 'none')
            return none
        art = np.asarray(self.abuf)
        seg = _longest_run(~art)
        if seg is None or seg.stop - seg.start < p.min_clean_s * fps:
            # artifacts break the window into pieces too short to trust: hold the tracker
            self.hmm.predict()
            self._mark_poor(t, 'none')
            return none
        f, P = power_spectrum(np.asarray(self.sbuf)[seg], fps)
        P[f < 0.9 * p.rate_bpm[0] / 60] = 0   # below the band: drift and the surgeon's own breathing
        sal = pitch_salience(P)
        band = (f >= p.rate_bpm[0] / 60) & (f <= p.rate_bpm[1] / 60)
        f_best = f[band][np.argmax(sal[band])]
        if harmonic_concentration(P, f, f_best) >= p.min_update_sqi:
            bpm = self.hmm.update(f, sal)
        else:                       # window not periodic: don't let it drag the tracker, just hold
            self.hmm.predict()
            bpm = self.hmm.current()
        sqi = harmonic_concentration(P, f, bpm / 60) if np.isfinite(bpm) else 0.0
        bt = np.array([b for b, _ in self.breath_log if b > t - p.rate_window_s])
        peak_bpm, cv = np.nan, np.inf
        if len(bt) >= 4:
            iv = np.diff(bt)
            iv = iv[iv < 2.5 * np.median(iv)]
            if len(iv) >= 3:
                peak_bpm = 60.0 / np.median(iv)
                cv = np.std(iv) / np.mean(iv)          # anaesthetised breathing is regular
                if cv < 0.25:
                    self.reg_bpm = (t, peak_bpm)
        agree = np.isfinite(peak_bpm) and abs(peak_bpm - bpm) <= 0.15 * bpm
        if sqi >= 0.5 and agree and cv < 0.25:
            conf = 'good'
        elif sqi >= 0.35 and agree and cv < 0.4:
            conf = 'fair'
        else:
            conf = 'poor'
        if conf in ('good', 'fair'):
            self.conf_bpm = (t, bpm)
        self._mark_poor(t, conf)
        return RateEstimate(t, bpm if conf in ('good', 'fair') else np.nan, conf, sqi, peak_bpm, bpm, self.state)

    def _mark_poor(self, t, conf):
        if conf in ('poor', 'none'):
            self.poor_since = self.poor_since if self.poor_since is not None else t
        else:
            self.poor_since = None
