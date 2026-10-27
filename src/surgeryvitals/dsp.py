"""Small DSP building blocks: harmonic-sum spectrum, causal filtering, HMM rate tracking.

Harmonic sum: Noll (1970), Hermes (1988). HMM tracking of a spectral fundamental: as in pitch
trackers such as pYIN (Mauch & Dixon 2014) and PPG heart-rate tracking (Zhang et al. 2015).
Full references in docs/ALGORITHM.md."""
import numpy as np
from scipy import signal


def power_spectrum(x, fs, nfft=8192, axis=0):
    """Hann-windowed power spectrum of a detrended segment (along `axis`)."""
    x = signal.detrend(x, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = x.shape[axis]
    w = np.hanning(x.shape[axis]).reshape(shape)
    P = np.abs(np.fft.rfft(x * w, n=nfft, axis=axis)) ** 2
    return np.fft.rfftfreq(nfft, 1 / fs), P


def harmonic_sum(P, n_harm=3):
    """H[f] = P[f] + P[2f] + P[3f] along axis 0. Impulsive breathing waveforms put a lot of
    power in harmonics; summing them makes the fundamental win (avoids octave errors)."""
    H = P.copy()
    n = P.shape[0]
    for k in range(2, n_harm + 1):
        m = n // k
        H[:m] += P[::k][:m]
    return H


def pitch_salience(P, n_harm=3):
    """Harmonic sum weighted by the candidate's own power. Plain harmonic sums also reward
    sub-harmonics (f0/3 collects P(f0) as its 3rd harmonic); requiring power at the candidate
    itself removes that failure."""
    return harmonic_sum(P, n_harm) * np.sqrt(P / (P.max() + 1e-12))


def harmonic_concentration(P, f, f0, bw=0.1, n_harm=3, lo=0.1, hi=6.0):
    """Fraction of band power within +-bw Hz of f0 and its harmonics (0..1). Signal quality index."""
    num = sum(P[(f > k * f0 - bw) & (f < k * f0 + bw)].sum() for k in range(1, n_harm + 1))
    den = P[(f > lo) & (f < hi)].sum()
    return float(num / den) if den > 0 else 0.0


class CausalBandpass:
    """Sample-by-sample Butterworth band-pass (for the live waveform)."""

    def __init__(self, lo, hi, fs, order=2):
        self.sos = signal.butter(order, [lo, min(hi, 0.45 * fs)], 'band', fs=fs, output='sos')
        self.zi = None

    def reset(self):
        self.zi = None

    def __call__(self, x):
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos) * x
        y, self.zi = signal.sosfilt(self.sos, [x], zi=self.zi)
        return float(y[0])


class RateTracker:
    """Forward HMM over a breaths-per-minute grid. Emission = harmonic-sum spectrum, transition =
    small Gaussian drift plus a tiny uniform jump probability so it can re-lock after a change."""

    def __init__(self, lo, hi, step=0.5, sigma=2.0, jump=2e-3, sharpen=2.0, floor=0.02):
        self.grid = np.arange(lo, hi + step / 2, step)
        k = np.arange(-int(4 * sigma / step), int(4 * sigma / step) + 1) * step
        self.kern = np.exp(-0.5 * (k / sigma) ** 2)
        self.kern /= self.kern.sum()
        self.jump, self.sharpen, self.floor = jump, sharpen, floor
        self.post = None

    def predict(self):
        if self.post is not None:
            p = np.convolve(self.post, self.kern, mode='same')
            p = (1 - self.jump) * p + self.jump / len(p)
            self.post = p / p.sum()

    def current(self):
        return float(self.grid[np.argmax(self.post)]) if self.post is not None else np.nan

    def update(self, f, H):
        lik = np.interp(self.grid / 60.0, f, H)
        lik = (lik / (lik.max() + 1e-12)) ** self.sharpen + self.floor
        if self.post is None:
            self.post = lik / lik.sum()
        else:
            self.predict()
            p = self.post * lik
            self.post = p / p.sum()
        return float(self.grid[np.argmax(self.post)])
