"""Regenerate the figures and animations in docs/ALGORITHM.md from the recordings.

    .venv/bin/python docs/figures/make_figures.py run       # process recordings, cache internals
    .venv/bin/python docs/figures/make_figures.py figures   # static figures -> docs/figures/*.png
    .venv/bin/python docs/figures/make_figures.py anims     # animations     -> docs/figures/*.gif
    .venv/bin/python docs/figures/make_figures.py all

Everything is computed by the package itself (RespirationMonitor, BreathLocalizer, AnchorTracker,
dsp). To expose intermediate values without changing the package, the runner taps two calls inside
surgeryvitals.monitor: the region optical flow and the rate spectrum. Cached runs go to
results/figcache/ (git-ignored). Needs videos/Recording_{1,2,3}.mp4 and
videos/synthetic/dynamic_test.mp4 (exploration/scripts/make_dynamic_video.py).
"""
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

import surgeryvitals.monitor as M  # noqa: E402
from surgeryvitals.config import Params  # noqa: E402

OUT = ROOT / 'docs' / 'figures'
CACHE = ROOT / 'results' / 'figcache'
VID = ROOT / 'videos'

# (cache name, video, start s, end s or None)
RUNS = [
    ('rec1', VID / 'Recording_1.mp4', 0, None),
    ('rec2', VID / 'Recording_2.mp4', 0, None),
    ('rec3', VID / 'Recording_3.mp4', 0, None),
    ('dyn', VID / 'synthetic' / 'dynamic_test.mp4', 60, 170),
]
F_KEEP = 4.0   # Hz; spectra are stored up to here


# ----------------------------------------------------------------------------- instrumented run
class _FlowTap:
    """Stands in for the cv2 module inside surgeryvitals.monitor and keeps the last region flow."""

    def __init__(self):
        self.last = None

    def __getattr__(self, name):
        return getattr(cv2, name)

    def calcOpticalFlowFarneback(self, *a, **kw):
        self.last = cv2.calcOpticalFlowFarneback(*a, **kw)
        return self.last


class _SpecTap:
    """Wraps monitor.pitch_salience to keep the spectrum (as used, after the floor) and salience."""

    def __init__(self, fn):
        self.fn, self.P, self.sal = fn, None, None

    def __call__(self, P, *a, **kw):
        self.P = P.copy()
        self.sal = self.fn(P, *a, **kw)
        return self.sal


def run(name, video, t0=0.0, t1=None):
    flow, spec = _FlowTap(), _SpecTap(M.pitch_salience)
    M.cv2, M.pitch_salience = flow, spec
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if t0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t0 * fps))
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mon = M.RespirationMonitor((H, W), fps)
    p = mon.p
    fr = defaultdict(list)
    rates, breaths, maps, locks = [], [], [], []
    f_spec, prev_box, i = None, None, 0
    while True:
        ok, frame = cap.read()
        t = t0 + i / fps
        if not ok or (t1 is not None and t >= t1):
            break
        flow.last, spec.sal = None, None
        res = mon.process(frame, t)
        fr['t'].append(t)
        fr['state'].append(res.state)
        fr['quad'].append(res.quad.ravel() if res.quad is not None else np.full(8, np.nan))
        fr['y'].append(res.y)
        fr['art'].append(res.artifact)
        locked = res.state in (M.LOCKED, M.ARTIFACT)
        fr['s'].append(mon.s if locked else np.nan)
        if flow.last is not None:
            fl = flow.last
            fr['v'].append(fl.reshape(-1, 2).mean(0))
            fr['mag'].append(float(np.sqrt((fl ** 2).sum(2)).mean()))
        else:
            fr['v'].append(np.full(2, np.nan))
            fr['mag'].append(np.nan)
        sm = mon.sec_max
        fr['thr'].append(float(np.clip(p.artifact_k * np.median(sm), p.artifact_floor, p.artifact_ceiling))
                         if len(sm) >= 5 else p.artifact_ceiling)
        fr['axis'].append(mon.axis.copy())
        fr['pol'].append(mon.polarity if mon.polarity_set else 0.0)
        reg = mon.trk.initialised and (mon.idx - 1) % p.reg_every == 0 and mon.idx > 1
        fr['inliers'].append(mon.trk.inliers if reg else -1)
        if mon.trk.box is not None and mon.trk.box != prev_box:
            locks.append((t, tuple(int(v) for v in mon.trk.box), res.roi_source))
            prev_box = mon.trk.box
        breaths += res.breaths
        if res.map_updated and mon.loc.score_map is not None:
            maps.append((t, mon.loc.score_map.astype(np.float16)))
        if res.rate is not None:
            r = dict(vars(res.rate))
            if spec.sal is not None:
                n = len(spec.P)
                f = np.fft.rfftfreq(2 * (n - 1), 1 / fps)
                keep = f <= F_KEEP
                f_spec = f[keep]
                r['P'] = spec.P[keep].astype(np.float32)
                r['sal'] = spec.sal[keep].astype(np.float32)
            r['post'] = None if mon.hmm.post is None else mon.hmm.post.astype(np.float32)
            rates.append(r)
        i += 1
    cap.release()
    out = {k: np.asarray(v) for k, v in fr.items()}
    out.update(fps=fps, shape=(H, W), rates=rates, breaths=np.asarray(breaths), maps=maps, locks=locks,
               f_spec=f_spec, hmm_grid=mon.hmm.grid, video=str(video), t0=t0)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f'{name}.pkl', 'wb') as fh:
        pickle.dump(out, fh)
    return name, len(fr['t'])


def load(name):
    with open(CACHE / f'{name}.pkl', 'rb') as fh:
        return pickle.load(fh)


def run_all():
    with ProcessPoolExecutor(max_workers=len(RUNS) + 1) as ex:
        jobs = [ex.submit(run, *r) for r in RUNS] + [ex.submit(loc_run, 'loc3', VID / 'Recording_3.mp4', 0, 24)]
        for j in jobs:
            print('%s: %d frames' % j.result())


# ----------------------------------------------------------------------------- localiser internals
def loc_run(name, video, t0, t1):
    """Whole-frame block flow exactly as BreathLocalizer computes it, kept for the whole segment,
    plus the 480-px colour frames it was computed from (for display)."""
    from surgeryvitals.localizer import BreathLocalizer
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(t0 * fps))
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    p = Params()
    loc = BreathLocalizer((H, W), fps, p)
    flows, times, frames, prev = [], [], [], None
    for i in range(int((t1 - t0) * fps)):
        ok, f = cap.read()
        if not ok:
            break
        if i % p.loc_stride:
            continue
        small = cv2.resize(f, (loc.w, loc.rh), interpolation=cv2.INTER_AREA)[:loc.h]
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flows.append(loc._block_flow(prev, g))
            times.append(t0 + i / fps)
            frames.append(small[..., ::-1])
        prev = g
    out = dict(flows=np.float32(flows), t=np.asarray(times), frames=np.uint8(frames), fs=loc.fs,
               scale=loc.sx, block=p.loc_block, shape=(H, W))
    with open(CACHE / f'{name}.pkl', 'wb') as fh:
        pickle.dump(out, fh)
    return name, len(times)


def periodicity(flows, fs, p=None):
    """The scoring steps of BreathLocalizer.localize, keeping the intermediate arrays.
    Checked against the localiser's own output in check_periodicity()."""
    from scipy import signal
    from surgeryvitals.dsp import harmonic_sum
    p = p or Params()
    sos = signal.butter(2, [0.1, min(5.0, 0.45 * fs)], 'band', fs=fs, output='sos')
    D = signal.sosfiltfilt(sos, np.cumsum(np.asarray(flows, np.float32), axis=0), axis=0)
    D -= D.mean(0)
    cxx, cyy, cxy = (D[..., 0] ** 2).mean(0), (D[..., 1] ** 2).mean(0), (D[..., 0] * D[..., 1]).mean(0)
    th = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    v = D[..., 0] * np.cos(th) + D[..., 1] * np.sin(th)
    amp = v.std(0)
    T, nfft = v.shape[0], 2048
    P = np.abs(np.fft.rfft(v * np.hanning(T)[:, None, None], n=nfft, axis=0)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    Hs = harmonic_sum(P)
    band = np.nonzero((f > p.search_bpm[0] / 60) & (f < p.search_bpm[1] / 60))[0]
    i0 = band[0] + np.argmax(Hs[band], axis=0)
    bpm = f[i0] * 60
    bw = int(round(0.08 / f[1]))
    cs = np.concatenate([np.zeros((1,) + P.shape[1:]), np.cumsum(P, axis=0)])
    num = np.zeros(i0.shape)
    for k in (1, 2, 3):
        c = np.clip(k * i0, 0, len(f) - 1)
        lo, hi = np.clip(c - bw, 0, len(f)), np.clip(c + bw + 1, 0, len(f))
        num += np.take_along_axis(cs, hi[None], 0)[0] - np.take_along_axis(cs, lo[None], 0)[0]
    tot = cs[np.searchsorted(f, min(6.0, f[-1]))] - cs[np.searchsorted(f, 0.1)]
    score = num / np.maximum(tot, 1e-12)
    floor = amp > 3 * np.percentile(amp, 10)
    act = score * floor
    thr = min(0.5, max(0.3, np.percentile(act, 98) * 0.8))
    best, comps = None, []
    for r0 in np.unique(np.round(bpm[act > thr])):
        cand = ((act > thr) & (np.abs(bpm - r0) < 4)).astype(np.uint8)
        n, lab = cv2.connectedComponents(cand, connectivity=8)
        for k in range(1, n):
            reg = lab == k
            if reg.sum() < 3:
                continue
            val = act[reg].mean() * np.sqrt(reg.sum())
            comps.append((val, reg, r0))
            if best is None or val > best[0]:
                best = (val, reg, r0)
    return dict(v=v, amp=amp, P=P, f=f, H=Hs, bpm=bpm, score=score, floor=floor, act=act, thr=thr,
                best=best, comps=comps, i0=i0)


def check_periodicity(L, k1):
    """The replica must give the localiser's own map and region."""
    from collections import deque
    from surgeryvitals.localizer import BreathLocalizer
    p = Params()
    loc = BreathLocalizer(L['shape'], L['fs'] * p.loc_stride, p)
    n = int(p.loc_window_s * loc.fs)
    loc.buf = deque(L['flows'][max(0, k1 - n):k1], maxlen=n)
    prop = loc.localize()
    if loc.score_map is None:
        return None, None
    mine = periodicity(L['flows'][max(0, k1 - n):k1], L['fs'])
    assert np.allclose(mine['act'], loc.score_map, atol=1e-5), 'periodicity replica diverged from localizer'
    return prop, mine


# ----------------------------------------------------------------------------- style
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch  # noqa: E402

SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'      # categorical slots 1-3
GOOD, FAIR, POOR, LOST, NONE = '#0ca30c', '#fab219', '#ec835a', '#d03b3b', '#c3c2b7'
CONF = {'good': GOOD, 'fair': FAIR, 'poor': POOR, 'none': NONE}
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])

plt.rcParams.update({
    'font.family': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 9,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 10,
    'axes.titlelocation': 'left', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2,
    'ytick.labelcolor': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'savefig.facecolor': SURF,
    'legend.frameon': False, 'legend.fontsize': 8, 'lines.solid_capstyle': 'round', 'grid.color': GRID,
    'grid.linewidth': 0.6,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=110, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    print('wrote', OUT / name)


def faded(img, keep=0.55):
    """Photo pushed toward grey-white so coloured overlays carry the information."""
    g = img.astype(np.float32).mean(2, keepdims=True)
    out = keep * (0.35 * img + 0.65 * g) + (1 - keep) * 245
    return np.clip(out, 0, 255).astype(np.uint8)


def label(ax, x, y, text, color=INK, **kw):
    """Text on a photo, with a light halo."""
    import matplotlib.patheffects as pe
    kw.setdefault('fontsize', 8.5)
    return ax.text(x, y, text, color=color, path_effects=[pe.withStroke(linewidth=2.6, foreground=SURF)], **kw)


def overlay(mask, color, alpha=1.0):
    """RGBA image: `color` where mask is true, transparent elsewhere."""
    out = np.zeros(mask.shape + (4,))
    out[..., :3] = to_rgb(color)
    out[..., 3] = np.asarray(mask, float) * alpha
    return out


def minus(x, spec):
    """Number with a typographic minus sign."""
    return format(x, spec).replace('-', '−')


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ----------------------------------------------------------------------------- stage A figures
LOC_T = 20.0     # Recording 3: first 20 s, the window the localiser uses at t = 20 s


def _loc_at(L, t):
    k = int(np.searchsorted(L['t'], t))
    prop, m = check_periodicity(L, k)
    return k, prop, m


def fig_blocks():
    L = load('loc3')
    k, prop, m = _loc_at(L, LOC_T)
    fr = L['frames'][k - 1]
    B, fs, sc = L['block'], L['fs'], L['scale']
    f_bpm = m['f'] * 60
    # four illustrative blocks, picked from the data rather than by hand
    by, bx = m['act'].shape
    yy, xx = np.mgrid[:by, :bx]
    region = m['best'][1]
    P = m['P']
    pick = {
        'mouse': np.unravel_index(np.argmax(np.where(region, m['act'], -1)), region.shape),
        'glove': np.unravel_index(np.argmax(np.where((yy < 20) & (xx > 8) & (xx < 30), m['amp'], -1)), region.shape),
        'background': np.unravel_index(np.argmin(np.where((yy < 12) & (xx > 50), m['amp'], np.inf)), region.shape),
    }
    rows = [('mouse', '1  Mouse (back, beside the ear bar)', BLUE),
            ('glove', '2  Surgeon\'s glove', INK2),
            ('background', '3  Static background', MUTED)]

    fig = plt.figure(figsize=(10, 9.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[4.6, 1, 1, 1], width_ratios=[1.25, 1], hspace=0.55, wspace=0.16)
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(faded(fr, 0.75))
    for x in range(0, fr.shape[1] + 1, B):
        ax.axvline(x - 0.5, color='white', lw=0.35, alpha=0.45)
    for y in range(0, fr.shape[0] + 1, B):
        ax.axhline(y - 0.5, color='white', lw=0.35, alpha=0.45)
    for n, (key, title, col) in enumerate(rows, 1):
        r, c = pick[key]
        ax.add_patch(Rectangle((c * B - 0.5, r * B - 0.5), B, B, fill=False, ec=col, lw=2.6))
        label(ax, c * B + B + 3, r * B + B / 2, str(n), va='center', fontsize=12, fontweight='bold')
    ax.set_title(f'Recording 3, t = {LOC_T:.0f} s: the frame at 480 px wide, cut into {by}×{bx} blocks of '
                 f'{B}×{B} px (≈{B * sc:.0f} px at full resolution)')
    bare(ax)

    t = np.arange(m['v'].shape[0]) / fs + (L['t'][k - 1] - m['v'].shape[0] / fs)
    p = Params()
    for i, (key, title, col) in enumerate(rows):
        r, c = pick[key]
        a1 = fig.add_subplot(gs[i + 1, 0])
        a1.set_xticks(np.arange(0, 21, 5))
        a1.plot(t, m['v'][:, r, c] * sc, color=col, lw=1.2)
        a1.set_title(title, fontsize=9.5)
        a1.set_ylabel('px')
        a1.set_xlim(t[0], t[-1])
        a1.grid(axis='y')
        if i == len(rows) - 1:
            a1.set_xlabel('time (s)')
        a2 = fig.add_subplot(gs[i + 1, 1])
        sel = f_bpm <= 240
        Pn = P[:, r, c] / P[sel, r, c].max()
        a2.axvspan(0, p.search_bpm[0], color=GRID, lw=0)
        f0 = m['bpm'][r, c]
        for h in (1, 2, 3):
            a2.axvspan(h * f0 - 4.8, h * f0 + 4.8, color=col, alpha=0.18, lw=0)
        a2.fill_between(f_bpm[sel], Pn[sel], color=col, alpha=0.10, lw=0)
        a2.plot(f_bpm[sel], Pn[sel], color=col, lw=1.2)
        a2.set_xlim(0, 240)
        a2.set_ylim(0, 1.08)
        a2.set_yticks([])
        a2.spines['left'].set_visible(False)
        verdict = 'in the region' if region[r, c] else ('below noise floor' if not m['floor'][r, c] else 'rejected')
        a2.text(0.99, 0.93, f'f₀ = {f0:.0f}/min   score = {m["act"][r, c]:.2f}   ({verdict})', transform=a2.transAxes,
                ha='right', va='top', fontsize=8.5, color=INK)
        if i == 0:
            a2.text(p.search_bpm[0] / 2, 0.5, 'below\nsearch\nband', ha='center', va='center', fontsize=7.5, color=INK2)
            a2.set_title('Power spectrum (shaded: f₀, 2f₀, 3f₀ ± 0.08 Hz)', fontsize=9.5)
        if i == len(rows) - 1:
            a2.set_xlabel('breaths / min')
    save(fig, 'A1_block_periodicity.png')


def fig_region():
    L = load('loc3')
    k, prop, m = _loc_at(L, LOC_T)
    fr = L['frames'][k - 1]
    B = L['block']
    up = lambda a: np.kron(a, np.ones((B, B)))
    zoom = (200, 480, 250, 106)      # x0, x1, y1, y0 of panels b and c (480-px frame)
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 2.75), gridspec_kw={'wspace': 0.05})
    # (a) periodicity score, whole frame
    ax = axs[0]
    ax.imshow(faded(fr))
    im = ax.imshow(np.ma.masked_less(up(m['act']), 0.1), cmap=SEQ, vmin=0, vmax=1, alpha=0.9)
    ax.add_patch(Rectangle((zoom[0], zoom[3]), zoom[1] - zoom[0], zoom[2] - zoom[3], fill=False, ec=INK2, lw=0.8))
    label(ax, zoom[0] + 3, zoom[3] - 3, 'b, c', fontsize=7.5, color=INK2)
    ax.set_title('a  Periodicity score per block (whole frame)')
    cax = ax.inset_axes([0.03, 0.9, 0.3, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0.1, 0.5, 1])
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=7, length=2, colors=MUTED, labelcolor=INK2)
    # (b) above threshold, grouped by rate
    ax = axs[1]
    ax.imshow(faded(fr))
    above = m['act'] > m['thr']
    val, win, r0 = m['best']
    ax.imshow(overlay(up(above & ~win), MUTED, 0.9))
    ax.imshow(overlay(up(win), BLUE, 0.9))
    ax.set_title(f'b  Score > τ = {m["thr"]:.2f}, grouped by rate (±4/min)')
    ys, xs = np.nonzero(win)
    label(ax, xs.min() * B - 5, ys.min() * B + 4,
          f'{win.sum()} connected blocks\nat {r0:.0f} breaths/min,\nmean score {m["act"][win].mean():.2f}',
          va='top', ha='right')
    if (above & ~win).any():
        label(ax, zoom[0] + 4, zoom[2] - 4, 'grey: other blocks above τ', va='bottom', fontsize=7.5)
    # (c) region
    ax = axs[2]
    ax.imshow(faded(fr, 0.9))
    x0, y0, x1, y1 = np.asarray(prop.box) / L['scale']
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=GOOD, lw=2))
    label(ax, x0 - 5, y0 + 4, f'region {prop.box[2] - prop.box[0]}×{prop.box[3] - prop.box[1]} px\n'
          f'(full resolution)', va='top', ha='right')
    ax.set_title('c  Bounding box of the winning group')
    for ax in axs[1:]:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    for ax in axs:
        bare(ax)
    save(fig, 'A2_region_selection.png')


# ----------------------------------------------------------------------------- stage B figures
def read_frames(video, times):
    """Frames (BGR) at the given times, in order."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
        out.append(cap.read()[1])
    cap.release()
    return out


def mouse_tracker_test(t0=30.0, dur=15.0, box=(1248, 736, 1376, 992)):
    """What happens if the region follows the mouse instead of the scene: a frame-to-frame template
    tracker (sub-pixel) is compared with the fixed, scene-anchored region, using the monitor's region
    flow settings on both."""
    from scipy import signal
    cache = CACHE / 'mouse_tracker_test.npz'
    if cache.exists():
        return dict(np.load(cache))
    cap = cv2.VideoCapture(str(VID / 'Recording_3.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(t0 * fps))
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    s = max(0.25, Params().roi_target_px / min(w, h))
    pos, tmpl, prev = np.array([x0, y0], float), None, None
    va, vt, P = [], [], []

    def patch(g, x, y):
        return cv2.warpAffine(g, np.float32([[1, 0, -x], [0, 1, -y]]), (w, h), flags=cv2.INTER_LINEAR)

    def small(p):
        return cv2.resize(p, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.uint8)

    def vertex(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d == 0 else 0.5 * (a - c) / d

    for i in range(int(dur * fps)):
        g = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if tmpl is not None:
            sx, sy = int(pos[0]) - 12, int(pos[1]) - 12
            r = cv2.matchTemplate(g[sy:sy + h + 24, sx:sx + w + 24], tmpl, cv2.TM_CCOEFF_NORMED)
            mx, my = cv2.minMaxLoc(r)[3]
            dx = vertex(*r[my, mx - 1:mx + 2]) if 0 < mx < r.shape[1] - 1 else 0.0
            dy = vertex(*r[my - 1:my + 2, mx]) if 0 < my < r.shape[0] - 1 else 0.0
            pos = np.array([sx + mx + dx, sy + my + dy])
        tmpl = patch(g, *pos)
        P.append(pos.copy())
        cur = (small(patch(g, x0, y0)), small(tmpl))
        if prev is not None:
            fl = [cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 9, 3, 5, 1.1, 0).reshape(-1, 2).mean(0)
                  for a, b in zip(prev, cur)]
            va.append(fl[0])
            vt.append(fl[1])
        prev = cur
    va, vt, P = np.array(va), np.array(vt), np.array(P)
    a = np.linalg.eigh(np.cov(signal.detrend(np.cumsum(va, 0), axis=0).T))[1][:, -1]
    sos = signal.butter(2, Params().display_band, 'band', fs=fps, output='sos')
    ya, yt = signal.sosfilt(sos, np.cumsum(va @ a)), signal.sosfilt(sos, np.cumsum(vt @ a))
    lo, med, hi = np.percentile(ya[int(2 * fps):], [1, 50, 99])
    sign = 1.0 if (hi - med) > (med - lo) else -1.0          # breaths up, as in the monitor
    out = dict(t=t0 + np.arange(1, len(ya) + 1) / fps, ya=sign * ya, yt=sign * yt, pos=P - P[0],
               tpos=t0 + np.arange(len(P)) / fps)
    np.savez(cache, **out)
    return out


def fig_mouse_tracker():
    d = mouse_tracker_test()
    k = d['t'] >= d['t'][0] + 2          # skip the filter start-up
    fig, axs = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True, gridspec_kw={'hspace': 0.45,
                                                                                'height_ratios': [1, 1.5]})
    ax = axs[0]
    kp = d['tpos'] >= d['t'][0] + 2
    ax.plot(d['tpos'][kp], d['pos'][kp, 1] - np.median(d['pos'][kp, 1]), color=INK2, lw=1.4)
    ax.set_ylabel('px')
    ax.set_title('a  A tracker that follows the mouse moves with every breath (vertical position of its window)')
    ax.grid(axis='y')
    ax = axs[1]
    ax.plot(d['t'][k], d['ya'][k], color=BLUE, lw=1.6)
    ax.plot(d['t'][k], d['yt'][k], color=MUTED, lw=1.4)
    te = d['t'][k][-1]
    ax.text(te + 0.15, d['ya'][k][-30:].max(), 'region fixed to the\nscene (SurgeryVitals)', color=INK, va='center',
            fontsize=8.5)
    ax.text(te + 0.15, d['yt'][k][-30:].min(), 'region that follows\nthe mouse', color=INK2, va='center', fontsize=8.5)
    ratio = d['yt'][k].std() / d['ya'][k].std()
    ax.set_title(f'b  Breathing waveform from each region: following the mouse keeps only {ratio * 100:.0f}% of '
                 'the signal (s.d.)')
    ax.set_xlabel('time in Recording 3 (s)')
    ax.set_ylabel('waveform (a.u.)')
    ax.set_yticks([])
    ax.set_xlim(d['t'][k][0], te)
    save(fig, 'B1_track_scene_not_mouse.png')


DYN_BOX_T, DYN_REF_T, DYN_CUR_T = 72.0, 89.5, 91.0    # dynamic test video: lock, before / after the bump


def fig_registration():
    from surgeryvitals.tracker import AnchorTracker
    d = load('dyn')
    box = next(b for t, b, src in d['locks'] if t <= DYN_REF_T)
    ref, cur = read_frames(d['video'], [DYN_REF_T, DYN_CUR_T])
    gref, gcur = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (ref, cur))
    p = Params()
    trk = AnchorTracker(p)
    trk.init(gref, box)
    ok = trk.register(gcur)
    # the same matching, kept to draw it
    s = p.reg_scale
    kp, des = trk.orb.detectAndCompute(cv2.resize(gcur, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), None)
    pairs = [m for m in trk.bf.knnMatch(trk.ref_des, des, k=2) if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    p0 = trk.ref_kp[[m[0].queryIdx for m in pairs]]
    p1 = np.float32([kp[m[0].trainIdx].pt for m in pairs]) / s
    M, inl = cv2.estimateAffinePartial2D(p0, p1, ransacReprojThreshold=4)
    inl = inl.ravel().astype(bool)
    assert ok and np.allclose(M, trk.A), 'registration replica diverged from AnchorTracker'
    ang = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    scale = np.hypot(M[0, 0], M[1, 0])

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.7), gridspec_kw={'wspace': 0.04})
    x0, y0, x1, y1 = box
    q0 = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    ax = axs[0]
    ax.imshow(faded(ref[..., ::-1], 0.8))
    ax.plot(trk.ref_kp[:, 0], trk.ref_kp[:, 1], '.', color=BLUE, ms=2.2, alpha=0.8)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=SURF, alpha=0.75, ec=GOOD, lw=2))
    label(ax, x1 + 12, y0 + 10, 'region (masked out:\nno features used)', va='top')
    ax.set_title(f'a  Reference frame: {len(trk.ref_kp)} ORB features on the scene around the region')
    ax = axs[1]
    ax.imshow(faded(cur[..., ::-1], 0.8))
    ax.plot(p1[inl, 0], p1[inl, 1], '.', color=BLUE, ms=2.2, alpha=0.8)
    ax.add_patch(Polygon(q0, closed=True, fill=False, ec=MUTED, lw=1.2))
    label(ax, x1 + 10, y1, 'old position', va='bottom', color=INK2, fontsize=8)
    q1 = cv2.transform(q0[None], M)[0]
    ax.add_patch(Polygon(q1, closed=True, fill=False, ec=GOOD, lw=2))
    ax.add_patch(FancyArrowPatch(q0.mean(0), q1.mean(0), arrowstyle='-|>', mutation_scale=14, color=INK2, lw=1.4,
                                 shrinkA=4, shrinkB=4))
    label(ax, q1[:, 0].max() + 10, q1[:, 1].min() + 10, 'region carried\nby the transform', va='top')
    ax.set_title(f'b  After a sudden camera bump: {inl.sum()} of {len(pairs)} matches agree (RANSAC inliers, blue)')
    label(ax, 30, cur.shape[0] - 40,
          f'estimated: shift ({minus(M[0, 2], "+.0f")}, {minus(M[1, 2], "+.0f")}) px, '
          f'rotation {minus(ang, "+.1f")}°, scale {scale:.3f}\n'
          f'applied to the video: shift (−300, −150) px', va='bottom', fontsize=8.5)
    for ax in axs:
        ax.set_xlim(0, ref.shape[1])
        ax.set_ylim(ref.shape[0], 0)
        bare(ax)
    save(fig, 'B2_anchor_registration.png')


# ----------------------------------------------------------------------------- stage C figures
def window(d, t0, t1):
    return (d['t'] >= t0) & (d['t'] < t1)


def rate_table(d):
    keys = ('t', 'bpm', 'raw_bpm', 'peak_bpm', 'sqi')
    out = {k: np.array([r[k] for r in d['rates']], float) for k in keys}
    out['conf'] = np.array([r['confidence'] for r in d['rates']])
    return out


def conf_strip(ax, R, t0, t1, labels=True):
    m = (R['t'] >= t0) & (R['t'] < t1)
    ax.bar(R['t'][m], 1, width=1.0, align='edge', color=[CONF[c] for c in R['conf'][m]], lw=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    if labels:
        import matplotlib.patches as mp
        ax.legend(handles=[mp.Patch(color=CONF[c], label=c) for c in CONF], loc='upper left', ncols=4,
                  bbox_to_anchor=(0, -0.9), fontsize=8, handlelength=1, columnspacing=1.2)


def runs_of(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


ART_T = (190.0, 240.0)    # Recording 2: a hand intrusion


def fig_artifact():
    d = load('rec2')
    m = window(d, *ART_T)
    t, art = d['t'][m], d['art'][m]
    br = d['breaths'][(d['breaths'][:, 0] >= ART_T[0]) & (d['breaths'][:, 0] < ART_T[1])]
    fig, axs = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={'hspace': 0.5})
    for ax in axs:
        for a, b in runs_of(art):
            ax.axvspan(t[a], t[b - 1], color=GRID, lw=0)
    ax = axs[0]
    ax.plot(t, d['mag'][m], color=INK2, lw=0.9)
    ax.plot(t, d['thr'][m], color=INK, lw=1.4)
    ax.set_yscale('log')
    ax.set_ylabel('px / frame')
    ax.text(t[-1] + 0.4, d['thr'][m][-1], 'threshold T', color=INK, va='center', fontsize=8.5)
    ax.text(t[-1] + 0.4, np.nanmedian(d['mag'][m]), 'flow\nmagnitude', color=INK2, va='center', fontsize=8.5)
    ax.set_title('a  Mean optical-flow magnitude in the region, and the artifact threshold '
                 'T = clip(3 × median per-second peak, 0.3, 3)')
    a0, b0 = max(runs_of(art), key=lambda r: r[1] - r[0])
    ax.text((t[a0] + t[b0 - 1]) / 2, 0.97, 'artifact (shaded)', transform=ax.get_xaxis_transform(), ha='center',
            va='top', fontsize=8.5, color=INK2)
    ax = axs[1]
    ax.plot(t, d['s'][m], color=BLUE, lw=1.1)
    ax.set_title('b  Integrated breathing-axis displacement s(t): increments are set to zero during artifacts, '
                 'so it holds still')
    ax.set_ylabel('px (patch)')
    ax = axs[2]
    y = d['y'][m]
    ax.plot(t, np.where(art, np.nan, y), color=BLUE, lw=1.0)
    ax.plot(t, np.where(art, y, np.nan), color=MUTED, lw=1.0)
    ax.plot(br[:, 0], br[:, 1], 'o', color=ORANGE, ms=4, mec=SURF, mew=1)
    ok = ~art & np.isfinite(y)
    lo, hi = np.percentile(y[ok], [1, 99])
    ax.set_ylim(lo - 0.6 * (hi - lo), hi + 0.6 * (hi - lo))
    ax.set_yticks([])
    ax.set_title('c  Displayed waveform y(t) and detected breaths (orange); no breaths are taken from artifact '
                 'samples (grey)')
    ax.set_xlabel('time in Recording 2 (s)')
    ax.set_xlim(*ART_T)
    save(fig, 'C1_artifact_rejection.png')


AXIS_T, AXIS_REC = 466.0, 'rec2'


def breathing_axes(d, t, f0_bpm, win_s=15.0):
    """Both versions of the axis from the monitor's _update_axis, for the same 15 s of increments."""
    from scipy import signal
    fps = d['fps']
    m = window(d, t - win_s, t + 1e-6) & np.isfinite(d['v'][:, 0])
    v = np.where(d['art'][m][:, None], 0, d['v'][m])
    D = signal.detrend(np.cumsum(v, 0), axis=0)
    X = np.fft.rfft(D * np.hanning(len(D))[:, None], n=8192, axis=0)
    f = np.fft.rfftfreq(8192, 1 / fps)
    sel = np.zeros(len(f), bool)
    for k in (1, 2, 3):
        sel |= np.abs(f - k * f0_bpm / 60) < 0.08
    Cs = np.real(X[sel].conj().T @ X[sel])
    Db = signal.sosfiltfilt(signal.butter(2, [0.2, 5], 'band', fs=fps, output='sos'), D, axis=0)
    Cb = np.cov(Db.T)
    ax_s, ax_b = (np.linalg.eigh(C)[1][:, -1] for C in (Cs, Cb))
    return d['t'][m], D, Db, ax_s, ax_b


def fig_axis():
    from surgeryvitals.dsp import harmonic_concentration, power_spectrum
    d = load(AXIS_REC)
    R = rate_table(d)
    f0 = R['raw_bpm'][np.searchsorted(R['t'], AXIS_T)]
    t, D, Db, a_s, a_b = breathing_axes(d, AXIS_T, f0)
    if a_s @ a_b < 0:
        a_b = -a_b
    fig = plt.figure(figsize=(10.5, 3.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.6], wspace=0.22, hspace=0.6)
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(D[:, 0], D[:, 1], color=MUTED, lw=0.6, alpha=0.8)
    r = np.abs(D).max() * 0.95
    for a, col, name in ((a_s, BLUE, 'steered to f₀'), (a_b, INK2, 'broadband')):
        ax.annotate('', xy=a * r, xytext=-a * r, arrowprops=dict(arrowstyle='-|>', color=col, lw=2, mutation_scale=12))
        ax.text(*(a * r * 1.06), name, color=INK, fontsize=8.5, ha='center', va='bottom' if a[1] < 0 else 'top')
    ax.set_aspect('equal')
    ax.set_xlim(-r * 1.4, r * 1.4)
    ax.set_ylim(r * 1.4, -r * 1.4)
    ax.set_xlabel('x displacement (px)')
    ax.set_ylabel('y displacement (px)')
    ang = np.degrees(np.arccos(abs(a_s @ a_b)))
    ax.set_title(f'a  2-D motion, last 15 s\n    (axes {ang:.0f}° apart)')
    ax.grid(True)
    pol = d['pol'][np.searchsorted(d['t'], AXIS_T)] or 1.0
    for i, (a, col, name) in enumerate(((a_s, BLUE, 'axis steered to f₀ (used)'), (a_b, INK2, 'broadband axis'))):
        ax2 = fig.add_subplot(gs[i, 1])
        z = pol * (Db @ a)
        f, P = power_spectrum(D @ a, d['fps'])
        P[f < 0.9 * Params().rate_bpm[0] / 60] = 0          # as in the monitor
        q = harmonic_concentration(P, f, f0 / 60)
        ax2.plot(t, z, color=col, lw=1.1)
        ax2.set_yticks([])
        ax2.set_xlim(t[0], t[-1])
        ax2.set_title(f'{"bc"[i]}  Projection on the {name}: spectral quality (SQI) {q:.2f}', fontsize=9.5)
        if i == 1:
            ax2.set_xlabel(f'time in Recording 2 (s)')
    save(fig, 'C2_breathing_axis.png')


POL_REC, POL_T = 'rec3', 18.0


def fig_polarity():
    from scipy import signal
    d = load(POL_REC)
    lock_t = d['locks'][0][0]
    i = np.searchsorted(d['t'], POL_T)
    a = d['axis'][i]
    m = window(d, lock_t, POL_T + 1e-6) & np.isfinite(d['v'][:, 0])
    v = np.where(d['art'][m][:, None], 0, d['v'][m])
    D = signal.detrend(np.cumsum(v, 0), axis=0)
    z = signal.sosfiltfilt(signal.butter(2, [0.2, 5], 'band', fs=d['fps'], output='sos'), D, axis=0) @ a
    t = d['t'][m]
    pol = d['pol'][i]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 2.9), sharey=True, gridspec_kw={'wspace': 0.08})
    for ax, sgn, title in ((axs[0], 1.0, 'a  Projection as first computed (sign arbitrary)'),
                           (axs[1], pol, f'b  After choosing the polarity p = {minus(pol, "+.0f")}: breaths point up')):
        zz = sgn * z
        lo, med, hi = np.percentile(zz, [1, 50, 99])
        alpha = ((hi - med) - (med - lo)) / (hi - lo)
        for q, name in ((hi, 'q₉₉'), (med, 'q₅₀'), (lo, 'q₁')):
            ax.axhline(q, color=AXIS, lw=0.8)
            ax.text(t[-1] + 0.1, q, name, va='center', fontsize=8, color=INK2)
        ax.plot(t, zz, color=BLUE if sgn == pol else MUTED, lw=1.2)
        ax.set_title(title, fontsize=9.5)
        ax.text(0.02, 0.04, f'α = {minus(alpha, "+.2f")}', transform=ax.transAxes, fontsize=9, color=INK)
        ax.set_xlabel('time in Recording 3 (s)')
        ax.set_xlim(t[0], t[-1])
        ax.set_yticks([])
    save(fig, 'C3_polarity.png')


# ----------------------------------------------------------------------------- stage D figure
DET_REC, DET_T = 'rec3', 300.0


def fig_breaths():
    from scipy import signal
    d = load(DET_REC)
    fps = d['fps']
    m = window(d, DET_T - 15, DET_T + 1e-6)
    t, y, art = d['t'][m], d['y'][m], d['art'][m]
    ok = ~art & np.isfinite(y)
    q5, q95 = np.percentile(y[ok], [5, 95])
    prom = 0.3 * (q95 - q5)
    pk, props = signal.find_peaks(np.nan_to_num(y), prominence=prom, distance=int(0.3 * fps), wlen=int(2.5 * fps) | 1)
    allpk, allprops = signal.find_peaks(np.nan_to_num(y), prominence=0.02 * (q95 - q5), wlen=int(2.5 * fps) | 1)
    weak = np.setdiff1d(allpk, pk)
    br = d['breaths'][(d['breaths'][:, 0] > DET_T - 15) & (d['breaths'][:, 0] <= DET_T)]
    fig, axs = plt.subplots(2, 1, figsize=(10.5, 5), sharex=True, gridspec_kw={'hspace': 0.45, 'height_ratios': [2.2, 1]})
    ax = axs[0]
    ax.axvspan(DET_T - 0.35, DET_T, color=GRID, lw=0)
    ax.text(DET_T - 0.4, 0.98, 'younger than\n0.35 s: wait', transform=ax.get_xaxis_transform(), ha='right',
            va='top', fontsize=8, color=INK2)
    ax.plot(t, y, color=BLUE, lw=1.3)
    for i, b in zip(pk, props['prominences']):
        ax.plot([t[i], t[i]], [y[i] - b, y[i]], color=ORANGE, lw=1, alpha=0.55)
    ax.plot(t[weak], y[weak], 'o', mfc='none', mec=MUTED, ms=5, mew=1)
    ax.plot(br[:, 0], br[:, 1], 'o', color=ORANGE, ms=6, mec=SURF, mew=1.2)
    ax.axhline(q5, color=AXIS, lw=0.8)
    ax.axhline(q95, color=AXIS, lw=0.8)
    ax.text(t[0] - 0.1, q95, 'q₉₅', ha='right', va='center', fontsize=8, color=INK2)
    ax.text(t[0] - 0.1, q5, 'q₅', ha='right', va='center', fontsize=8, color=INK2)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_title(f'a  Peaks need prominence ≥ 0.3 × (q₉₅ − q₅) (orange bars) and ≥ 0.3 s spacing; '
                 f'filled = confirmed breath, hollow = too small', fontsize=9.5)
    ax = axs[1]
    iv = np.diff(br[:, 0])
    mid = br[1:, 0] - iv / 2
    med = np.median(iv)
    ax.plot(mid, iv, 'o', color=ORANGE, ms=5, mec=SURF, mew=1)
    ax.axhline(med, color=INK2, lw=1)
    cv = iv.std() / iv.mean()
    ax.text(0.01, 0.95, f'median {med:.2f} s → {60 / med:.1f} breaths/min,   CV = {cv:.2f}', va='top',
            transform=ax.transAxes, fontsize=8.5, color=INK)
    ax.set_ylim(med * 0.6, med * 1.4)
    ax.set_ylabel('interval (s)')
    ax.set_title('b  Breath-to-breath intervals in the 15 s window', fontsize=9.5)
    ax.set_xlabel('time in Recording 3 (s)')
    ax.set_xlim(t[0], t[-1])
    ax.grid(axis='y')
    save(fig, 'D1_breath_detection.png')


# ----------------------------------------------------------------------------- stage E figures
SAL_REC, SAL_T = 'rec2', 60.0


def fig_salience():
    from surgeryvitals.dsp import harmonic_sum
    d = load(SAL_REC)
    r = min((r for r in d['rates'] if r.get('P') is not None), key=lambda r: abs(r['t'] - SAL_T))
    f = d['f_spec'] * 60
    P, sal = r['P'].astype(float), r['sal'].astype(float)
    H = harmonic_sum(P)[:len(P)]
    # the harmonic sum near the top of the stored range lacks its upper harmonics; only show up to 140/min
    show = (f >= 15) & (f <= 140)
    f0 = r['raw_bpm']
    fig, axs = plt.subplots(3, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={'hspace': 0.55})
    rows = [(P, 'a  Power spectrum P(f) of the 15 s window', INK2),
            (H, 'b  Harmonic sum H(f) = P(f) + P(2f) + P(3f): the fundamental is strongest, but sub-harmonics '
                'collect power too', INK2),
            (sal, 'c  Salience Λ(f) = H(f) · √(P(f) / max P): needs power at the candidate itself', BLUE)]
    for ax, (Y, title, col) in zip(axs, rows):
        Yn = Y / Y[show].max()
        ax.axvspan(0, Params().rate_bpm[0], color=GRID, lw=0)
        ax.fill_between(f[show], Yn[show], color=col, alpha=0.1, lw=0)
        ax.plot(f[show], Yn[show], color=col, lw=1.3)
        for k, name in ((1 / 3, 'f₀/3'), (1 / 2, 'f₀/2'), (1, 'f₀')):
            x = f0 * k
            yk = Yn[np.argmin(np.abs(f - x))]
            ax.plot([x], [yk], 'o', color=ORANGE if k == 1 else INK2, ms=5, mec=SURF, mew=1)
            ax.text(x, yk + 0.08, name, ha='center', va='bottom', fontsize=8.5, color=INK)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_title(title, fontsize=9.5)
    axs[0].text(Params().rate_bpm[0] / 2 + 7, 0.6, 'below\n25/min', ha='center', fontsize=7.5, color=INK2)
    axs[-1].set_xlabel('breaths / min')
    axs[-1].set_xlim(15, 140)
    fig.suptitle(f'Recording 2, window ending at {r["t"]:.0f} s: rate ≈ {f0:.0f} breaths/min', x=0.125, ha='left',
                 y=0.99, fontsize=10, color=INK)
    save(fig, 'E1_harmonic_salience.png')


def fig_tracking():
    d = load('rec2')
    R = rate_table(d)
    rs = [r for r in d['rates'] if r.get('sal') is not None]
    f = d['f_spec'] * 60
    band = (f >= 25) & (f <= 130)
    S = np.array([r['sal'][band] / (r['sal'][band].max() + 1e-12) for r in rs]).T
    ts = np.array([r['t'] for r in rs])
    fig, axs = plt.subplots(2, 1, figsize=(10.5, 4.6), sharex=True, gridspec_kw={'hspace': 0.15,
                                                                                  'height_ratios': [1, 0.07]})
    ax = axs[0]
    ax.imshow(S, aspect='auto', origin='lower', cmap=SEQ, vmin=0, vmax=1, interpolation='nearest',
              extent=[ts[0] - 0.5, ts[-1] + 0.5, f[band][0], f[band][-1]])
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=3.5, foreground=SURF)]
    ax.plot(R['t'], R['bpm'], color=ORANGE, lw=2, path_effects=halo)
    ax.set_ylabel('breaths / min')
    ax.set_title('Recording 2: salience Λ per 1 s update (blue, normalised per column) and the reported rate '
                 '(orange) from the HMM tracker')
    ax.set_xlim(0, R['t'][-1] + 1)
    ax.set_ylim(25, 130)
    conf_strip(axs[1], R, 0, R['t'][-1] + 1)
    axs[1].set_xlabel('time (s)', labelpad=2)
    axs[1].xaxis.set_label_coords(0.5, -2.2)
    save(fig, 'E2_rate_tracking.png')


def fig_confidence():
    d = load('rec1')
    R = rate_table(d)
    t = R['t']
    dis = np.abs(R['peak_bpm'] - R['raw_bpm']) / R['raw_bpm'] * 100
    # interval CV per update, as computed by the monitor (breaths in the last 15 s)
    bt = d['breaths'][:, 0] if len(d['breaths']) else np.array([])
    cv = np.full(len(t), np.nan)
    for j, tj in enumerate(t):
        b = bt[(bt > tj - 15) & (bt <= tj)]
        if len(b) >= 4:
            iv = np.diff(b)
            iv = iv[iv < 2.5 * np.median(iv)]
            if len(iv) >= 3:
                cv[j] = iv.std() / iv.mean()
    fig, axs = plt.subplots(5, 1, figsize=(10.5, 7.6), sharex=True,
                            gridspec_kw={'hspace': 0.6, 'height_ratios': [2, 1, 1, 1, 0.22]})
    ax = axs[0]
    ax.plot(t, R['raw_bpm'], color=MUTED, lw=1.2)
    ax.plot(t, R['peak_bpm'], 'o', color=ORANGE, ms=2.5)
    ax.plot(t, R['bpm'], color=BLUE, lw=2.2)
    ax.set_ylim(0, 80)
    ax.set_ylabel('breaths / min')
    ax.grid(axis='y')
    ax.set_title('a  Spectral estimate (grey), breath-to-breath rate (orange) and the reported rate (blue)')
    import matplotlib.lines as ml
    ax.legend(handles=[ml.Line2D([], [], color=MUTED, lw=1.2, label='spectral (HMM tracker)'),
                       ml.Line2D([], [], color=ORANGE, marker='o', ls='', ms=4, label='breath-to-breath'),
                       ml.Line2D([], [], color=BLUE, lw=2.2, label='reported')], loc='lower left', ncols=3)
    rows = [(axs[1], R['sqi'], (0.35, 0.5), 'b  Spectral quality (SQI): fair ≥ 0.35, good ≥ 0.5', (0, 1)),
            (axs[2], dis, (15,), 'c  Disagreement between the two estimates (%): must be ≤ 15', (0, 60)),
            (axs[3], cv, (0.25, 0.4), 'd  Irregularity of breath intervals (CV): good < 0.25, fair < 0.4', (0, 0.8))]
    for ax, y, thr, title, lim in rows:
        ax.plot(t, y, color=INK2, lw=1.1)
        for v in thr:
            ax.axhline(v, color=INK, lw=1)
        ax.set_ylim(*lim)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis='y')
    conf_strip(axs[4], R, 0, t[-1] + 1)
    axs[4].set_title('e  Confidence: a rate is reported only when good or fair', fontsize=9.5)
    axs[4].set_xlabel('time in Recording 1 (s)', labelpad=2)
    axs[4].xaxis.set_label_coords(0.5, -2.6)
    axs[0].set_xlim(0, t[-1] + 1)
    save(fig, 'E3_confidence.png')


FIGURES = [fig_blocks, fig_region, fig_mouse_tracker, fig_registration, fig_artifact, fig_axis, fig_polarity,
           fig_breaths, fig_salience, fig_tracking, fig_confidence]


# ----------------------------------------------------------------------------- animations
def write_anim(fig, frames_fn, n, name, fps=15, quality=78):
    """Render n frames (frames_fn(i) updates the figure) into a looping animated WebP: full colour and
    a fraction of the size of a GIF; GitHub and current browsers show it inline."""
    from PIL import Image
    imgs = []
    for i in range(n):
        frames_fn(i)
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    plt.close(fig)
    imgs[0].save(OUT / name, save_all=True, append_images=imgs[1:], duration=int(round(1000 / fps)), loop=0,
                 quality=quality, method=6)
    print('wrote', OUT / name, f'{(OUT / name).stat().st_size / 1e6:.2f} MB, {n} frames')


def anim_localiser(speed=2.0, fps=12):
    """Recording 3 from the first frame: block motion -> periodicity map -> region."""
    import matplotlib.patheffects as pe
    L = load('loc3')
    p = Params()
    F, ts, B, fs = L['flows'], L['t'], L['block'], L['fs']
    by, bx = F.shape[1:3]
    step = max(1, int(round(fs * speed / fps)))
    idx = np.arange(step, len(ts), step)
    # blocks to trace: the mouse (from the final region) and the surgeon's glove (largest motion)
    final = periodicity(F[-int(p.loc_window_s * fs):], fs)
    reg = final['best'][1]
    mb = np.unravel_index(np.argmax(np.where(reg, final['act'], -1)), reg.shape)
    yy, xx = np.mgrid[:by, :bx]
    gb = np.unravel_index(np.argmax(np.where((yy < 20) & (xx > 8) & (xx < 30), final['amp'], -1)), reg.shape)
    disp = np.cumsum(F, axis=0)

    def along(r, c):                  # cumulative displacement along the block's dominant direction
        D = disp[:, r, c] - disp[:, r, c].mean(0)
        a = np.linalg.eigh(np.cov(D.T))[1][:, -1]
        return D @ (a if a[1] >= 0 else -a)
    tr_m, tr_g = along(*mb), along(*gb)

    fig = plt.figure(figsize=(9.6, 5.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.6, 0.55, 0.55], hspace=0.32, wspace=0.04, left=0.09, right=0.99,
                          top=0.89, bottom=0.09)
    a_vid, a_map = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    a_tr, a_tr2 = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
    for ax in (a_vid, a_map):
        bare(ax)
    im_vid = a_vid.imshow(faded(L['frames'][0], 0.85))
    sub = (slice(0, None, 2), slice(0, None, 2))
    gy, gx = ((np.mgrid[:by, :bx] + 0.5) * B)[(slice(None),) + sub]
    qv = a_vid.quiver(gx, gy, np.zeros(gx.shape), np.zeros(gx.shape), color=BLUE, angles='xy', scale_units='xy',
                      scale=0.25, width=0.004, headwidth=3, headlength=3, headaxislength=2.5)
    a_vid.set_title('Motion of the 32-px blocks (every other block; arrows capped)', fontsize=10)
    im_bg = a_map.imshow(faded(L['frames'][0], 0.55))
    im_map = a_map.imshow(np.zeros((by * B, bx * B)), cmap=SEQ, vmin=0, vmax=1, alpha=0.0)
    a_map.set_title('Periodicity score at 30–210 breaths/min', fontsize=10)
    halo = [pe.withStroke(linewidth=3, foreground=SURF)]
    msg = a_map.text(0.5, 0.5, '', transform=a_map.transAxes, ha='center', va='center', fontsize=11, color=INK,
                     path_effects=halo)
    boxes = [ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=GOOD, lw=2.2, visible=False)) for ax in (a_vid, a_map)]
    box_lbl = a_map.text(0, 0, '', fontsize=9, color=INK, path_effects=halo, va='top', ha='right')
    for (r, c), col, name in ((mb, BLUE, '1'), (gb, INK2, '2')):
        a_vid.add_patch(Rectangle((c * B - 0.5, r * B - 0.5), B, B, fill=False, ec=col, lw=2))
        a_vid.text(c * B + B + 2, r * B + B / 2, name, fontsize=10, fontweight='bold', va='center', color=INK,
                   path_effects=halo)
    win = 10.0
    l_m, = a_tr.plot([], [], color=BLUE, lw=1.5)
    l_g, = a_tr2.plot([], [], color=INK2, lw=1.3)
    sc = L['scale']
    for ax, tr, name in ((a_tr, tr_m, '1 mouse'), (a_tr2, tr_g, '2 glove')):
        rng = np.ptp(tr) * sc
        ax.set_ylim(-0.6 * rng, 0.6 * rng)
        ax.set_ylabel(f'{name}\n(px)', fontsize=8, rotation=0, ha='right', va='center', color=INK)
        ax.tick_params(axis='y', labelsize=7.5)
    a_tr.tick_params(axis='x', labelbottom=False)
    a_tr2.set_xlabel('time (s)')
    a_tr.set_title('Displacement of the two marked blocks along their main direction, last 10 s '
                   '(full-resolution px)', fontsize=10)
    title = fig.suptitle('', x=0.01, y=0.985, ha='left', fontsize=10.5, color=INK)

    def draw(j):
        k = idx[j]
        t = ts[k]
        v = F[k - step + 1:k + 1].mean(0)[sub]
        mag = np.hypot(v[..., 0], v[..., 1])
        v = v * (np.minimum(mag, 4.0) / np.maximum(mag, 1e-9))[..., None]
        qv.set_UVC(v[..., 0], v[..., 1])
        im_vid.set_data(faded(L['frames'][k], 0.85))
        im_bg.set_data(faded(L['frames'][k], 0.55))
        title.set_text(f'Recording 3, t = {t:4.1f} s   (played at {speed:.0f}× speed)')
        n_hist = k + 1
        need = int(p.loc_min_s * fs)
        if n_hist < need:
            msg.set_text(f'collecting motion history\n{n_hist / fs:4.1f} of {p.loc_min_s:.0f} s')
            im_map.set_alpha(0.0)
            for b in boxes:
                b.set_visible(False)
            box_lbl.set_text('')
        else:
            m = periodicity(F[max(0, k + 1 - int(p.loc_window_s * fs)):k + 1], fs)
            im_map.set_data(np.kron(m['act'], np.ones((B, B))))
            im_map.set_alpha(np.clip(np.kron(m['act'], np.ones((B, B))) * 1.6, 0, 0.92))
            msg.set_text('')
            if m['best'] is not None and m['act'][m['best'][1]].mean() >= p.loc_min_score:
                ys, xs = np.nonzero(m['best'][1])
                for b in boxes:
                    b.set_bounds(xs.min() * B, ys.min() * B, (xs.max() - xs.min() + 1) * B, (ys.max() - ys.min() + 1) * B)
                    b.set_visible(True)
                box_lbl.set_position((xs.min() * B - 4, ys.min() * B))
                box_lbl.set_text(f'region\n{m["best"][2]:.0f}/min')
        lo = max(0, k - int(win * fs))
        tt = ts[lo:k + 1]
        dm, dg = tr_m[lo:k + 1], tr_g[lo:k + 1]
        l_m.set_data(tt, (dm - tr_m.mean()) * sc)
        l_g.set_data(tt, (dg - tr_g.mean()) * sc)
        for ax in (a_tr, a_tr2):
            ax.set_xlim(max(0.0, t - win), max(win, t))

    write_anim(fig, draw, len(idx), 'A_localiser.webp', fps=fps)


def anim_tracking(t0=146.0, t1=163.0, speed=2.0, fps=12):
    """Dynamic test video: the camera pans 500 px; the region follows the scene around the mouse."""
    from surgeryvitals.tracker import AnchorTracker
    d = load('dyn')
    fps_v = d['fps']
    step = int(round(fps_v * speed / fps))
    frames_t = np.arange(t0, t1, step / fps_v)
    cap = cv2.VideoCapture(d['video'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(t0 * fps_v))
    p = Params()
    trk = AnchorTracker(p)
    box = d['locks'][0][1]
    W = 640
    fig = plt.figure(figsize=(9.6, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.05, left=0.01, right=0.99, top=0.86, bottom=0.03)
    ax, a_p = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    bare(ax)
    bare(a_p)
    im = ax.imshow(np.zeros((360, W, 3), np.uint8))
    pts, = ax.plot([], [], '.', color=BLUE, ms=2.5, alpha=0.85)
    quad = ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, fill=False, ec=GOOD, lw=2))
    ev = ax.text(0.015, 0.035, '', transform=ax.transAxes, fontsize=9.5, color=INK,
                 bbox=dict(boxstyle='round,pad=0.3', fc=SURF, ec='none', alpha=0.9))
    ax.set_title('Blue: scene features matched to the reference (RANSAC inliers). Green: the region.', fontsize=9.5)
    im_p = a_p.imshow(np.zeros((10, 10), np.uint8), cmap='gray', vmin=0, vmax=255)
    a_p.set_title('What the region\nanalysis sees', fontsize=9.5)
    title = fig.suptitle('', x=0.01, y=0.985, ha='left', fontsize=10.5, color=INK)
    sc = W / d['shape'][1]
    first = {}

    def frame_at(t):
        i = round((t - t0) * fps_v)
        while first.get('i', -1) < i:
            f = cap.read()[1]
            first['i'] = first.get('i', -1) + 1
            first['f'] = f
        return first['f']

    def draw(j):
        t = frames_t[j]
        f = frame_at(t)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        k = np.searchsorted(d['t'], t - 1e-6)
        q = d['quad'][k].reshape(4, 2)
        if j == 0:
            trk.init(g, box)
            trk.A = cv2.getAffineTransform(np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]]]),
                                           q[:3].astype(np.float32))
            trk.ref_kp = cv2.transform(trk.ref_kp[None], cv2.invertAffineTransform(trk.A))[0]
        s = p.reg_scale
        kp, des = trk.orb.detectAndCompute(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), None)
        pairs = [m for m in trk.bf.knnMatch(trk.ref_des, des, k=2) if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(pairs) > 15:
            p0 = trk.ref_kp[[m[0].queryIdx for m in pairs]]
            p1 = np.float32([kp[m[0].trainIdx].pt for m in pairs]) / s
            M, inl = cv2.estimateAffinePartial2D(p0, p1, ransacReprojThreshold=4)
            p1 = p1[inl.ravel().astype(bool)] if M is not None else p1[:0]
        else:
            p1 = np.zeros((0, 2))
        im.set_data(faded(cv2.resize(f, (W, int(f.shape[0] * sc)), interpolation=cv2.INTER_AREA)[..., ::-1], 0.9))
        pts.set_data(p1[:, 0] * sc, p1[:, 1] * sc)
        quad.set_xy(q * sc)
        # the patch exactly as the monitor builds it (region resampled through the current transform)
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        A = cv2.getAffineTransform(q[:3].astype(np.float32), np.float32([[0, 0], [w, 0], [w, h]]))
        im_p.set_data(cv2.warpAffine(g, A, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE))
        im_p.set_extent((0, w, h, 0))
        state = d['state'][k]
        title.set_text(f'Synthetic test video (Recording 3 footage), t = {t:5.1f} s, {speed:.0f}× speed:  '
                       f'state "{state}"')
        ev.set_text('camera pans 500 px to the right (150–160 s)' if 149 <= t < 161 else '')

    write_anim(fig, draw, len(frames_t), 'B_anchor_tracking.webp', fps=fps)
    cap.release()


def anim_waveform(t0=40.0, t1=50.0, fps=15):
    """Recording 3 in real time: flow in the region -> 2-D displacement -> breathing axis -> waveform -> breaths."""
    d = load('rec3')
    fps_v = d['fps']
    step = int(round(fps_v / fps))
    k0 = np.searchsorted(d['t'], t0)
    k1 = np.searchsorted(d['t'], t1)
    q = d['quad'][k0].reshape(4, 2)
    x0, y0 = np.floor(q.min(0)).astype(int)
    x1, y1 = np.ceil(q.max(0)).astype(int)
    mx, my = int(0.9 * (x1 - x0)), int(0.25 * (y1 - y0))
    cx0, cy0, cx1, cy1 = x0 - mx, y0 - my, x1 + mx, y1 + my
    cap = cv2.VideoCapture(d['video'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, k0 - step)
    p = Params()
    s = max(0.25, p.roi_target_px / min(x1 - x0, y1 - y0))
    fig = plt.figure(figsize=(9.6, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 2.3], height_ratios=[1, 1], wspace=0.25, hspace=0.45,
                          left=0.01, right=0.98, top=0.88, bottom=0.11)
    a_img = fig.add_subplot(gs[:, 0])
    a_xy = fig.add_subplot(gs[:, 1])
    a_s = fig.add_subplot(gs[0, 2])
    a_y = fig.add_subplot(gs[1, 2])
    bare(a_img)
    im = a_img.imshow(np.zeros((cy1 - cy0, cx1 - cx0, 3), np.uint8))
    a_img.add_patch(Rectangle((x0 - cx0, y0 - cy0), x1 - x0, y1 - y0, fill=False, ec=GOOD, lw=2))
    gy, gx = np.mgrid[2:int((y1 - y0) * s) - 1:4, 2:int((x1 - x0) * s) - 1:4]
    qv = a_img.quiver(gx / s + x0 - cx0, gy / s + y0 - cy0, np.zeros(gx.shape), np.zeros(gx.shape), color=BLUE,
                      angles='xy', scale_units='xy', scale=0.035, width=0.02, headwidth=3, headlength=3,
                      headaxislength=2.5)
    a_img.set_xlabel('arrows ×30', fontsize=8, color=INK2)
    a_img.set_title('Region: optical flow', fontsize=10)
    # 2-D displacement trail and the breathing axis
    v = np.where(d['art'][:, None], 0, np.nan_to_num(d['v']))
    D = np.cumsum(v[k0 - int(15 * fps_v):k1], axis=0)
    D = D - D[int(15 * fps_v)]
    trail, = a_xy.plot([], [], color=MUTED, lw=1)
    dot, = a_xy.plot([], [], 'o', color=BLUE, ms=7, mec=SURF, mew=1.5)
    axl, = a_xy.plot([], [], color=INK2, lw=1.4)
    n3 = int(3 * fps_v)
    r = 1.3 * np.percentile([np.abs(D[i - n3:i + 1] - D[i - n3:i + 1].mean(0)).max() for i in range(n3, len(D))], 90)
    a_xy.set_aspect('equal')
    a_xy.set_xlim(-r, r)
    a_xy.set_ylim(r, -r)
    a_xy.set_xlabel('x (px)')
    a_xy.set_ylabel('y (px)', labelpad=1)
    a_xy.grid(True)
    a_xy.set_title('2-D displacement, last 3 s\n(line: breathing axis)', fontsize=10)
    win = 8.0
    l_s, = a_s.plot([], [], color=INK2, lw=1.2)
    a_s.set_title('Projection on the axis, integrated: s(t)', fontsize=10)
    a_s.set_yticks([])
    l_y, = a_y.plot([], [], color=BLUE, lw=1.6)
    br_pts, = a_y.plot([], [], 'o', color=ORANGE, ms=6, mec=SURF, mew=1.2)
    a_y.set_title('Band-passed waveform y(t); confirmed breaths', fontsize=10)
    a_y.set_yticks([])
    a_y.set_xlabel('time in Recording 3 (s)')
    rate_txt = fig.text(0.98, 0.955, '', ha='right', va='bottom', fontsize=10.5, color=INK)
    for ax in (a_s, a_y):
        ax.spines['left'].set_visible(False)
    tt, ss, yy = d['t'], d['s'], d['y']
    ylo, yhi = np.nanpercentile(yy[k0:k1], [0.5, 99.5])
    title = fig.suptitle('', x=0.01, y=0.985, ha='left', fontsize=10.5, color=INK)
    R = rate_table(d)
    st = {'prev': None}

    def read_small():
        f = cap.read()[1]
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        return f, cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def draw(j):
        k = k0 + j * step
        for _ in range(step - 1):
            read_small()
        f, sm = read_small()
        if st['prev'] is not None and st['prev'].shape == sm.shape:
            fl = cv2.calcOpticalFlowFarneback(st['prev'], sm, None, 0.5, 2, 9, 3, 5, 1.1, 0)
            qv.set_UVC(fl[gy, gx, 0] / s, fl[gy, gx, 1] / s)
        st['prev'] = sm
        im.set_data(f[cy0:cy1, cx0:cx1, ::-1])
        t = tt[k]
        i = k - (k0 - int(15 * fps_v))
        tr = D[max(0, i - int(3 * fps_v)):i + 1]
        c = tr.mean(0)
        trail.set_data(tr[:, 0] - c[0], tr[:, 1] - c[1])
        dot.set_data([D[i, 0] - c[0]], [D[i, 1] - c[1]])
        a = d['axis'][k] * (d['pol'][k] or 1)
        axl.set_data([-a[0] * r, a[0] * r], [-a[1] * r, a[1] * r])
        m = (tt > t - win) & (tt <= t)
        l_s.set_data(tt[m], ss[m])
        l_y.set_data(tt[m], yy[m])
        b = d['breaths']
        b = b[(b[:, 0] > t - win) & (b[:, 0] <= t - 0.35)]
        br_pts.set_data(b[:, 0], b[:, 1])
        for ax in (a_s, a_y):
            ax.set_xlim(t - win, t)
        sm_ = ss[m][np.isfinite(ss[m])]
        a_s.set_ylim(sm_.min() - 0.1 * np.ptp(sm_) - 1e-3, sm_.max() + 0.1 * np.ptp(sm_) + 1e-3)
        a_y.set_ylim(ylo - 0.15 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))
        ri = np.searchsorted(R['t'], t, side='right') - 1
        rate_txt.set_text(f'reported rate {R["bpm"][ri]:.0f} breaths/min ({R["conf"][ri]})'
                          if ri >= 0 and np.isfinite(R['bpm'][ri]) else '')
        title.set_text(f'From pixels to breaths: Recording 3, t = {t:5.1f} s (real time)')

    write_anim(fig, draw, (k1 - k0) // step, 'C_waveform.webp', fps=fps)
    cap.release()


ANIMATIONS = [anim_localiser, anim_tracking, anim_waveform]


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('run', 'all'):
        run_all()
    if what in ('figures', 'all'):
        for fn in FIGURES:
            fn()
    if what in ('anims', 'all'):
        for fn in ANIMATIONS:
            fn()
