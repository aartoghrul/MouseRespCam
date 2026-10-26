# How fast does the monitor move its region when the mouse's position in the image changes?
# Runs the monitor frame by frame on videos/synthetic/dynamic_test.mp4 and compares the region
# with the frame-by-frame ground truth written by make_dynamic_video.py.
#
#   .venv/bin/python exploration/scripts/eval_dynamic.py
#   -> exploration/14_dynamic_region_test.png + printed latency table
import csv
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from surgeryvitals.monitor import RespirationMonitor
from surgeryvitals.source import VideoFileSource

VID = 'videos/synthetic/dynamic_test.mp4'
MARGIN = 40          # px: region centre may sit this far outside the body box (e.g. on the ear-bar post)
STABLE_S = 2.0       # on target continuously this long counts as "found"

truth = list(csv.DictReader(open('videos/synthetic/dynamic_test_truth.csv')))
events = json.load(open('videos/synthetic/dynamic_test_events.json'))
gt = np.array([[int(r[k]) for k in ('x0', 'y0', 'x1', 'y1')] for r in truth])
src_rec = np.array([int(r['source_rec']) for r in truth])
src_t = np.array([float(r['source_t_s']) for r in truth])

src = VideoFileSource(VID)
mon = RespirationMonitor(src.shape, src.fps)
T, state, cen, rates = [], [], [], []
while True:
    f, t = src.read()
    if f is None:
        break
    r = mon.process(f, t)
    T.append(t)
    state.append(r.state)
    cen.append(r.quad.mean(0) if r.quad is not None else (np.nan, np.nan))
    if r.rate is not None:
        rates.append((t, r.rate.bpm, r.rate.confidence))
T, cen, state = np.array(T), np.array(cen), np.array(state)
n = len(T)
g = gt[:n]
tracking = np.isin(state, ['locked', 'artifact'])
on = tracking & (cen[:, 0] >= g[:, 0] - MARGIN) & (cen[:, 0] <= g[:, 2] + MARGIN) \
    & (cen[:, 1] >= g[:, 1] - MARGIN) & (cen[:, 1] <= g[:, 3] + MARGIN)
wrong = tracking & ~on

# reference rate: the earlier headless run on the source recording, at the matching source time
ref = {}
for rec in set(src_rec.tolist()):
    rr = list(csv.DictReader(open(f'results/Recording_{rec}/rate.csv')))
    ref[rec] = (np.array([float(x['t_s']) for x in rr]),
                np.array([float(x['rate_bpm']) if x['rate_bpm'] else np.nan for x in rr]))
rt = np.array([x[0] for x in rates])
rb = np.array([x[1] for x in rates])
fi = np.minimum((rt * src.fps).astype(int), n - 1)
rref = np.array([np.interp(src_t[i], *ref[src_rec[i]]) if np.isfinite(
    ref[src_rec[i]][1][np.argmin(np.abs(ref[src_rec[i]][0] - src_t[i]))]) else np.nan for i in fi])

fps = src.fps
stable = int(STABLE_S * fps)


def first_stable(mask, i0, i1):
    run = 0
    for i in range(i0, i1):
        run = run + 1 if mask[i] else 0
        if run >= stable:
            return (i - stable + 1) / fps
    return None


bounds = [0] + [int(e['t_s'] * fps) for e in events] + [n]
print(f'{"event":44s} {"handled by":>12s} {"old region dropped":>19s} {"on target after":>16s} '
      f'{"fresh rate after":>17s} {"no-rate gap":>12s} {"wrong-spot time":>16s} {"|rate-ref|":>11s}')
rows = [('start (no region yet)', 0, bounds[1])] + [(e['event'], bounds[k + 1], bounds[k + 2]) for k, e in enumerate(events)]
for name, i0, i1 in rows:
    t0, t1 = i0 / fps, i1 / fps
    lost_idx = np.nonzero(state[i0:i1] == 'lost')[0]
    handled = 're-search' if len(lost_idx) else ('tracking' if i0 > 0 else 'search')
    dropped = f'{lost_idx[0] / fps:.1f} s' if len(lost_idx) else '-'
    ts = first_stable(on, i0, i1)
    # fresh rate: first valid estimate once the region is back on target
    m = (rt >= (ts if ts is not None else t1)) & (rt < t1) & np.isfinite(rb)
    tr = rt[m][0] - t0 if m.any() else None
    # longest stretch without any valid rate inside the segment
    valid_t = np.r_[t0, rt[(rt >= t0) & (rt < t1) & np.isfinite(rb)], t1]
    gap = np.max(np.diff(valid_t))
    seg = (rt >= t0) & (rt < t1) & np.isfinite(rb) & np.isfinite(rref)
    err = np.nanmedian(np.abs(rb[seg] - rref[seg])) if seg.any() else np.nan
    print(f'{name[:44]:44s} {handled:>12s} {dropped:>19s} '
          f'{("%.1f s" % (ts - t0)) if ts is not None else "never":>16s} '
          f'{("%.0f s" % tr) if tr is not None else "never":>17s} {gap:>10.0f} s '
          f'{wrong[i0:i1].sum() / fps:>14.1f} s {err:>9.1f}')
np.savez('/tmp/sv/dynamic_eval.npz', T=T, state=state, cen=cen, rt=rt, rb=rb, rref=rref)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={'height_ratios': [2, 2, 0.5, 2]})
for k, (lab, a, b) in enumerate([('x (px)', 0, 2), ('y (px)', 1, 3)]):
    ax[k].fill_between(T, g[:, a], g[:, b], color='#0ca30c', alpha=0.18, lw=0, label='true mouse box')
    ax[k].plot(T, cen[:, k], color='#2a78d6', lw=1.2, label='region centre (method)')
    ax[k].set_ylabel(lab)
    ax[k].legend(loc='upper left', fontsize=8, frameon=False)
ax[0].set_title('Region centre vs true mouse position', loc='left')
colors = {'searching': '#c3c2b7', 'locked': '#0ca30c', 'artifact': '#fab219', 'lost': '#d03b3b'}
for s, c in colors.items():
    m = state == s
    ax[2].fill_between(T, 0, 1, where=m, color=c, step='mid', lw=0, label=s)
ax[2].fill_between(T, 1.1, 1.5, where=wrong, color='#d03b3b', step='mid', lw=0, label='tracking a wrong spot')
ax[2].set_yticks([])
ax[2].legend(loc='upper left', fontsize=8, frameon=False, ncols=5, bbox_to_anchor=(0, 1.9))
ax[3].plot(rt, rref, color='#898781', lw=1.5, ls='--', label='reference (source recording)')
ax[3].plot(rt, rb, color='#2a78d6', lw=2, label='reported rate (test video)')
ax[3].set_ylabel('breaths / min')
ax[3].set_ylim(0, 110)
ax[3].legend(loc='upper left', fontsize=8, frameon=False)
ax[3].set_xlabel('time in test video (s)')
for a in ax:
    for e in events:
        a.axvline(e['t_s'], color='k', lw=0.8, ls=':')
for e in events:
    ax[0].text(e['t_s'] + 1, ax[0].get_ylim()[1], e['event'].split(':')[0], fontsize=8, va='top')
plt.tight_layout()
plt.savefig('exploration/14_dynamic_region_test.png', dpi=75)
