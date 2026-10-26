"""Static summary figure of a processed recording (for exploration / validation)."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BLUE, ORANGE = '#2a78d6', '#eb6834'
STATUS = {'good': '#0ca30c', 'fair': '#fab219', 'poor': '#ec835a', 'none': '#c3c2b7'}


def summary_plot(rec, fps, title, path, zoom_s=20):
    t, y, art = np.array(rec.t), np.array(rec.y, float), np.array(rec.art, bool)
    rt = np.array([r.t for r in rec.rates])
    bpm = np.array([r.bpm for r in rec.rates])
    raw = np.array([r.raw_bpm for r in rec.rates])
    conf = [r.confidence for r in rec.rates]
    bt = np.array([b for b, _ in rec.breaths])
    by = np.array([v for _, v in rec.breaths])

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig = plt.figure(figsize=(16, 9), facecolor='#fcfcfb')
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.2, 1.6, 0.25], hspace=0.55, wspace=0.12)

    # two zoomed waveform excerpts where the signal was good
    good_t = rt[[c == 'good' for c in conf]]
    picks = [good_t[len(good_t) // 4], good_t[3 * len(good_t) // 4]] if len(good_t) > 4 else [t[-1] / 3, 2 * t[-1] / 3]
    for k, tc in enumerate(picks):
        ax = fig.add_subplot(gs[0, k])
        m = (t > tc - zoom_s / 2) & (t < tc + zoom_s / 2)
        ax.plot(t[m], np.where(art[m], np.nan, y[m]), color=BLUE, lw=1.4, label='breathing motion')
        ax.plot(t[m], np.where(art[m], y[m], np.nan), color=MUTED, lw=1.4, label='artifact (ignored)')
        mb = (bt > tc - zoom_s / 2) & (bt < tc + zoom_s / 2)
        ax.plot(bt[mb], by[mb], 'o', color=ORANGE, ms=5, label=f'detected breath ({mb.sum()} in {zoom_s} s)')
        ax.set_title(f'Waveform excerpt {k + 1}: {tc - zoom_s / 2:.0f}–{tc + zoom_s / 2:.0f} s', loc='left', color=INK)
        ax.set_yticks([]); ax.set_xlabel('time (s)')
        ax.grid(axis='x', color=GRID, lw=0.6)
        ax.legend(loc='upper right', fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, np.where(art, np.nan, y), color=BLUE, lw=0.4)
    ax.plot(t, np.where(art, y, np.nan), color=MUTED, lw=0.4)
    ax.set_title('Breathing motion, whole recording (grey = artifact)', loc='left', color=INK)
    ax.set_yticks([]); ax.set_xlim(0, t[-1])
    ok = np.isfinite(y) & ~art
    if ok.any():
        lo, hi = np.percentile(y[ok], [0.5, 99.5]); ax.set_ylim(lo - (hi - lo) * 0.3, hi + (hi - lo) * 0.3)

    ax = fig.add_subplot(gs[2, :])
    ax.plot(rt, raw, color=MUTED, lw=0.8, ls=':', label='tracker (all windows)')
    ax.plot(rt, bpm, color=BLUE, lw=2, label='reported rate (good/fair confidence)')
    if len(bt) > 2:
        ib = np.diff(bt); keep = ib < 3
        ax.plot(bt[1:][keep], 60 / ib[keep], '.', color=ORANGE, ms=2.5, alpha=0.6, label='breath-to-breath')
    ax.set_ylabel('breaths / min'); ax.set_xlim(0, t[-1])
    fin = bpm[np.isfinite(bpm)]
    top = max(100, np.percentile(fin, 99) * 1.4) if len(fin) else 150
    ax.set_ylim(0, top); ax.grid(color=GRID, lw=0.6)
    ax.set_title('Respiratory rate', loc='left', color=INK)
    ax.legend(loc='upper right', fontsize=8, frameon=False, ncols=3)

    ax = fig.add_subplot(gs[3, :], sharex=ax)
    ax.bar(rt, 1, width=1.0, color=[STATUS[c] for c in conf], align='edge')
    ax.set_yticks([]); ax.set_xlabel('time (s)')
    ax.set_title('Confidence: ' + '   '.join(f'■ {k}' for k in STATUS), loc='left', color=INK2, fontsize=8)
    cover = np.mean([c in ('good', 'fair') for c in conf]) * 100 if conf else 0
    fig.suptitle(f'{title} — rate reported for {cover:.0f}% of the recording', x=0.01, ha='left', color=INK, fontsize=12)
    fig.savefig(path, dpi=80, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
