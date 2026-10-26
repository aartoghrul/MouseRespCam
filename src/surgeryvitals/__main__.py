"""Usage:
    python -m surgeryvitals [VIDEO] [--speed 1]          # GUI, simulated real time
    python -m surgeryvitals VIDEO --headless [--plot P]  # as fast as possible, write results
"""
import argparse
import sys
import time
from pathlib import Path


def run_headless(path, out_root, plot=None):
    from .monitor import RespirationMonitor
    from .recorder import ResultRecorder
    from .source import VideoFileSource

    src = VideoFileSource(path)
    mon = RespirationMonitor(src.shape, src.fps)
    rec = ResultRecorder()
    t0 = time.time()
    while True:
        frame, t = src.read()
        if frame is None:
            break
        res = mon.process(frame, t)
        rec.add(res, mon.bbox() if res.rate is not None else None)
    el = time.time() - t0
    out = rec.save(Path(out_root) / Path(path).stem, meta={'fps': src.fps})
    print(f'{Path(path).name}: {src.duration:.0f} s of video in {el:.0f} s ({src.duration / el:.1f}x real time) -> {out}')
    if plot:
        from .report import summary_plot
        summary_plot(rec, src.fps, Path(path).name, plot)
    return rec


def main():
    ap = argparse.ArgumentParser(prog='surgeryvitals')
    ap.add_argument('video', nargs='?')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--out', default='results')
    ap.add_argument('--plot')
    ap.add_argument('--speed', type=float, default=1.0, help='playback speed; 0 = as fast as possible')
    ap.add_argument('--screenshot', help=argparse.SUPPRESS)
    ap.add_argument('--after', type=float, default=20.0, help=argparse.SUPPRESS)
    ap.add_argument('--screen-size', type=lambda v: tuple(map(int, v.split('x'))), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.headless:
        if not a.video:
            ap.error('--headless needs a video')
        run_headless(a.video, a.out, a.plot)
    else:
        from .gui import run_gui
        sys.exit(run_gui(a.video, a.speed, a.out, a.screenshot, a.after, a.screen_size))


if __name__ == '__main__':
    main()
