"""Usage:
    python -m surgeryvitals [VIDEO] [--speed 1]          # GUI, simulated real time
    python -m surgeryvitals VIDEO --headless [--plot P]  # as fast as possible, write results
    python -m surgeryvitals --camera [N]                 # GUI with camera N selected; press Record to start
    python -m surgeryvitals --camera [N] --headless      # live, print the rate; Ctrl+C stops and saves
    python -m surgeryvitals --list-cameras
"""
import argparse
import sys
import time
from pathlib import Path


def session_meta(params, opts):
    from dataclasses import asdict
    return {'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'save': asdict(opts), 'params': asdict(params)}


def run_headless(path, out_root, plot=None):
    from .monitor import RespirationMonitor
    from .recorder import ResultRecorder, SaveOptions
    from .source import VideoFileSource

    src = VideoFileSource(path)
    mon = RespirationMonitor(src.shape, src.fps)
    rec = ResultRecorder()
    opts = SaveOptions(footage=False)
    meta = session_meta(mon.p, opts)
    t0 = time.time()
    while True:
        frame, t = src.read()
        if frame is None:
            break
        res = mon.process(frame, t)
        rec.add(res, mon.bbox() if res.rate is not None else None)
    el = time.time() - t0
    out = rec.save(Path(out_root) / Path(path).stem, {**meta, **src.describe()}, opts, Path(path).name)
    print(f'{Path(path).name}: {src.duration:.0f} s of video in {el:.0f} s ({src.duration / el:.1f}x real time) -> {out}')
    if plot:
        from .report import summary_plot
        summary_plot(rec, src.fps, Path(path).name, plot)
    return rec


def run_live_headless(index, size, max_fps, out_root, duration=None, plot=None, footage='half', name=None,
                      every_s=5):
    from .monitor import RespirationMonitor
    from .recorder import FootageWriter, ResultRecorder, SaveOptions
    from .source import live_session_name, open_camera_source

    print(f'Opening camera {index}...')
    src = open_camera_source(index, size, max_fps)
    mon = RespirationMonitor(src.shape, src.fps)
    rec = ResultRecorder()
    opts = SaveOptions(footage=footage != 'off', footage_scale=1.0 if footage == 'full' else 0.5)
    meta = session_meta(mon.p, opts)
    out_dir = Path(out_root) / live_session_name(index, name)
    writer = FootageWriter(out_dir, src.fps, src.shape, opts.footage_scale) if opts.footage else None
    print(f'Monitoring, saving to {out_dir}. Press Ctrl+C to stop and save.')
    try:
        while duration is None or not rec.t or rec.t[-1] < duration:
            frame, t = src.read()
            if frame is None:
                if src.ended:
                    print(src.error or 'Camera stopped.')
                    break
                continue
            if writer:
                writer.write(frame)
            res = mon.process(frame, t)
            rec.add(res, mon.bbox() if res.rate is not None else None)
            r = res.rate
            if r is not None and round(r.t) % every_s == 0:
                rate = f'{r.bpm:5.0f} /min' if r.bpm == r.bpm else '    — /min'
                print(f'{int(r.t // 60):02d}:{int(r.t % 60):02d}  {rate}  confidence {r.confidence:<5}  {r.state}'
                      + (f'  ({src.gaps} gaps: processing too slow)' if src.gaps else ''), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        if writer:
            writer.close()
    if not rec.t:
        return rec
    out = rec.save(out_dir, {**meta, **src.describe()}, opts, out_dir.name)
    print(f'{rec.t[-1]:.0f} s monitored -> {out}')
    if plot:
        from .report import summary_plot
        summary_plot(rec, src.fps, out_dir.name, plot)
    return rec


def size_arg(v):
    try:
        w, h = map(int, v.lower().split('x'))
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError('expected WIDTHxHEIGHT, e.g. 1920x1080')


def main():
    ap = argparse.ArgumentParser(prog='surgeryvitals')
    ap.add_argument('video', nargs='?')
    ap.add_argument('--camera', type=int, nargs='?', const=0, metavar='N',
                    help='camera index (see --list-cameras). GUI: selects it, recording starts with the '
                         'Record button. Headless: starts recording at once')
    ap.add_argument('--camera-size', type=size_arg, metavar='WxH',
                    help='requested camera resolution (default 1920x1080; the camera picks its nearest mode)')
    ap.add_argument('--max-fps', type=float,
                    help='upper limit for the live processing rate (default 30; lowered automatically on slow machines)')
    ap.add_argument('--name', help='session name for the results folder, e.g. an animal ID (live)')
    ap.add_argument('--footage', choices=('off', 'half', 'full'), default='half',
                    help='save the live camera footage at half (default) or full resolution (headless)')
    ap.add_argument('--duration', type=float, help='stop a headless live session after this many seconds')
    ap.add_argument('--list-cameras', action='store_true')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--out', help='results folder (default: results)')
    ap.add_argument('--plot')
    ap.add_argument('--speed', type=float, default=1.0, help='playback speed; 0 = as fast as possible')
    ap.add_argument('--screenshot', help=argparse.SUPPRESS)
    ap.add_argument('--after', type=float, default=20.0, help=argparse.SUPPRESS)
    ap.add_argument('--screen-size', type=size_arg, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.video and a.camera is not None:
        ap.error('give either a video or --camera, not both')
    try:
        run(ap, a)
    except OSError as e:          # camera / file problems: the message says what to do
        sys.exit(f'Error: {e}')


def run(ap, a):
    if a.list_cameras:
        from .source import list_cameras
        cams = list_cameras()
        for i, label in cams:
            print(f'{i}: {label}')
        if not cams:
            print('no cameras found')
        return
    if a.headless:
        out = a.out or 'results'
        if a.camera is not None:
            run_live_headless(a.camera, a.camera_size or (1920, 1080), a.max_fps or 30.0, out, a.duration, a.plot,
                              a.footage, a.name)
        elif not a.video:
            ap.error('--headless needs a video or --camera')
        else:
            run_headless(a.video, out, a.plot)
    else:
        from .gui import run_gui
        # options given on the command line override the saved GUI settings
        overrides = {k: v for k, v in (('camera_size', a.camera_size), ('max_fps', a.max_fps), ('out_root', a.out),
                                       ('session_name', a.name)) if v is not None}
        sys.exit(run_gui(a.video, a.speed, a.screenshot, a.after, a.screen_size, a.camera, overrides))


if __name__ == '__main__':
    main()
