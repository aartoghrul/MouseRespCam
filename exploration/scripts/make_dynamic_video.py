# Build a test video in which the mouse changes position, with frame-by-frame ground truth.
# Real footage is stitched together and known camera motions are applied (sudden bump, slow pan,
# zoom + tilt, cuts to other sessions). The true mouse box per frame is a hand-marked body box in
# the source frame, mapped through the exact transform applied.
#
#   .venv/bin/python exploration/scripts/make_dynamic_video.py
#   -> videos/synthetic/dynamic_test.mp4, dynamic_test_truth.csv, dynamic_test_events.json
import csv
import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path('videos/synthetic')
W, H, FPS = 1920, 1080, 30

# hand-marked mouse body boxes (x0, y0, x1, y1) in source coordinates (from gridded frames)
MOUSE = {3: (1000, 800, 1360, 950), 5: (1100, 450, 1400, 720), 4: (1030, 740, 1420, 950)}

# (start s in test video, duration s, source recording, source start s)
SEGMENTS = [(0, 270, 3, 100), (270, 60, 5, 200), (330, 60, 4, 400)]
EVENTS = [
    (90, 'sudden camera bump: shift (-300, -150) px'),
    (150, 'slow pan: +500 px to the right over 10 s'),
    (210, 'sudden zoom 1.2x + 3 deg tilt'),
    (270, 'cut to another session (Recording 5)'),
    (330, 'cut to another session (Recording 4)'),
]


def transform(t):
    """2x3 affine applied to the source frame at test-video time t."""
    if t < 90 or t >= 270:
        return np.float32([[1, 0, 0], [0, 1, 0]])
    if t < 150:
        dx, dy = -300, -150
    elif t < 160:
        dx, dy = -300 + 500 * (t - 150) / 10, -150
    else:
        dx, dy = 200, -150
    A = np.float32([[1, 0, dx], [0, 1, dy]])
    if t >= 210:
        Z = cv2.getRotationMatrix2D((W / 2, H / 2), 3.0, 1.2)
        A = (np.vstack([Z, [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]]))[:2].astype(np.float32)
    return A


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(OUT / 'dynamic_test.mp4'), cv2.VideoWriter_fourcc(*'avc1'), FPS, (W, H))
    rows = []
    n = 0
    for start, dur, rec, src_t0 in SEGMENTS:
        cap = cv2.VideoCapture(f'videos/Recording_{rec}.mp4')
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_t0 * FPS))
        x0, y0, x1, y1 = MOUSE[rec]
        corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        for i in range(int(dur * FPS)):
            ok, f = cap.read()
            if not ok:
                raise RuntimeError(f'Recording_{rec} ended early')
            t = n / FPS
            A = transform(t)
            if not np.allclose(A, [[1, 0, 0], [0, 1, 0]]):
                f = cv2.warpAffine(f, A, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            q = cv2.transform(corners[None], A)[0]
            rows.append([n, f'{t:.3f}', rec, f'{src_t0 + i / FPS:.3f}', *np.round(q.min(0)).astype(int),
                         *np.round(q.max(0)).astype(int)])
            vw.write(f)
            n += 1
        cap.release()
    vw.release()
    with open(OUT / 'dynamic_test_truth.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['frame', 't_s', 'source_rec', 'source_t_s', 'x0', 'y0', 'x1', 'y1'])
        w.writerows(rows)
    json.dump([{'t_s': t, 'event': e} for t, e in EVENTS], open(OUT / 'dynamic_test_events.json', 'w'), indent=1)
    print(f'{n} frames ({n / FPS:.0f} s) -> {OUT}/dynamic_test.mp4')


if __name__ == '__main__':
    main()
