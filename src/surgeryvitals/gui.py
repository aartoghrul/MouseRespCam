"""Monitoring GUI. Runs a live camera, or replays a recording in simulated real time (or faster),
through the same frame-by-frame pipeline."""
import os
import queue
import sys
import tempfile
import time
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QByteArray, QPointF, QRect, QRectF, QSettings, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (QColor, QCursor, QDesktopServices, QFont, QGuiApplication, QIcon, QImage, QKeySequence,
                           QPainter, QPalette, QPen, QPixmap, QPolygonF, QShortcut)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QProgressBar, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from .monitor import ARTIFACT, LOCKED, LOST, SEARCHING, RespirationMonitor
from .recorder import FootageWriter, ResultRecorder, SaveOptions
from .source import VideoFileSource, list_cameras, live_session_name, mac_camera_names, open_camera_source

# ---------------------------------------------------------------- theme (dark, validated palette)
PAGE, SURFACE, RAISED = '#0d0d0d', '#1a1a19', '#232321'
INK, INK2, MUTED = '#f0efec', '#c3c2b7', '#898781'
GRID, AXIS, BORDER = '#2c2c2a', '#383835', 'rgba(255,255,255,0.10)'
BLUE, ORANGE = '#3987e5', '#d95926'
GOOD, WARNING, SERIOUS, CRITICAL = '#0ca30c', '#fab219', '#ec835a', '#d03b3b'

SIGNAL_WINDOW_S = 20.0
DISPLAY_W = 960
DISABLED = '#5a5955'

# ---------------------------------------------------------------- toolbar icons (24 px grid, stroke style)
ICONS = {
    'open': '<path d="M3 6.5A1.5 1.5 0 0 1 4.5 5H9l2 2h8.5A1.5 1.5 0 0 1 21 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-15'
            'A1.5 1.5 0 0 1 3 17.5z"/>',
    'camera': '<rect x="2.5" y="6.5" width="13" height="11" rx="2"/><path d="M15.5 10.5 21.5 7v10l-6-3.5"/>',
    'play': '<path d="M7.5 4.8v14.4L19 12z" fill="{c}"/>',
    'pause': '<rect x="6" y="5" width="4" height="14" rx="1" fill="{c}" stroke="none"/>'
             '<rect x="14" y="5" width="4" height="14" rx="1" fill="{c}" stroke="none"/>',
    'stop': '<rect x="6" y="6" width="12" height="12" rx="2" fill="{c}" stroke="none"/>',
    'record': f'<circle cx="12" cy="12" r="6.5" fill="{CRITICAL}" stroke="none"/>',
    'replay': '<path d="M3.5 12a8.5 8.5 0 1 0 2.5-6L3.5 8.5"/><path d="M3.5 3.5v5h5"/>',
    'auto': '<path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3"/>'
            '<circle cx="12" cy="12" r="3.5"/>',
    'draw': '<rect x="4" y="4" width="16" height="16" rx="1" stroke-dasharray="3.2 3"/>'
            '<path d="M12 9v6M9 12h6"/>',
    'map': '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M15 3v18M3 9h18M3 15h18"/>'
           '<rect x="9" y="9" width="6" height="6" fill="{c}"/><rect x="15" y="9" width="6" height="6" fill="{c}" '
           'fill-opacity="0.45"/><rect x="9" y="15" width="6" height="6" fill="{c}" fill-opacity="0.45"/>',
    'results': '<path d="M3 6.5A1.5 1.5 0 0 1 4.5 5H9l2 2h8.5A1.5 1.5 0 0 1 21 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-15'
               'A1.5 1.5 0 0 1 3 17.5z"/><path d="M10 15.5l4.5-4.5M11 11h3.5v3.5"/>',
    'settings': '<path d="M4 6.5h8M16 6.5h4M4 12h2.5M10.5 12H20M4 17.5h10M18 17.5h2"/><circle cx="14" cy="6.5" r="2"/>'
                '<circle cx="8.5" cy="12" r="2"/><circle cx="16" cy="17.5" r="2"/>',
}


def make_icon(name):
    """QIcon from ICONS, in ink (normal) and a dim grey (disabled), rendered for high-DPI screens."""
    icon = QIcon()
    for mode, color in ((QIcon.Normal, INK), (QIcon.Active, INK), (QIcon.Disabled, DISABLED)):
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{c}" '
               'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' + ICONS[name] + '</svg>')
        pm = QPixmap(72, 72)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        QSvgRenderer(QByteArray(svg.replace('{c}', color).encode())).render(p)
        p.end()
        icon.addPixmap(pm, mode)
    return icon


def checkbox_image(checked):
    """Checkbox indicator as a PNG file (style sheets can only load images from files)."""
    path = Path(tempfile.gettempdir()) / f'surgeryvitals_check_{int(checked)}.png'
    if not path.exists():
        box = (f'<rect x="1" y="1" width="22" height="22" rx="5" fill="{BLUE}"/>'
               f'<path d="M6.5 12.5l3.5 3.5 7.5-8" fill="none" stroke="{INK}" stroke-width="2.6" '
               f'stroke-linecap="round" stroke-linejoin="round"/>' if checked else
               f'<rect x="1.75" y="1.75" width="20.5" height="20.5" rx="4.5" fill="{RAISED}" stroke="{MUTED}" '
               f'stroke-width="1.5"/>')
        pm = QPixmap(48, 48)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        QSvgRenderer(QByteArray(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{box}</svg>'
                                .encode())).render(p)
        p.end()
        pm.save(str(path))
    return path.as_posix()


def camera_label(index):
    """Camera name for display (macOS reports names without opening the camera)."""
    names = mac_camera_names()
    return names[index] if names and 0 <= index < len(names) else f'camera {index}'


def shortcut_text(key):
    """Shortcut as the platform writes it (⌘O on macOS, Ctrl+O on Windows)."""
    return QKeySequence(key).toString(QKeySequence.NativeText)

def make_qss(s):
    """Stylesheet with every size multiplied by the UI scale s (see ui_scale)."""
    px = lambda v: max(1, round(v * s))
    fpx = lambda v, lo: max(lo, round(v * s))      # font sizes never drop below a readable floor
    return f"""
QWidget {{ background: {PAGE}; color: {INK}; font-size: {fpx(13, 11)}px; }}
QLabel {{ background: transparent; }}
QFrame#card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: {px(12)}px; }}
QFrame#topbar {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
QLabel#appTitle {{ font-size: {fpx(16, 13)}px; font-weight: 700; }}
QLabel#cardTitle {{ color: {INK}; font-size: {fpx(14, 12)}px; font-weight: 600; }}
QLabel#cardSub {{ color: {MUTED}; font-size: {fpx(12, 10)}px; }}
QLabel#statKey {{ color: {MUTED}; font-size: {fpx(12, 11)}px; }}
QLabel#statVal {{ color: {INK}; font-size: {fpx(16, 13)}px; font-weight: 600; }}
QLabel#unit {{ color: {INK2}; font-size: {fpx(15, 12)}px; }}
QPushButton, QComboBox {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: {px(8)}px;
    padding: {px(6)}px {px(12)}px; color: {INK}; }}
QPushButton:hover, QComboBox:hover {{ background: {GRID}; }}
QPushButton:checked {{ background: #16304f; border-color: {BLUE}; }}
QPushButton:disabled {{ color: {AXIS}; }}
QPushButton#icon {{ padding: 0px; background: transparent; border: 1px solid transparent; }}
QPushButton#icon:hover {{ background: {RAISED}; border-color: {BORDER}; }}
QPushButton#icon:checked {{ background: #16304f; border-color: {BLUE}; }}
QFrame#sep {{ background: {AXIS}; border: none; }}
QWidget#group {{ background: transparent; }}
QFrame#tile {{ background: {RAISED}; border-radius: {px(8)}px; }}
QLabel#timeChip {{ background: {RAISED}; border-radius: {px(6)}px; padding: {px(3)}px {px(9)}px; color: {INK2}; }}
QLabel#section {{ color: {INK}; font-size: {fpx(14, 12)}px; font-weight: 600; }}
QDialog {{ background: {SURFACE}; }}
QLineEdit {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: {px(8)}px; padding: {px(5)}px {px(8)}px;
    color: {INK}; selection-background-color: #16304f; }}
QLineEdit:focus {{ border-color: {BLUE}; }}
QCheckBox {{ spacing: {px(8)}px; background: transparent; }}
QCheckBox::indicator {{ width: {px(16)}px; height: {px(16)}px; }}
QCheckBox::indicator:unchecked {{ image: url({checkbox_image(False)}); }}
QCheckBox::indicator:checked {{ image: url({checkbox_image(True)}); }}
QComboBox:disabled {{ color: {MUTED}; }}
QToolTip {{ background: {RAISED}; color: {INK}; border: 1px solid {BORDER}; padding: 4px 6px; }}
QComboBox::drop-down {{ border: none; width: {px(18)}px; }}
QComboBox QAbstractItemView {{ background: {RAISED}; selection-background-color: #16304f; }}
QProgressBar {{ background: {RAISED}; border: none; border-radius: 2px; }}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 2px; }}
"""


def dark_palette():
    """Palette for the controls Qt draws itself (check marks, popups, scroll bars) in the theme colours."""
    pal = QPalette()
    for role, color in ((QPalette.Window, PAGE), (QPalette.WindowText, INK), (QPalette.Base, RAISED),
                        (QPalette.AlternateBase, SURFACE), (QPalette.Text, INK), (QPalette.Button, RAISED),
                        (QPalette.ButtonText, INK), (QPalette.Highlight, BLUE), (QPalette.HighlightedText, INK),
                        (QPalette.ToolTipBase, RAISED), (QPalette.ToolTipText, INK), (QPalette.Link, BLUE),
                        (QPalette.PlaceholderText, MUTED)):
        pal.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(MUTED))
    return pal


def ui_scale(avail):
    """UI scale from the available screen area (logical px). 1.0 is designed for ~1700x1080."""
    return float(np.clip(min(avail.height() / 1080, avail.width() / 1700), 0.62, 1.3))


pg.setConfigOptions(antialias=True, background=SURFACE, foreground=MUTED, imageAxisOrder='row-major')


def fmt_time(t):
    t = max(0, t)
    return f'{int(t // 60):02d}:{int(t % 60):02d}'


# ---------------------------------------------------------------- processing thread
class Batch:
    """Everything produced since the last GUI refresh."""

    def __init__(self):
        self.t, self.y, self.art = [], [], []
        self.breaths, self.rates = [], []
        self.frame = self.quad = self.heat = None
        self.state, self.roi_source = SEARCHING, None
        self.now = 0.0
        self.duration = 0.0
        self.xrt = None
        self.scale = None       # display px per full-resolution px
        self.finished = False
        self.live = False
        self.fps = None
        self.load = None        # live: share of real time spent processing
        self.gaps = 0           # live: times processing fell behind and frames were dropped
        self.error = None


class Worker(QThread):
    """Runs the monitor on a recording (paced to `speed`) or a live camera (paced by the camera).
    `camera` = (index, requested size, max fps) selects live mode. Results go to a folder in
    `out_root` (the recording's name, or <session name or camera N>_<date-time> when live)."""
    batch = Signal(object)
    failed = Signal(str)
    saved = Signal(str)
    opened = Signal(str)        # description of the source once it is running
    info = Signal(str)

    def __init__(self, path, speed, out_root, opts, camera=None, session_name=None):
        super().__init__()
        self.path, self.speed, self.out_root, self.camera = path, speed, out_root, camera
        self.opts, self.session_name = opts, session_name
        self.out_dir = None
        self.meta = {}
        self.cmds = queue.Queue()
        self._stop = False

    def stop(self):
        self._stop = True
        self.wait(10000)

    def send(self, *cmd):
        self.cmds.put(cmd)

    def run(self):
        try:
            self._run()
        except Exception as e:      # surface errors in the UI instead of dying silently
            msg = str(e) if isinstance(e, OSError) else f'{type(e).__name__}: {e}'
            self.failed.emit(msg)

    def _save(self, rec, src):
        out = rec.save(self.out_dir, {**self.meta, **src.describe()}, self.opts, self.out_dir.name)
        self.saved.emit(str(out))

    def _open(self):
        if self.camera is None:
            src = VideoFileSource(self.path)
            self.out_dir = Path(self.out_root) / Path(self.path).stem
            self.opened.emit(Path(self.path).name)
            return src
        index, size, max_fps = self.camera
        src = open_camera_source(index, size, max_fps, log=self.info.emit)
        self.out_dir = Path(self.out_root) / live_session_name(index, self.session_name)
        h, w = src.shape
        self.opened.emit(f'{src.name} · {w}×{h} · {src.fps:g} fps')
        return src

    def _run(self):
        src = self._open()
        writer = None
        try:
            mon = RespirationMonitor(src.shape, src.fps)
            self.meta = {'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'save': asdict(self.opts),
                         'params': asdict(mon.p)}
            if src.live and self.opts.footage:
                writer = FootageWriter(self.out_dir, src.fps, src.shape, self.opts.footage_scale)
            self._loop(src, mon, writer)
        finally:
            src.release()
            if writer:
                writer.close()

    def _loop(self, src, mon, writer):
        live = src.live
        rec = ResultRecorder()
        scale = DISPLAY_W / src.shape[1]
        disp_size = (DISPLAY_W, int(round(src.shape[0] * scale)))
        cell = mon.loc.p.loc_block * mon.loc.sx * scale       # heat-map block size in display px
        acc, paused, last_t = Batch(), False, 0.0
        wall0, video0 = time.perf_counter(), 0.0               # pacing anchor
        rate_wall, rate_video, xrt = time.perf_counter(), 0.0, None
        busy, load = 0.0, None                                 # live: share of wall time spent working
        stopped = False                                        # by the user (vs. window closed / replaced)
        last_emit = 0.0
        while not self._stop:
            while not self.cmds.empty():
                cmd = self.cmds.get()
                if cmd[0] == 'speed':
                    self.speed = cmd[1]
                    wall0, video0 = time.perf_counter(), last_t
                elif cmd[0] == 'pause':
                    paused = cmd[1]
                    wall0, video0 = time.perf_counter(), last_t
                elif cmd[0] == 'roi':
                    mon.set_manual_roi(cmd[1], last_t)
                elif cmd[0] == 'redetect':
                    mon.redetect()
                elif cmd[0] == 'stop':
                    self._stop = stopped = True
            if self._stop:
                break
            if paused:
                self.msleep(30)
                continue
            frame, t = src.read()
            if frame is None:
                if src.ended:
                    break
                continue                                       # live: no frame yet
            tic = time.perf_counter()
            if writer:
                writer.write(frame)
            last_t = t
            res = mon.process(frame, t)
            rec.add(res, mon.bbox() if res.rate is not None else None)
            acc.t.append(t)
            acc.y.append(res.y)
            acc.art.append(res.artifact)
            acc.breaths.extend(res.breaths)
            if res.rate is not None:
                acc.rates.append(res.rate)
            if res.map_updated and mon.loc.score_map is not None:
                acc.heat = (mon.loc.score_map.copy(), cell)
            if self.speed > 0 and not live:
                lag = wall0 + (t - video0) / self.speed - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
            now = time.perf_counter()
            if now - rate_wall > 1.0:
                xrt = (t - rate_video) / (now - rate_wall)
                load = busy / (now - rate_wall)
                rate_wall, rate_video, busy = now, t, 0.0
            if now - last_emit >= 1 / 30:
                last_emit = now
                small = cv2.resize(frame, disp_size, interpolation=cv2.INTER_AREA)
                acc.frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                acc.quad = None if res.quad is None else res.quad * scale
                acc.state, acc.roi_source = res.state, res.roi_source
                acc.now, acc.duration, acc.xrt, acc.scale = t, src.duration, xrt, scale
                acc.live, acc.fps, acc.load, acc.gaps = live, src.fps, load, mon.gaps
                self.batch.emit(acc)
                acc = Batch()
            busy += time.perf_counter() - tic
        if self._stop and not stopped and not (live and rec.t):
            return          # closed or replaced; a live session is still saved (it cannot be re-run)
        acc.finished = True
        acc.now, acc.duration, acc.live = last_t, src.duration, live
        acc.error = getattr(src, 'error', None)
        if rec.t:
            self._save(rec, src)
        self.batch.emit(acc)


# ---------------------------------------------------------------- video view
STATE_STYLE = {
    LOCKED: (GOOD, 'Breathing region'),
    ARTIFACT: (WARNING, 'Motion artifact'),
    LOST: (CRITICAL, 'Region lost'),
    SEARCHING: (MUTED, 'Searching'),
}


class VideoView(QWidget):
    roiDrawn = Signal(tuple)    # (x0, y0, x1, y1) in display-image pixels

    def __init__(self):
        super().__init__()
        self.setMinimumSize(160, 90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img = None
        self.quad = None
        self.state = SEARCHING
        self.source = None
        self.now = 0.0
        self.heat = None
        self.show_heat = False
        self.drawing = False
        self.drag = None
        self.placeholder = f'Start the camera, or open a recording  ({shortcut_text(QKeySequence.Open)})'
        self.ended = False      # session over: the last frame stays, without live overlays
        self.s = 1.0

    def set_frame(self, rgb, quad, state, source, now):
        h, w, _ = rgb.shape
        self.img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.quad, self.state, self.source, self.now = quad, state, source, now
        self.ended = False
        self.update()

    def aspect(self):
        return self.img.width() / self.img.height() if self.img is not None else 16 / 9

    def set_heat(self, heat):
        score, cell = heat
        a = np.clip((score - 0.25) / 0.5, 0, 1)
        rgba = np.zeros(score.shape + (4,), np.uint8)
        rgba[..., :3] = (57, 135, 229)
        rgba[..., 3] = (a * 170).astype(np.uint8)
        hh, ww = score.shape
        self.heat = (QImage(rgba.data, ww, hh, 4 * ww, QImage.Format_RGBA8888).copy(), cell)
        self.update()

    def _target(self):
        if self.img is None:
            return QRectF(self.rect())
        iw, ih = self.img.width(), self.img.height()
        s = min(self.width() / iw, self.height() / ih)
        return QRectF((self.width() - iw * s) / 2, (self.height() - ih * s) / 2, iw * s, ih * s)

    def _to_img(self, pt):
        r = self._target()
        s = r.width() / self.img.width()
        return (pt.x() - r.x()) / s, (pt.y() - r.y()) / s

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(SURFACE))
        if self.img is None:
            p.setPen(QColor(MUTED))
            p.drawText(self.rect().adjusted(20, 0, -20, 0), Qt.AlignCenter | Qt.TextWordWrap, self.placeholder)
            return
        r = self._target()
        p.drawImage(r, self.img)
        s = r.width() / self.img.width()
        if self.show_heat and self.heat is not None:
            himg, cell = self.heat
            p.drawImage(QRectF(r.x(), r.y(), himg.width() * cell * s, himg.height() * cell * s), himg)
        color, label = STATE_STYLE[self.state]
        if self.quad is not None:
            poly = QPolygonF([QPointF(r.x() + x * s, r.y() + y * s) for x, y in self.quad])
            fill = QColor(color)
            fill.setAlpha(28)
            p.setBrush(fill)
            pen = QPen(QColor(color), 2.5)
            if self.state == LOST:
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawPolygon(poly)
            text = f'{label} · {"manual" if self.source == "manual" else "auto"}' if self.state != LOST else label
            top = min(pt.y() for pt in poly)
            left = min(pt.x() for pt in poly)
            self._chip(p, QPointF(left, top - 8), text, color, above=True)
        elif self.state == SEARCHING and not self.ended:
            self._chip(p, QPointF(r.center().x(), r.y() + 24), 'Searching for breathing motion… needs ~15 s of visible breathing',
                       MUTED, centered=True)
        self._chip(p, QPointF(r.x() + 12, r.bottom() - 12), fmt_time(self.now) + f'.{int(self.now * 10) % 10}',
                   RAISED, above=True, ink=INK)
        if self.drawing:
            p.setPen(QPen(QColor(BLUE), 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            if self.drag:
                p.drawRect(QRectF(self.drag[0], self.drag[1]).normalized())
            self._chip(p, QPointF(r.center().x(), r.y() + 24), 'Drag a box over the breathing area  ·  Esc to cancel',
                       BLUE, centered=True, ink=INK)

    def _chip(self, p, anchor, text, color, above=False, centered=False, ink=PAGE):
        f = QFont(self.font())
        f.setPixelSize(max(10, round(13 * self.s)))
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        fm = p.fontMetrics()
        pad = max(6, round(8 * self.s))
        w, h = fm.horizontalAdvance(text) + 2 * pad, fm.height() + pad
        x = anchor.x() - w / 2 if centered else anchor.x()
        y = anchor.y() - h if above else anchor.y()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(color))
        p.drawRoundedRect(QRectF(x, y, w, h), 6, 6)
        p.setPen(QColor(ink))
        p.drawText(QRectF(x, y, w, h), Qt.AlignCenter, text)

    def start_drawing(self, on):
        self.drawing = on
        self.drag = None
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        self.update()

    def mousePressEvent(self, e):
        if self.drawing and self.img is not None:
            self.drag = (e.position(), e.position())

    def mouseMoveEvent(self, e):
        if self.drawing and self.drag:
            self.drag = (self.drag[0], e.position())
            self.update()

    def mouseReleaseEvent(self, e):
        if self.drawing and self.drag:
            (x0, y0), (x1, y1) = self._to_img(self.drag[0]), self._to_img(e.position())
            self.start_drawing(False)
            if abs(x1 - x0) > 6 and abs(y1 - y0) > 6:
                self.roiDrawn.emit((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))


# ---------------------------------------------------------------- plots
class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        if spacing < 5:                       # minor ticks (1 s gridlines) stay unlabeled
            return [''] * len(values)
        return [fmt_time(v) for v in values]


def styled_plot(s, height=None):
    w = pg.PlotWidget(axisItems={'bottom': TimeAxis('bottom')})
    tick_font = QFont()
    tick_font.setPixelSize(max(9, round(11 * s)))
    w.setMenuEnabled(False)
    w.setMouseEnabled(False, False)
    w.hideButtons()
    pi = w.getPlotItem()
    for name in ('bottom', 'left'):
        ax = pi.getAxis(name)
        ax.setPen(pg.mkPen(AXIS))
        ax.setTextPen(pg.mkPen(MUTED))
        ax.setStyle(tickLength=-4, tickFont=tick_font)
    if height:
        w.setFixedHeight(height)
    return w


def card(title, subtitle, s):
    """Card with a one-line header: title, muted subtitle, then extras (legends, counters) added at
    the right end, so they never sit on top of the title text."""
    fr = QFrame()
    fr.setObjectName('card')
    lay = QVBoxLayout(fr)
    lay.setContentsMargins(round(14 * s), round(10 * s), round(14 * s), round(10 * s))
    lay.setSpacing(round(6 * s))
    head = QHBoxLayout()
    head.setSpacing(round(10 * s))
    t = QLabel(title)
    t.setObjectName('cardTitle')
    head.addWidget(t)
    if subtitle:
        st = QLabel(subtitle)
        st.setObjectName('cardSub')
        st.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)     # gives way to legends on narrow screens
        head.addWidget(st, 1)
    else:
        head.addStretch(1)
    lay.addLayout(head)
    return fr, lay, head


def legend_label(items):
    """Static legend: coloured swatch + text, text in ink (never in the series colour)."""
    parts = []
    for color, text, kind in items:
        sw = '━━' if kind == 'line' else '●'
        parts.append(f'<span style="color:{color}">{sw}</span>&nbsp;<span style="color:{INK2}">{text}</span>')
    lab = QLabel('&nbsp;&nbsp;&nbsp;'.join(parts))
    lab.setTextFormat(Qt.RichText)
    return lab


def separator(s):
    sep = QFrame()
    sep.setObjectName('sep')
    sep.setFixedSize(1, round(22 * s))
    return sep


# ---------------------------------------------------------------- settings
RESOLUTIONS = [('1920 × 1080  (Full HD)', (1920, 1080)), ('1280 × 720', (1280, 720)), ('640 × 480', (640, 480))]
RATES = [('Up to 30 fps', 30.0), ('Up to 15 fps  (slower computers)', 15.0)]
SAVE_ITEMS = [  # SaveOptions field, label, file / detail
    ('rate', 'Breathing rate over time', 'rate.csv · every second, with confidence'),
    ('breaths', 'Detected breaths', 'breaths.csv · time of every breath'),
    ('motion', 'Motion signal, per frame', 'motion.csv · waveform, raw motion, artifacts'),
    ('roi', 'Tracked region', 'roi.csv · position of the region over time'),
    ('summary', 'Summary figure', 'summary.png'),
    ('footage', 'Camera footage', 'footage.mp4 · live sessions, can be re-analysed'),
]
FOOTAGE_SIZES = [('Half resolution  (≈ 1.3 GB / hour)', 0.5), ('Full resolution  (≈ 5 GB / hour)', 1.0)]


def builtin_settings():
    return {'camera': 0, 'camera_size': (1920, 1080), 'max_fps': 30.0, 'out_root': 'results', 'session_name': '',
            'opts': SaveOptions()}


def load_settings(overrides=None):
    """The user's default settings (saved with 'Save as default'; QSettings: a plist on macOS, the
    registry on Windows), falling back to the built-in ones."""
    q = QSettings('SurgeryVitals', 'SurgeryVitals')
    b = builtin_settings()
    opts = b['opts']
    for f in fields(SaveOptions):
        setattr(opts, f.name, q.value(f'save/{f.name}', getattr(opts, f.name), type=type(getattr(opts, f.name))))
    cfg = {
        'camera': q.value('camera/index', b['camera'], type=int),
        'camera_size': (q.value('camera/width', b['camera_size'][0], type=int),
                        q.value('camera/height', b['camera_size'][1], type=int)),
        'max_fps': q.value('camera/max_fps', b['max_fps'], type=float),
        'out_root': q.value('output/folder', b['out_root'], type=str),
        'session_name': q.value('output/session_name', b['session_name'], type=str),
        'opts': opts,
    }
    cfg.update(overrides or {})
    return cfg


def store_settings(cfg):
    q = QSettings('SurgeryVitals', 'SurgeryVitals')
    q.setValue('camera/index', cfg['camera'])
    q.setValue('camera/width', cfg['camera_size'][0])
    q.setValue('camera/height', cfg['camera_size'][1])
    q.setValue('camera/max_fps', cfg['max_fps'])
    q.setValue('output/folder', cfg['out_root'])
    q.setValue('output/session_name', cfg['session_name'])
    for k, v in asdict(cfg['opts']).items():
        q.setValue(f'save/{k}', v)


class SettingsDialog(QDialog):
    """Camera, output folder, session name and what each session saves. 'Use for this session'
    keeps the choice until the program closes; 'Save as default' also loads it at every start."""

    def __init__(self, parent, cfg, s):
        super().__init__(parent)
        self.setWindowTitle('Session settings')
        self.cfg = cfg
        self.save_default = False
        px = lambda v: max(1, round(v * s))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(px(22), px(18), px(22), px(16))
        lay.setSpacing(px(6))

        f = QFormLayout()
        f.setHorizontalSpacing(px(14))
        f.setVerticalSpacing(px(8))
        f.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addLayout(f)

        def section(text, first=False):
            lab = QLabel(text)
            lab.setObjectName('section')
            if not first:
                lab.setContentsMargins(0, px(12), 0, 0)
            f.addRow(lab)

        def hint(text):
            lab = QLabel(text)
            lab.setObjectName('cardSub')
            lab.setWordWrap(True)
            return lab

        section('Camera', first=True)
        self.cam_box = QComboBox()
        self.cam_box.setMinimumWidth(px(300))
        refresh = QPushButton()
        refresh.setObjectName('icon')
        refresh.setIcon(make_icon('replay'))
        refresh.setIconSize(QSize(px(16), px(16)))
        refresh.setToolTip('Look for cameras again')
        refresh.clicked.connect(self._fill_cameras)
        row = QHBoxLayout()
        row.addWidget(self.cam_box, 1)
        row.addWidget(refresh)
        f.addRow('Camera', row)
        self.cam_note = hint('')
        self.cam_note.setVisible(False)
        f.addRow('', self.cam_note)
        self.res_box = QComboBox()
        for name, size in RESOLUTIONS:
            self.res_box.addItem(name, size)
        f.addRow('Resolution', self.res_box)
        self.fps_box = QComboBox()
        for name, v in RATES:
            self.fps_box.addItem(name, v)
        f.addRow('Processing', self.fps_box)
        f.addRow('', hint('The camera uses its nearest supported mode. The processing rate is lowered '
                          'automatically if this computer cannot keep up.'))

        section('Output')
        self.folder = QLineEdit()
        browse = QPushButton('Choose…')
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        f.addRow('Folder', row)
        self.name = QLineEdit()
        self.name.setPlaceholderText('optional, e.g. an animal ID')
        f.addRow('Session name', self.name)
        self.preview = hint('')
        f.addRow('', self.preview)
        self.folder.textChanged.connect(self._update_preview)
        self.name.textChanged.connect(self._update_preview)

        section('Save')
        grid = QGridLayout()
        grid.setHorizontalSpacing(px(12))
        grid.setVerticalSpacing(px(7))
        self.checks = {}
        for i, (key, label, detail) in enumerate(SAVE_ITEMS):
            cb = QCheckBox(label)
            self.checks[key] = cb
            grid.addWidget(cb, i, 0)
            detail = hint(detail)
            detail.setWordWrap(False)
            grid.addWidget(detail, i, 1)
        self.footage_box = QComboBox()
        for name, v in FOOTAGE_SIZES:
            self.footage_box.addItem(name, v)
        grid.addWidget(self.footage_box, len(SAVE_ITEMS), 1)
        self.checks['footage'].toggled.connect(self.footage_box.setEnabled)
        grid.setColumnStretch(1, 1)
        f.addRow(grid)
        lay.addSpacing(px(4))
        lay.addWidget(hint('Every session also writes session.json (camera, settings and algorithm parameters). '
                           'Results are written when a session stops or a recording ends.'))

        lay.addWidget(hint('Save as default: these settings are loaded every time the program starts.'))

        lay.addSpacing(px(10))
        buttons = QDialogButtonBox()
        restore = buttons.addButton('Restore built-in defaults', QDialogButtonBox.ResetRole)
        buttons.addButton(QDialogButtonBox.Cancel)
        use = buttons.addButton('Use for this session', QDialogButtonBox.AcceptRole)
        default = buttons.addButton('Save as default', QDialogButtonBox.AcceptRole)
        default.setDefault(True)
        restore.clicked.connect(lambda: self._set_fields(builtin_settings()))
        use.clicked.connect(self.accept)
        default.clicked.connect(lambda: (setattr(self, 'save_default', True), self.accept()))
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._set_fields(cfg)
        QTimer.singleShot(0, self._fill_cameras)      # after the dialog is shown: probing can take a moment

    def _set_fields(self, cfg):
        self.cfg_camera = cfg['camera']
        if self.cam_box.findData(cfg['camera']) >= 0:
            self.cam_box.setCurrentIndex(self.cam_box.findData(cfg['camera']))
        self.res_box.setCurrentIndex(next((i for i, (_, v) in enumerate(RESOLUTIONS)
                                           if v == tuple(cfg['camera_size'])), 0))
        self.fps_box.setCurrentIndex(0 if cfg['max_fps'] >= 30 else 1)
        self.folder.setText(str(cfg['out_root']))
        self.name.setText(cfg['session_name'])
        for key, cb in self.checks.items():
            cb.setChecked(getattr(cfg['opts'], key))
        self.footage_box.setCurrentIndex(0 if cfg['opts'].footage_scale < 1 else 1)
        self.footage_box.setEnabled(self.checks['footage'].isChecked())
        self._update_preview()

    def _fill_cameras(self):
        self.cam_box.clear()
        self.cam_box.addItem('Looking for cameras…')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            cams, err = list_cameras(), None
        except OSError as e:
            cams, err = [], str(e)
        finally:
            QApplication.restoreOverrideCursor()
        self.cam_box.clear()
        for i, label in cams:
            self.cam_box.addItem(label, i)
        if not cams:
            self.cam_box.addItem('No camera found', None)
        sel = self.cam_box.findData(self.cfg_camera)
        self.cam_box.setCurrentIndex(max(0, sel))
        self.cam_box.setEnabled(len(cams) > 1)
        self.cam_note.setText(err or ('' if cams else 'Connect a camera, then press the refresh button.'))
        self.cam_note.setVisible(bool(err or not cams))

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Results folder', str(Path(self.folder.text()).resolve()))
        if d:
            self.folder.setText(d)

    def _update_preview(self):
        name = live_session_name(self.cam_box.currentData() or 0, self.name.text())
        self.preview.setText(f'Next live session: {Path(self.folder.text() or "results") / name}{os.sep}')

    def result_cfg(self):
        opts = SaveOptions(**{k: cb.isChecked() for k, cb in self.checks.items()},
                           footage_scale=self.footage_box.currentData())
        cam = self.cam_box.currentData()
        return {'camera': self.cfg_camera if cam is None else cam, 'camera_size': self.res_box.currentData(),
                'max_fps': self.fps_box.currentData(), 'out_root': self.folder.text().strip() or 'results',
                'session_name': self.name.text().strip(), 'opts': opts}


# ---------------------------------------------------------------- main window
STAT_TILES = [('peak', 'Breath-to-breath'), ('sqi', 'Signal quality'), ('last60', 'Last 60 s'),
              ('cover', 'Rate available'), ('region', 'Region'), ('speed', 'Processing')]


class MainWindow(QMainWindow):
    def __init__(self, speed, cfg, scale=1.0):
        super().__init__()
        self.s = scale
        self.cfg = cfg
        self.worker = None
        self.speed = speed
        self.live = False
        self.camera_index = None
        self.path = None
        self.last_out = None     # folder of the last saved session
        self.running = False     # a session (live or recording) is being processed now
        self.transport = None    # what the play button does for a recording: pause | play | replay
        self.note_hold = None    # (text, until): a message that status updates must not overwrite
        self.tile_cols = 0
        self.setWindowTitle('SurgeryVitals — respiratory monitor')
        self._build()
        self._reset_data()
        self._show_idle()
        QShortcut(QKeySequence('Space'), self, self.on_space)
        QShortcut(QKeySequence.Open, self, self.open_dialog)
        QShortcut(QKeySequence.Preferences, self, self.open_settings)
        QShortcut(QKeySequence('R'), self, lambda: self.btn_draw.click())
        QShortcut(QKeySequence('A'), self, self.redetect)
        QShortcut(QKeySequence('Escape'), self, lambda: self.btn_draw.setChecked(False))

    # ---------- layout
    def px(self, v):
        return max(1, round(v * self.s))

    def _hero_color(self, color):
        self.rate_label.setStyleSheet(f'font-size: {self.px(84)}px; font-weight: 600; color: {color};')

    def _icon_button(self, name, tip, checkable=False):
        b = QPushButton()
        b.setObjectName('icon')
        b.setIcon(make_icon(name))
        b.setIconSize(QSize(self.px(18), self.px(18)))
        b.setFixedSize(self.px(34), self.px(34))
        b.setToolTip(tip)
        b.setCheckable(checkable)
        return b

    def _build(self):
        px = self.px
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- top bar: identity and session time on the left, tools on the right
        bar = QFrame()
        bar.setObjectName('topbar')
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(px(16), px(7), px(12), px(7))
        bl.setSpacing(px(4))
        title = QLabel('SurgeryVitals')
        title.setObjectName('appTitle')
        self.file_label = QLabel('no source')
        self.file_label.setStyleSheet(f'color: {MUTED};')
        self.time_label = QLabel('')
        self.time_label.setObjectName('timeChip')
        bl.addWidget(title)
        bl.addSpacing(px(12))
        bl.addWidget(self.file_label)
        bl.addSpacing(px(10))
        bl.addWidget(self.time_label)
        bl.addStretch(1)
        open_key = shortcut_text(QKeySequence.Open)
        self.btn_open = self._icon_button('open', f'Open a recording  ({open_key})')
        self.btn_rec = self._icon_button('record', '')
        self.btn_play = self._icon_button('pause', '')
        self.speed_box = QComboBox()
        self.speed_box.setToolTip('Playback speed')
        self.speed_box.setFixedHeight(px(34))
        self.speed_items = [('0.5×', 0.5), ('1×', 1.0), ('2×', 2.0), ('4×', 4.0), ('Max', 0.0)]
        for name, _ in self.speed_items:
            self.speed_box.addItem(name)
        idx = min(range(len(self.speed_items)), key=lambda i: abs(self.speed_items[i][1] - self.speed))
        self.speed_box.setCurrentIndex(idx)
        self.btn_auto = self._icon_button('auto', 'Auto-detect the breathing region  (A)')
        self.btn_draw = self._icon_button('draw', 'Draw the region by hand  (R)', checkable=True)
        self.btn_heat = self._icon_button('map', 'Breathing map: show where rhythmic motion was found', checkable=True)
        self.btn_results = self._icon_button('results', 'Open the results folder of the last session')
        prefs = shortcut_text(QKeySequence.Preferences)
        self.btn_settings = self._icon_button('settings', 'Session settings: camera, output folder, what to save'
                                              + (f'  ({prefs})' if prefs else ''))
        self.transport_icons = {k: make_icon(k) for k in ('pause', 'play', 'stop', 'record', 'replay')}
        # groups: sources | playback of a recording (hidden otherwise) | region tools | results, settings
        groups = ((self.btn_open, self.btn_rec), (self.btn_play, self.speed_box),
                  (self.btn_auto, self.btn_draw, self.btn_heat), (self.btn_results, self.btn_settings))
        self.groups = []
        for gi, group in enumerate(groups):
            box = QWidget()
            box.setObjectName('group')
            gl = QHBoxLayout(box)
            gl.setContentsMargins(0, 0, 0, 0)
            gl.setSpacing(px(4))
            if gi:
                gl.addSpacing(px(6))
                gl.addWidget(separator(self.s))
                gl.addSpacing(px(6))
            for w in group:
                gl.addWidget(w)
            bl.addWidget(box)
            self.groups.append(box)
        self.playback_group = self.groups[1]
        outer.addWidget(bar)
        self.progress = QProgressBar()
        self.progress.setFixedHeight(3)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1000)
        outer.addWidget(self.progress)

        body = QWidget()
        bv = QVBoxLayout(body)
        bv.setContentsMargins(px(12), px(10), px(12), px(12))
        bv.setSpacing(px(10))

        # --- top row: video (as wide as its picture) | rate and stats (the rest)
        top = QHBoxLayout()
        top.setSpacing(px(10))
        self.vcard, vlay, vhead = card('Camera', 'shaded box = region used for the estimate', self.s)
        self.video = VideoView()
        self.video.setMinimumSize(px(320), px(180))
        self.video.s = self.s
        vlay.addWidget(self.video, 1)
        top.addWidget(self.vcard, 100)

        side = QVBoxLayout()
        side.setSpacing(px(10))
        hero = QFrame()
        hero.setObjectName('card')
        hl = QVBoxLayout(hero)
        hl.setContentsMargins(px(18), px(12), px(18), px(14))
        hl.setSpacing(px(6))
        k = QLabel('Respiratory rate')
        k.setObjectName('cardTitle')
        hl.addWidget(k)
        hl.addStretch(1)
        row = QHBoxLayout()
        self.rate_label = QLabel('—')
        self._hero_color(MUTED)
        unit = QLabel('breaths\n/ min')
        unit.setObjectName('unit')
        row.addWidget(self.rate_label)
        row.addWidget(unit, 0, Qt.AlignBottom)
        row.addStretch(1)
        hl.addLayout(row)
        self.status_chip = QLabel()
        self.status_chip.setTextFormat(Qt.RichText)
        self.status_chip.setStyleSheet(f'background: {RAISED}; border-radius: {px(8)}px; padding: {px(6)}px {px(10)}px;')
        hl.addWidget(self.status_chip)
        self.status_note = QLabel('')
        self.status_note.setObjectName('cardSub')
        self.status_note.setWordWrap(True)
        self.status_note.setTextFormat(Qt.RichText)
        self.status_note.linkActivated.connect(lambda _: self.open_results())
        hl.addWidget(self.status_note)
        hl.addStretch(2)
        side.addWidget(hero, 1)

        self.stats_card = QFrame()
        self.stats_card.setObjectName('card')
        self.tiles = QGridLayout(self.stats_card)
        self.tiles.setContentsMargins(px(10), px(10), px(10), px(10))
        self.tiles.setSpacing(px(8))
        self.stat, self.tile_widgets = {}, []
        for key, name in STAT_TILES:
            tile = QFrame()
            tile.setObjectName('tile')
            tl = QVBoxLayout(tile)
            tl.setContentsMargins(px(10), px(7), px(10), px(8))
            tl.setSpacing(px(2))
            a = QLabel(name)
            a.setObjectName('statKey')
            b = QLabel('—')
            b.setObjectName('statVal')
            tl.addWidget(a)
            tl.addWidget(b)
            self.stat[key] = b
            self.tile_widgets.append(tile)
        self._layout_tiles(2)
        side.addWidget(self.stats_card)
        sidew = QWidget()
        sidew.setLayout(side)
        side.setContentsMargins(0, 0, 0, 0)
        sidew.setMinimumWidth(px(340))
        top.addWidget(sidew, 1)
        bv.addLayout(top, 6)

        # --- live waveform
        scard, slay, shead = card('Breathing motion', f'last {SIGNAL_WINDOW_S:.0f} s · each upward peak is one breath '
                                  '· vertical lines every second', self.s)
        self.count_label = QLabel('')
        self.count_label.setStyleSheet(f'color: {INK2};')
        shead.addWidget(self.count_label)
        shead.addSpacing(px(12))
        shead.addWidget(legend_label([(BLUE, 'breathing motion', 'line'), (ORANGE, 'detected breath', 'dot'),
                                      (MUTED, 'artifact (ignored)', 'line')]))
        self.sig_plot = styled_plot(self.s)
        self.sig_plot.setMinimumHeight(px(110))
        spi = self.sig_plot.getPlotItem()
        spi.hideAxis('left')
        spi.getAxis('bottom').setTickSpacing(5, 1)
        spi.showGrid(x=True, y=False, alpha=0.35)
        self.sig_ok = spi.plot(pen=pg.mkPen(BLUE, width=2), connect='finite')
        self.sig_art = spi.plot(pen=pg.mkPen(MUTED, width=2), connect='finite')
        self.sig_breaths = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(ORANGE), pen=pg.mkPen(SURFACE, width=2))
        spi.addItem(self.sig_breaths)
        self.sig_hint = pg.TextItem('The waveform appears here once the breathing region is found', color=MUTED,
                                    anchor=(0.5, 0.5))
        spi.addItem(self.sig_hint)
        slay.addWidget(self.sig_plot, 1)
        bv.addWidget(scard, 3)

        # --- trend
        tcard, tlay, thead = card('Respiratory rate since start', 'breaths / min · 15 s window, updated every '
                                  'second · gaps = no trustworthy estimate', self.s)
        thead.addWidget(legend_label([(BLUE, 'rate', 'line'), (ORANGE, 'breath-to-breath', 'dot')]))
        self.trend_plot = styled_plot(self.s)
        self.trend_plot.setMinimumHeight(px(90))
        tpi = self.trend_plot.getPlotItem()
        tpi.showGrid(x=False, y=True, alpha=0.35)
        tpi.getAxis('left').setWidth(36)
        self.b2b = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(217, 89, 38, 120), pen=None)
        tpi.addItem(self.b2b)
        self.trend = tpi.plot(pen=pg.mkPen(BLUE, width=2.5), connect='finite')
        tlay.addWidget(self.trend_plot, 1)
        self.strip = styled_plot(self.s, height=px(20))
        st = self.strip.getPlotItem()
        st.hideAxis('left')
        st.hideAxis('bottom')
        st.getViewBox().setYRange(0, 1, padding=0)
        self.strip_img = pg.ImageItem()
        st.addItem(self.strip_img)
        self.strip.setXLink(self.trend_plot)
        strip_row = QHBoxLayout()
        sp = QLabel('confidence')
        sp.setObjectName('cardSub')
        strip_row.addWidget(sp)
        strip_row.addWidget(self.strip, 1)
        strip_row.addWidget(legend_label([(GOOD, 'good', 'dot'), (WARNING, 'fair', 'dot'),
                                          (SERIOUS, 'poor / artifact', 'dot'), (AXIS, 'searching', 'dot')]))
        tlay.addLayout(strip_row)
        bv.addWidget(tcard, 3)
        outer.addWidget(body, 1)
        self.setCentralWidget(root)

        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_rec.clicked.connect(self.on_record)
        self.btn_play.clicked.connect(self.on_transport)
        self.speed_box.currentIndexChanged.connect(self.set_speed)
        self.btn_auto.clicked.connect(self.redetect)
        self.btn_draw.toggled.connect(self.video.start_drawing)
        self.btn_heat.toggled.connect(self.toggle_heat)
        self.btn_results.clicked.connect(self.open_results)
        self.btn_settings.clicked.connect(self.open_settings)
        self.video.roiDrawn.connect(self.on_roi)
        self._set_controls(False)
        self.btn_results.setEnabled(False)
        self.playback_group.setVisible(False)

    def _layout_tiles(self, cols):
        if cols == self.tile_cols:
            return
        self.tile_cols = cols
        for i, tile in enumerate(self.tile_widgets):
            self.tiles.addWidget(tile, i // cols, i % cols)

    def _fit_video(self):
        """Make the camera card exactly as wide as its picture; the rate panel takes the rest."""
        m = self.vcard.layout().contentsMargins()
        aspect = self.video.aspect()
        self.vcard.setMaximumWidth(int(self.video.height() * aspect) + m.left() + m.right() + 2)
        self._layout_tiles(3 if self.stats_card.width() > self.px(600) else 2)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        QTimer.singleShot(0, self._fit_video)

    def _set_controls(self, on):
        """`on`: a session is running. During a live recording its settings and the source are fixed."""
        self.running = on
        for w in (self.speed_box, self.btn_auto, self.btn_draw, self.btn_heat):
            w.setEnabled(on)
        recording = on and self.live
        self.btn_settings.setEnabled(not recording)
        self.btn_open.setEnabled(not recording)
        self.btn_settings.setToolTip('Stop the recording to change settings' if recording else
                                     'Session settings: camera, output folder, what to save'
                                     + (f'  ({shortcut_text(QKeySequence.Preferences)})'
                                        if shortcut_text(QKeySequence.Preferences) else ''))
        self._set_record('stop' if recording else 'record')

    def _set_record(self, action):
        self.btn_rec.setIcon(self.transport_icons[action])
        self.btn_rec.setEnabled(True)
        self.btn_rec.setToolTip('Stop and save the recording' + ('  (Space)' if self.live else '') if action == 'stop'
                                else f'Start recording from {camera_label(self.cfg["camera"])}')

    def _show_idle(self):
        """Nothing running yet: say which camera Record will use."""
        cam = camera_label(self.cfg['camera'])
        self.file_label.setText(f'{cam} · ready')
        self.video.placeholder = (f'Press the red record button to start recording from {cam},\n'
                                  f'or open a recording ({shortcut_text(QKeySequence.Open)})')
        self.video.update()
        self._set_record('record')
        self.status_chip.setText(f'<span style="color:{MUTED}; font-size:{self.px(15)}px">○</span>&nbsp;&nbsp;'
                                 f'<span style="color:{INK}; font-weight:600">Not recording</span>')
        self.status_note.setText('')

    # ---------- data
    def _reset_data(self):
        self.t, self.y, self.art = [], [], []
        self.breaths = []
        self.rates = []
        self.last_valid = None   # (t, bpm)
        self.ylim = None
        self.scale = None
        self.paused = False
        self._set_transport('pause')
        self.rate_label.setText('—')
        self._hero_color(MUTED)
        self.video.img = None
        self.video.heat = None
        for item in (self.sig_ok, self.sig_art, self.sig_breaths, self.trend, self.b2b):
            item.setData([], [])
        self.strip_img.clear()
        self.count_label.setText('')
        self._set_time('')
        self.progress.setValue(0)
        self.progress.setVisible(bool(self.path))
        for v in self.stat.values():
            v.setText('—')
        self.sig_plot.setXRange(0, SIGNAL_WINDOW_S, padding=0)     # empty axes already show their real scale
        self.sig_plot.setYRange(0, 1, padding=0)
        self.sig_hint.setPos(SIGNAL_WINDOW_S / 2, 0.5)
        self.sig_hint.setVisible(True)
        self.trend_plot.setXRange(0, 60, padding=0.01)
        self.trend_plot.setYRange(0, 100, padding=0)
        self._set_status(SEARCHING, None)

    def _set_time(self, html):
        self.time_label.setText(html)
        self.time_label.setVisible(bool(html))

    def start(self, path=None, camera=None):
        """Start a recording (`path`) or a live camera (`camera` = index)."""
        if self.worker:
            self.worker.batch.disconnect()   # a stopping live session still saves; keep only its 'saved' note
            self.worker.stop()
        self.live, self.camera_index, self.path = camera is not None, camera, path
        self.playback_group.setVisible(not self.live)
        self.note_hold = None
        self._reset_data()
        cfg = self.cfg
        if self.live:
            self.file_label.setText(f'Camera {camera}')
            self.video.placeholder = 'Opening camera…  (allow camera access if the system asks)'
            cam = (camera, cfg['camera_size'], cfg['max_fps'])
        else:
            self.file_label.setText(Path(path).name)
            self.video.placeholder = 'Loading…'
            cam = None
        self.video.update()
        self.worker = Worker(path, self.speed_items[self.speed_box.currentIndex()][1], cfg['out_root'], cfg['opts'],
                             cam, cfg['session_name'])
        self.worker.batch.connect(self.on_batch)
        self.worker.failed.connect(self.on_failed)
        self.worker.saved.connect(self.on_saved)     # methods, not lambdas: run on the GUI thread
        self.worker.opened.connect(self.file_label.setText)
        self.worker.info.connect(self.on_info)
        self.worker.start()
        self._set_controls(True)
        self.btn_play.setEnabled(True)

    def on_record(self):
        if self.running and self.live:
            self.worker.send('stop')          # live sessions are stopped (and saved), not paused
            self.btn_rec.setEnabled(False)
        else:
            self.start(camera=self.cfg['camera'])

    def on_space(self):
        if self.live and self.running:
            self.on_record()
        elif self.path:
            self.on_transport()

    def open_settings(self):
        if self.running and self.live:
            return                           # settings are fixed while recording
        dlg = SettingsDialog(self, self.cfg, self.s)
        if not dlg.exec():
            return
        self.cfg = dlg.result_cfg()
        if dlg.save_default:
            store_settings(self.cfg)
        if not self.running and not self.path:
            self._show_idle()
        self._set_record('stop' if self.running and self.live else 'record')

    def open_results(self):
        if self.last_out and Path(self.last_out).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_out).resolve())))

    def on_failed(self, msg):
        self._hold_note(f'Error: {msg}', float('inf'))
        self.video.placeholder = 'Could not start. See the note on the right.'
        self.video.img = None
        self.video.update()
        self._set_controls(False)
        self._set_time('')
        if not self.live:
            self._set_transport('replay')

    def on_saved(self, out_dir):
        self.last_out = out_dir
        self.btn_results.setEnabled(True)
        try:
            shown = Path(out_dir).resolve().relative_to(Path.cwd())
        except ValueError:
            shown = Path(out_dir)
        self._hold_note(f'Saved to <a href="open" style="color:{BLUE}; text-decoration:none">{shown}{os.sep}</a>',
                        float('inf'))

    def on_info(self, msg):
        if msg.startswith('Warning'):
            self._hold_note(msg, float('inf'))

    def _hold_note(self, text, seconds):
        self.note_hold = (text, time.monotonic() + seconds)
        self.status_note.setText(text)

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open recording', str(Path('videos').resolve()),
                                              'Video (*.mp4 *.avi *.mov *.mkv)')
        if path:
            self.start(path)

    def _set_transport(self, action):
        """Play button of a recording: pause | play | replay."""
        self.transport = action
        self.btn_play.setIcon(self.transport_icons[action])
        self.btn_play.setToolTip({'pause': 'Pause  (Space)', 'play': 'Play  (Space)',
                                  'replay': 'Replay the recording from the start  (Space)'}[action])

    def on_transport(self):
        if not self.worker or not self.btn_play.isEnabled():
            return
        if self.transport == 'replay':
            self.start(self.path)
        else:
            self.paused = not self.paused
            self.worker.send('pause', self.paused)
            self._set_transport('play' if self.paused else 'pause')

    def set_speed(self, i):
        if self.worker:
            self.worker.send('speed', self.speed_items[i][1])

    def redetect(self):
        if self.worker:
            self.worker.send('redetect')
            self.btn_draw.setChecked(False)

    def toggle_heat(self, on):
        self.video.show_heat = on
        self.video.update()

    def on_roi(self, box):
        self.btn_draw.setChecked(False)
        if self.worker and self.scale:
            self.worker.send('roi', tuple(v / self.scale for v in box))

    # ---------- refresh
    def on_batch(self, b):
        self.t.extend(b.t)
        self.y.extend(b.y)
        self.art.extend(b.art)
        self.breaths.extend(b.breaths)
        self.rates.extend(b.rates)
        if b.frame is not None:
            self.scale = b.scale
            first = self.video.img is None
            self.video.set_frame(b.frame, b.quad, b.state, b.roi_source, b.now)
            if first:
                self._fit_video()
        if b.heat is not None:
            self.video.set_heat(b.heat)
        if b.live:
            self._set_time(f'<span style="color:{CRITICAL}">●</span>&nbsp; LIVE &nbsp;{fmt_time(b.now)}')
        else:
            self._set_time(f'{fmt_time(b.now)} / {fmt_time(b.duration)}')
        if b.duration:
            self.progress.setValue(int(1000 * b.now / b.duration))
        self._update_signal(b.now)
        if b.rates:
            self._update_trend(b.now)
        if b.finished:
            self._finish(b)
        else:
            self._update_hero(b)

    def _finish(self, b):
        self._set_controls(False)
        if not b.live:
            self._set_transport('replay')
        if b.live:
            self._set_time('')
        self.rate_label.setText('—')
        self._hero_color(MUTED)
        self.video.state, self.video.quad, self.video.ended = SEARCHING, None, True
        self.video.update()
        text = 'Session stopped' if b.live else 'Recording finished'
        self.status_chip.setText(f'<span style="color:{MUTED}; font-size:{self.px(15)}px">■</span>&nbsp;&nbsp;'
                                 f'<span style="color:{INK}; font-weight:600">{text}</span>')
        if b.error:
            self._hold_note(f'Error: {b.error}', float('inf'))
        elif not self.note_hold:         # 'saved' arrives before this batch, so nothing was written
            self.status_note.setText('Nothing was recorded, so nothing was saved.')

    def _update_signal(self, now):
        n = int(SIGNAL_WINDOW_S * 32)
        t = np.asarray(self.t[-n:])
        if len(t) < 2:
            return
        y = np.asarray(self.y[-n:], float)
        art = np.asarray(self.art[-n:], bool)
        self.sig_ok.setData(t, np.where(art, np.nan, y))
        # artifact samples plus one neighbour each side so grey segments join the blue line
        art_wide = art | np.r_[art[1:], False] | np.r_[False, art[:-1]]
        self.sig_art.setData(t, np.where(art_wide, y, np.nan))
        t0 = max(0.0, now - SIGNAL_WINDOW_S)          # window starts at 00:00, scrolls after 20 s
        bt = np.array([b[0] for b in self.breaths if b[0] > t0])
        by = np.array([b[1] for b in self.breaths if b[0] > t0])
        self.sig_breaths.setData(bt, by)
        ok = np.isfinite(y) & ~art
        if ok.sum() > 10:
            lo, hi = np.percentile(y[ok], [1, 99.5])
            pad = 0.2 * (hi - lo) + 1e-6
            target = np.array([lo - pad, hi + pad])
            self.ylim = target if self.ylim is None else 0.9 * self.ylim + 0.1 * target
            self.sig_plot.setYRange(*self.ylim, padding=0)
        self.sig_plot.setXRange(t0, t0 + SIGNAL_WINDOW_S, padding=0)
        has_wave = ok.any()
        self.sig_hint.setVisible(not has_wave)
        if not has_wave:
            self.sig_plot.setYRange(0, 1, padding=0)
            self.sig_hint.setPos(t0 + SIGNAL_WINDOW_S / 2, 0.5)
        if len(bt) >= 3:
            # rate from the spacing of the marked breaths (what you get counting peaks by hand)
            per_min = 60 * (len(bt) - 1) / (bt[-1] - bt[0])
            self.count_label.setText(f'<b>{len(bt)}</b> breaths marked  '
                                     f'<span style="color:{MUTED}">(≈ {per_min:.0f} / min from their spacing)</span>')
        else:
            self.count_label.setText('')

    def _update_trend(self, now):
        rt = np.array([r.t for r in self.rates])
        bpm = np.array([r.bpm for r in self.rates])
        self.trend.setData(rt, bpm)
        bt = np.array([b[0] for b in self.breaths])
        if len(bt) > 2:
            iv = np.diff(bt)
            keep = iv < 3
            self.b2b.setData(bt[1:][keep], 60 / iv[keep])
        fin = bpm[np.isfinite(bpm)]
        top = max(100.0, float(np.percentile(fin, 99)) * 1.3) if len(fin) else 100.0
        xmax = max(60.0, now)
        self.trend_plot.setYRange(0, top, padding=0)
        self.trend_plot.setXRange(0, xmax, padding=0.01)
        colors = {'good': GOOD, 'fair': WARNING, 'poor': SERIOUS}
        rgba = np.array([QColor(colors.get(r.confidence, SERIOUS if r.state in (LOCKED, ARTIFACT) else AXIS)).getRgb()
                         for r in self.rates], np.uint8)[None]
        self.strip_img.setImage(rgba, autoLevels=False)
        self.strip_img.setRect(QRectF(rt[0], 0, max(1.0, rt[-1] - rt[0] + 1), 1))

    def _set_status(self, state, conf):
        if state == SEARCHING:
            icon, text, color, note = '◌', 'Searching for breathing', MUTED, \
                'Looking for rhythmic motion in the whole image. Needs ~15 s of visible breathing.'
        elif state == LOST:
            icon, text, color, note = '✕', 'Region lost — re-acquiring', CRITICAL, \
                'The scene changed (camera moved or view blocked).'
        elif state == ARTIFACT:
            icon, text, color, note = '▲', 'Motion artifact', WARNING, \
                'Large movement in the region; these samples are ignored.'
        elif conf == 'good':
            icon, text, color, note = '●', 'Good signal', GOOD, ''
        elif conf == 'fair':
            icon, text, color, note = '●', 'Fair signal', WARNING, ''
        else:
            icon, text, color, note = '▲', 'Weak or irregular signal', SERIOUS, \
                'Rate withheld until the breathing rhythm is clear again.'
        self.status_chip.setText(f'<span style="color:{color}; font-size:{self.px(15)}px">{icon}</span>&nbsp;&nbsp;'
                                 f'<span style="color:{INK}; font-weight:600">{text}</span>')
        if self.note_hold and time.monotonic() < self.note_hold[1]:
            note = self.note_hold[0]
        self.status_note.setText(note)

    def _update_hero(self, b):
        last = self.rates[-1] if self.rates else None
        conf = last.confidence if last else None
        self._set_status(b.state, conf)
        if last is not None and np.isfinite(last.bpm):
            self.last_valid = (last.t, last.bpm)
            self.rate_label.setText(f'{last.bpm:.0f}')
            self._hero_color(INK)
        elif self.last_valid and b.now - self.last_valid[0] < 15:
            self.rate_label.setText(f'{self.last_valid[1]:.0f}')       # hold briefly, dimmed
            self._hero_color(MUTED)
        else:
            self.rate_label.setText('—')
            self._hero_color(MUTED)
        if last is not None:
            self.stat['peak'].setText(f'{last.peak_bpm:.0f} / min' if np.isfinite(last.peak_bpm) else '—')
            self.stat['sqi'].setText(f'{last.sqi * 100:.0f} %' if last.sqi else '—')
            recent = [r.bpm for r in self.rates if r.t > b.now - 60 and np.isfinite(r.bpm)]
            self.stat['last60'].setText(f'{np.mean(recent):.0f}  ({min(recent):.0f}–{max(recent):.0f})' if recent else '—')
            cover = np.mean([np.isfinite(r.bpm) for r in self.rates]) * 100
            self.stat['cover'].setText(f'{cover:.0f} % of time')
        region = {LOCKED: 'tracking', ARTIFACT: 'tracking', LOST: 'lost', SEARCHING: 'searching'}[b.state]
        if b.roi_source and b.state != SEARCHING:
            region = f'{b.roi_source} · {region}'
        self.stat['region'].setText(region)
        if b.live and b.load is not None:
            gaps = f' · {b.gaps} gap{"s" if b.gaps > 1 else ""}' if b.gaps else ''
            self.stat['speed'].setText(f'{b.fps:g} fps · {b.load * 100:.0f} % busy{gaps}')
        elif b.xrt and not b.live:
            self.stat['speed'].setText(f'{b.xrt:.1f}× real time')

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop()
        super().closeEvent(e)


def fit_window(app, speed, cfg, avail):
    """Build the window at a scale that fits `avail` (the screen's free area, logical px).
    If the layout's minimum size still exceeds it, rebuild one step smaller."""
    s = ui_scale(avail)
    while True:
        app.setStyleSheet(make_qss(s))
        win = MainWindow(speed, cfg, s)
        m = win.minimumSizeHint()
        # leave room for the title bar (~30 px) that the window manager adds
        if (m.width() <= avail.width() and m.height() <= avail.height() - 30) or s <= 0.62:
            return win
        win.deleteLater()
        s = max(0.62, s * 0.92)


def run_gui(path=None, speed=1.0, screenshot=None, after=20.0, screen_size=None, camera=None, overrides=None):
    """`camera` = index starts a live camera instead of a recording. `overrides` replace saved settings
    (camera_size, max_fps, out_root, session_name) for this run."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setPalette(dark_palette())
    screen = QGuiApplication.screenAt(QCursor.pos()) or app.primaryScreen()
    avail = screen.availableGeometry()
    if screen_size:                      # development aid: emulate another screen size
        avail = QRect(0, 0, *screen_size)
    cfg = load_settings(overrides)
    if camera is not None:
        cfg['camera'] = camera
    win = fit_window(app, speed, cfg, avail)
    win.setGeometry(avail.adjusted(0, 30, 0, 0) if screen_size else avail)
    if screenshot:
        win.show()
    else:
        win.showMaximized()              # fills the screen it opened on, whatever its size
    if path:                             # started only now: fit_window may build several windows
        win.start(path)                  # a live recording is never started automatically: press Record
    if screenshot:      # development aid: grab the window after `after` seconds and quit
        def grab():
            win.grab().save(screenshot)
            win.close()
            app.quit()
        QTimer.singleShot(int(after * 1000), grab)
    return app.exec()
