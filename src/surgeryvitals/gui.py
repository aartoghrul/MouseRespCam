"""Monitoring GUI. Replays a recording in simulated real time (or faster) through the same
frame-by-frame pipeline a live camera would use."""
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (QColor, QCursor, QFont, QGuiApplication, QImage, QKeySequence, QPainter, QPen,
                           QPolygonF, QShortcut)
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QMainWindow, QProgressBar, QPushButton, QSizePolicy, QVBoxLayout,
                               QWidget)

from .monitor import ARTIFACT, LOCKED, LOST, SEARCHING, RespirationMonitor
from .recorder import ResultRecorder
from .source import VideoFileSource

# ---------------------------------------------------------------- theme (dark, validated palette)
PAGE, SURFACE, RAISED = '#0d0d0d', '#1a1a19', '#232321'
INK, INK2, MUTED = '#f0efec', '#c3c2b7', '#898781'
GRID, AXIS, BORDER = '#2c2c2a', '#383835', 'rgba(255,255,255,0.10)'
BLUE, ORANGE = '#3987e5', '#d95926'
GOOD, WARNING, SERIOUS, CRITICAL = '#0ca30c', '#fab219', '#ec835a', '#d03b3b'

SIGNAL_WINDOW_S = 20.0
DISPLAY_W = 960

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
QLabel#statVal {{ color: {INK}; font-size: {fpx(14, 12)}px; }}
QLabel#unit {{ color: {INK2}; font-size: {fpx(15, 12)}px; }}
QPushButton, QComboBox {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: {px(8)}px;
    padding: {px(6)}px {px(12)}px; color: {INK}; }}
QPushButton:hover, QComboBox:hover {{ background: {GRID}; }}
QPushButton:checked {{ background: #16304f; border-color: {BLUE}; }}
QPushButton:disabled {{ color: {AXIS}; }}
QComboBox::drop-down {{ border: none; width: {px(18)}px; }}
QComboBox QAbstractItemView {{ background: {RAISED}; selection-background-color: #16304f; }}
QProgressBar {{ background: {RAISED}; border: none; border-radius: 2px; }}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 2px; }}
"""


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


class Worker(QThread):
    batch = Signal(object)
    failed = Signal(str)
    saved = Signal(str)

    def __init__(self, path, speed, out_root):
        super().__init__()
        self.path, self.speed, self.out_root = path, speed, out_root
        self.cmds = queue.Queue()
        self._stop = False

    def stop(self):
        self._stop = True
        self.wait(3000)

    def send(self, *cmd):
        self.cmds.put(cmd)

    def run(self):
        try:
            self._run()
        except Exception as e:      # surface errors in the UI instead of dying silently
            self.failed.emit(f'{type(e).__name__}: {e}')

    def _save(self, rec, src):
        out = rec.save(Path(self.out_root) / Path(self.path).stem, meta={'fps': src.fps})
        self.saved.emit(str(out))

    def _run(self):
        src = VideoFileSource(self.path)
        mon = RespirationMonitor(src.shape, src.fps)
        rec = ResultRecorder()
        scale = DISPLAY_W / src.shape[1]
        disp_size = (DISPLAY_W, int(round(src.shape[0] * scale)))
        cell = mon.loc.p.loc_block * mon.loc.sx * scale       # heat-map block size in display px
        acc, paused, last_t = Batch(), False, 0.0
        wall0, video0 = time.perf_counter(), 0.0               # pacing anchor
        rate_wall, rate_video, xrt = time.perf_counter(), 0.0, None
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
                elif cmd[0] == 'export':
                    self._save(rec, src)
            if paused:
                self.msleep(30)
                continue
            frame, t = src.read()
            if frame is None:
                acc.finished = True
                acc.now, acc.duration = last_t, src.duration
                self._save(rec, src)
                self.batch.emit(acc)
                return
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
            if self.speed > 0:
                lag = wall0 + (t - video0) / self.speed - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
            now = time.perf_counter()
            if now - rate_wall > 1.0:
                xrt = (t - rate_video) / (now - rate_wall)
                rate_wall, rate_video = now, t
            if now - last_emit >= 1 / 30:
                last_emit = now
                small = cv2.resize(frame, disp_size, interpolation=cv2.INTER_AREA)
                acc.frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                acc.quad = None if res.quad is None else res.quad * scale
                acc.state, acc.roi_source = res.state, res.roi_source
                acc.now, acc.duration, acc.xrt, acc.scale = t, src.duration, xrt, scale
                self.batch.emit(acc)
                acc = Batch()


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
        self.placeholder = 'Open a recording to start  (⌘O)'
        self.s = 1.0

    def set_frame(self, rgb, quad, state, source, now):
        h, w, _ = rgb.shape
        self.img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.quad, self.state, self.source, self.now = quad, state, source, now
        self.update()

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
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
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
        elif self.state == SEARCHING:
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
    """Card with a two-line header: title row and subtitle row. Extra widgets (legends, counters)
    are added to the right end of either row, so they never sit on top of the title text."""
    fr = QFrame()
    fr.setObjectName('card')
    lay = QVBoxLayout(fr)
    lay.setContentsMargins(round(16 * s), round(10 * s), round(16 * s), round(10 * s))
    lay.setSpacing(round(6 * s))
    head, sub = QHBoxLayout(), QHBoxLayout()
    t = QLabel(title)
    t.setObjectName('cardTitle')
    head.addWidget(t)
    head.addStretch(1)
    lay.addLayout(head)
    if subtitle:
        st = QLabel(subtitle)
        st.setObjectName('cardSub')
        sub.addWidget(st)
        sub.addStretch(1)
        lay.addLayout(sub)
    return fr, lay, head, sub


def legend_label(items):
    """Static legend: coloured swatch + text, text in ink (never in the series colour)."""
    parts = []
    for color, text, kind in items:
        sw = '━━' if kind == 'line' else '●'
        parts.append(f'<span style="color:{color}">{sw}</span>&nbsp;<span style="color:{INK2}">{text}</span>')
    lab = QLabel('&nbsp;&nbsp;&nbsp;'.join(parts))
    lab.setTextFormat(Qt.RichText)
    return lab


# ---------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self, path, speed, out_root, scale=1.0):
        super().__init__()
        self.s = scale
        self.out_root = out_root
        self.worker = None
        self.speed = speed
        self.setWindowTitle('SurgeryVitals — respiratory monitor')
        self._build()
        self._reset_data()
        QShortcut(QKeySequence('Space'), self, self.toggle_pause)
        QShortcut(QKeySequence.Open, self, self.open_dialog)
        QShortcut(QKeySequence('R'), self, lambda: self.btn_draw.click())
        QShortcut(QKeySequence('A'), self, self.redetect)
        QShortcut(QKeySequence('Escape'), self, lambda: self.btn_draw.setChecked(False))
        if path:
            self.start(path)

    # ---------- layout
    def px(self, v):
        return max(1, round(v * self.s))

    def _hero_color(self, color):
        self.rate_label.setStyleSheet(f'font-size: {self.px(84)}px; font-weight: 600; color: {color};')

    def _build(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bar = QFrame()
        bar.setObjectName('topbar')
        bl = QHBoxLayout(bar)
        px = self.px
        bl.setContentsMargins(px(16), px(8), px(16), px(8))
        bl.setSpacing(px(8))
        title = QLabel('SurgeryVitals')
        title.setObjectName('appTitle')
        self.file_label = QLabel('no recording')
        self.file_label.setStyleSheet(f'color: {MUTED};')
        bl.addWidget(title)
        bl.addSpacing(8)
        bl.addWidget(self.file_label)
        bl.addStretch(1)
        self.btn_open = QPushButton('Open…')
        self.btn_play = QPushButton('Pause')
        self.speed_box = QComboBox()
        self.speed_items = [('0.5×', 0.5), ('1× real time', 1.0), ('2×', 2.0), ('4×', 4.0), ('Max', 0.0)]
        for name, _ in self.speed_items:
            self.speed_box.addItem(name)
        idx = min(range(len(self.speed_items)), key=lambda i: abs(self.speed_items[i][1] - self.speed))
        self.speed_box.setCurrentIndex(idx)
        self.btn_auto = QPushButton('Auto-detect')
        self.btn_draw = QPushButton('Draw region')
        self.btn_draw.setCheckable(True)
        self.btn_heat = QPushButton('Breathing map')
        self.btn_heat.setCheckable(True)
        self.btn_export = QPushButton('Export')
        for w in (self.btn_open, self.btn_play, self.speed_box, self.btn_auto, self.btn_draw, self.btn_heat,
                  self.btn_export):
            bl.addWidget(w)
        self.time_label = QLabel('00:00 / 00:00')
        self.time_label.setStyleSheet(f'color: {INK2};')
        self.time_label.setMinimumWidth(110)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bl.addSpacing(8)
        bl.addWidget(self.time_label)
        outer.addWidget(bar)
        self.progress = QProgressBar()
        self.progress.setFixedHeight(3)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1000)
        outer.addWidget(self.progress)

        body = QWidget()
        grid = QGridLayout(body)
        grid.setContentsMargins(px(14), px(12), px(14), px(12))
        grid.setSpacing(px(10))

        # video
        vcard, vlay, _, _ = card('Camera', 'shaded box = region used for the estimate', self.s)
        self.video = VideoView()
        self.video.setMinimumSize(px(320), px(180))
        self.video.s = self.s
        vlay.addWidget(self.video, 1)
        grid.addWidget(vcard, 0, 0)

        # hero / stats
        side = QFrame()
        side.setObjectName('card')
        sl = QVBoxLayout(side)
        sl.setContentsMargins(px(18), px(12), px(18), px(12))
        sl.setSpacing(px(4))
        k = QLabel('Respiratory rate')
        k.setObjectName('cardTitle')
        sl.addWidget(k)
        row = QHBoxLayout()
        self.rate_label = QLabel('—')
        self._hero_color(MUTED)
        unit = QLabel('breaths\n/ min')
        unit.setObjectName('unit')
        row.addWidget(self.rate_label)
        row.addWidget(unit, 0, Qt.AlignBottom)
        row.addStretch(1)
        sl.addLayout(row)
        self.status_chip = QLabel()
        self.status_chip.setTextFormat(Qt.RichText)
        self.status_chip.setStyleSheet(f'background: {RAISED}; border-radius: {px(8)}px; padding: {px(6)}px {px(10)}px;')
        sl.addWidget(self.status_chip)
        self.status_note = QLabel('')
        self.status_note.setObjectName('cardSub')
        self.status_note.setWordWrap(True)
        sl.addWidget(self.status_note)
        sl.addSpacing(px(10))
        stats = QGridLayout()
        stats.setVerticalSpacing(px(8))
        self.stat = {}
        for i, (key, name) in enumerate([('peak', 'Breath-to-breath'), ('sqi', 'Signal quality'),
                                         ('last60', 'Last 60 s'), ('cover', 'Rate available'),
                                         ('region', 'Region'), ('speed', 'Processing')]):
            a = QLabel(name)
            a.setObjectName('statKey')
            b = QLabel('—')
            b.setObjectName('statVal')
            stats.addWidget(a, i, 0)
            stats.addWidget(b, i, 1, Qt.AlignRight)
            self.stat[key] = b
        sl.addLayout(stats)
        sl.addStretch(1)
        # wide enough for its own text at any scale (fonts have floors, so it can't just scale down)
        side.setFixedWidth(max(px(330), side.sizeHint().width() + px(12)))
        grid.addWidget(side, 0, 1)

        # live waveform
        scard, slay, shead, ssub = card('Breathing motion', f'last {SIGNAL_WINDOW_S:.0f} s · each upward peak is '
                                        'one breath · vertical lines every second', self.s)
        self.count_label = QLabel('')
        self.count_label.setStyleSheet(f'color: {INK2};')
        shead.addWidget(legend_label([(BLUE, 'breathing motion', 'line'), (ORANGE, 'detected breath', 'dot'),
                                      (MUTED, 'artifact (ignored)', 'line')]))
        ssub.addWidget(self.count_label)
        self.sig_plot = styled_plot(self.s)
        self.sig_plot.setMinimumHeight(px(120))
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
        grid.addWidget(scard, 1, 0, 1, 2)

        # trend
        tcard, tlay, thead, _ = card('Respiratory rate since start', 'breaths / min · 15 s window, updated every '
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
        self.strip = styled_plot(self.s, height=px(24))
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
        grid.addWidget(tcard, 2, 0, 1, 2)

        grid.setRowStretch(0, 6)
        grid.setRowStretch(1, 3)
        grid.setRowStretch(2, 3)
        grid.setColumnStretch(0, 1)
        outer.addWidget(body, 1)
        self.setCentralWidget(root)

        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_play.clicked.connect(self.toggle_pause)
        self.speed_box.currentIndexChanged.connect(self.set_speed)
        self.btn_auto.clicked.connect(self.redetect)
        self.btn_draw.toggled.connect(self.video.start_drawing)
        self.btn_heat.toggled.connect(self.toggle_heat)
        self.btn_export.clicked.connect(lambda: self.worker and self.worker.send('export'))
        self.video.roiDrawn.connect(self.on_roi)
        self._set_controls(False)

    def _set_controls(self, on):
        for w in (self.btn_play, self.speed_box, self.btn_auto, self.btn_draw, self.btn_heat, self.btn_export):
            w.setEnabled(on)

    # ---------- data
    def _reset_data(self):
        self.t, self.y, self.art = [], [], []
        self.breaths = []
        self.rates = []
        self.last_valid = None   # (t, bpm)
        self.ylim = None
        self.scale = None
        self.paused = False
        self.btn_play.setText('Pause')
        self.rate_label.setText('—')
        self._set_status(SEARCHING, None)

    def start(self, path):
        if self.worker:
            self.worker.stop()
        self._reset_data()
        self.file_label.setText(Path(path).name)
        self.video.placeholder = 'Loading…'
        self.worker = Worker(path, self.speed_items[self.speed_box.currentIndex()][1], self.out_root)
        self.worker.batch.connect(self.on_batch)
        self.worker.failed.connect(lambda m: self.status_note.setText(f'Error: {m}'))
        self.worker.saved.connect(lambda d: self.status_note.setText(f'Results saved to {d}/'))
        self.worker.start()
        self._set_controls(True)

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open recording', str(Path('videos').resolve()),
                                              'Video (*.mp4 *.avi *.mov *.mkv)')
        if path:
            self.start(path)

    def toggle_pause(self):
        if not self.worker:
            return
        self.paused = not self.paused
        self.worker.send('pause', self.paused)
        self.btn_play.setText('Play' if self.paused else 'Pause')

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
            self.video.set_frame(b.frame, b.quad, b.state, b.roi_source, b.now)
        if b.heat is not None:
            self.video.set_heat(b.heat)
        self.time_label.setText(f'{fmt_time(b.now)} / {fmt_time(b.duration)}')
        if b.duration:
            self.progress.setValue(int(1000 * b.now / b.duration))
        self._update_signal(b.now)
        if b.rates:
            self._update_trend(b.now)
        self._update_hero(b)
        if b.finished:
            self.btn_play.setText('Finished')
            self._set_controls(False)
            self.btn_export.setEnabled(True)

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
        if not self.status_note.text().startswith(('Results saved', 'Error')):
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
        if b.xrt:
            self.stat['speed'].setText(f'{b.xrt:.1f}× real time')

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop()
        super().closeEvent(e)


def fit_window(app, path, speed, out_root, avail):
    """Build the window at a scale that fits `avail` (the screen's free area, logical px).
    If the layout's minimum size still exceeds it, rebuild one step smaller."""
    s = ui_scale(avail)
    while True:
        app.setStyleSheet(make_qss(s))
        win = MainWindow(path, speed, out_root, s)
        m = win.minimumSizeHint()
        # leave room for the title bar (~30 px) that the window manager adds
        if (m.width() <= avail.width() and m.height() <= avail.height() - 30) or s <= 0.62:
            return win
        win.deleteLater()
        s = max(0.62, s * 0.92)


def run_gui(path=None, speed=1.0, out_root='results', screenshot=None, after=20.0, screen_size=None):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle('Fusion')
    screen = QGuiApplication.screenAt(QCursor.pos()) or app.primaryScreen()
    avail = screen.availableGeometry()
    if screen_size:                      # development aid: emulate another screen size
        avail = QRect(0, 0, *screen_size)
    win = fit_window(app, path, speed, out_root, avail)
    win.setGeometry(avail.adjusted(0, 30, 0, 0) if screen_size else avail)
    if screenshot:
        win.show()
    else:
        win.showMaximized()              # fills the screen it opened on, whatever its size
    if screenshot:      # development aid: grab the window after `after` seconds and quit
        def grab():
            win.grab().save(screenshot)
            win.close()
            app.quit()
        QTimer.singleShot(int(after * 1000), grab)
    return app.exec()
