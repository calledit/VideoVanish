#!/usr/bin/env python3
# vanishvideo.py
#
# Feature set:
# - Dark, editor-style UI theme.
# - Master/original video (with audio) + optional infilled video + optional mask video.
# - All layers PLAY together when you hit Play. Followers (infilled/mask) are resynced
#   periodically during playback to reduce drift, and snap perfectly on pause/seek/stop.
# - Exact frame timestamps are read from the master (via QVideoSink) to drive UI & overlay.
# - Overlay for positive/negative clicks and rectangles, including live rectangle preview.
# - Keyframe list with thumbnails annotated with the points/rects for that time.
# - Single-click-to-seek slider that immediately updates UI/overlay and all layers.
# - “Poster-frame on load”: when a new master is loaded, we show frame 0 immediately.
#
# Dependencies: PySide6 (Qt6), a system multimedia stack (gstreamer/ffmpeg backends vary by OS).
# Tested on Linux; video backend capabilities may differ by platform/codec.

import sys, json, argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PySide6.QtCore import Qt, QSize, QPointF, QRectF, Signal, QTimer, QUrl
from PySide6.QtGui import QAction, QIcon, QPainter, QPen, QColor, QPixmap, QPalette
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QFileDialog, QPushButton, QDockWidget, QStyle,
    QToolBar, QMessageBox, QListWidget, QListWidgetItem, QButtonGroup,
    QToolButton, QGraphicsView, QGraphicsScene, QGraphicsObject, QFrame,
    QCheckBox, QRadioButton, QGroupBox
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem


# --------- Small helper to format ms -> "mm:ss" or "hh:mm:ss" ----------
def fmt_ms(ms: int) -> str:
    s = max(0, ms // 1000)
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --------- Annotation data we store per keyframe ----------
@dataclass
class Keyframe:
    t_ms: int
    pos_clicks: List[Tuple[float, float]] = field(default_factory=list)   # normalized [0..1]
    neg_clicks: List[Tuple[float, float]] = field(default_factory=list)
    rects: List[Tuple[float, float, float, float]] = field(default_factory=list)  # (x,y,w,h) normalized


# --------- The overlay drawn on top of video to show/edit points/rectangles ----------
class OverlayItem(QGraphicsObject):
    # Signals emitted when the user annotates:
    addPositive = Signal(float, float)
    addNegative = Signal(float, float)
    addRectangle = Signal(float, float, float, float)
    requestDelete = Signal(float, float)

    # Tools
    TOOL_POS = 1
    TOOL_NEG = 2
    TOOL_RECT = 3

    def __init__(self, rect: QRectF, parent=None):
        super().__init__(parent)
        self._rect = QRectF(rect)
        self._tool = self.TOOL_POS
        self._kf: Optional[Keyframe] = None
        self._drawing = False
        self._drag_start: Optional[QPointF] = None
        self._drag_cur: Optional[QPointF] = None

        # Mouse/keyboard handling
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setFlag(self.GraphicsItemFlag.ItemIsFocusable, True)
        self.setZValue(20.0)  # draw above video

        # Visual sizing
        self.dot_radius_px = 10.0
        self.rect_pen_width = 4.0
        self.live_rect_pen_width = 3.0

    def setTool(self, tool: int):
        self._tool = tool

    def setKeyframe(self, kf: Optional[Keyframe]):
        """Swap to show annotations for this keyframe, or None for none."""
        self._kf = kf
        self.update()

    def setRect(self, rect: QRectF):
        """Called when the view resizes; keeps overlay aligned to letterbox region."""
        if rect != self._rect:
            self.prepareGeometryChange()
            self._rect = QRectF(rect)
            self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(self._rect)

    def paint(self, p: QPainter, _opt, _widget=None):
        # Draw the live rectangle while dragging, then the saved annotations.
        p.setRenderHint(QPainter.Antialiasing, True)

        # Live, dashed rectangle preview while drawing
        if self._drawing and self._drag_start and self._drag_cur:
            r = self._make_rect(self._drag_start, self._drag_cur)
            p.setPen(QPen(QColor(255, 255, 255, 220), self.live_rect_pen_width, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)

        # Saved annotations on the current keyframe
        if self._kf:
            W = max(1.0, self._rect.width())
            H = max(1.0, self._rect.height())

            # Rectangles (cyan)
            p.setPen(QPen(QColor(0, 200, 255, 220), self.rect_pen_width))
            p.setBrush(Qt.NoBrush)
            for (nx, ny, nw, nh) in self._kf.rects:
                x = self._rect.left() + nx * W
                y = self._rect.top()  + ny * H
                p.drawRect(QRectF(x, y, nw * W, nh * H))

            # Dots: positive = green; negative = red
            r_px = self.dot_radius_px
            p.setPen(Qt.NoPen)

            p.setBrush(QColor(55, 200, 90, 235))  # positive
            for (nx, ny) in self._kf.pos_clicks:
                p.drawEllipse(QPointF(self._rect.left()+nx*W, self._rect.top()+ny*H), r_px, r_px)

            p.setBrush(QColor(230, 70, 70, 235))  # negative
            for (nx, ny) in self._kf.neg_clicks:
                p.drawEllipse(QPointF(self._rect.left()+nx*W, self._rect.top()+ny*H), r_px, r_px)

    # ---- Mouse handling: add/remove annotations ----
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            if e.modifiers() & Qt.ControlModifier:
                # Ctrl+click deletes the nearest point (pos or neg) or rect edge
                nx, ny = self._normalize_point(e.pos())
                self.requestDelete.emit(nx, ny)
                e.accept()
                return
            if self._tool == self.TOOL_RECT:
                # Begin rectangle drag
                self._drawing = True
                self._drag_start = e.pos()
                self._drag_cur = self._drag_start
                self.update()
            else:
                # Add a dot
                nx, ny = self._normalize_point(e.pos())
                if self._tool == self.TOOL_POS:
                    self.addPositive.emit(nx, ny)
                else:
                    self.addNegative.emit(nx, ny)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drawing and self._drag_start:
            self._drag_cur = e.pos()
            self.update()
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._drawing and e.button() == Qt.LeftButton and self._drag_start and self._drag_cur:
            # Complete rectangle and emit normalized rect
            r = self._make_rect(self._drag_start, self._drag_cur)
            nx, ny, nw, nh = self._normalize_rect(r)
            if nw > 0 and nh > 0:
                self.addRectangle.emit(nx, ny, nw, nh)
        self._drawing = False
        self._drag_start = None
        self._drag_cur = None
        self.update()
        super().mouseReleaseEvent(e)

    # ---- Normalization utilities (pixels <-> normalized [0..1]) ----
    def _normalize_point(self, pt: QPointF) -> Tuple[float, float]:
        W = max(1.0, self._rect.width())
        H = max(1.0, self._rect.height())
        nx = (pt.x() - self._rect.left()) / W
        ny = (pt.y() - self._rect.top()) / H
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def _normalize_rect(self, r: QRectF) -> Tuple[float, float, float, float]:
        W = max(1.0, self._rect.width())
        H = max(1.0, self._rect.height())
        nx = (r.left() - self._rect.left()) / W
        ny = (r.top() - self._rect.top()) / H
        nw = r.width() / W
        nh = r.height() / H
        return (
            max(0.0, min(1.0, nx)),
            max(0.0, min(1.0, ny)),
            max(0.0, min(1.0, nw)),
            max(0.0, min(1.0, nh)),
        )

    @staticmethod
    def _make_rect(a: QPointF, b: QPointF) -> QRectF:
        x1, y1 = a.x(), a.y()
        x2, y2 = b.x(), b.y()
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x1-x2), abs(y1-y2)
        return QRectF(x, y, w, h)


# --------- The video view with three layers + overlay ----------
class VideoView(QGraphicsView):
    """
    A QGraphicsView that contains:
      - Original base (QGraphicsVideoItem)
      - Infilled base (QGraphicsVideoItem) — toggled visible as needed
      - Mask overlay (QGraphicsVideoItem) — toggled visible + opacity
      - OverlayItem for drawing annotations on top
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Smooth transforms and antialiasing make overlays & thumbnails look clean
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        self.setFrameShape(QFrame.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # Scene setup
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setBackgroundBrush(QColor(18, 18, 22))
        self.scene_.setSceneRect(QRectF(0, 0, 1280, 720))

        # Base video items
        self.video_item_orig = QGraphicsVideoItem();  self.video_item_orig.setZValue(0.0)
        self.video_item_infill = QGraphicsVideoItem(); self.video_item_infill.setZValue(0.0); self.video_item_infill.setVisible(False)
        self.scene_.addItem(self.video_item_orig)
        self.scene_.addItem(self.video_item_infill)

        # Mask as a separate layer on top
        self.mask_item = QGraphicsVideoItem(); self.mask_item.setZValue(10.0)
        self.mask_item.setOpacity(0.4); self.mask_item.setVisible(False)
        self.scene_.addItem(self.mask_item)

        # Annotation overlay above everything
        rect = QRectF(0, 0, 1280, 720)
        for it in (self.video_item_orig, self.video_item_infill, self.mask_item):
            it.setSize(rect.size())
        self.overlay_item = OverlayItem(rect)
        self.overlay_item.setZValue(20.0)
        self.scene_.addItem(self.overlay_item)

    def set_base_visible(self, which: str):
        """Switch between showing original vs infilled as the visible base layer."""
        if which == "original":
            self.video_item_orig.setVisible(True)
            self.video_item_infill.setVisible(False)
        else:
            self.video_item_orig.setVisible(False)
            self.video_item_infill.setVisible(True)

    def set_mask_visible(self, vis: bool):
        self.mask_item.setVisible(vis)

    def set_mask_opacity(self, alpha: float):
        self.mask_item.setOpacity(alpha)

    def resizeEvent(self, e):
        """Letterbox while preserving source aspect; keep overlay aligned with the video rectangle."""
        super().resizeEvent(e)
        view_rect = self.viewport().rect()
        if view_rect.isEmpty():
            return

        base_item = self.video_item_infill if self.video_item_infill.isVisible() else self.video_item_orig
        native = base_item.nativeSize()
        if native.isValid():
            src_w, src_h = native.width(), native.height()
        else:
            # Fallback aspect when we don't yet know native size
            src_w, src_h = 16, 9

        dst_w, dst_h = view_rect.width(), view_rect.height()
        src_ratio = src_w / src_h
        dst_ratio = dst_w / max(1, dst_h)

        # Compute a centered target rect that respects aspect
        if src_ratio > dst_ratio:
            new_h = int(dst_w / src_ratio)
            x = 0
            y = (dst_h - new_h) // 2
            target = QRectF(x, y, dst_w, new_h)
        else:
            new_w = int(dst_h * src_ratio)
            x = (dst_w - new_w) // 2
            y = 0
            target = QRectF(x, y, new_w, dst_h)

        for it in (self.video_item_orig, self.video_item_infill, self.mask_item):
            it.setPos(target.topLeft())
            it.setSize(target.size())
        self.overlay_item.setRect(target)
        self.scene_.setSceneRect(QRectF(0, 0, view_rect.width(), view_rect.height()))

    # ----- Thumbnail capture, with annotations painted onto the pixmap -----
    def _draw_annotations_on_pixmap(self, pix: QPixmap, kf: Optional[Keyframe]) -> QPixmap:
        if pix.isNull() or not kf:
            return pix
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.overlay_item.boundingRect()
        vw = self.viewport().width()
        vh = self.viewport().height()
        if vw <= 0 or vh <= 0:
            return pix
        sx = pix.width() / vw
        sy = pix.height() / vh

        # Rectangles
        p.setPen(QPen(QColor(0, 200, 255, 220), 2))
        p.setBrush(Qt.NoBrush)
        for (nx, ny, nw, nh) in kf.rects:
            x = (rect.left() + nx * rect.width()) * sx
            y = (rect.top()  + ny * rect.height()) * sy
            w = (nw * rect.width()) * sx
            h = (nh * rect.height()) * sy
            p.drawRect(QRectF(x, y, w, h))

        # Dots
        r_px = 6.0
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(55, 200, 90, 235))  # positive
        for (nx, ny) in kf.pos_clicks:
            x = (rect.left() + nx * rect.width()) * sx
            y = (rect.top()  + ny * rect.height()) * sy
            p.drawEllipse(QPointF(x, y), r_px, r_px)

        p.setBrush(QColor(230, 70, 70, 235))  # negative
        for (nx, ny) in kf.neg_clicks:
            x = (rect.left() + nx * rect.width()) * sx
            y = (rect.top()  + ny * rect.height()) * sy
            p.drawEllipse(QPointF(x, y), r_px, r_px)
        p.end()
        return pix

    def grabThumbWithOverlay(self, kf: Optional[Keyframe], size: QSize) -> Optional[QIcon]:
        pix: QPixmap = self.viewport().grab()
        if pix.isNull():
            return None
        pix = self._draw_annotations_on_pixmap(pix, kf)
        thumb = pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QIcon(thumb)


# --------- Slider with clickable seek (single click jumps to that time) ----------
class SeekSlider(QSlider):
    clickedValue = Signal(int)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.orientation() == Qt.Horizontal:
            x = e.position().x() if hasattr(e, "position") else e.pos().x()
            ratio = max(0.0, min(1.0, x / max(1, self.width())))
            val = int(self.minimum() + ratio * (self.maximum() - self.minimum()))
            self.setValue(val)
            self.clickedValue.emit(val)
            e.accept()
            return
        super().mousePressEvent(e)


# --------- Core player widget that wires everything together ----------
class VideoPlayer(QWidget):
    # External “position changed” signal (not strictly needed but can be handy)
    positionChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = VideoView(self)

        # --- Players: master + followers ---
        # Master/original (with audio)
        self.player_orig = QMediaPlayer(self)
        # Infilled follower (video only)
        self.player_infill = QMediaPlayer(self)
        # Mask follower (video only)
        self.mask_player = QMediaPlayer(self)

        # Audio: only for master
        self.audio = QAudioOutput(self); self.audio.setVolume(0.9)
        self.player_orig.setAudioOutput(self.audio)
        self.player_infill.setAudioOutput(None)
        self.mask_audio = QAudioOutput(self); self.mask_audio.setVolume(0.0)
        self.mask_player.setAudioOutput(self.mask_audio)

        # Route video to QGraphicsVideoItems
        self.player_orig.setVideoOutput(self.view.video_item_orig)
        self.player_infill.setVideoOutput(self.view.video_item_infill)
        self.mask_player.setVideoOutput(self.view.mask_item)

        # Read exact decoded frame timestamps from the master, so UI/overlay are frame-accurate
        self._master_sink: QVideoSink = self.view.video_item_orig.videoSink()
        self._master_sink.videoFrameChanged.connect(self._on_master_frame_changed)
        if hasattr(self.player_orig, "setNotifyInterval"):
            # Optional: reduce timer callbacks; we rely on frame callbacks anyway
            try: self.player_orig.setNotifyInterval(0)
            except Exception: pass

        # Last decoded frame timestamp (ms) from the master
        self._last_frame_ms = 0

        # While playing, followers play and we periodically “nudge” them if drift grows:
        self.resync_interval_ms = 120  # how often we check drift (ms)
        self.resync_threshold_ms = 35  # only nudge if more than this drift (ms)
        self._resync_timer = QTimer(self)
        self._resync_timer.setInterval(self.resync_interval_ms)
        self._resync_timer.timeout.connect(self._playing_resync)

        # Which base is visible
        self.show_mode = "original"  # "original" | "infilled"
        self.view.set_base_visible("original")

        # --- Keyframe bar / thumbnails ---
        self.kf_list = QListWidget()
        self.kf_list.setFlow(QListWidget.LeftToRight)
        self.kf_list.setWrapping(False)
        self.kf_list.setFixedHeight(78)
        self.kf_list.setIconSize(QSize(128, 72))
        self.kf_list.itemClicked.connect(self._on_kf_clicked)

        # --- Transport UI ---
        self.play_btn = QPushButton(self.style().standardIcon(QStyle.SP_MediaPlay), "")
        self.play_btn.setFixedSize(QSize(36, 36))
        self.play_btn.clicked.connect(self.toggle_play)
        self.stop_btn = QPushButton(self.style().standardIcon(QStyle.SP_MediaStop), "")
        self.stop_btn.setFixedSize(QSize(36, 36))
        self.stop_btn.clicked.connect(self.stop)

        self.pos_slider = SeekSlider(Qt.Horizontal)
        self.pos_slider.setRange(0, 0)
        self.pos_slider.setSingleStep(100)
        self.pos_slider.setPageStep(1000)
        self.pos_slider.sliderMoved.connect(self.seek)
        self.pos_slider.sliderPressed.connect(self._slider_pressed)
        self.pos_slider.sliderReleased.connect(self._slider_released)
        self._slider_down = False
        self.pos_slider.clickedValue.connect(self.seek)

        self.time_label = QLabel("00:00 / 00:00")

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.kf_list)
        controls = QHBoxLayout()
        controls.setContentsMargins(10, 8, 10, 8)
        controls.setSpacing(12)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.pos_slider, 1)
        controls.addWidget(self.time_label)
        layout.addLayout(controls)

        # --- Master Qt signals -> our handlers ---
        self.player_orig.positionChanged.connect(self._on_master_position_changed)
        self.player_orig.durationChanged.connect(self._on_master_duration_changed)
        self.player_orig.playbackStateChanged.connect(self._on_master_state_changed)
        # Poster frame on load: run once per media
        self.player_orig.mediaStatusChanged.connect(self._on_master_media_status)
        self._poster_ready = False

        # --- Overlay slots ---
        ov = self.view.overlay_item
        ov.addPositive.connect(self._on_add_positive)
        ov.addNegative.connect(self._on_add_negative)
        ov.addRectangle.connect(self._on_add_rect)
        ov.requestDelete.connect(self._on_delete_at)

        # Annotation store
        self.keyframes: Dict[int, Keyframe] = {}
        self._pending_icon_updates: set[int] = set()

    # ---------- Public API to load media ----------
    def load_original(self, path: str):
        """Load the master/original video (with audio)."""
        p = str(Path(path).resolve())
        self._poster_ready = False  # ensure poster logic runs once for this source
        self.player_orig.setSource(QUrl.fromLocalFile(p))

    def load_infilled(self, path: str):
        p = str(Path(path).resolve())
        self.player_infill.setSource(QUrl.fromLocalFile(p))

    def load_mask(self, path: str):
        p = str(Path(path).resolve())
        self.mask_player.setSource(QUrl.fromLocalFile(p))

    # ---------- Slider press/drag/release handlers ----------
    def _slider_pressed(self):
        self._slider_down = True

    def _slider_released(self):
        self._slider_down = False
        # Seek to the dropped position; slider value already points to it
        self.seek(self.pos_slider.value())
        # Minor adjustment so slider reflects exact last-frame timestamp
        QTimer.singleShot(30, lambda: self.pos_slider.setValue(self._last_frame_ms))

    # ---------- Change which base is visible ----------
    def set_mode(self, mode: str):
        if mode == self.show_mode:
            return
        was_playing = (self.player_orig.playbackState() == QMediaPlayer.PlayingState)
        # Pause to switch cleanly and show a snapped frame
        self.pause_all()
        self.show_mode = mode
        self.view.set_base_visible(mode)
        if was_playing:
            # If we were playing, resume followers + resync timer
            self.play_all()
        else:
            # Stay snapped at the current master frame
            self._snap_all_to(self._last_frame_ms or self.player_orig.position())

    # ---------- Mask visibility/opacity ----------
    def set_mask_visible(self, vis: bool):
        self.view.set_mask_visible(vis)
        if vis:
            if self.player_orig.playbackState() == QMediaPlayer.PlayingState:
                # Playing: roughly align then play
                cur = self._last_frame_ms or self.player_orig.position()
                self.mask_player.setPosition(cur)
                self.mask_player.play()
            else:
                # Paused: exact snap
                cur = self._last_frame_ms or self.player_orig.position()
                self.mask_player.pause()
                self.mask_player.setPosition(cur)
        else:
            self.mask_player.pause()

    def set_mask_opacity(self, value: int):
        self.view.set_mask_opacity(max(0.0, min(1.0, value / 100.0)))

    # ---------- Transport ----------
    def toggle_play(self):
        if self.player_orig.playbackState() == QMediaPlayer.PlayingState:
            self.pause_all()
        else:
            self.play_all()

    def play_all(self):
        """Start master + followers, aligned to the last known master frame."""
        cur = self._last_frame_ms or self.player_orig.position()
        # Master plays with audio
        self.player_orig.setPosition(cur)
        self.player_orig.play()

        # Infilled follower plays only if visible
        if self.show_mode == "infilled":
            self.player_infill.setPosition(cur)
            self.player_infill.play()
        else:
            self.player_infill.pause()

        # Mask follower plays if visible
        if self.view.mask_item.isVisible():
            self.mask_player.setPosition(cur)
            self.mask_player.play()
        else:
            self.mask_player.pause()

        # Enable periodic drift correction
        self._resync_timer.start()

    def pause_all(self):
        """Pause everything and snap followers to the exact master frame."""
        self.player_orig.pause()
        self.player_infill.pause()
        self.mask_player.pause()
        self._resync_timer.stop()
        self._snap_all_to(self._last_frame_ms or self.player_orig.position())

    def stop(self):
        self._resync_timer.stop()
        self.player_orig.stop()
        self.player_infill.stop()
        self.mask_player.stop()

    def seek(self, pos_ms: int):
        """
        Jump to a position:
         - Update master & followers immediately (so UI reflects it right away),
         - If paused, also do an exact snap to avoid stale frames on some backends.
        """
        self.player_orig.setPosition(pos_ms)
        if self.show_mode == "infilled":
            self.player_infill.setPosition(pos_ms)
        if self.view.mask_item.isVisible():
            self.mask_player.setPosition(pos_ms)

        # Immediate UI/overlay refresh (don't wait for next decoded frame)
        self._last_frame_ms = pos_ms
        if not self._slider_down:
            self.pos_slider.setValue(pos_ms)
        self.time_label.setText(f"{fmt_ms(pos_ms)} / {fmt_ms(self.player_orig.duration() or 0)}")
        t_key = self._round_now(pos_ms)
        self.view.overlay_item.setKeyframe(self.keyframes.get(t_key))
        self.view.viewport().update()

        if self.player_orig.playbackState() != QMediaPlayer.PlayingState:
            # If paused, exact snap helps backends that only paint on frame decode
            self._snap_all_to(pos_ms)

    def set_volume(self, value: int):
        self.audio.setVolume(max(0.0, min(1.0, value / 100.0)))

    # ---------- FRAME-DRIVEN UI (master decoded frames) ----------
    def _on_master_frame_changed(self, frame):
        """
        Called by Qt when the master decodes a new frame.
        We use the frame timestamp as ground truth for:
          - last-frame ms,
          - UI slider/time label,
          - overlay keyframe selection.
        Periodic resync nudges followers separately via a timer to keep things smooth.
        """
        ts_us = frame.startTime()
        pos_ms = int(ts_us // 1000) if (ts_us is not None and ts_us >= 0) else self.player_orig.position()
        self._last_frame_ms = pos_ms

        if not self._slider_down:
            self.pos_slider.setValue(pos_ms)
            self.time_label.setText(f"{fmt_ms(pos_ms)} / {fmt_ms(self.player_orig.duration() or 0)}")

        t_key = self._round_now(pos_ms)
        self.view.overlay_item.setKeyframe(self.keyframes.get(t_key))

    # ---------- Periodic resync (while playing) ----------
    def _playing_resync(self):
        """
        While playing, check follower drift relative to master and nudge
        them if their positions differ by more than resync_threshold_ms.
        """
        if self.player_orig.playbackState() != QMediaPlayer.PlayingState:
            return
        master_ms = self._last_frame_ms or self.player_orig.position()

        # Infilled drift correction
        if self.show_mode == "infilled":
            d = abs((self.player_infill.position() or 0) - master_ms)
            if d > self.resync_threshold_ms:
                self.player_infill.setPosition(master_ms)

        # Mask drift correction
        if self.view.mask_item.isVisible():
            d = abs((self.mask_player.position() or 0) - master_ms)
            if d > self.resync_threshold_ms:
                self.mask_player.setPosition(master_ms)

    def _snap_all_to(self, pos_ms: int, snap_orig=False):
        """
        Exact snap for paused/seek state: we explicitly set follower positions,
        refresh UI, and force a repaint of the overlay region.
        """
        if snap_orig:
            self.player_orig.setPosition(pos_ms)
        if self.show_mode == "infilled":
            self.player_infill.setPosition(pos_ms)
        if self.view.mask_item.isVisible():
            self.mask_player.setPosition(pos_ms)

        self.pos_slider.setValue(pos_ms)
        self.time_label.setText(f"{fmt_ms(pos_ms)} / {fmt_ms(self.player_orig.duration() or 0)}")
        t_key = self._round_now(pos_ms)
        self.view.overlay_item.setKeyframe(self.keyframes.get(t_key))
        self.view.viewport().update()

    # ---------- Poster frame on load (run once per media) ----------
    def _on_master_media_status(self, status):
        """
        When the master is Loaded/Buffered and not already playing, we force
        the very first frame to be visible by briefly toggling play→pause at t=0.
        This avoids the “black window until first play” behavior some backends have.
        """
        from PySide6.QtMultimedia import QMediaPlayer
        if self._poster_ready:
            return
        if status in (getattr(QMediaPlayer, "LoadedMedia", None),
                      getattr(QMediaPlayer, "BufferedMedia", None)):
            if self.player_orig.playbackState() != QMediaPlayer.PlayingState:
                QTimer.singleShot(0, self._show_first_master_frame)

    def _show_first_master_frame(self):
        # If user hit Play already, do nothing.
        from PySide6.QtMultimedia import QMediaPlayer
        if self.player_orig.playbackState() == QMediaPlayer.PlayingState:
            self._poster_ready = True
            return

        # Nudge a frame to render reliably across backends: play→pause at t=0
        try:
            self.player_orig.setPosition(0)
            self.player_orig.play()
            QTimer.singleShot(0, self.player_orig.pause)
        except Exception:
            # Fallback: just set position (some backends render on seek alone)
            self.player_orig.setPosition(0)

        self._last_frame_ms = 0

        # Keep followers at t=0 without starting playback
        if self.show_mode == "infilled":
            self.player_infill.pause()
            self.player_infill.setPosition(0)
        if self.view.mask_item.isVisible():
            self.mask_player.pause()
            self.mask_player.setPosition(0)

        # Update UI/overlay
        self.pos_slider.setValue(0)
        self.time_label.setText(f"{fmt_ms(0)} / {fmt_ms(self.player_orig.duration() or 0)}")
        self.view.overlay_item.setKeyframe(self.keyframes.get(0))
        self.view.viewport().update()

        self._poster_ready = True

    # ---------- Basic UI signal handlers ----------
    def _on_master_position_changed(self, position):
        # We update UI off decoded frames primarily; still reflect slider when dragging.
        if not self._slider_down:
            self.pos_slider.setValue(position)
        self.positionChanged.emit(position)

    def _on_master_duration_changed(self, duration):
        self.pos_slider.setRange(0, duration)
        self.time_label.setText(f"{fmt_ms(self._last_frame_ms or self.player_orig.position() or 0)} / {fmt_ms(duration)}")

    def _on_master_state_changed(self, state):
        self.play_btn.setIcon(self.style().standardIcon(
            QStyle.SP_MediaPause if state == QMediaPlayer.PlayingState else QStyle.SP_MediaPlay
        ))
        if state != QMediaPlayer.PlayingState:
            # Ensure a final exact snap when pausing/stopping
            self._snap_all_to(self._last_frame_ms or self.player_orig.position(), True)

    # ---------- Keyframe utility ----------
    def _round_now(self, pos: Optional[int] = None) -> int:
        """10 ms rounding to group near-identical timestamps into a single keyframe."""
        if pos is None:
            pos = self._last_frame_ms or self.player_orig.position()
        return int(round(pos / 10.0) * 10)

    def _get_or_make_kf(self, t_ms: Optional[int] = None, add_chip=True) -> Keyframe:
        if t_ms is None:
            t_ms = self._round_now()
        kf = self.keyframes.get(t_ms)
        if not kf:
            kf = Keyframe(t_ms=t_ms)
            self.keyframes[t_ms] = kf
            if add_chip:
                self._add_kf_chip(kf, with_icon=True)
        return kf

    def _add_kf_chip(self, kf: Keyframe, with_icon: bool = True):
        t_new = kf.t_ms
        if self._find_kf_item_by_time(t_new) is not None:
            return
        # Insert sorted by time ascending
        insert_row = self.kf_list.count()
        for row in range(self.kf_list.count()):
            it = self.kf_list.item(row)
            t = it.data(Qt.UserRole)
            if t is not None and t_new < t:
                insert_row = row
                break
        item = QListWidgetItem(fmt_ms(t_new))
        item.setToolTip(f"Keyframe @ {fmt_ms(t_new)}")
        item.setData(Qt.UserRole, t_new)
        if with_icon:
            icon = self.view.grabThumbWithOverlay(kf, self.kf_list.iconSize())
            if icon is not None:
                item.setIcon(icon)
            else:
                self._pending_icon_updates.add(t_new)
        self.kf_list.insertItem(insert_row, item)

    def _remove_kf_chip(self, t_ms: int):
        for i in range(self.kf_list.count()):
            it = self.kf_list.item(i)
            if it.data(Qt.UserRole) == t_ms:
                self.kf_list.takeItem(i)
                break
        self._pending_icon_updates.discard(t_ms)

    def _find_kf_item_by_time(self, t_ms: int) -> Optional[QListWidgetItem]:
        for i in range(self.kf_list.count()):
            it = self.kf_list.item(i)
            if it.data(Qt.UserRole) == t_ms:
                return it
        return None

    def _ensure_icon_for_time(self, t_ms: int):
        it = self._find_kf_item_by_time(t_ms)
        if it:
            icon = self.view.grabThumbWithOverlay(self.keyframes.get(t_ms), self.kf_list.iconSize())
            if icon is not None:
                it.setIcon(icon)

    def _prune_if_empty(self, kf: Keyframe):
        if not kf.pos_clicks and not kf.neg_clicks and not kf.rects:
            t = kf.t_ms
            self.keyframes.pop(t, None)
            self._remove_kf_chip(t)
            cur = self.keyframes.get(self._round_now())
            self.view.overlay_item.setKeyframe(cur)
            self.view.viewport().update()
        else:
            self._ensure_icon_for_time(kf.t_ms)

    # ---------- Annotation slots (called by OverlayItem signals) ----------
    def _on_add_positive(self, nx: float, ny: float):
        kf = self._get_or_make_kf()
        kf.pos_clicks.append((nx, ny))
        self.view.overlay_item.setKeyframe(kf)
        self._ensure_icon_for_time(kf.t_ms)
        print(f"[KF {fmt_ms(kf.t_ms)}] +POS ({nx:.3f},{ny:.3f})")

    def _on_add_negative(self, nx: float, ny: float):
        kf = self._get_or_make_kf()
        kf.neg_clicks.append((nx, ny))
        self.view.overlay_item.setKeyframe(kf)
        self._ensure_icon_for_time(kf.t_ms)
        print(f"[KF {fmt_ms(kf.t_ms)}] -NEG ({nx:.3f},{ny:.3f})")

    def _on_add_rect(self, nx: float, ny: float, nw: float, nh: float):
        kf = self._get_or_make_kf()
        kf.rects.append((nx, ny, nw, nh))
        self.view.overlay_item.setKeyframe(kf)
        self._ensure_icon_for_time(kf.t_ms)
        print(f"[KF {fmt_ms(kf.t_ms)}] RECT ({nx:.3f},{ny:.3f},{nw:.3f},{nh:.3f})")

    def _on_delete_at(self, nx: float, ny: float):
        t_key = self._round_now()
        kf = self.keyframes.get(t_key)
        if not kf:
            return

        rect = self.view.overlay_item.boundingRect()
        W = max(1.0, rect.width())
        H = max(1.0, rect.height())
        R_px = 8.0  # near-distance in pixels for deleting points/edge

        def pop_near_point(points):
            for i, (px, py) in enumerate(points):
                dx = abs(nx - px) * W
                dy = abs(ny - py) * H
                if (dx*dx + dy*dy) ** 0.5 <= R_px:
                    points.pop(i)
                    return True
            return False

        # Try deleting a point first
        if pop_near_point(kf.pos_clicks) or pop_near_point(kf.neg_clicks):
            print(f"[KF {fmt_ms(kf.t_ms)}] deleted point near ({nx:.3f},{ny:.3f})")
            self.view.overlay_item.setKeyframe(kf)
            self._prune_if_empty(kf)
            return

        # Then try deleting a rectangle by clicking near its border
        def near_rect_edge(nx_, ny_, r):
            rx, ry, rw, rh = r
            Rx = R_px / W
            Ry = R_px / H
            left   = abs(nx_ - rx)      <= Rx and (ry - Ry) <= ny_ <= (ry + rh + Ry)
            right  = abs(nx_ - (rx+rw)) <= Rx and (ry - Ry) <= ny_ <= (ry + rh + Ry)
            top    = abs(ny_ - ry)      <= Ry and (rx - Rx) <= nx_ <= (rx + rw + Rx)
            bottom = abs(ny_ - (ry+rh)) <= Ry and (rx - Rx) <= nx_ <= (rx + rw + Rx)
            return left or right or top or bottom

        for i, r in enumerate(kf.rects):
            if near_rect_edge(nx, ny, r):
                kf.rects.pop(i)
                print(f"[KF {fmt_ms(kf.t_ms)}] deleted rectangle")
                self.view.overlay_item.setKeyframe(kf)
                self._prune_if_empty(kf)
                return

        # If we reach here, nothing was deleted; refresh chip anyway
        self._ensure_icon_for_time(kf.t_ms)

    # ---------- Keyframe list click -> seek ----------
    def _on_kf_clicked(self, item: QListWidgetItem):
        t_ms = item.data(Qt.UserRole)
        if t_ms is not None:
            self.seek(int(t_ms))
            # After the frame is visible, refresh that chip (ensures overlay in the thumb)
            QTimer.singleShot(80, lambda: self._ensure_icon_for_time(int(t_ms)))

    # ---------- Save/Load annotations ----------
    def to_json_obj(self, video_path: Optional[str]) -> dict:
        return {
            "video": str(video_path) if video_path else None,
            "keyframes": [
                {
                    "t_ms": k.t_ms,
                    "pos_clicks": k.pos_clicks,
                    "neg_clicks": k.neg_clicks,
                    "rects": k.rects
                }
                for _, k in sorted(self.keyframes.items())
            ],
        }

    def load_from_json_obj(self, obj: dict):
        self.keyframes.clear()
        self.kf_list.clear()
        self._pending_icon_updates.clear()
        for entry in obj.get("keyframes", []):
            kf = Keyframe(
                t_ms=int(entry["t_ms"]),
                pos_clicks=[tuple(map(float, p)) for p in entry.get("pos_clicks", [])],
                neg_clicks=[tuple(map(float, p)) for p in entry.get("neg_clicks", [])],
                rects=[tuple(map(float, r)) for r in entry.get("rects", [])],
            )
            self.keyframes[kf.t_ms] = kf
            self._add_kf_chip(kf, with_icon=False)
            self._pending_icon_updates.add(kf.t_ms)
        QTimer.singleShot(100, lambda: [self._ensure_icon_for_time(t) for t in list(self._pending_icon_updates)])
        self.view.overlay_item.setKeyframe(self.keyframes.get(self._round_now()))
        self.view.viewport().update()


# --------- Side panel with tools and file loaders ----------
class SideDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Tools", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea)

        self.container = QWidget(self)
        lay = QVBoxLayout(self.container)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # Tool selection
        lay.addWidget(QLabel("Annotate Tool"))
        self.btn_pos = QToolButton(text="Positive"); self.btn_pos.setCheckable(True)
        self.btn_neg = QToolButton(text="Negative"); self.btn_neg.setCheckable(True)
        self.btn_rect = QToolButton(text="Rectangle"); self.btn_rect.setCheckable(True)
        group = QButtonGroup(self); group.setExclusive(True)
        group.addButton(self.btn_pos, OverlayItem.TOOL_POS)
        group.addButton(self.btn_neg, OverlayItem.TOOL_NEG)
        group.addButton(self.btn_rect, OverlayItem.TOOL_RECT)
        self.btn_pos.setChecked(True)
        row = QHBoxLayout()
        row.addWidget(self.btn_pos); row.addWidget(self.btn_neg); row.addWidget(self.btn_rect)
        lay.addLayout(row)

        # File pickers
        lay.addSpacing(8); lay.addWidget(QLabel("Files"))
        self.open_color_btn = QPushButton("Open Color Video…")
        self.open_infilled_btn = QPushButton("Open Infilled Video…")
        self.open_mask_btn = QPushButton("Open Mask Video…")
        lay.addWidget(self.open_color_btn)
        lay.addWidget(self.open_infilled_btn)
        lay.addWidget(self.open_mask_btn)

        # View mode
        mode_group = QGroupBox("View Mode")
        mg_lay = QHBoxLayout(mode_group)
        self.rb_original = QRadioButton("Original")
        self.rb_infilled = QRadioButton("Infilled")
        self.rb_original.setChecked(True)
        mg_lay.addWidget(self.rb_original)
        mg_lay.addWidget(self.rb_infilled)
        lay.addWidget(mode_group)

        # Mask controls
        mask_group = QGroupBox("Mask Overlay")
        mk_lay = QVBoxLayout(mask_group)
        self.cb_show_mask = QCheckBox("Show Mask"); self.cb_show_mask.setChecked(False)
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity"))
        self.mask_opacity = QSlider(Qt.Horizontal); self.mask_opacity.setRange(0, 100); self.mask_opacity.setValue(40)
        op_row.addWidget(self.mask_opacity)
        mk_lay.addWidget(self.cb_show_mask)
        mk_lay.addLayout(op_row)
        lay.addWidget(mask_group)

        lay.addStretch(1)

        # Placeholder processing buttons
        self.btn_generate_mask = QPushButton("Generate Mask")
        self.btn_make_vanish   = QPushButton("Make Vanish")
        lay.addWidget(self.btn_generate_mask)
        lay.addWidget(self.btn_make_vanish)

        self.setWidget(self.container)
        self.tool_group = group


# --------- MainWindow wiring menus, toolbar, and the player ----------
class MainWindow(QMainWindow):
    def __init__(self, color_video: Optional[str] = None,
                 mask_video: Optional[str] = None,
                 infilled_video: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("VideoVanish – Playing Followers + Resync")
        self.resize(1280, 820)

        self.current_video_path: Optional[Path] = Path(color_video) if color_video else None
        self.infilled_video_path: Optional[Path] = None
        self.mask_video_path: Optional[Path] = None

        # Central player widget
        self.player_widget = VideoPlayer(self)
        self.setCentralWidget(self.player_widget)

        # Side tools
        self.tools = SideDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.tools)

        self._apply_dark_theme()  # <- global dark palette + style sheet
        self._make_menu()
        self._make_toolbar()

        # Connect sidebar widgets
        self.tools.tool_group.idToggled.connect(self._on_tool_changed)
        self.tools.open_color_btn.clicked.connect(self.open_color_video)
        self.tools.open_infilled_btn.clicked.connect(self.open_infilled_video)
        self.tools.open_mask_btn.clicked.connect(self.open_mask_video)

        self.tools.rb_original.toggled.connect(lambda checked: checked and self.set_mode("original"))
        self.tools.rb_infilled.toggled.connect(lambda checked: checked and self.set_mode("infilled"))

        self.tools.cb_show_mask.toggled.connect(self.player_widget.set_mask_visible)
        self.tools.mask_opacity.valueChanged.connect(self.player_widget.set_mask_opacity)

        self.tools.btn_generate_mask.clicked.connect(self.generate_mask)
        self.tools.btn_make_vanish.clicked.connect(self.make_vanish)

        # Autoload CLI arguments if provided
        if color_video:
            self.load_color_video(color_video)
        if mask_video:
            self.mask_video_path = Path(mask_video)
            self.player_widget.load_mask(mask_video)
            self.tools.cb_show_mask.setChecked(True)
            self.player_widget.set_mask_visible(True)
        if infilled_video:
            self.infilled_video_path = Path(infilled_video)
            self.player_widget.load_infilled(infilled_video)

    # ---------- Simple stubs for future processing integration ----------
    def generate_mask(self):
        QMessageBox.information(self, "Generate Mask", "Mask generation not yet implemented.")

    def make_vanish(self):
        QMessageBox.information(self, "Make Vanish", "Inpainting not yet implemented.")

    # ---------- Menu/toolbar ----------
    def _make_menu(self):
        m = self.menuBar()
        file_menu = m.addMenu("&File")
        act_open = QAction("Open Color Video…", self)
        act_open.triggered.connect(self.open_color_video)
        file_menu.addAction(act_open)

        act_open_inf = QAction("Open Infilled Video…", self)
        act_open_inf.triggered.connect(self.open_infilled_video)
        file_menu.addAction(act_open_inf)

        act_open_mask = QAction("Open Mask Video…", self)
        act_open_mask.triggered.connect(self.open_mask_video)
        file_menu.addAction(act_open_mask)

        file_menu.addSeparator()
        act_save = QAction("Save Annotations…", self)
        act_save.triggered.connect(self.save_annotations)
        file_menu.addAction(act_save)
        act_load = QAction("Load Annotations…", self)
        act_load.triggered.connect(self.load_annotations)
        file_menu.addAction(act_load)

        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    def _make_toolbar(self):
        tb = QToolBar("Main")
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(Qt.TopToolBarArea, tb)

        act_open = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), "Open Color Video…", self)
        act_open.triggered.connect(self.open_color_video)
        tb.addAction(act_open)

        act_play = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), "Play/Pause (Space)", self)
        act_play.triggered.connect(self.player_widget.toggle_play)
        tb.addAction(act_play)

        act_stop = QAction(self.style().standardIcon(QStyle.SP_MediaStop), "Stop", self)
        act_stop.triggered.connect(self.player_widget.stop)
        tb.addAction(act_stop)

        tb.addSeparator()
        vol_label = QLabel("Vol")
        vol_slider = QSlider(Qt.Horizontal)
        vol_slider.setRange(0, 100)
        vol_slider.setValue(90)
        vol_slider.setFixedWidth(120)
        vol_slider.valueChanged.connect(self.player_widget.set_volume)
        tb.addWidget(vol_label)
        tb.addWidget(vol_slider)

    # ---------- Apply a dark editor-style theme ----------
    def _apply_dark_theme(self):
        # palette
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(24, 25, 28))
        pal.setColor(QPalette.WindowText, Qt.white)
        pal.setColor(QPalette.Base, QColor(30, 32, 35))
        pal.setColor(QPalette.AlternateBase, QColor(38, 41, 45))
        pal.setColor(QPalette.ToolTipBase, Qt.white)
        pal.setColor(QPalette.ToolTipText, Qt.white)
        pal.setColor(QPalette.Text, Qt.white)
        pal.setColor(QPalette.Button, QColor(48, 50, 55))
        pal.setColor(QPalette.ButtonText, Qt.white)
        pal.setColor(QPalette.BrightText, Qt.red)
        pal.setColor(QPalette.Highlight, QColor(80, 120, 200))
        pal.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(pal)

        # subtle stylesheet for widgets
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1E2023; color: #E8E8E8; }
            QDockWidget { titlebar-close-icon: url(none); titlebar-normal-icon: url(none); }
            QGroupBox { border: 1px solid #3A3D41; margin-top: 8px; border-radius: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #C9CDD1; }
            QPushButton {
                background: #2C2F33; color: #E8E8E8; border: 1px solid #3A3D41; padding: 6px 10px; border-radius: 6px;
            }
            QPushButton:hover { background: #34383D; }
            QPushButton:pressed { background: #3D4147; }
            QToolButton {
                background: #2C2F33; color: #E8E8E8; border: 1px solid #3A3D41; padding: 6px 10px; border-radius: 6px;
            }
            QToolButton:checked { background: #3A3F46; border-color: #4A5060; }
            QSlider::groove:horizontal {
                border: 1px solid #3A3D41; height: 6px; background: #2C2F33; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #A9C1FF; border: 1px solid #4A5060; width: 12px; margin: -6px 0; border-radius: 6px;
            }
            QSlider::sub-page:horizontal { background: #556C99; border: 1px solid #3A3D41; height: 6px; border-radius: 3px; }
            QListWidget { background: #222428; border: 1px solid #3A3D41; }
            QMenuBar { background: #1E2023; }
            QMenu { background: #24262A; color: #E8E8E8; }
            QMenu::item:selected { background: #373B40; }
            QToolBar { background: #1E2023; border-bottom: 1px solid #32353A; }
            QLabel { color: #C9CDD1; }
        """)

    # ---------- File open/load helpers ----------
    def open_color_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Color Video", "", "Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*)")
        if path:
            self.load_color_video(path)

    def load_color_video(self, path: str):
        self.current_video_path = Path(path)
        self.player_widget.load_original(path)
        # Make sure we’re viewing the original layer so the poster frame is visible
        self.tools.rb_original.setChecked(True)
        self.set_mode("original")
        self.setWindowTitle(f"VideoVanish – {self.current_video_path.name}")

    def open_infilled_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Infilled Video", "", "Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*)")
        if path:
            self.infilled_video_path = Path(path)
            self.player_widget.load_infilled(path)
            if self.tools.rb_infilled.isChecked():
                self.set_mode("infilled")

    def open_mask_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Mask Video", "", "Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*)")
        if path:
            self.mask_video_path = Path(path)
            self.player_widget.load_mask(path)
            if self.tools.cb_show_mask.isChecked():
                self.player_widget.set_mask_visible(True)

    # ---------- Mode & mask ----------
    def set_mode(self, mode: str):
        if mode not in ("original", "infilled"):
            return
        if mode == "infilled" and not self.infilled_video_path:
            QMessageBox.warning(self, "Infilled missing", "Load an infilled video first.")
            self.tools.rb_original.setChecked(True)
            return
        self.player_widget.set_mode(mode)

    # ---------- Save/Load annotations ----------
    def save_annotations(self):
        base = self.current_video_path or Path.cwd() / "annotations"
        path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", str(base.with_suffix(".annotations.json")), "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        obj = self.player_widget.to_json_obj(str(self.current_video_path) if self.current_video_path else None)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
            QMessageBox.information(self, "Saved", f"Annotations saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def load_annotations(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Annotations", "", "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read JSON:\n{e}")
            return

        video_in_json = obj.get("video")
        if video_in_json and (not self.current_video_path or str(self.current_video_path) != video_in_json):
            if Path(video_in_json).exists():
                self.load_color_video(video_in_json)
            else:
                QMessageBox.warning(
                    self, "Video Missing",
                    f"The JSON references a video that doesn't exist:\n{video_in_json}\n"
                    f"Annotations will load but thumbnails may be off."
                )
        self.player_widget.load_from_json_obj(obj)
        QMessageBox.information(self, "Loaded", "Annotations loaded.")

    # ---------- Overlay tool selection ----------
    def _on_tool_changed(self, tool_id: int, checked: bool):
        if checked:
            self.player_widget.view.overlay_item.setTool(tool_id)

    # ---------- Keyboard shortcuts ----------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.player_widget.toggle_play()
            event.accept()
            return
        super().keyPressEvent(event)


# --------- CLI parsing & main entry ----------
def parse_args(argv):
    p = argparse.ArgumentParser(description="VideoVanish (Playing Followers + Resync)")
    p.add_argument("--color_video", type=str, default=None, help="Path to color/original video")
    p.add_argument("--mask_video", type=str, default=None, help="Path to mask video")
    p.add_argument("--infilled_video", type=str, default=None, help="Path to infilled video")
    args, _ = p.parse_known_args(argv)
    return args

def main():
    args = parse_args(sys.argv[1:])
    app = QApplication(sys.argv)
    win = MainWindow(
        color_video=args.color_video,
        mask_video=args.mask_video,
        infilled_video=args.infilled_video
    )
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

