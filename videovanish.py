#!/usr/bin/env python3
# videovanish.py — frame-indexed keyframes + object labels + RAM previews for mask & infill
#
# New:
# - In-memory preview layers for both Mask and Infilled:
#     * set_mask_preview_frames(list_of_np_arrays, start_frame=None)
#     * clear_mask_preview()
#     * set_infill_preview_frames(list_of_np_arrays, start_frame=None)
#     * clear_infill_preview()
# - Mask preview respects Mask checkbox + opacity slider and sits above the base.
# - Infill preview acts as the visible base when View Mode = "Infilled" (replaces file infill while present).
#
# Retained:
# - Dark editor-style UI.
# - Master/original (with audio) + optional file-backed infilled + optional file-backed mask.
# - Followers PLAY during playback with periodic drift correction; exact snap on pause/seek.
# - Frame-accurate UI via QVideoSink; keyframes by frame index.
# - Object selector & labels drawn next to points/rects; JSON includes obj.
# - Clickable seek slider; poster frame shown on load.

import sys, json, argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from PySide6.QtCore import Qt, QSize, QPointF, QRectF, Signal, QTimer, QUrl
from PySide6.QtGui import (
    QAction, QIcon, QPainter, QPen, QColor, QPixmap, QPalette, QFont, QImage
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QFileDialog, QPushButton, QDockWidget, QStyle,
    QToolBar, QMessageBox, QListWidget, QListWidgetItem, QButtonGroup,
    QToolButton, QGraphicsView, QGraphicsScene, QGraphicsObject, QFrame,
    QCheckBox, QRadioButton, QGroupBox, QComboBox, QGraphicsPixmapItem
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink, QMediaMetaData
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem

import sam2_masker, tools, diffuerase


# ---------- Helpers ----------
def fmt_ms(ms: int) -> str:
    s = max(0, ms // 1000)
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def ms_to_frame(ms: int, fps: float) -> int:
    return int(round((ms / 1000.0) * fps))

def frame_to_ms(fi: int, fps: float) -> int:
    return int(round((fi / fps) * 1000.0))


# ---------- Annotation data (by frame index) ----------
@dataclass
class Keyframe:
    frame_idx: int
    # Each point: (x_norm, y_norm, obj_id)
    pos_clicks: List[Tuple[float, float, int]] = field(default_factory=list)
    neg_clicks: List[Tuple[float, float, int]] = field(default_factory=list)
    # Each rect: (x_norm, y_norm, w_norm, h_norm, obj_id)
    rects: List[Tuple[float, float, float, float, int]] = field(default_factory=list)


# ---------- Overlay for drawing points/rectangles ----------
class OverlayItem(QGraphicsObject):
    addPositive = Signal(float, float)
    addNegative = Signal(float, float)
    addRectangle = Signal(float, float, float, float)
    requestDelete = Signal(float, float)

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

        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.setAcceptHoverEvents(True)
        self.setFlag(self.GraphicsItemFlag.ItemIsFocusable, True)
        self.setZValue(20.0)

        self.dot_radius_px = 10.0
        self.rect_pen_width = 4.0
        self.live_rect_pen_width = 3.0

        # Label styling
        self.label_font = QFont()
        self.label_font.setPointSize(9)
        self.label_pen = QPen(QColor(0, 0, 0, 220), 3)  # outline
        self.label_color = QColor(255, 255, 255, 230)

    def setTool(self, tool: int): self._tool = tool
    def setKeyframe(self, kf: Optional[Keyframe]): self._kf = kf; self.update()

    def setRect(self, rect: QRectF):
        if rect != self._rect:
            self.prepareGeometryChange()
            self._rect = QRectF(rect)
            self.update()

    def boundingRect(self) -> QRectF: return QRectF(self._rect)

    def _draw_label(self, p: QPainter, x: float, y: float, text: str):
        p.save()
        p.setFont(self.label_font)
        p.setPen(self.label_pen);   p.drawText(x + 1, y + 1, text)  # outline
        p.setPen(self.label_color); p.drawText(x, y, text)           # fill
        p.restore()

    def paint(self, p: QPainter, _opt, _widget=None):
        p.setRenderHint(QPainter.Antialiasing, True)

        # Live dashed rectangle
        if self._drawing and self._drag_start and self._drag_cur:
            r = self._make_rect(self._drag_start, self._drag_cur)
            p.setPen(QPen(QColor(255, 255, 255, 220), self.live_rect_pen_width, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)

        # Saved annotations
        if self._kf:
            W = max(1.0, self._rect.width()); H = max(1.0, self._rect.height())

            # Rectangles (cyan) + object label
            p.setPen(QPen(QColor(0, 200, 255, 220), self.rect_pen_width))
            p.setBrush(Qt.NoBrush)
            for (nx, ny, nw, nh, obj) in self._kf.rects:
                x = self._rect.left() + nx * W
                y = self._rect.top()  + ny * H
                w = nw * W; h = nh * H
                p.drawRect(QRectF(x, y, w, h))
                self._draw_label(p, x + w + 4, y + 12, f"{obj}")

            # Dots: positive = green; negative = red; with object labels
            r_px = self.dot_radius_px
            p.setPen(Qt.NoPen)

            p.setBrush(QColor(55, 200, 90, 235))  # positive
            for (nx, ny, obj) in self._kf.pos_clicks:
                cx = self._rect.left()+nx*W; cy = self._rect.top()+ny*H
                p.drawEllipse(QPointF(cx, cy), r_px, r_px)
                self._draw_label(p, cx + r_px + 3, cy + 4, f"{obj}")

            p.setBrush(QColor(230, 70, 70, 235))  # negative
            for (nx, ny, obj) in self._kf.neg_clicks:
                cx = self._rect.left()+nx*W; cy = self._rect.top()+ny*H
                p.drawEllipse(QPointF(cx, cy), r_px, r_px)
                self._draw_label(p, cx + r_px + 3, cy + 4, f"{obj}")

    # Mouse handling
    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            # right-click = delete nearest point or rect edge
            nx, ny = self._normalize_point(e.pos())
            self.requestDelete.emit(nx, ny)
            e.accept()
            return

        if e.button() == Qt.LeftButton:
            if self._tool == self.TOOL_RECT:
                self._drawing = True
                self._drag_start = e.pos()
                self._drag_cur = self._drag_start
                self.update()
            else:
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
            self._drag_cur = e.pos(); self.update(); e.accept(); return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._drawing and e.button() == Qt.LeftButton and self._drag_start and self._drag_cur:
            r = self._make_rect(self._drag_start, self._drag_cur)
            nx, ny, nw, nh = self._normalize_rect(r)
            if nw > 0 and nh > 0: self.addRectangle.emit(nx, ny, nw, nh)
        self._drawing = False; self._drag_start = None; self._drag_cur = None; self.update()
        super().mouseReleaseEvent(e)

    # Normalization utilities
    def _normalize_point(self, pt: QPointF) -> Tuple[float, float]:
        W = max(1.0, self._rect.width()); H = max(1.0, self._rect.height())
        nx = (pt.x() - self._rect.left()) / W; ny = (pt.y() - self._rect.top()) / H
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def _normalize_rect(self, r: QRectF) -> Tuple[float, float, float, float]:
        W = max(1.0, self._rect.width()); H = max(1.0, self._rect.height())
        nx = (r.left() - self._rect.left()) / W; ny = (r.top() - self._rect.top()) / H
        nw = r.width() / W; nh = r.height() / H
        return (max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny)),
                max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh)))

    @staticmethod
    def _make_rect(a: QPointF, b: QPointF) -> QRectF:
        x1, y1 = a.x(), a.y(); x2, y2 = b.x(), b.y()
        x, y = min(x1, x2), min(y1, y2); w, h = abs(x1-x2), abs(y1-y2)
        return QRectF(x, y, w, h)


# ---------- VideoView with base layers + previews + overlay ----------
class VideoView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        self.setFrameShape(QFrame.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.scene_ = QGraphicsScene(self); self.setScene(self.scene_)
        self.setBackgroundBrush(QColor(18, 18, 22))
        self.scene_.setSceneRect(QRectF(0, 0, 1280, 720))

        # Base video items
        self.video_item_orig = QGraphicsVideoItem();  self.video_item_orig.setZValue(0.0)
        self.video_item_infill = QGraphicsVideoItem(); self.video_item_infill.setZValue(0.0); self.video_item_infill.setVisible(False)
        self.scene_.addItem(self.video_item_orig)
        self.scene_.addItem(self.video_item_infill)

        # Infill RAM preview as a base pixmap (replaces file infill when active)
        self.infill_preview_item = QGraphicsPixmapItem()
        self.infill_preview_item.setZValue(0.0)
        self.infill_preview_item.setVisible(False)
        self.scene_.addItem(self.infill_preview_item)

        # Mask (file-backed) as a video layer on top
        self.mask_item = QGraphicsVideoItem(); self.mask_item.setZValue(10.0)
        self.mask_item.setOpacity(0.4); self.mask_item.setVisible(False)
        self.scene_.addItem(self.mask_item)

        # Mask RAM preview as a pixmap layer on top
        self.mask_preview_item = QGraphicsPixmapItem()
        self.mask_preview_item.setZValue(15.0)
        self.mask_preview_item.setVisible(False)
        self.scene_.addItem(self.mask_preview_item)

        # Annotation overlay above everything
        rect = QRectF(0, 0, 1280, 720)
        for it in (self.video_item_orig, self.video_item_infill, self.mask_item):
            it.setSize(rect.size())
        self.overlay_item = OverlayItem(rect); self.overlay_item.setZValue(20.0); self.scene_.addItem(self.overlay_item)

    # Base visibility (original vs infilled)
    def set_base_visible(self, which: str, use_infill_preview: bool = False):
        if which == "original":
            self.video_item_orig.setVisible(True)
            self.video_item_infill.setVisible(False)
            self.infill_preview_item.setVisible(False)
        else:
            self.video_item_orig.setVisible(False)
            # If preview frames present, show preview pixmap instead of file infill
            self.infill_preview_item.setVisible(use_infill_preview)
            self.video_item_infill.setVisible(not use_infill_preview)

    # Mask visibility / opacity
    def set_mask_visible(self, vis: bool): self.mask_item.setVisible(vis)
    def set_mask_opacity(self, alpha: float): self.mask_item.setOpacity(alpha)

    # Mask preview helpers
    def set_mask_preview_visible(self, vis: bool): self.mask_preview_item.setVisible(vis)
    def set_mask_preview_opacity(self, alpha: float): self.mask_preview_item.setOpacity(alpha)

    # Common pixmap scaling for both preview layers
    def _fit_and_set_pixmap(self, item: QGraphicsPixmapItem, pix: QPixmap):
        target = self.overlay_item.boundingRect()
        if not target.isEmpty() and not pix.isNull():
            scaled = pix.scaled(int(target.width()), int(target.height()),
                                Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            item.setPixmap(scaled)
            item.setPos(target.topLeft())

    def set_mask_preview_pixmap(self, pix: QPixmap):
        self._fit_and_set_pixmap(self.mask_preview_item, pix)

    def set_infill_preview_pixmap(self, pix: QPixmap):
        self._fit_and_set_pixmap(self.infill_preview_item, pix)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        view_rect = self.viewport().rect()
        if view_rect.isEmpty(): return
        base_item = self.video_item_infill if self.video_item_infill.isVisible() else self.video_item_orig
        native = base_item.nativeSize()
        if native.isValid(): src_w, src_h = native.width(), native.height()
        else: src_w, src_h = 16, 9

        dst_w, dst_h = view_rect.width(), view_rect.height()
        src_ratio = src_w / src_h; dst_ratio = dst_w / max(1, dst_h)
        if src_ratio > dst_ratio:
            new_h = int(dst_w / src_ratio); x = 0; y = (dst_h - new_h) // 2; target = QRectF(x, y, dst_w, new_h)
        else:
            new_w = int(dst_h * src_ratio); x = (dst_w - new_w) // 2; y = 0; target = QRectF(x, y, new_w, dst_h)

        for it in (self.video_item_orig, self.video_item_infill, self.mask_item):
            it.setPos(target.topLeft()); it.setSize(target.size())
        self.overlay_item.setRect(target)
        self.scene_.setSceneRect(QRectF(0, 0, view_rect.width(), view_rect.height()))

        # Reposition/scale current preview pixmaps to fit
        if not self.mask_preview_item.pixmap().isNull():
            self.set_mask_preview_pixmap(self.mask_preview_item.pixmap())
        if not self.infill_preview_item.pixmap().isNull():
            self.set_infill_preview_pixmap(self.infill_preview_item.pixmap())

    # Thumbnails (draw overlay annotations)
    def _draw_annotations_on_pixmap(self, pix: QPixmap, kf: Optional[Keyframe]) -> QPixmap:
        if pix.isNull() or not kf: return pix
        p = QPainter(pix); p.setRenderHint(QPainter.Antialiasing, True)

        # label style
        font = QFont(); font.setPointSize(8)
        outline_pen = QPen(QColor(0,0,0,220), 2)
        text_color = QColor(255,255,255,230)

        rect = self.overlay_item.boundingRect()
        vw = self.viewport().width(); vh = self.viewport().height()
        if vw <= 0 or vh <= 0: return pix
        sx = pix.width() / vw; sy = pix.height() / vh

        # Rectangles
        p.setPen(QPen(QColor(0, 200, 255, 220), 2)); p.setBrush(Qt.NoBrush)
        for (nx, ny, nw, nh, obj) in kf.rects:
            x = (rect.left() + nx * rect.width()) * sx
            y = (rect.top()  + ny * rect.height()) * sy
            w = (nw * rect.width()) * sx; h = (nh * rect.height()) * sy
            p.drawRect(QRectF(x, y, w, h))
            p.setFont(font); p.setPen(outline_pen); p.drawText(x + w + 3, y + 10, f"{obj}")
            p.setPen(text_color); p.drawText(x + w + 2, y + 9, f"{obj}")

        # Dots
        r_px = 5.0; p.setPen(Qt.NoPen)
        p.setBrush(QColor(55, 200, 90, 235))
        for (nx, ny, obj) in kf.pos_clicks:
            x = (rect.left() + nx * rect.width()) * sx
            y = (rect.top()  + ny * rect.height()) * sy
            p.drawEllipse(QPointF(x, y), r_px, r_px)
            p.setFont(font); p.setPen(outline_pen); p.drawText(x + r_px + 2, y + 4, f"{obj}")
            p.setPen(text_color); p.drawText(x + r_px + 1, y + 3, f"{obj}")

        p.setBrush(QColor(230, 70, 70, 235))
        for (nx, ny, obj) in kf.neg_clicks:
            x = (rect.left() + nx * rect.width()) * sx
            y = (rect.top()  + ny * rect.height()) * sy
            p.drawEllipse(QPointF(x, y), r_px, r_px)
            p.setFont(font); p.setPen(outline_pen); p.drawText(x + r_px + 2, y + 4, f"{obj}")
            p.setPen(text_color); p.drawText(x + r_px + 1, y + 3, f"{obj}")

        p.end(); return pix

    def grabThumbWithOverlay(self, kf: Optional[Keyframe], size: QSize) -> Optional[QIcon]:
        pix: QPixmap = self.viewport().grab()
        if pix.isNull(): return None
        pix = self._draw_annotations_on_pixmap(pix, kf)
        thumb = pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QIcon(thumb)


# ---------- Clickable seek slider ----------
class SeekSlider(QSlider):
    clickedValue = Signal(int)
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.orientation() == Qt.Horizontal:
            x = e.position().x() if hasattr(e, "position") else e.pos().x()
            ratio = max(0.0, min(1.0, x / max(1, self.width())))
            val = int(self.minimum() + ratio * (self.maximum() - self.minimum()))
            self.setValue(val); self.clickedValue.emit(val); e.accept(); return
        super().mousePressEvent(e)


# ---------- Core Player ----------
class VideoPlayer(QWidget):
    positionChanged = Signal(int)  # ms for UI

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = VideoView(self)

        # Players
        self.player_orig = QMediaPlayer(self)    # master (audio)
        self.player_infill = QMediaPlayer(self)  # follower (file-backed infill)
        self.mask_player = QMediaPlayer(self)    # follower (file-backed mask)

        # Audio
        self.audio = QAudioOutput(self); self.audio.setVolume(0.9)
        self.player_orig.setAudioOutput(self.audio)
        self.player_infill.setAudioOutput(None)
        self.mask_audio = QAudioOutput(self); self.mask_audio.setVolume(0.0)
        self.mask_player.setAudioOutput(self.mask_audio)

        # Video routing
        self.player_orig.setVideoOutput(self.view.video_item_orig)
        self.player_infill.setVideoOutput(self.view.video_item_infill)
        self.mask_player.setVideoOutput(self.view.mask_item)

        # Frame callbacks from master
        self._master_sink: QVideoSink = self.view.video_item_orig.videoSink()
        self._master_sink.videoFrameChanged.connect(self._on_master_frame_changed)
        if hasattr(self.player_orig, "setNotifyInterval"):
            try: self.player_orig.setNotifyInterval(0)
            except Exception: pass

        # Time/frame tracking
        self._last_frame_ms: int = 0
        self._last_frame_idx: int = 0
        self._fps: Optional[float] = None  # enforced from metadata

        # Playback resync
        self.resync_interval_ms = 120
        self.resync_threshold_ms = 35
        self._resync_timer = QTimer(self); self._resync_timer.setInterval(self.resync_interval_ms)
        self._resync_timer.timeout.connect(self._playing_resync)

        # Mode
        self.show_mode = "original"
        self.view.set_base_visible("original")

        # Current object selection (1-based)
        self.current_object: int = 1

        # --- RAM mask preview state ---
        self._mask_preview_frames: Optional[List[np.ndarray]] = None
        self._mask_preview_start_frame: int = 0

        # --- RAM infill preview state ---
        self._infill_preview_frames: Optional[List[np.ndarray]] = None
        self._infill_preview_start_frame: int = 0

        # Keyframe bar
        self.kf_list = QListWidget(); self.kf_list.setFlow(QListWidget.LeftToRight); self.kf_list.setWrapping(False)
        self.kf_list.setFixedHeight(78); self.kf_list.setIconSize(QSize(128, 72))
        self.kf_list.itemClicked.connect(self._on_kf_clicked)

        # Transport
        self.play_btn = QPushButton(self.style().standardIcon(QStyle.SP_MediaPlay), ""); self.play_btn.setFixedSize(QSize(36, 36))
        self.stop_btn = QPushButton(self.style().standardIcon(QStyle.SP_MediaStop), "");  self.stop_btn.setFixedSize(QSize(36, 36))
        self.play_btn.clicked.connect(self.toggle_play); self.stop_btn.clicked.connect(self.stop)

        self.pos_slider = SeekSlider(Qt.Horizontal)
        self.pos_slider.setRange(0, 0); self.pos_slider.setSingleStep(100); self.pos_slider.setPageStep(1000)
        self.pos_slider.sliderMoved.connect(self.seek); self.pos_slider.sliderPressed.connect(self._slider_pressed)
        self.pos_slider.sliderReleased.connect(self._slider_released); self._slider_down = False
        self.pos_slider.clickedValue.connect(self.seek)

        self.time_label = QLabel("00:00 / 00:00")

        # Layout
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view, 1); layout.addWidget(self.kf_list)
        controls = QHBoxLayout(); controls.setContentsMargins(10, 8, 10, 8); controls.setSpacing(12)
        controls.addWidget(self.play_btn); controls.addWidget(self.stop_btn)
        controls.addWidget(self.pos_slider, 1); controls.addWidget(self.time_label)
        layout.addLayout(controls)

        # Master signals
        self.player_orig.positionChanged.connect(self._on_master_position_changed)
        self.player_orig.durationChanged.connect(self._on_master_duration_changed)
        self.player_orig.playbackStateChanged.connect(self._on_master_state_changed)
        self.player_orig.mediaStatusChanged.connect(self._on_master_media_status)
        self._poster_ready = False

        # Overlay
        ov = self.view.overlay_item
        ov.addPositive.connect(self._on_add_positive)
        ov.addNegative.connect(self._on_add_negative)
        ov.addRectangle.connect(self._on_add_rect)
        ov.requestDelete.connect(self._on_delete_at)

        # Annotation store
        self.keyframes: Dict[int, Keyframe] = {}
        self._pending_icon_updates: set[int] = set()

    # ---------- Object selection ----------
    def set_current_object(self, obj_id: int):
        if obj_id >= 1:
            self.current_object = obj_id

    # ---------- Public API: load media ----------
    def load_original(self, path: str):
        p = str(Path(path).resolve())
        self._poster_ready = False
        self._fps = None
        self.player_orig.setSource(QUrl.fromLocalFile(p))

    def load_infilled(self, path: str):
        # If a RAM preview exists, clear it so file-backed infill takes over.
        if self._infill_preview_frames is not None:
            self.clear_infill_preview()
        self.player_infill.setSource(QUrl.fromLocalFile(str(Path(path).resolve())))

    def load_mask(self, path: str):
        # If a RAM preview exists, clear it so file-backed mask takes over.
        if self._mask_preview_frames is not None:
            self.clear_mask_preview()
        self.mask_player.setSource(QUrl.fromLocalFile(str(Path(path).resolve())))

    # ---------- RAM preview: shared utilities ----------
    def _np_to_qpixmap(self, arr: np.ndarray) -> QPixmap:
        """Accept (H,W), (H,W,3), or (H,W,4) uint8 and return a QPixmap."""
        if arr is None:
            return QPixmap()
        a = np.asarray(arr)
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)

        if a.ndim == 2:
            h, w = a.shape
            a_rgb = np.repeat(a[:, :, None], 3, axis=2)
            bytes_per_line = 3 * w
            qimg = QImage(a_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)

        if a.ndim == 3 and a.shape[2] == 3:
            h, w, _ = a.shape
            bytes_per_line = 3 * w
            qimg = QImage(a.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)

        if a.ndim == 3 and a.shape[2] == 4:
            h, w, _ = a.shape
            bytes_per_line = 4 * w
            qimg = QImage(a.data, w, h, bytes_per_line, QImage.Format_RGBA8888).copy()
            return QPixmap.fromImage(qimg)

        # Fallback: coerce to RGB
        a = a.reshape(a.shape[0], a.shape[1], -1)
        if a.shape[2] >= 3:
            return self._np_to_qpixmap(a[:, :, :3].copy())
        return QPixmap()

    # ---------- RAM preview: MASK ----------
    def set_mask_preview_frames(self, frames: List[np.ndarray], start_frame: Optional[int] = None):
        """Provide a short RAM mask sequence. frames[i] -> mask for (start_frame + i)."""
        if not frames:
            self.clear_mask_preview()
            return
        if start_frame is None:
            start_frame = self._current_frame_idx() if (self._fps and self._fps > 0) else 0

        self._mask_preview_frames = frames
        self._mask_preview_start_frame = int(start_frame)

        # If mask is visible, switch to preview layer; else keep hidden until user toggles
        if self.view.mask_item.isVisible() or True:
            # hide file-mask (we'll manage it in set_mask_visible)
            self.view.set_mask_visible(False)
            self.view.set_mask_preview_visible(True)
            self.view.set_mask_preview_opacity(0.5)#we overwrite the opactity here 
            self._update_mask_preview_for_frame(self._current_frame_idx())

    def clear_mask_preview(self):
        self._mask_preview_frames = None
        self._mask_preview_start_frame = 0
        self.view.set_mask_preview_visible(False)

    def _update_mask_preview_for_frame(self, frame_idx: int):
        if not self._mask_preview_frames:
            return
        i = frame_idx - self._mask_preview_start_frame
        if i < 0 or i >= len(self._mask_preview_frames):
            self.view.set_mask_preview_visible(False)
            return
        arr = self._mask_preview_frames[i]
        pix = self._np_to_qpixmap(arr)
        if not pix.isNull():
            self.view.set_mask_preview_visible(True)
            self.view.set_mask_preview_pixmap(pix)

    # ---------- RAM preview: INFILL ----------
    def set_infill_preview_frames(self, frames: List[np.ndarray], start_frame: Optional[int] = None):
        """Provide a short RAM infill sequence that acts as the base when mode='infilled'."""
        if not frames:
            self.clear_infill_preview()
            return
        if start_frame is None:
            start_frame = self._current_frame_idx() if (self._fps and self._fps > 0) else 0

        self._infill_preview_frames = frames
        self._infill_preview_start_frame = int(start_frame)

        # If in 'infilled' mode, show the preview pixmap base instead of file-backed infill
        if self.show_mode == "infilled":
            self.view.set_base_visible("infilled", use_infill_preview=True)
            self._update_infill_preview_for_frame(self._current_frame_idx())

    def clear_infill_preview(self):
        self._infill_preview_frames = None
        self._infill_preview_start_frame = 0
        # If currently in 'infilled' mode, switch back to file-backed infill visibility
        if self.show_mode == "infilled":
            self.view.set_base_visible("infilled", use_infill_preview=False)

    def _update_infill_preview_for_frame(self, frame_idx: int):
        if not self._infill_preview_frames:
            return
        if self.show_mode != "infilled":
            return
        i = frame_idx - self._infill_preview_start_frame
        if i < 0 or i >= len(self._infill_preview_frames):
            # out of range; hide preview (black) to avoid stale frame
            self.view.infill_preview_item.setVisible(False)
            return
        arr = self._infill_preview_frames[i]
        pix = self._np_to_qpixmap(arr)
        if not pix.isNull():
            self.view.infill_preview_item.setVisible(True)
            self.view.set_infill_preview_pixmap(pix)

    # ---------- Slider handlers ----------
    def _slider_pressed(self): self._slider_down = True
    def _slider_released(self):
        self._slider_down = False
        self.seek(self.pos_slider.value())
        QTimer.singleShot(30, lambda: self.pos_slider.setValue(self._last_frame_ms))

    # ---------- Mode ----------
    def set_mode(self, mode: str):
        if mode == self.show_mode: return
        was_playing = (self.player_orig.playbackState() == QMediaPlayer.PlayingState)
        self.pause_all()
        self.show_mode = mode
        use_preview = (mode == "infilled" and self._infill_preview_frames is not None)
        self.view.set_base_visible(mode, use_infill_preview=use_preview)
        if was_playing: self.play_all()
        else: self._snap_all_to(self._last_frame_ms or self.player_orig.position())

    # ---------- Mask visibility/opacity ----------
    def set_mask_visible(self, vis: bool):
        # If RAM preview exists, toggle that; else toggle file-backed mask player
        has_preview = (self._mask_preview_frames is not None)
        self.view.set_mask_preview_visible(vis and has_preview)
        self.view.set_mask_visible(vis and not has_preview)

        cur = self._last_frame_ms or self.player_orig.position()
        if vis and not has_preview:
            if self.player_orig.playbackState() == QMediaPlayer.PlayingState:
                self.mask_player.setPosition(cur); self.mask_player.play()
            else:
                self.mask_player.pause(); self.mask_player.setPosition(cur)
        else:
            self.mask_player.pause()

        if vis and has_preview:
            self._update_mask_preview_for_frame(self._current_frame_idx())

    def set_mask_opacity(self, value: int):
        alpha = max(0.0, min(1.0, value / 100.0))
        self.view.set_mask_opacity(alpha)
        self.view.set_mask_preview_opacity(alpha)

    # ---------- Transport ----------
    def toggle_play(self):
        if self.player_orig.playbackState() == QMediaPlayer.PlayingState: self.pause_all()
        else: self.play_all()

    def play_all(self):
        cur = self._last_frame_ms or self.player_orig.position()
        # Master
        self.player_orig.setPosition(cur); self.player_orig.play()

        # Infilled: only play file-backed when visible and no preview
        if self.show_mode == "infilled" and self._infill_preview_frames is None:
            self.player_infill.setPosition(cur); self.player_infill.play()
        else:
            self.player_infill.pause()

        # Mask: only play when visible AND no preview
        if self.view.mask_item.isVisible() and self._mask_preview_frames is None:
            self.mask_player.setPosition(cur); self.mask_player.play()
        else:
            self.mask_player.pause()

        self._resync_timer.start()

    def pause_all(self):
        self.player_orig.pause(); self.player_infill.pause(); self.mask_player.pause()
        self._resync_timer.stop()
        self._snap_all_to(self._last_frame_ms or self.player_orig.position())

    def stop(self):
        self.seek(0)

    def seek(self, pos_ms: int):
        self.player_orig.setPosition(pos_ms)
        if self.show_mode == "infilled" and self._infill_preview_frames is None:
            self.player_infill.setPosition(pos_ms)
        if self.view.mask_item.isVisible() and self._mask_preview_frames is None:
            self.mask_player.setPosition(pos_ms)

        self._last_frame_ms = pos_ms
        if not self._slider_down: self.pos_slider.setValue(pos_ms)
        self.time_label.setText(f"{fmt_ms(pos_ms)} / {fmt_ms(self.player_orig.duration() or 0)}")

        if self._fps is not None and self._fps > 0:
            self._last_frame_idx = ms_to_frame(pos_ms, self._fps)
            self.view.overlay_item.setKeyframe(self.keyframes.get(self._last_frame_idx))

        # Update RAM previews (if present)
        self._update_mask_preview_for_frame(self._last_frame_idx)
        self._update_infill_preview_for_frame(self._last_frame_idx)

        self.view.viewport().update()

        if self.player_orig.playbackState() != QMediaPlayer.PlayingState:
            self._snap_all_to(pos_ms)

    def set_volume(self, value: int): self.audio.setVolume(max(0.0, min(1.0, value / 100.0)))

    # ---------- Frame-driven UI (decoded master frames) ----------
    def _on_master_frame_changed(self, frame):
        ts_us = frame.startTime()
        pos_ms = int(ts_us // 1000) if (ts_us is not None and ts_us >= 0) else self.player_orig.position()
        self._last_frame_ms = pos_ms

        if self._fps is not None and self._fps > 0:
            self._last_frame_idx = ms_to_frame(pos_ms, self._fps)

        if not self._slider_down:
            self.pos_slider.setValue(pos_ms)
            self.time_label.setText(f"{fmt_ms(pos_ms)} / {fmt_ms(self.player_orig.duration() or 0)}")

        self.view.overlay_item.setKeyframe(self.keyframes.get(self._last_frame_idx))

        # Advance RAM previews if present
        self._update_mask_preview_for_frame(self._last_frame_idx)
        self._update_infill_preview_for_frame(self._last_frame_idx)

    # ---------- Periodic resync while playing ----------
    def _playing_resync(self):
        if self.player_orig.playbackState() != QMediaPlayer.PlayingState: return
        master_ms = self._last_frame_ms or self.player_orig.position()

        # Infilled drift correction only if file-backed infill is active
        if self.show_mode == "infilled" and self._infill_preview_frames is None:
            d = abs((self.player_infill.position() or 0) - master_ms)
            if d > self.resync_threshold_ms: self.player_infill.setPosition(master_ms)

        # Mask drift correction only if file-backed mask is active & visible
        if self.view.mask_item.isVisible() and self._mask_preview_frames is None:
            d = abs((self.mask_player.position() or 0) - master_ms)
            if d > self.resync_threshold_ms: self.mask_player.setPosition(master_ms)

    def _snap_all_to(self, pos_ms: int, snap_orig=False):
        if snap_orig: self.player_orig.setPosition(pos_ms)
        if self.show_mode == "infilled" and self._infill_preview_frames is None:
            self.player_infill.setPosition(pos_ms)
        if self.view.mask_item.isVisible() and self._mask_preview_frames is None:
            self.mask_player.setPosition(pos_ms)

        if self._fps is not None and self._fps > 0:
            self._last_frame_idx = ms_to_frame(pos_ms, self._fps)
            self.view.overlay_item.setKeyframe(self.keyframes.get(self._last_frame_idx))

        # Update RAM previews too
        self._update_mask_preview_for_frame(self._last_frame_idx)
        self._update_infill_preview_for_frame(self._last_frame_idx)

        self.pos_slider.setValue(pos_ms)
        self.time_label.setText(f"{fmt_ms(pos_ms)} / {fmt_ms(self.player_orig.duration() or 0)}")
        self.view.viewport().update()

    # ---------- Enforce FPS on media load + poster frame ----------
    def _on_master_media_status(self, status):
        from PySide6.QtMultimedia import QMediaPlayer
        if self._poster_ready: return
        if status in (getattr(QMediaPlayer, "LoadedMedia", None),
                      getattr(QMediaPlayer, "BufferedMedia", None)):
            # enforce FPS from metadata
            if self._fps is None:
                fps = self.player_orig.metaData().value(QMediaMetaData.VideoFrameRate)
                if fps is None:
                    raise ValueError("Video FPS could not be determined from metadata. Re-encode or provide valid FPS.")
                try:
                    fps_val = float(fps)
                except Exception:
                    raise ValueError(f"Invalid FPS metadata value: {fps!r}")
                if fps_val <= 0:
                    raise ValueError(f"Non-positive FPS in metadata: {fps_val}")
                self._fps = fps_val
                print(f"[INFO] FPS (metadata): {self._fps:.3f}")

            if self.player_orig.playbackState() != QMediaPlayer.PlayingState:
                QTimer.singleShot(0, self._show_first_master_frame)

    def _show_first_master_frame(self):
        from PySide6.QtMultimedia import QMediaPlayer
        if self.player_orig.playbackState() == QMediaPlayer.PlayingState:
            self._poster_ready = True; return
        try:
            self.player_orig.setPosition(0)
            self.player_orig.play()
            QTimer.singleShot(0, self.player_orig.pause)
        except Exception:
            self.player_orig.setPosition(0)

        self._last_frame_ms = 0
        if self._fps is not None and self._fps > 0:
            self._last_frame_idx = 0
            self.view.overlay_item.setKeyframe(self.keyframes.get(0))

        # Ensure base visibility reflects current mode + preview state
        use_preview = (self.show_mode == "infilled" and self._infill_preview_frames is not None)
        self.view.set_base_visible(self.show_mode, use_infill_preview=use_preview)

        # Keep file followers at t=0 (if active)
        if self.show_mode == "infilled" and self._infill_preview_frames is None:
            self.player_infill.pause(); self.player_infill.setPosition(0)
        if self.view.mask_item.isVisible() and self._mask_preview_frames is None:
            self.mask_player.pause(); self.mask_player.setPosition(0)

        # Update RAM previews for frame 0
        self._update_mask_preview_for_frame(0)
        self._update_infill_preview_for_frame(0)

        self.pos_slider.setValue(0)
        self.time_label.setText(f"{fmt_ms(0)} / {fmt_ms(self.player_orig.duration() or 0)}")
        self.view.viewport().update()
        self._poster_ready = True

    # ---------- Master UI handlers ----------
    def _on_master_position_changed(self, position):
        if not self._slider_down: self.pos_slider.setValue(position)
        self.positionChanged.emit(position)

    def _on_master_duration_changed(self, duration):
        self.pos_slider.setRange(0, duration)
        self.time_label.setText(f"{fmt_ms(self._last_frame_ms or self.player_orig.position() or 0)} / {fmt_ms(duration)}")

    def _on_master_state_changed(self, state):
        self.play_btn.setIcon(self.style().standardIcon(
            QStyle.SP_MediaPause if state == QMediaPlayer.PlayingState else QStyle.SP_MediaPlay
        ))
        if state != QMediaPlayer.PlayingState:
            self._snap_all_to(self._last_frame_ms or self.player_orig.position(), True)

    # ---------- Keyframe helpers (by frame index) ----------
    def _current_frame_idx(self) -> int:
        return self._last_frame_idx

    def _get_or_make_kf(self, frame_idx: Optional[int] = None, add_chip=True) -> Keyframe:
        if frame_idx is None: frame_idx = self._current_frame_idx()
        kf = self.keyframes.get(frame_idx)
        if not kf:
            kf = Keyframe(frame_idx=frame_idx); self.keyframes[frame_idx] = kf
            if add_chip: self._add_kf_chip(kf, with_icon=True)
        return kf

    def _add_kf_chip(self, kf: Keyframe, with_icon: bool = True):
        fi = kf.frame_idx
        if self._find_kf_item_by_frame(fi) is not None: return
        insert_row = self.kf_list.count()
        for row in range(self.kf_list.count()):
            it = self.kf_list.item(row); t = it.data(Qt.UserRole)
            if t is not None and fi < t: insert_row = row; break

        label = f"#{fi}"
        if self._fps: label = f"{fmt_ms(frame_to_ms(fi, self._fps))}"
        item = QListWidgetItem(label)
        item.setToolTip(f"Keyframe @ frame {fi}" + (f" ({label})" if self._fps else ""))
        item.setData(Qt.UserRole, fi)

        if with_icon:
            icon = self.view.grabThumbWithOverlay(kf, self.kf_list.iconSize())
            if icon is not None: item.setIcon(icon)
            else: self._pending_icon_updates.add(fi)
        self.kf_list.insertItem(insert_row, item)

    def _remove_kf_chip(self, frame_idx: int):
        for i in range(self.kf_list.count()):
            it = self.kf_list.item(i)
            if it.data(Qt.UserRole) == frame_idx:
                self.kf_list.takeItem(i); break
        self._pending_icon_updates.discard(frame_idx)

    def _find_kf_item_by_frame(self, frame_idx: int) -> Optional[QListWidgetItem]:
        for i in range(self.kf_list.count()):
            it = self.kf_list.item(i)
            if it.data(Qt.UserRole) == frame_idx: return it
        return None

    def _ensure_icon_for_frame(self, frame_idx: int):
        it = self._find_kf_item_by_frame(frame_idx)
        if it:
            icon = self.view.grabThumbWithOverlay(self.keyframes.get(frame_idx), self.kf_list.iconSize())
            if icon is not None: it.setIcon(icon)

    def _prune_if_empty(self, kf: Keyframe):
        if not kf.pos_clicks and not kf.neg_clicks and not kf.rects:
            fi = kf.frame_idx; self.keyframes.pop(fi, None); self._remove_kf_chip(fi)
            self.view.overlay_item.setKeyframe(self.keyframes.get(self._current_frame_idx())); self.view.viewport().update()
        else:
            self._ensure_icon_for_frame(kf.frame_idx)

    # Annotation slots (by frame) — include object id
    def _on_add_positive(self, nx: float, ny: float):
        kf = self._get_or_make_kf(); kf.pos_clicks.append((nx, ny, self.current_object))
        self.view.overlay_item.setKeyframe(kf); self._ensure_icon_for_frame(kf.frame_idx)
        print(f"[KF frame {kf.frame_idx}] +POS obj={self.current_object} ({nx:.3f},{ny:.3f})")

    def _on_add_negative(self, nx: float, ny: float):
        kf = self._get_or_make_kf(); kf.neg_clicks.append((nx, ny, self.current_object))
        self.view.overlay_item.setKeyframe(kf); self._ensure_icon_for_frame(kf.frame_idx)
        print(f"[KF frame {kf.frame_idx}] -NEG obj={self.current_object} ({nx:.3f},{ny:.3f})")

    def _on_add_rect(self, nx: float, ny: float, nw: float, nh: float):
        kf = self._get_or_make_kf(); kf.rects.append((nx, ny, nw, nh, self.current_object))
        self.view.overlay_item.setKeyframe(kf); self._ensure_icon_for_frame(kf.frame_idx)
        print(f"[KF frame {kf.frame_idx}] RECT obj={self.current_object} ({nx:.3f},{ny:.3f},{nw:.3f},{nh:.3f})")

    def _on_delete_at(self, nx: float, ny: float):
        fi = self._current_frame_idx(); kf = self.keyframes.get(fi)
        if not kf: return
        rect = self.view.overlay_item.boundingRect()
        W = max(1.0, rect.width()); H = max(1.0, rect.height()); R_px = 8.0
        def pop_near_point(points):
            for i, (px, py, pobj) in enumerate(points):
                dx = abs(nx - px) * W; dy = abs(ny - py) * H
                if (dx*dx + dy*dy) ** 0.5 <= R_px: points.pop(i); return True
            return False
        if pop_near_point(kf.pos_clicks) or pop_near_point(kf.neg_clicks):
            print(f"[KF frame {kf.frame_idx}] deleted point near ({nx:.3f},{ny:.3f})")
            self.view.overlay_item.setKeyframe(kf); self._prune_if_empty(kf); return
        def near_rect_edge(nx_, ny_, r):
            rx, ry, rw, rh, robj = r; Rx = R_px / W; Ry = R_px / H
            left   = abs(nx_ - rx)      <= Rx and (ry - Ry) <= ny_ <= (ry + rh + Ry)
            right  = abs(nx_ - (rx+rw)) <= Rx and (ry - Ry) <= ny_ <= (ry + rh + Ry)
            top    = abs(ny_ - ry)      <= Ry and (rx - Rx) <= nx_ <= (rx + rw + Rx)
            bottom = abs(ny_ - (ry+rh)) <= Ry and (rx - Rx) <= nx_ <= (rx + rw + Rx)
            return left or right or top or bottom
        for i, r in enumerate(kf.rects):
            if near_rect_edge(nx, ny, r):
                kf.rects.pop(i); print(f"[KF frame {kf.frame_idx}] deleted rectangle")
                self.view.overlay_item.setKeyframe(kf); self._prune_if_empty(kf); return
        self._ensure_icon_for_frame(kf.frame_idx)

    # Keyframe list click -> seek (by frame index)
    def _on_kf_clicked(self, item: QListWidgetItem):
        fi = item.data(Qt.UserRole)
        if fi is not None and self._fps:
            pos_ms = frame_to_ms(int(fi), self._fps)
            self.seek(pos_ms)
            QTimer.singleShot(80, lambda: self._ensure_icon_for_frame(int(fi)))

    # Save / Load JSON (frame-based) with object ids
    def to_json_obj(self, video_path: Optional[str]) -> dict:
        def pts_to_list(pts):
            return [{"x": x, "y": y, "obj": obj} for (x, y, obj) in pts]
        def rects_to_list(rects):
            return [{"x": x, "y": y, "w": w, "h": h, "obj": obj} for (x, y, w, h, obj) in rects]

        return {
            "video": str(video_path) if video_path else None,
            "fps": self._fps,
            "keyframes": [
                {
                    "frame_idx": k.frame_idx,
                    "pos_clicks": pts_to_list(k.pos_clicks),
                    "neg_clicks": pts_to_list(k.neg_clicks),
                    "rects": rects_to_list(k.rects),
                }
                for _, k in sorted(self.keyframes.items())
            ],
        }

    def load_from_json_obj(self, obj: dict):
        self.keyframes.clear(); self.kf_list.clear(); self._pending_icon_updates.clear()

        for entry in obj.get("keyframes", []):
            fi = int(entry["frame_idx"])
            def parse_pts(L):
                out = []
                for v in L:
                    if isinstance(v, dict):
                        out.append((float(v["x"]), float(v["y"]), int(v.get("obj", 1))))
                    else:
                        x, y = v[0], v[1]
                        out.append((float(x), float(y), 1))
                return out
            def parse_rects(L):
                out = []
                for v in L:
                    if isinstance(v, dict):
                        out.append((float(v["x"]), float(v["y"]), float(v["w"]), float(v["h"]), int(v.get("obj", 1))))
                    else:
                        x, y, w, h = v[0], v[1], v[2], v[3]
                        out.append((float(x), float(y), float(w), float(h), 1))
                return out

            kf = Keyframe(
                frame_idx=fi,
                pos_clicks=parse_pts(entry.get("pos_clicks", [])),
                neg_clicks=parse_pts(entry.get("neg_clicks", [])),
                rects=parse_rects(entry.get("rects", [])),
            )
            self.keyframes[kf.frame_idx] = kf
            self._add_kf_chip(kf, with_icon=False)
            self._pending_icon_updates.add(kf.frame_idx)

        QTimer.singleShot(100, lambda: [self._ensure_icon_for_frame(t) for t in list(self._pending_icon_updates)])
        self.view.overlay_item.setKeyframe(self.keyframes.get(self._current_frame_idx()))
        self.view.viewport().update()


# ---------- SideDock (includes Object selector) ----------
class SideDock(QDockWidget):
    objectChanged = Signal(int)
    addObjectRequested = Signal()

    def __init__(self, parent=None):
        super().__init__("Tools", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea)
        self.container = QWidget(self); lay = QVBoxLayout(self.container)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(10)

        lay.addWidget(QLabel("Annotate Tool"))

        # Object group
        obj_group = QGroupBox("Object")
        obj_lay = QHBoxLayout(obj_group)
        self.object_combo = QComboBox()
        self.object_combo.addItems(["Object 1", "Object 2", "Object 3"])
        self.object_combo.setCurrentIndex(0)
        self.btn_add_object = QPushButton("Add Object")
        obj_lay.addWidget(self.object_combo, 1)
        obj_lay.addWidget(self.btn_add_object)
        lay.addWidget(obj_group)

        # Tool selection
        self.btn_pos = QToolButton(text="Positive"); self.btn_pos.setCheckable(True)
        self.btn_neg = QToolButton(text="Negative"); self.btn_neg.setCheckable(True)
        self.btn_rect = QToolButton(text="Rectangle"); self.btn_rect.setCheckable(True)
        group = QButtonGroup(self); group.setExclusive(True)
        group.addButton(self.btn_pos, OverlayItem.TOOL_POS)
        group.addButton(self.btn_neg, OverlayItem.TOOL_NEG)
        group.addButton(self.btn_rect, OverlayItem.TOOL_RECT)
        self.btn_pos.setChecked(True)
        row = QHBoxLayout(); row.addWidget(self.btn_pos); row.addWidget(self.btn_neg); row.addWidget(self.btn_rect); lay.addLayout(row)

        lay.addSpacing(8); lay.addWidget(QLabel("Files"))
        self.open_color_btn = QPushButton("Open Color Video…")
        self.open_infilled_btn = QPushButton("Open Infilled Video…")
        self.open_mask_btn = QPushButton("Open Mask Video…")
        lay.addWidget(self.open_color_btn); lay.addWidget(self.open_infilled_btn); lay.addWidget(self.open_mask_btn)

        mode_group = QGroupBox("View Mode"); mg_lay = QHBoxLayout(mode_group)
        self.rb_original = QRadioButton("Original"); self.rb_infilled = QRadioButton("Infilled")
        self.rb_original.setChecked(True); mg_lay.addWidget(self.rb_original); mg_lay.addWidget(self.rb_infilled)
        lay.addWidget(mode_group)

        mask_group = QGroupBox("Mask Overlay"); mk_lay = QVBoxLayout(mask_group)
        self.cb_show_mask = QCheckBox("Show Mask"); self.cb_show_mask.setChecked(False)
        op_row = QHBoxLayout(); op_row.addWidget(QLabel("Opacity"))
        self.mask_opacity = QSlider(Qt.Horizontal); self.mask_opacity.setRange(0, 100); self.mask_opacity.setValue(40)
        op_row.addWidget(self.mask_opacity)
        mk_lay.addWidget(self.cb_show_mask); mk_lay.addLayout(op_row)
        lay.addWidget(mask_group)

        # --- compact action rows with preview buttons ---
        lay.addStretch(1)

        # Row 1: Mask (Generate + Preview)
        row_mask = QHBoxLayout()
        self.btn_generate_mask = QPushButton("Generate Mask")
        self.btn_generate_mask.setFixedWidth(140)     # narrower
        self.btn_preview_mask  = QPushButton("Preview")
        self.btn_preview_mask.setFixedWidth(100)      # new
        row_mask.addWidget(self.btn_generate_mask)
        row_mask.addWidget(self.btn_preview_mask)
        row_mask.addStretch(1)
        lay.addLayout(row_mask)

        # Row 2: Infill (Make Vanish + Preview)
        row_infill = QHBoxLayout()
        self.btn_make_vanish   = QPushButton("Make Vanish")
        self.btn_make_vanish.setFixedWidth(140)       # narrower
        self.btn_preview_infill = QPushButton("Preview")
        self.btn_preview_infill.setFixedWidth(100)    # new
        row_infill.addWidget(self.btn_make_vanish)
        row_infill.addWidget(self.btn_preview_infill)
        row_infill.addStretch(1)
        lay.addLayout(row_infill)


        # Optional: expose preview clicks upward
        self.btn_preview_mask.clicked.connect(lambda: getattr(self.parent(), "on_preview_mask_clicked", lambda: None)())
        self.btn_preview_infill.clicked.connect(lambda: getattr(self.parent(), "on_preview_infill_clicked", lambda: None)())


        self.setWidget(self.container)
        self.tool_group = group

        # Signals
        self.object_combo.currentIndexChanged.connect(self._on_object_changed)
        self.btn_add_object.clicked.connect(self._on_add_object)

    def _on_object_changed(self, idx: int):
        self.objectChanged.emit(idx + 1)  # 1-based

    def _on_add_object(self):
        count = self.object_combo.count()
        next_label = f"Object {count + 1}"
        self.object_combo.addItem(next_label)
        self.object_combo.setCurrentIndex(count)
        self.addObjectRequested.emit()


# ---------- MainWindow ----------
class MainWindow(QMainWindow):
    def __init__(self, color_video: Optional[str] = None,
                 mask_video: Optional[str] = None,
                 infilled_video: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("VideoVanish – Playing Followers + Resync (Frame Keyframes + Objects + RAM Previews)")
        self.resize(1280, 820)

        self.current_video_path: Optional[Path] = Path(color_video) if color_video else None
        self.infilled_video_path: Optional[Path] = None
        self.mask_video_path: Optional[Path] = None

        self.player_widget = VideoPlayer(self); self.setCentralWidget(self.player_widget)
        self.tools = SideDock(self); self.addDockWidget(Qt.RightDockWidgetArea, self.tools)

        self._apply_dark_theme()
        self._make_menu(); self._make_toolbar()

        # Wire tools
        self.tools.tool_group.idToggled.connect(self._on_tool_changed)
        self.tools.objectChanged.connect(self.player_widget.set_current_object)
        self.tools.addObjectRequested.connect(lambda: None)

        self.tools.open_color_btn.clicked.connect(self.open_color_video)
        self.tools.open_infilled_btn.clicked.connect(self.open_infilled_video)
        self.tools.open_mask_btn.clicked.connect(self.open_mask_video)

        self.tools.rb_original.toggled.connect(lambda checked: checked and self.set_mode("original"))
        self.tools.rb_infilled.toggled.connect(lambda checked: checked and self.set_mode("infilled"))
        self.tools.cb_show_mask.toggled.connect(self.player_widget.set_mask_visible)
        self.tools.mask_opacity.valueChanged.connect(self.player_widget.set_mask_opacity)

        self.tools.btn_generate_mask.clicked.connect(self.generate_mask)
        self.tools.btn_make_vanish.clicked.connect(self.make_vanish)

        # Autoload from CLI
        if color_video: self.load_color_video(color_video)
        if mask_video:
            self.mask_video_path = Path(mask_video)
            self.player_widget.load_mask(mask_video)
            self.tools.cb_show_mask.setChecked(True)
            self.player_widget.set_mask_visible(True)
        if infilled_video:
            self.infilled_video_path = Path(infilled_video)
            self.player_widget.load_infilled(infilled_video)

    def _annotations_dict_for_frames(self, frame_indices: list[int] | None = None) -> dict:
        """
        Build the annotations dict expected by masker:

        If frame_indices is None, include *all* keyframes with annotations.
        Otherwise, only include frames in frame_indices.
        """
        vp = self.player_widget
        out_keyframes = []

        # which frames to check?
        frames_to_check = (
            frame_indices if frame_indices is not None
            else sorted(vp.keyframes.keys())
        )

        for fi in frames_to_check:
            kf = vp.keyframes.get(fi)
            if not kf:
                continue

            def pack_pts(pts):
                return [{"x": float(x), "y": float(y), "obj": int(obj)}
                        for (x, y, obj) in pts]

            def pack_rects(rects):
                return [{"x": float(x), "y": float(y), "w": float(w), "h": float(h), "obj": int(obj)}
                        for (x, y, w, h, obj) in rects]

            has_any = (len(kf.pos_clicks) + len(kf.neg_clicks) + len(kf.rects)) > 0
            if not has_any:
                continue

            out_keyframes.append({
                "frame_idx": int(kf.frame_idx),
                "pos_clicks": pack_pts(kf.pos_clicks),
                "neg_clicks": pack_pts(kf.neg_clicks),
                "rects":     pack_rects(kf.rects),
            })

        return {"keyframes": out_keyframes}
   
    # Stubs for processing
    def generate_mask(self):
        frames, fps = tools.load_video_frames_from_path(self.current_video_path, start_frame=0, max_frames=-1)
        H0, W0 = frames[0].shape[:2]
        annotations = self._annotations_dict_for_frames(None)
        mask_frames = sam2_masker.run_sam2_on_frames(frames, annotations)

        out_video = str(self.current_video_path) + "_generated_mask.mkv"
        tools.write_video_frames_to_path(out_video, mask_frames, fps, H0, W0)
        self.mask_video_path = Path(out_video)
        self.player_widget.load_mask(out_video)
        self.player_widget.set_mask_visible(True)
        QMessageBox.information(self, "Mask generated", "The mask video file: "+out_video+" generated")

    def make_vanish(self):
        frames, fps = tools.load_video_frames_from_path(self.current_video_path)
        H0, W0 = frames[0].shape[:2]
        mask_frames, fps = tools.load_video_frames_from_path(str(self.mask_video_path))

        infill_frames = diffuerase.run_infill_on_frames(frames, mask_frames)
        out_video = str(self.current_video_path) + "_vanished.mkv"
        tools.write_video_frames_to_path(out_video, infill_frames, fps, H0, W0)
        print("generated infill")

        self.infilled_video_path = Path(out_video)
        self.player_widget.load_infilled(out_video)
        self.tools.rb_infilled.setChecked(True)
        self.set_mode("infilled")

    def on_preview_mask_clicked(self):
        # TODO: trigger your RAM mask preview generation/show here
        frames, fps = tools.load_video_frames_from_path(self.current_video_path, start_frame=self.player_widget._last_frame_idx, max_frames=1)
        annotations = self._annotations_dict_for_frames([self.player_widget._last_frame_idx])#only do curent frame
        if not annotations["keyframes"]:
            QMessageBox.warning(self, "No keyframe selected", "You have to create a keyframe by adding annotations to do a preview")
            return
        mask_frames = sam2_masker.run_sam2_on_frames(frames, annotations)
        self.player_widget.set_mask_preview_frames(mask_frames)

    def on_preview_infill_clicked(self):
        # TODO: trigger your RAM infill preview generation/show here
        frames, fps = tools.load_video_frames_from_path(self.current_video_path, start_frame=self.player_widget._last_frame_idx, max_frames=22)
        H0, W0 = frames[0].shape[:2]
        mask_frames, fps = tools.load_video_frames_from_path(str(self.mask_video_path), start_frame=self.player_widget._last_frame_idx, max_frames=22)

        infill_frames = diffuerase.run_infill_on_frames(frames, mask_frames)
        print("generated infill")
        self.player_widget.set_infill_preview_frames(infill_frames)
        self.tools.rb_infilled.setChecked(True)
        self.set_mode("infilled")


    # Menus/toolbar
    def _make_menu(self):
        m = self.menuBar(); file_menu = m.addMenu("&File")
        act_open = QAction("Open Color Video…", self); act_open.triggered.connect(self.open_color_video); file_menu.addAction(act_open)
        act_open_inf = QAction("Open Infilled Video…", self); act_open_inf.triggered.connect(self.open_infilled_video); file_menu.addAction(act_open_inf)
        act_open_mask = QAction("Open Mask Video…", self); act_open_mask.triggered.connect(self.open_mask_video); file_menu.addAction(act_open_mask)
        file_menu.addSeparator()
        act_save = QAction("Save Annotations…", self); act_save.triggered.connect(self.save_annotations); file_menu.addAction(act_save)
        act_load = QAction("Load Annotations…", self); act_load.triggered.connect(self.load_annotations); file_menu.addAction(act_load)
        file_menu.addSeparator(); act_quit = QAction("Quit", self); act_quit.triggered.connect(self.close); file_menu.addAction(act_quit)

    def _make_toolbar(self):
        tb = QToolBar("Main"); tb.setIconSize(QSize(18, 18)); self.addToolBar(Qt.TopToolBarArea, tb)
        act_open = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), "Open Color Video…", self); act_open.triggered.connect(self.open_color_video); tb.addAction(act_open)
        act_play = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), "Play/Pause (Space)", self); act_play.triggered.connect(self.player_widget.toggle_play); tb.addAction(act_play)
        act_stop = QAction(self.style().standardIcon(QStyle.SP_MediaStop), "Stop", self); act_stop.triggered.connect(self.player_widget.stop); tb.addAction(act_stop)
        tb.addSeparator(); vol_label = QLabel("Vol"); vol_slider = QSlider(Qt.Horizontal)
        vol_slider.setRange(0, 100); vol_slider.setValue(90); vol_slider.setFixedWidth(120); vol_slider.valueChanged.connect(self.player_widget.set_volume)
        tb.addWidget(vol_label); tb.addWidget(vol_slider)

    # Dark theme
    def _apply_dark_theme(self):
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

    # File helpers
    def open_color_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Color Video", "", "Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*)")
        if path: self.load_color_video(path)

    def load_color_video(self, path: str):
        self.current_video_path = Path(path)
        self.player_widget.load_original(path)
        self.tools.rb_original.setChecked(True); self.set_mode("original")
        self.setWindowTitle(f"VideoVanish – {self.current_video_path.name}")

    def open_infilled_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Infilled Video", "", "Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*)")
        if path:
            self.infilled_video_path = Path(path); self.player_widget.load_infilled(path)
            if self.tools.rb_infilled.isChecked(): self.set_mode("infilled")

    def open_mask_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Mask Video", "", "Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*)")
        if path:
            self.mask_video_path = Path(path); self.player_widget.load_mask(path)
            if self.tools.cb_show_mask.isChecked(): self.player_widget.set_mask_visible(True)

    # Mode
    def set_mode(self, mode: str):
        if mode not in ("original", "infilled"): return
        if mode == "infilled" and not self.infilled_video_path and self.player_widget._infill_preview_frames is None:
            QMessageBox.warning(self, "Infilled missing", "Load an infilled video first, or set infill preview frames.")
            self.tools.rb_original.setChecked(True); return
        self.player_widget.set_mode(mode)

    # Save/Load annotations (frame-based)
    def save_annotations(self):
        base = self.current_video_path or Path.cwd() / "annotations"
        path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", str(base.with_suffix(".annotations.json")), "JSON Files (*.json);;All Files (*)")
        if not path: return
        obj = self.player_widget.to_json_obj(str(self.current_video_path) if self.current_video_path else None)
        try:
            with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2)
            QMessageBox.information(self, "Saved", f"Annotations saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def load_annotations(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Annotations", "", "JSON Files (*.json);;All Files (*)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f: obj = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read JSON:\n{e}"); return

        video_in_json = obj.get("video")
        if video_in_json and (not self.current_video_path or str(self.current_video_path) != video_in_json):
            if Path(video_in_json).exists(): self.load_color_video(video_in_json)
            else:
                QMessageBox.warning(self, "Video Missing",
                    f"The JSON references a video that doesn't exist:\n{video_in_json}\n"
                    f"Annotations will load but thumbnails may be off.")
        self.player_widget.load_from_json_obj(obj); QMessageBox.information(self, "Loaded", "Annotations loaded.")

    # Overlay tool selection
    def _on_tool_changed(self, tool_id: int, checked: bool):
        if checked: self.player_widget.view.overlay_item.setTool(tool_id)

    # Shortcuts
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space: self.player_widget.toggle_play(); event.accept(); return
        super().keyPressEvent(event)


# ---------- CLI & main ----------
def parse_args(argv):
    p = argparse.ArgumentParser(description="VideoVanish (Frame keyframes + Objects + RAM Previews)")
    p.add_argument("--color_video", type=str, default=None, help="Path to color/original video")
    p.add_argument("--mask_video", type=str, default=None, help="Path to mask video")
    p.add_argument("--infilled_video", type=str, default=None, help="Path to infilled video")
    args, _ = p.parse_known_args(argv); return args

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

