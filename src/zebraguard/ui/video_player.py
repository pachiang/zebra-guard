"""OpenCV 自繪影片播放器(Qt)。

為何不用 QMediaPlayer:
  · 需要精準 loop 一個時間區間(±3 秒)
  · 需要在每幀上疊 bbox / mask overlay(QMediaPlayer 很難做)
  · 需要穩定的 seek(QMediaPlayer 在 Windows 上 seek 有時不可靠)

Layout:
  · `VideoCanvas` (QLabel) 佔滿整個 PlayerView 區域
  · 浮動控制:
      - `_ControlsBar` 貼底,hover 時淡入,播放時無活動 2.5s 後淡出
      - `_LoopToggle`  貼右上,小圖示 on/off
      - `_PausedBadge` 貼正中,暫停時浮現(播放 icon 當作「暫停中」提示)
  · 點影片 = 切換播放/暫停
  · 解碼在 `DecoderThread` 子執行緒;frame_ready signal 把 QImage 推回主執行緒畫
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QRect,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPixmap,
    QPolygon,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)


def _bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _fmt_time(sec: float) -> str:
    m = int(sec) // 60
    s = sec - 60 * m
    return f"{m:02d}:{s:05.2f}"


# =========================================================================
# Decoder thread
# =========================================================================


class DecoderThread(QThread):
    frame_ready = Signal(QImage, int, float)  # image, frame_idx, time_sec
    ended = Signal()
    opened = Signal(int, float)  # total_frames, fps

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._path: Path | None = None
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0
        self._total: int = 0
        self._playing: bool = False
        self._stopping: bool = False
        self._loop_start: int = 0
        self._loop_end: int = -1
        self._target_seek: int | None = None
        self._speed: float = 1.0

    @Slot(str)
    def open(self, path: str) -> None:
        self._close()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._path = Path(path)
        self._playing = False
        self._target_seek = 0
        self.opened.emit(self._total, self._fps)

    @Slot(bool)
    def set_playing(self, flag: bool) -> None:
        self._playing = flag

    @Slot(int)
    def seek_frame(self, frame_idx: int) -> None:
        self._target_seek = max(0, min(self._total - 1, int(frame_idx)))

    @Slot(int, int)
    def set_loop(self, start_frame: int, end_frame: int) -> None:
        self._loop_start = max(0, start_frame)
        self._loop_end = int(end_frame)

    @Slot(float)
    def set_speed(self, speed: float) -> None:
        self._speed = max(0.25, min(4.0, speed))

    def request_stop(self) -> None:
        self._stopping = True

    def run(self) -> None:
        last_t = time.monotonic()
        while not self._stopping:
            if self._cap is None:
                self.msleep(20)
                continue

            if self._target_seek is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._target_seek)
                cur_idx = self._target_seek
                self._target_seek = None
                ok, frame = self._cap.read()
                if ok:
                    self.frame_ready.emit(
                        _bgr_to_qimage(frame), cur_idx, cur_idx / self._fps
                    )
                last_t = time.monotonic()
                continue

            if not self._playing:
                self.msleep(20)
                continue

            cur_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            if self._loop_end >= 0 and cur_idx >= self._loop_end:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._loop_start)
                cur_idx = self._loop_start
                last_t = time.monotonic()

            ok, frame = self._cap.read()
            if not ok:
                if self._loop_end >= 0:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._loop_start)
                    last_t = time.monotonic()
                    continue
                self._playing = False
                self.ended.emit()
                continue

            self.frame_ready.emit(
                _bgr_to_qimage(frame), cur_idx, cur_idx / self._fps
            )

            target_interval = 1.0 / max(1.0, self._fps * self._speed)
            elapsed = time.monotonic() - last_t
            sleep_for = target_interval - elapsed
            if sleep_for > 0:
                self.msleep(int(sleep_for * 1000))
            last_t = time.monotonic()

        self._close()

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# =========================================================================
# Canvas — 只負責顯示 pixmap + 向上冒泡 hover / click
# =========================================================================


class VideoCanvas(QLabel):
    clicked = Signal()
    cursor_moved = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(480, 270))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #000; border-radius: 10px;")
        self.setMouseTracking(True)
        self._src: QImage | None = None

    def set_image(self, img: QImage) -> None:
        self._src = img
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._src is None or self._src.isNull():
            return
        pm = QPixmap.fromImage(self._src)
        scaled = pm.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self.cursor_moved.emit()
        super().mouseMoveEvent(event)


# =========================================================================
# Overlay widgets
# =========================================================================


class _PlayPauseButton(QPushButton):
    """Overlay 上的 play / pause 圓按鈕;不使用 emoji,改自繪三角形與兩條 bar。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(36, 36)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._playing = False
        self.setStyleSheet(
            """
            QPushButton {
                background: rgba(0, 0, 0, 130);
                border: 1px solid rgba(255, 255, 255, 45);
                border-radius: 18px;
                padding: 0;
            }
            QPushButton:hover {
                background: rgba(245, 165, 36, 200);
                border-color: #f5a524;
            }
            """
        )

    def set_playing(self, playing: bool) -> None:
        if self._playing == playing:
            return
        self._playing = playing
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2

        # hover 時換深色 icon,呼應橘底
        icon_color = QColor("#1a1200") if self.underMouse() else QColor(255, 255, 255, 235)
        p.setBrush(icon_color)
        p.setPen(Qt.PenStyle.NoPen)

        if self._playing:
            bar_w = 4
            bar_h = 13
            gap = 4
            p.drawRoundedRect(cx - gap // 2 - bar_w, cy - bar_h // 2, bar_w, bar_h, 1, 1)
            p.drawRoundedRect(cx + gap // 2, cy - bar_h // 2, bar_w, bar_h, 1, 1)
        else:
            # 稍微偏右讓視覺重心平衡
            size = 13
            pts = QPolygon([
                QPoint(cx - size // 3, cy - size // 2 - 1),
                QPoint(cx - size // 3, cy + size // 2 + 1),
                QPoint(cx + size * 2 // 3, cy),
            ])
            p.drawPolygon(pts)
        p.end()


class _PausedBadge(QWidget):
    """正中央的「暫停中」浮水印:半透明深色圓 + 兩條白 bar。

    暫停時顯示、播放時隱藏。click-through(不攔滑鼠)。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.hide()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r = min(w, h) // 2 - 2
        cx, cy = w // 2, h // 2

        # 外圈半透明黑
        p.setBrush(QColor(0, 0, 0, 150))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPoint(cx, cy), r, r)

        # 兩條白色 bar(pause 圖示)
        p.setBrush(QColor(255, 255, 255, 235))
        bar_w = max(6, r // 5)
        bar_h = int(r * 1.2)
        gap = max(6, r // 4)
        p.drawRoundedRect(
            cx - gap // 2 - bar_w, cy - bar_h // 2, bar_w, bar_h, 2, 2
        )
        p.drawRoundedRect(
            cx + gap // 2, cy - bar_h // 2, bar_w, bar_h, 2, 2
        )
        p.end()


class _OverlaySlider(QSlider):
    """播放進度 slider。

    - 扁平、透明 groove + amber fill
    - 按下即 seek(不用等放開)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setMinimumHeight(18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(255,255,255,55);
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #f5a524;
                border-radius: 2px;
            }
            QSlider::add-page:horizontal {
                background: rgba(255,255,255,55);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #f5a524;
                width: 12px;
                height: 12px;
                margin: -5px 0;
                border-radius: 6px;
                border: 1px solid #1a1200;
            }
            QSlider::handle:horizontal:hover {
                background: #ffb534;
            }
            """
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """點擊任意位置直接 seek 到那點(Qt 預設只能按住 handle 拖)。"""
        if event.button() == Qt.MouseButton.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            sr = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self
            )
            if not sr.contains(event.position().toPoint()):
                # jump-to-click
                if self.orientation() == Qt.Orientation.Horizontal:
                    pos = event.position().toPoint().x()
                    span = self.width() - sr.width()
                    value = self.minimum() + (self.maximum() - self.minimum()) * max(
                        0, min(span, pos - sr.width() // 2)
                    ) / max(1, span)
                    self.setSliderPosition(int(round(value)))
        super().mousePressEvent(event)


class _ControlsBar(QFrame):
    """底部浮動控制列:play/pause + time + slider。

    背景:透明 → 深色漸層(下緣),讓字在亮影片上也看得見。
    """

    play_toggle = Signal()
    seek_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PlayerControls")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            QFrame#PlayerControls {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0,0,0,0),
                    stop:0.35 rgba(0,0,0,120),
                    stop:1 rgba(0,0,0,190));
                border: none;
            }
            QLabel { color: rgba(255,255,255,235); font-size: 12px; }
            """
        )

        self.play_btn = _PlayPauseButton(self)
        self.play_btn.clicked.connect(self.play_toggle.emit)

        self.time_label = QLabel("00:00.00 / 00:00.00", self)
        self.time_label.setMinimumWidth(150)

        self.slider = _OverlaySlider(self)
        self.slider.setTracking(True)
        self.slider.sliderMoved.connect(self.seek_requested.emit)
        self.slider.sliderPressed.connect(self._on_slider_press)
        self._dragging = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 14, 12, 8)
        layout.setSpacing(10)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.time_label)

    def _on_slider_press(self) -> None:
        self._dragging = True
        self.seek_requested.emit(self.slider.value())
        # 放開時 mouseRelease 不發訊號,但在 sliderReleased 時收到
        self.slider.sliderReleased.connect(self._on_slider_release)

    def _on_slider_release(self) -> None:
        self._dragging = False
        try:
            self.slider.sliderReleased.disconnect(self._on_slider_release)
        except (RuntimeError, TypeError):
            pass

    def set_play_icon(self, playing: bool) -> None:
        self.play_btn.set_playing(playing)

    def set_time(self, current_sec: float, total_sec: float) -> None:
        self.time_label.setText(
            f"{_fmt_time(current_sec)}  /  {_fmt_time(total_sec)}"
        )

    def set_slider_range(self, total_frames: int) -> None:
        self.slider.setRange(0, max(0, total_frames - 1))

    def set_slider_value(self, frame: int) -> None:
        if self._dragging:
            return
        self.slider.blockSignals(True)
        self.slider.setValue(frame)
        self.slider.blockSignals(False)


# =========================================================================
# Main PlayerView
# =========================================================================


class PlayerView(QWidget):
    """包 canvas + overlay 控制,對外介面與舊版一致。"""

    time_changed = Signal(float)  # seconds

    _HIDE_DELAY_MS = 2500

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PlayerView")
        self.setMouseTracking(True)

        # Canvas 佔滿整個區域
        self.canvas = VideoCanvas(self)
        self.canvas.clicked.connect(self.toggle)
        self.canvas.cursor_moved.connect(self._on_cursor_activity)

        # Overlay children(position in resizeEvent)
        self.controls = _ControlsBar(self)
        self.controls.play_toggle.connect(self.toggle)
        self.controls.seek_requested.connect(self._on_slider_seek)
        self.controls.hide()

        # Loop 控制已搬到 Review view 的 EventTimeline 右側(由 Review 透過
        # set_loop_enabled() 打開 / 關閉;player 自己不再有按鈕)
        self._loop_enabled = True

        self.paused_badge = _PausedBadge(self)

        # Decoder
        self._thread = DecoderThread(self)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.opened.connect(self._on_opened)
        self._thread.ended.connect(self._on_ended)
        self._thread.start()

        # State
        self._fps: float = 30.0
        self._total: int = 0
        self._current_frame: int = 0
        self._playing = False
        self._loop_start_sec: float | None = None
        self._loop_end_sec: float | None = None
        self._mouse_over = False

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._maybe_hide_controls)

        # 讓所有子元件滑鼠移動都算「活動」以延長 overlay 顯示
        self.installEventFilter(self)

    # ---- public --------------------------------------------------

    def open(self, video_path: str | Path) -> None:
        self._thread.open(str(video_path))

    def play(self) -> None:
        if not self._playing:
            # 若在影片尾端(loop 關閉時),按 play 先回到起點再播,否則 thread
            # 下一次 read() 會立刻 EOF 再觸發 pause,看起來 play/pause 壞掉。
            # loop 開啟時回 loop_start,否則回 0。
            if self._total > 0 and self._current_frame >= self._total - 2:
                if (
                    self._loop_enabled
                    and self._loop_start_sec is not None
                ):
                    target = int(round(self._loop_start_sec * self._fps))
                else:
                    target = 0
                self._thread.seek_frame(max(0, target))

            self._playing = True
            self._thread.set_playing(True)
            self.controls.set_play_icon(True)
            self.paused_badge.hide()
            # 播放開始:若滑鼠不在影片上,2.5s 後自動收起 overlay
            if not self._mouse_over:
                self._hide_timer.start(self._HIDE_DELAY_MS)
            else:
                self._show_overlays()

    def pause(self) -> None:
        if self._playing:
            self._playing = False
            self._thread.set_playing(False)
            self.controls.set_play_icon(False)
            self._show_overlays()
            self.paused_badge.show()
            self.paused_badge.raise_()

    def toggle(self) -> None:
        (self.pause if self._playing else self.play)()

    def seek(self, sec: float) -> None:
        self._thread.seek_frame(int(round(sec * self._fps)))

    def set_loop(self, start_sec: float | None, end_sec: float | None) -> None:
        self._loop_start_sec = start_sec
        self._loop_end_sec = end_sec
        self._apply_loop()

    def set_loop_enabled(self, enabled: bool) -> None:
        """由外部(Review view)控制循環 on/off。"""
        self._loop_enabled = bool(enabled)
        self._apply_loop()

    def loop_enabled(self) -> bool:
        return self._loop_enabled

    def is_playing(self) -> bool:
        return self._playing

    def fps(self) -> float:
        return self._fps

    def current_time(self) -> float:
        return self._current_frame / self._fps if self._fps > 0 else 0.0

    def shutdown(self) -> None:
        self._thread.request_stop()
        self._thread.quit()
        self._thread.wait(1500)

    # ---- layout --------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        self.canvas.setGeometry(0, 0, w, h)

        ctl_h = 56
        self.controls.setGeometry(8, h - ctl_h - 4, w - 16, ctl_h)

        badge_size = min(100, max(60, min(w, h) // 6))
        self.paused_badge.setGeometry(
            (w - badge_size) // 2,
            (h - badge_size) // 2,
            badge_size,
            badge_size,
        )

        self._raise_overlays()

    def _raise_overlays(self) -> None:
        self.controls.raise_()
        self.paused_badge.raise_()

    # ---- overlay visibility -------------------------------------

    def enterEvent(self, event) -> None:  # noqa: N802
        self._mouse_over = True
        self._show_overlays()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._mouse_over = False
        if self._playing:
            self._hide_timer.start(400)
        super().leaveEvent(event)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.MouseMove:
            self._on_cursor_activity()
        return super().eventFilter(obj, event)

    def _on_cursor_activity(self) -> None:
        self._mouse_over = True
        self._show_overlays()
        if self._playing:
            self._hide_timer.start(self._HIDE_DELAY_MS)

    def _show_overlays(self) -> None:
        self.controls.show()
        self._raise_overlays()

    def _maybe_hide_controls(self) -> None:
        if self._playing and not self._mouse_over:
            self.controls.hide()

    # ---- decoder callbacks --------------------------------------

    @Slot(int, float)
    def _on_opened(self, total: int, fps: float) -> None:
        self._total = total
        self._fps = fps
        self.controls.set_slider_range(total)
        self.controls.set_slider_value(0)
        self._update_time_label(0)
        self._apply_loop()
        # 初始:顯示 paused badge(還沒播)
        self.paused_badge.show()
        self._show_overlays()

    @Slot(QImage, int, float)
    def _on_frame(self, img: QImage, frame_idx: int, t_sec: float) -> None:
        self.canvas.set_image(img)
        self._current_frame = frame_idx
        self.controls.set_slider_value(frame_idx)
        self._update_time_label(t_sec)
        self.time_changed.emit(t_sec)

    @Slot()
    def _on_ended(self) -> None:
        self.pause()

    # ---- internals ----------------------------------------------

    def _on_slider_seek(self, frame: int) -> None:
        self._thread.seek_frame(frame)

    def _apply_loop(self) -> None:
        if (
            not self._loop_enabled
            or self._loop_start_sec is None
            or self._loop_end_sec is None
        ):
            self._thread.set_loop(0, -1)
            return
        s = max(0, int(round(self._loop_start_sec * self._fps)))
        e = int(round(self._loop_end_sec * self._fps))
        self._thread.set_loop(s, e)

    def _update_time_label(self, t_sec: float) -> None:
        total_sec = self._total / self._fps if self._fps > 0 else 0.0
        self.controls.set_time(t_sec, total_sec)
