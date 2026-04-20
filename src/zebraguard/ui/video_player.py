"""OpenCV 自繪影片播放器(Qt)。

為何不用 QMediaPlayer:
  - 需要精準 loop 一個時間區間(±3 秒)
  - 需要在每幀上疊 bbox / mask overlay(QMediaPlayer 很難做)
  - 需要穩定的 seek(QMediaPlayer 在 Windows 上 seek 有時不可靠)

設計:
  - 主線程:PlayerView (QWidget) — QLabel 顯示 QImage + 播放控制列
  - 子線程:DecoderThread — 跑 cv2.VideoCapture 的迴圈,以 Qt signal 把每幀 QImage
    丟給主線程繪製
  - 指令 (play/pause/seek/setLoop) 透過 DecoderThread 的 slot(queued connection)
    丟進解碼執行緒
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (
    QObject,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def _bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # copy() — QImage 不 own data 時 numpy buffer 會被回收
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class DecoderThread(QThread):
    frame_ready = Signal(QImage, int, float)  # image, frame_idx, time_sec
    ended = Signal()  # 抵達末端(非 loop 情境)
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
        self._loop_end: int = -1  # -1 = 不 loop
        self._target_seek: int | None = None
        self._speed: float = 1.0

    # ---- public (called via QMetaObject.invokeMethod / queued signal) ----

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
        """end_frame < 0 表示停用 loop。"""
        self._loop_start = max(0, start_frame)
        self._loop_end = int(end_frame)

    @Slot(float)
    def set_speed(self, speed: float) -> None:
        self._speed = max(0.25, min(4.0, speed))

    def request_stop(self) -> None:
        self._stopping = True

    # ---- thread main ------------------------------------------------------

    def run(self) -> None:
        # 解碼主迴圈:即使 pause 也要持續 poll seek / loop 指令
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
                    self.frame_ready.emit(_bgr_to_qimage(frame), cur_idx, cur_idx / self._fps)
                last_t = time.monotonic()
                continue

            if not self._playing:
                self.msleep(20)
                continue

            cur_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            # loop wrap-around
            if self._loop_end >= 0 and cur_idx >= self._loop_end:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._loop_start)
                cur_idx = self._loop_start
                last_t = time.monotonic()

            ok, frame = self._cap.read()
            if not ok:
                # 抵達影片結尾
                if self._loop_end >= 0:
                    # loop 模式則回 loop_start
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._loop_start)
                    last_t = time.monotonic()
                    continue
                self._playing = False
                self.ended.emit()
                continue

            self.frame_ready.emit(_bgr_to_qimage(frame), cur_idx, cur_idx / self._fps)

            # 節流:讓真實播放 fps ≈ 影片 fps × speed
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


class VideoCanvas(QLabel):
    """顯示當前 QImage,保持比例置中。"""

    clicked_progress = Signal(float)  # 0.0-1.0(僅 progress bar 會用;此處備用)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(480, 270))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #000; border-radius: 10px;")
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


def _fmt_time(sec: float) -> str:
    m = int(sec) // 60
    s = sec - 60 * m
    return f"{m:02d}:{s:05.2f}"


class PlayerView(QWidget):
    """包 canvas + 控制列 + 進度列。

    主要介面:
      open(video_path)
      set_loop(start_sec, end_sec)    — 任一負數或 None 則停用
      seek(time_sec)
      play() / pause() / toggle()
    """

    time_changed = Signal(float)  # 每幀更新 callback

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PlayerView")

        self.canvas = VideoCanvas(self)
        self.play_btn = QPushButton("▶  播放")
        self.play_btn.setProperty("accent", True)
        self.play_btn.setFixedWidth(110)
        self.loop_btn = QPushButton("↻  循環播放")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setChecked(True)
        self.loop_btn.setFixedWidth(120)

        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.time_label.setObjectName("Hint")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setTracking(False)

        ctl = QHBoxLayout()
        ctl.setSpacing(10)
        ctl.addWidget(self.play_btn)
        ctl.addWidget(self.loop_btn)
        ctl.addWidget(self.slider, stretch=1)
        ctl.addWidget(self.time_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.canvas, stretch=1)
        layout.addLayout(ctl)

        self._thread = DecoderThread(self)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.opened.connect(self._on_opened)
        self._thread.ended.connect(self._on_ended)
        self._thread.start()

        self._fps: float = 30.0
        self._total: int = 0
        self._playing = False
        self._loop_start_sec: float | None = None
        self._loop_end_sec: float | None = None
        self._slider_dragging = False

        self.play_btn.clicked.connect(self.toggle)
        self.loop_btn.toggled.connect(self._on_loop_toggled)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_slider_dragging", True))
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_changed)

    # ---- public API -------------------------------------------------------

    def open(self, video_path: str | Path) -> None:
        self._thread.open(str(video_path))

    def play(self) -> None:
        if not self._playing:
            self._playing = True
            self.play_btn.setText("⏸  暫停")
            self._thread.set_playing(True)

    def pause(self) -> None:
        if self._playing:
            self._playing = False
            self.play_btn.setText("▶  播放")
            self._thread.set_playing(False)

    def toggle(self) -> None:
        (self.pause if self._playing else self.play)()

    def seek(self, sec: float) -> None:
        self._thread.seek_frame(int(round(sec * self._fps)))

    def set_loop(self, start_sec: float | None, end_sec: float | None) -> None:
        self._loop_start_sec = start_sec
        self._loop_end_sec = end_sec
        self._apply_loop()

    def is_playing(self) -> bool:
        return self._playing

    def fps(self) -> float:
        return self._fps

    def shutdown(self) -> None:
        """析構時呼叫:停掉 decoder thread。"""
        self._thread.request_stop()
        self._thread.quit()
        self._thread.wait(1500)

    # ---- internals --------------------------------------------------------

    def _apply_loop(self) -> None:
        if (
            not self.loop_btn.isChecked()
            or self._loop_start_sec is None
            or self._loop_end_sec is None
        ):
            self._thread.set_loop(0, -1)
            return
        s = max(0, int(round(self._loop_start_sec * self._fps)))
        e = int(round(self._loop_end_sec * self._fps))
        self._thread.set_loop(s, e)

    @Slot(int, float)
    def _on_opened(self, total: int, fps: float) -> None:
        self._total = total
        self._fps = fps
        self.slider.setRange(0, max(0, total - 1))
        self.slider.setValue(0)
        self._update_time_label(0)
        self._apply_loop()

    @Slot(QImage, int, float)
    def _on_frame(self, img: QImage, frame_idx: int, t_sec: float) -> None:
        self.canvas.set_image(img)
        if not self._slider_dragging:
            self.slider.blockSignals(True)
            self.slider.setValue(frame_idx)
            self.slider.blockSignals(False)
        self._update_time_label(t_sec)
        self.time_changed.emit(t_sec)

    @Slot()
    def _on_ended(self) -> None:
        self.pause()

    def _on_slider_changed(self, value: int) -> None:
        if self._slider_dragging:
            self._update_time_label(value / self._fps)

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        self._thread.seek_frame(self.slider.value())

    def _on_loop_toggled(self, _checked: bool) -> None:
        self._apply_loop()

    def _update_time_label(self, t_sec: float) -> None:
        total_sec = self._total / self._fps if self._fps > 0 else 0.0
        self.time_label.setText(f"{_fmt_time(t_sec)} / {_fmt_time(total_sec)}")
