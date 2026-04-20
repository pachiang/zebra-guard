"""Processing view:顯示 v7 pipeline 執行進度。"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zebraguard.core.worker import PipelineWorker

STAGE_LABEL = {
    "loading_mask": "載入斑馬線辨識模型",
    "loading_yolo": "載入物件偵測模型",
    "analyzing": "分析影片中",
    "annotating": "輸出標註影片",
    "done": "完成",
}


def _fmt_eta(sec: float) -> str:
    if sec <= 0 or sec > 3600 * 24:
        return "估算中…"
    m = int(sec) // 60
    s = int(sec) % 60
    if m >= 60:
        h = m // 60
        m = m % 60
        return f"{h} 時 {m} 分"
    if m > 0:
        return f"{m} 分 {s} 秒"
    return f"{s} 秒"


class MetricCard(QFrame):
    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setProperty("class", "card")
        self.setMinimumWidth(160)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)

        self.label = QLabel(label)
        self.label.setObjectName("MetricLabel")
        self.value = QLabel("—")
        self.value.setObjectName("Metric")

        layout.addWidget(self.label)
        layout.addWidget(self.value)

    def set_value(self, text: str) -> None:
        self.value.setText(text)


class ProcessingView(QWidget):
    completed = Signal(str)  # project path
    cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._project_path: Path | None = None
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._t_start: float = 0.0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 32, 40, 32)
        outer.setSpacing(20)

        title = QLabel("正在分析影片")
        title.setObjectName("SectionTitle")
        tfont = QFont()
        tfont.setPointSize(20)
        tfont.setWeight(QFont.Weight.DemiBold)
        title.setFont(tfont)

        self.stage_label = QLabel("準備中…")
        self.stage_label.setObjectName("Hint")
        sfont = QFont()
        sfont.setPointSize(12)
        self.stage_label.setFont(sfont)

        # 指標列
        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.m_progress = MetricCard("進度")
        self.m_speed = MetricCard("處理速度")
        self.m_eta = MetricCard("預估剩餘")
        self.m_hits = MetricCard("發現命中")
        for w in (self.m_progress, self.m_speed, self.m_eta, self.m_hits):
            metrics.addWidget(w)

        # 進度條
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")

        # Log
        log_label = QLabel("分析紀錄")
        log_label.setObjectName("MetricLabel")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(200)

        # 按鈕
        btn_row = QHBoxLayout()
        self.cancel_btn = QPushButton("取消分析")
        self.cancel_btn.setProperty("danger", True)
        self.cancel_btn.setMinimumWidth(120)
        btn_row.addStretch(1)
        btn_row.addWidget(self.cancel_btn)

        outer.addWidget(title)
        outer.addWidget(self.stage_label)
        outer.addLayout(metrics)
        outer.addWidget(self.progress)
        outer.addWidget(log_label)
        outer.addWidget(self.log, stretch=1)
        outer.addLayout(btn_row)

        self.cancel_btn.clicked.connect(self.cancel)

    # ---- public ----------------------------------------------------------

    def start(self, project_path: Path) -> None:
        self._project_path = project_path
        self._t_start = time.monotonic()
        self._reset_ui()
        self._append_log(f"開啟專案:{project_path}")

        # 建 worker + thread
        worker = PipelineWorker(project_path)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        worker.progress.connect(self._on_progress)
        worker.log_line.connect(self._append_log)
        worker.done.connect(self._on_done)
        worker.cancelled.connect(self._on_cancelled)
        worker.error.connect(self._on_error)

        self._worker = worker
        self._thread = thread
        thread.start()

    def cancel(self) -> None:
        if self._worker is not None:
            self._append_log("收到取消指令,等待當前幀處理完成…")
            self.cancel_btn.setEnabled(False)
            self._worker.cancel()

    def shutdown(self) -> None:
        """主視窗關閉時呼叫。"""
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)

    # ---- slots ----------------------------------------------------------

    def _reset_ui(self) -> None:
        self.progress.setValue(0)
        self.stage_label.setText("準備中…")
        self.m_progress.set_value("0%")
        self.m_speed.set_value("—")
        self.m_eta.set_value("—")
        self.m_hits.set_value("0")
        self.cancel_btn.setEnabled(True)
        self.log.clear()

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _on_progress(self, stage: str, current: int, total: int, hits: int) -> None:
        self.stage_label.setText(STAGE_LABEL.get(stage, stage))
        if stage in ("analyzing", "annotating") and total > 0:
            pct = 1000 * current / total
            self.progress.setValue(int(pct))
            self.m_progress.set_value(f"{current}/{total}  ({100*current/total:.1f}%)")
            elapsed = max(1e-6, time.monotonic() - self._t_start)
            if stage == "analyzing":
                rate = current / elapsed
                eta = (total - current) / rate if rate > 0 else 0
                self.m_speed.set_value(f"{rate:.1f} fps")
                self.m_eta.set_value(_fmt_eta(eta))
                self.m_hits.set_value(str(hits))
        elif stage in ("loading_mask", "loading_yolo"):
            self.progress.setValue(0)

    def _on_done(self, report: dict) -> None:
        n_events = len(report.get("events", []))
        self._append_log(f"分析完成:{n_events} 筆候選事件")
        self.stage_label.setText("完成")
        self.progress.setValue(1000)
        if self._project_path is not None:
            self.completed.emit(str(self._project_path))

    def _on_cancelled(self) -> None:
        self._append_log("分析已取消。")
        self.cancelled.emit()

    def _on_error(self, traceback_str: str) -> None:
        self._append_log("發生錯誤:")
        self._append_log(traceback_str)
        self.cancel_btn.setText("關閉")
        self.cancel_btn.setEnabled(True)
