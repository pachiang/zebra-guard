"""Export dialog:挑輸出資料夾,顯示進度,開啟結果。"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zebraguard.core.project import Project
from zebraguard.export import EXPORT_LAYOUT, export_accepted_events


class _ExportWorker(QObject):
    progress = Signal(int, int, str)
    done = Signal(str)  # 輸出根目錄
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        video_path: Path,
        events: list[dict[str, Any]],
        out_root: Path,
        fps: float | None,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._events = events
        self._out_root = out_root
        self._fps = fps

    def run(self) -> None:
        try:
            export_accepted_events(
                self._video_path,
                self._events,
                self._out_root,
                fps=self._fps,
                progress_cb=lambda c, t, msg: self.progress.emit(c, t, msg),
            )
            self.done.emit(str(self._out_root))
        except Exception:  # noqa: BLE001
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()


class ExportDialog(QDialog):
    def __init__(self, project_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("匯出檢舉輔助包")
        self.setModal(True)
        self.setMinimumSize(620, 560)

        self._project_path = project_path
        self._accepted: list[dict[str, Any]] = []
        self._video_path: Path | None = None
        self._fps: float | None = None
        self._thread: QThread | None = None
        self._worker: _ExportWorker | None = None

        self._build_ui()
        self._load_project_info()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 20)
        outer.setSpacing(14)

        title = QLabel("匯出採用的違規片段")
        title.setObjectName("SectionTitle")
        tf = QFont()
        tf.setPointSize(16)
        tf.setWeight(QFont.Weight.DemiBold)
        title.setFont(tf)

        self.summary = QLabel("—")
        self.summary.setObjectName("Hint")

        layout_hint = QLabel("輸出目錄結構:")
        layout_hint.setObjectName("MetricLabel")
        layout_preview = QPlainTextEdit(EXPORT_LAYOUT.strip())
        layout_preview.setReadOnly(True)
        layout_preview.setFixedHeight(110)

        # 路徑選擇
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("選擇輸出資料夾…")
        self.browse_btn = QPushButton("瀏覽…")
        self.browse_btn.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(self.browse_btn)

        # 進度
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status = QLabel("")
        self.status.setObjectName("Hint")

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)

        # 按鈕
        btns = QHBoxLayout()
        self.cancel_btn = QPushButton("關閉")
        self.cancel_btn.setProperty("ghost", True)
        self.cancel_btn.clicked.connect(self.reject)
        self.open_btn = QPushButton("開啟輸出資料夾")
        self.open_btn.setProperty("ghost", True)
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_out)
        self.run_btn = QPushButton("開始匯出")
        self.run_btn.setProperty("accent", True)
        self.run_btn.setMinimumWidth(140)
        self.run_btn.clicked.connect(self._start_export)
        btns.addWidget(self.cancel_btn)
        btns.addStretch(1)
        btns.addWidget(self.open_btn)
        btns.addWidget(self.run_btn)

        outer.addWidget(title)
        outer.addWidget(self.summary)
        outer.addWidget(layout_hint)
        outer.addWidget(layout_preview)
        outer.addLayout(path_row)
        outer.addWidget(self.progress)
        outer.addWidget(self.status)
        outer.addWidget(self.log, stretch=1)
        outer.addLayout(btns)

    def _load_project_info(self) -> None:
        project = Project.load(self._project_path)
        try:
            self._video_path = Path(project.meta.video_path)
            self._fps = project.meta.video_fps or 30.0
            all_events = project.load_events()
        finally:
            project.close()

        for i, e in enumerate(all_events):
            e["_display_index"] = i + 1
        self._accepted = [e for e in all_events if e.get("user_status") == "accepted"]

        self.summary.setText(
            f"影片:{self._video_path.name}   ·   採用事件 {len(self._accepted)} 筆"
        )
        # 預設輸出位置:專案資料夾/exports/<timestamp>/
        from datetime import datetime

        default = self._project_path / "exports" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path_edit.setText(str(default))

        if not self._accepted:
            self.run_btn.setEnabled(False)
            self.status.setText("沒有採用中的事件,請先於審查頁按下「採用」。")

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", self.path_edit.text())
        if path:
            self.path_edit.setText(path)

    def _start_export(self) -> None:
        if self._video_path is None or not self._accepted:
            return
        out_root = Path(self.path_edit.text().strip())
        if not out_root:
            QMessageBox.warning(self, "路徑無效", "請先選擇輸出資料夾。")
            return
        if not self._video_path.exists():
            QMessageBox.critical(self, "找不到影片", f"專案記錄的影片不存在:\n{self._video_path}")
            return

        self.run_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.path_edit.setEnabled(False)
        self.progress.setValue(0)
        self.log.clear()
        self._out_root = out_root

        worker = _ExportWorker(self._video_path, self._accepted, out_root, self._fps)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        worker.progress.connect(self._on_progress)
        worker.done.connect(self._on_done)
        worker.error.connect(self._on_error)

        self._worker = worker
        self._thread = thread
        thread.start()

    def _on_progress(self, current: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress.setValue(int(100 * current / total))
        self.status.setText(msg)
        self.log.appendPlainText(f"[{current}/{total}] {msg}")

    def _on_done(self, out_root: str) -> None:
        self.status.setText("完成。")
        self.progress.setValue(100)
        self.open_btn.setEnabled(True)
        self.log.appendPlainText(f"輸出完成:{out_root}")
        self.cancel_btn.setText("關閉")

    def _on_error(self, tb: str) -> None:
        self.status.setText("匯出過程發生錯誤,請見下方紀錄。")
        self.log.appendPlainText("錯誤:")
        self.log.appendPlainText(tb)
        self.run_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.path_edit.setEnabled(True)

    def _open_out(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.path_edit.text()))
