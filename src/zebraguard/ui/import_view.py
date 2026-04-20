"""Import view:選擇 / 拖放行車記錄器影片並建立新專案。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zebraguard.core.project import Project
from zebraguard.ui.new_project_dialog import NewProjectDialog

SUPPORTED_EXT = {".mp4", ".mov", ".mkv", ".avi"}


class DropZone(QFrame):
    file_dropped = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(300)
        self._apply_style(hover=False)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        icon = QLabel("📁")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 48px;")

        title = QLabel("拖放行車記錄器影片到此")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tfont = QFont()
        tfont.setPointSize(16)
        tfont.setWeight(QFont.Weight.DemiBold)
        title.setFont(tfont)

        sub = QLabel("支援 mp4 / mov / mkv / avi")
        sub.setObjectName("Hint")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(sub)

    def _apply_style(self, *, hover: bool) -> None:
        if hover:
            self.setStyleSheet(
                "QFrame#DropZone { background: rgba(245,165,36,0.08);"
                "border: 2px dashed #f5a524; border-radius: 14px; }"
            )
        else:
            self.setStyleSheet(
                "QFrame#DropZone { background: #1a1d24;"
                "border: 2px dashed #2e333f; border-radius: 14px; }"
            )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and any(
            Path(u.toLocalFile()).suffix.lower() in SUPPORTED_EXT
            for u in event.mimeData().urls()
        ):
            self._apply_style(hover=True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._apply_style(hover=False)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._apply_style(hover=False)
        urls = event.mimeData().urls()
        for u in urls:
            p = u.toLocalFile()
            if Path(p).suffix.lower() in SUPPORTED_EXT:
                self.file_dropped.emit(p)
                return
        event.ignore()


class ImportView(QWidget):
    project_created = Signal(str)  # project dir path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 40, 60, 40)
        outer.setSpacing(18)
        outer.addStretch(1)

        title = QLabel("開始新的檢舉分析")
        title.setObjectName("SectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tfont = QFont()
        tfont.setPointSize(22)
        tfont.setWeight(QFont.Weight.DemiBold)
        title.setFont(tfont)

        subtitle = QLabel(
            "選擇一段行車記錄器影片。ZebraGuard 會自動偵測影片中的斑馬線,\n"
            "找出可疑的未禮讓片段,再由您逐一審閱。"
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setObjectName("Hint")
        sfont = QFont()
        sfont.setPointSize(12)
        subtitle.setFont(sfont)

        self.drop = DropZone()

        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_row.setSpacing(12)
        self.choose_btn = QPushButton("選擇影片檔案…")
        self.choose_btn.setProperty("accent", True)
        self.choose_btn.setMinimumWidth(200)
        self.open_proj_btn = QPushButton("開啟既有專案")
        self.open_proj_btn.setProperty("ghost", True)
        self.open_proj_btn.setMinimumWidth(180)

        btn_row.addWidget(self.choose_btn)
        btn_row.addWidget(self.open_proj_btn)

        notes = QLabel(
            "分析會將專案資料夾(.zgproj)存到您選擇的位置;影片本身不會被複製。\n"
            "首次執行會下載約 850 MB 的斑馬線辨識模型。"
        )
        notes.setObjectName("Hint")
        notes.setAlignment(Qt.AlignmentFlag.AlignCenter)

        outer.addWidget(title)
        outer.addWidget(subtitle)
        outer.addSpacing(16)
        outer.addWidget(self.drop, stretch=2)
        outer.addLayout(btn_row)
        outer.addWidget(notes)
        outer.addStretch(1)

        self.drop.file_dropped.connect(self._on_video_selected)
        self.choose_btn.clicked.connect(self.prompt_open)
        self.open_proj_btn.clicked.connect(self._open_existing)

    # ---- public -----------------------------------------------------------

    def prompt_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇行車記錄器影片",
            "",
            "影片 (*.mp4 *.mov *.mkv *.avi)",
        )
        if path:
            self._on_video_selected(path)

    def reset(self) -> None:
        """關閉專案後呼叫,目前無狀態可清。"""
        pass

    def _open_existing(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "開啟 ZebraGuard 專案",
            "",
            "ZebraGuard 專案 (project.json)",
        )
        if not path:
            return
        # 委派給 main window 經由樹狀往上找;這裡直接 emit 會走「新專案」邏輯
        # 所以改用 window 的 action_open_project。
        mw = self.window()
        if hasattr(mw, "_load_existing_project"):
            mw._load_existing_project(Path(path).parent)  # type: ignore[attr-defined]

    # ---- internals --------------------------------------------------------

    def _on_video_selected(self, video_path: str) -> None:
        video = Path(video_path)
        if not video.is_file():
            QMessageBox.warning(self, "檔案不存在", f"找不到:{video}")
            return

        # Step 1:選分析模式 + backend + weights
        wizard = NewProjectDialog(video, self)
        if wizard.exec() != NewProjectDialog.DialogCode.Accepted:
            return
        opts = wizard.result_options()
        if opts is None:
            return

        # Step 2:決定專案存放位置
        default_name = f"{video.stem}_{datetime.now():%Y%m%d_%H%M}.zgproj"
        default_dir = video.parent / default_name
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "選擇專案存放位置",
            str(default_dir),
            "ZebraGuard 專案 (*.zgproj)",
        )
        if not chosen:
            return
        proj_path = Path(chosen)
        if proj_path.suffix != ".zgproj":
            proj_path = proj_path.with_suffix(".zgproj")
        if proj_path.exists():
            QMessageBox.warning(
                self,
                "已存在",
                f"{proj_path.name} 已存在;請改名或刪除既有資料夾。",
            )
            return

        try:
            project = Project.create(
                proj_path,
                video,
                mode=opts.mode,
                crosswalk_backend=opts.crosswalk_backend,
                yolo_seg_weights=opts.yolo_seg_weights,
            )
            project.close()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "建立專案失敗", str(exc))
            return

        self.project_created.emit(str(proj_path))
