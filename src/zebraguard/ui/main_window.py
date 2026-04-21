"""主視窗:頂部工具列 + 中央 QStackedWidget 在三個 view 之間切換。

View 切換由 MainWindow 掌控,各 view 透過 signal 告訴 MainWindow 要前往下一步。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from zebraguard import __version__
from zebraguard.core.project import Project
from zebraguard.ui.import_view import ImportView
from zebraguard.ui.parking_review_view import ParkingReviewView
from zebraguard.ui.processing_view import ProcessingView
from zebraguard.ui.review_view import ReviewView
from zebraguard.ui.roi_editor_view import RoiEditorView
from zebraguard.ui.static_stub_view import StaticStubView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"ZebraGuard v{__version__}")
        self.resize(1280, 820)

        self._project_path: Path | None = None

        self._build_menu()
        self._build_ui()

    # ---- layout -----------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("檔案(&F)")
        new_proj = QAction("新專案…", self)
        new_proj.setShortcut("Ctrl+N")
        new_proj.triggered.connect(self.action_new_project)
        file_menu.addAction(new_proj)

        open_proj = QAction("開啟專案…", self)
        open_proj.setShortcut("Ctrl+O")
        open_proj.triggered.connect(self.action_open_project)
        file_menu.addAction(open_proj)

        file_menu.addSeparator()

        close_proj = QAction("關閉專案", self)
        close_proj.triggered.connect(self.action_close_project)
        file_menu.addAction(close_proj)

        file_menu.addSeparator()
        quit_act = QAction("結束", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        help_menu = menubar.addMenu("說明(&H)")
        about = QAction("關於 ZebraGuard", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

        user_guide = QAction("使用手冊", self)
        user_guide.triggered.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path.cwd() / "docs" / "user-guide" / "README.md"))
        ))
        help_menu.addAction(user_guide)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("ZebraGuardRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Stacked views(TopBar 已移除;專案名顯示於 window title,關閉專案於
        # 選單或 Review 左下)
        self.stack = QStackedWidget()
        self.import_view = ImportView()
        self.processing_view = ProcessingView()
        self.review_view = ReviewView()
        self.parking_review_view = ParkingReviewView()
        self.roi_editor_view = RoiEditorView()
        self.static_stub_view = StaticStubView()
        self.stack.addWidget(self.import_view)
        self.stack.addWidget(self.processing_view)
        self.stack.addWidget(self.review_view)
        self.stack.addWidget(self.parking_review_view)
        self.stack.addWidget(self.roi_editor_view)
        self.stack.addWidget(self.static_stub_view)

        self.import_view.project_created.connect(self._on_project_created)
        self.import_view.static_project_created.connect(self._on_static_project_created)
        self.processing_view.completed.connect(self._on_analysis_done)
        self.processing_view.cancelled.connect(self._on_analysis_cancelled)
        self.review_view.request_close_project.connect(self.action_close_project)
        self.review_view.request_rerun.connect(self._on_rerun_requested)
        self.parking_review_view.request_close_project.connect(self.action_close_project)
        self.roi_editor_view.rois_saved.connect(self._on_rois_saved)
        self.roi_editor_view.cancelled.connect(self.action_close_project)
        self.static_stub_view.request_close_project.connect(self.action_close_project)

        root_layout.addWidget(self.stack, stretch=1)
        self.setCentralWidget(root)

        self.stack.setCurrentWidget(self.import_view)

    # ---- actions ----------------------------------------------------------

    def action_new_project(self) -> None:
        # 讓 ImportView 主動處理(觸發其「選擇影片」按鈕)
        self.stack.setCurrentWidget(self.import_view)
        self.import_view.prompt_open()

    def action_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "開啟 ZebraGuard 專案",
            "",
            "ZebraGuard 專案 (project.json)",
        )
        if not path:
            return
        proj_dir = Path(path).parent
        self._load_existing_project(proj_dir)

    def action_close_project(self) -> None:
        if self._project_path is None:
            return
        # 若尚在分析中,先詢問取消
        if self.stack.currentWidget() is self.processing_view:
            reply = QMessageBox.question(
                self, "分析進行中", "目前仍在分析,關閉專案會取消分析。確定要關閉嗎?"
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self.processing_view.cancel()

        self._project_path = None
        self._set_window_title(None)
        self.stack.setCurrentWidget(self.import_view)
        self.import_view.reset()

    # ---- view coordination -----------------------------------------------

    def _on_project_created(self, project_path: str) -> None:
        path = Path(project_path)
        self._project_path = path
        try:
            proj = Project.load(path)
            backend = proj.meta.crosswalk_backend
            proj.close()
        except Exception:  # noqa: BLE001
            backend = "?"
        self._set_window_title(f"{path.stem}  ·  {backend}")
        self.stack.setCurrentWidget(self.processing_view)
        self.processing_view.start(path)

    def _on_static_project_created(self, project_path: str) -> None:
        """違停(static mode)新專案建立 → 進 ROI 編輯器。"""
        path = Path(project_path)
        self._project_path = path
        self._set_window_title(f"{path.stem}  ·  違停檢測 · ROI 編輯")
        self.roi_editor_view.load_project(path)
        self.stack.setCurrentWidget(self.roi_editor_view)

    def _on_rois_saved(self, project_path: str) -> None:
        """ROI 存完 → 跑違停檢測 pipeline。"""
        path = Path(project_path)
        self._project_path = path
        self._set_window_title(f"{path.stem}  ·  違停檢測 · 分析中")
        self.stack.setCurrentWidget(self.processing_view)
        self.processing_view.start(path)

    def _on_analysis_done(self, project_path: str) -> None:
        path = Path(project_path)
        self._project_path = path
        try:
            proj = Project.load(path)
            mode = proj.meta.mode or "dashcam"
            proj.close()
        except Exception:  # noqa: BLE001
            mode = "dashcam"

        self._set_window_title(path.stem)
        if mode == "static":
            self.parking_review_view.load_project(path)
            self.stack.setCurrentWidget(self.parking_review_view)
        else:
            self.review_view.load_project(path)
            self.stack.setCurrentWidget(self.review_view)

    def _on_analysis_cancelled(self) -> None:
        self.action_close_project()

    def _on_rerun_requested(self, new_params: dict) -> None:
        """Review 發的重跑請求。清舊事件、套用新 config、切 ProcessingView。"""
        if self._project_path is None:
            return
        try:
            project = Project.load(self._project_path)
            try:
                if new_params:
                    # 保留 mode / crosswalk_backend 等中繼鍵
                    existing = dict(project.meta.pipeline_config or {})
                    existing.update(new_params)
                    project.save_config(existing)
                project.clear_events()
                project.update_progress("rerun_requested")
            finally:
                project.close()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "重跑失敗", f"無法準備重跑:\n{exc}")
            return

        self.stack.setCurrentWidget(self.processing_view)
        self.processing_view.start(self._project_path)

    def _load_existing_project(self, proj_dir: Path) -> None:
        try:
            project = Project.load(proj_dir)
            stage = project.meta.progress.get("stage", "created")
            mode = project.meta.mode or "dashcam"
            has_events = len(project.load_events()) > 0
            parking_zones = list(project.meta.parking_zones or [])
            project.close()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "開啟失敗", f"無法開啟專案:\n{exc}")
            return

        self._project_path = proj_dir
        self._set_window_title(proj_dir.stem)

        # Static mode(違停檢測)分流
        if mode == "static":
            if not parking_zones:
                # 還沒畫 ROI → 編輯器
                self.roi_editor_view.load_project(proj_dir)
                self.stack.setCurrentWidget(self.roi_editor_view)
            elif has_events or stage == "done":
                # 已跑完 → ParkingReviewView
                self.parking_review_view.load_project(proj_dir)
                self.stack.setCurrentWidget(self.parking_review_view)
            else:
                # ROI 有、但還沒跑過 → 問使用者要不要直接開始分析
                reply = QMessageBox.question(
                    self,
                    "開始分析?",
                    "已畫好合法停車區但尚未分析。要現在開始分析嗎?",
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.stack.setCurrentWidget(self.processing_view)
                    self.processing_view.start(proj_dir)
                else:
                    # ROI 有但選不跑 → 開 review(可能空清單,讓使用者自己決定)
                    self.parking_review_view.load_project(proj_dir)
                    self.stack.setCurrentWidget(self.parking_review_view)
            return

        # Dashcam 流程(既有)
        if has_events or stage == "done":
            self.review_view.load_project(proj_dir)
            self.stack.setCurrentWidget(self.review_view)
        else:
            reply = QMessageBox.question(
                self,
                "重新分析?",
                "此專案尚未完成分析。是否要現在開始分析?",
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.stack.setCurrentWidget(self.processing_view)
                self.processing_view.start(proj_dir)
            else:
                self.stack.setCurrentWidget(self.import_view)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "關於 ZebraGuard",
            f"<h3>ZebraGuard v{__version__}</h3>"
            "<p>從行車記錄器影片識別車輛未禮讓斑馬線行人,協助產生檢舉輔助包。</p>"
            "<p>本軟體僅供合法檢舉用途;詳見啟動時顯示之免責聲明。</p>",
        )

    def _set_window_title(self, project_tag: str | None) -> None:
        base = f"ZebraGuard v{__version__}"
        self.setWindowTitle(f"{base}  —  {project_tag}" if project_tag else base)

    def closeEvent(self, event) -> None:  # noqa: N802
        # 清掉 decoder thread 等
        self.review_view.shutdown()
        self.parking_review_view.shutdown()
        self.processing_view.shutdown()
        self.roi_editor_view.shutdown()
        self.static_stub_view.shutdown()
        super().closeEvent(event)
