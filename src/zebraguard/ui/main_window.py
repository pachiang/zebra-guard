"""主視窗:頂部工具列 + 中央 QStackedWidget 在三個 view 之間切換。

View 切換由 MainWindow 掌控,各 view 透過 signal 告訴 MainWindow 要前往下一步。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from zebraguard import __version__
from zebraguard.core.project import Project
from zebraguard.ui.import_view import ImportView
from zebraguard.ui.processing_view import ProcessingView
from zebraguard.ui.review_view import ReviewView


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

        # Top bar
        top = QFrame()
        top.setObjectName("TopBar")
        top.setFixedHeight(58)
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(24, 8, 24, 8)
        top_layout.setSpacing(16)

        logo = QLabel("🦓")  # placeholder;正式版會改 QSvgWidget
        logo.setStyleSheet("font-size: 22px;")

        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        title = QLabel("ZebraGuard")
        title.setObjectName("AppTitle")
        subtitle = QLabel("行車記錄器未禮讓檢舉輔助")
        subtitle.setObjectName("AppSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        self.project_label = QLabel("尚未開啟專案")
        self.project_label.setObjectName("ProjectName")

        top_layout.addWidget(logo)
        top_layout.addLayout(title_col)
        top_layout.addStretch(1)
        top_layout.addWidget(self.project_label)

        self.close_proj_btn = QPushButton("關閉專案")
        self.close_proj_btn.setProperty("ghost", True)
        self.close_proj_btn.setVisible(False)
        self.close_proj_btn.clicked.connect(self.action_close_project)
        top_layout.addWidget(self.close_proj_btn)

        # Stacked views
        self.stack = QStackedWidget()
        self.import_view = ImportView()
        self.processing_view = ProcessingView()
        self.review_view = ReviewView()
        self.stack.addWidget(self.import_view)
        self.stack.addWidget(self.processing_view)
        self.stack.addWidget(self.review_view)

        # Signals
        self.import_view.project_created.connect(self._on_project_created)
        self.processing_view.completed.connect(self._on_analysis_done)
        self.processing_view.cancelled.connect(self._on_analysis_cancelled)
        self.review_view.request_close_project.connect(self.action_close_project)

        root_layout.addWidget(top)
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
        self.project_label.setText("尚未開啟專案")
        self.close_proj_btn.setVisible(False)
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
        self.project_label.setText(f"{path.stem}  ·  {backend}")
        self.close_proj_btn.setVisible(True)
        self.stack.setCurrentWidget(self.processing_view)
        self.processing_view.start(path)

    def _on_analysis_done(self, project_path: str) -> None:
        path = Path(project_path)
        self._project_path = path
        self.project_label.setText(path.stem)
        self.close_proj_btn.setVisible(True)
        self.review_view.load_project(path)
        self.stack.setCurrentWidget(self.review_view)

    def _on_analysis_cancelled(self) -> None:
        self.action_close_project()

    def _load_existing_project(self, proj_dir: Path) -> None:
        try:
            project = Project.load(proj_dir)
            stage = project.meta.progress.get("stage", "created")
            has_events = len(project.load_events()) > 0
            project.close()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "開啟失敗", f"無法開啟專案:\n{exc}")
            return

        self._project_path = proj_dir
        self.project_label.setText(proj_dir.stem)
        self.close_proj_btn.setVisible(True)

        if has_events or stage == "done":
            self.review_view.load_project(proj_dir)
            self.stack.setCurrentWidget(self.review_view)
        else:
            # 之前沒跑完 / 沒跑過 — 回到 import view 讓使用者決定是否重跑
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

    def closeEvent(self, event) -> None:  # noqa: N802
        # 清掉 decoder thread 等
        self.review_view.shutdown()
        self.processing_view.shutdown()
        super().closeEvent(event)
