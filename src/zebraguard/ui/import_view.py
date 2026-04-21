"""Import view — 功能首頁三卡:未禮讓行人 / 違停檢測 / 開啟既有。

- 未禮讓行人 → NewProjectDialog(現有 dashcam 流程)
- 違停檢測 → 直接選影片建 mode=static 專案(ROI 編輯器 M2 才做)
- 開啟既有專案 → 依 project.meta.mode 路由到對應 Review / Stub view
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QEnterEvent, QFont, QMouseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from zebraguard.core.project import Project
from zebraguard.ui.new_project_dialog import NewProjectDialog

SUPPORTED_EXT = {".mp4", ".mov", ".mkv", ".avi"}


class _FeatureCard(QFrame):
    """功能卡。整張卡點擊觸發 clicked signal。"""

    clicked = Signal()

    def __init__(
        self,
        title: str,
        subtitle: str,
        description: str,
        cta: str,
        accent: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("FeatureCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumSize(260, 220)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._accent = accent
        self._hovered = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(10)

        self.title_lbl = QLabel(title)
        tf = QFont()
        tf.setPointSize(16)
        tf.setWeight(QFont.Weight.DemiBold)
        self.title_lbl.setFont(tf)

        self.subtitle_lbl = QLabel(subtitle)
        self.subtitle_lbl.setObjectName("Hint")
        sf = QFont()
        sf.setPointSize(11)
        self.subtitle_lbl.setFont(sf)

        self.desc_lbl = QLabel(description)
        self.desc_lbl.setObjectName("Hint")
        self.desc_lbl.setWordWrap(True)

        self.cta_lbl = QLabel(cta)
        cf = QFont()
        cf.setPointSize(12)
        cf.setWeight(QFont.Weight.DemiBold)
        self.cta_lbl.setFont(cf)
        self.cta_lbl.setStyleSheet("color: #f5a524;" if accent else "color: #e8eaed;")

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.subtitle_lbl)
        layout.addStretch(1)
        layout.addWidget(self.desc_lbl)
        layout.addSpacing(6)
        layout.addWidget(self.cta_lbl)

        self._apply_style()

    def _apply_style(self) -> None:
        border = "#f5a524" if (self._accent and self._hovered) else (
            "#4a5162" if self._hovered else "#2e333f"
        )
        bg = "#242832" if self._hovered else "#1a1d24"
        self.setStyleSheet(
            f"""
            QFrame#FeatureCard {{
                background: {bg};
                border: 1.5px solid {border};
                border-radius: 12px;
            }}
            """
        )

    def enterEvent(self, event: QEnterEvent) -> None:  # noqa: N802
        self._hovered = True
        self._apply_style()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovered = False
        self._apply_style()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ImportView(QWidget):
    # dashcam 新專案建立完成 → 交給 MainWindow 切到 Processing
    project_created = Signal(str)
    # 靜態(違停檢測)新專案建立完成 → 交給 MainWindow 切到 StaticStubView
    static_project_created = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 50, 60, 50)
        outer.setSpacing(16)

        outer.addStretch(1)

        title = QLabel("ZebraGuard")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tf = QFont()
        tf.setPointSize(30)
        tf.setWeight(QFont.Weight.DemiBold)
        title.setFont(tf)

        subtitle = QLabel("選擇要使用的功能")
        subtitle.setObjectName("Hint")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sf = QFont()
        sf.setPointSize(13)
        subtitle.setFont(sf)

        outer.addWidget(title)
        outer.addWidget(subtitle)
        outer.addSpacing(22)

        # 三張卡
        cards_row = QHBoxLayout()
        cards_row.setSpacing(18)

        self.card_dashcam = _FeatureCard(
            title="未禮讓行人",
            subtitle="行車記錄器(Dashcam)",
            description="從行車紀錄器影片中找出車輛未禮讓斑馬線行人的片段,"
                        "AI 自動偵測斑馬線與行人車輛,由您審查採用。",
            cta="新建 →",
        )
        self.card_static = _FeatureCard(
            title="違停檢測",
            subtitle="固定式攝影機",
            description="從固定相機影片中找出疑似違停的車輛(並排 / 路口 / "
                        "紅黃線等)。您先畫合法停車區,系統列出候選後由您標註分類。",
            cta="新建 →",
        )
        self.card_open = _FeatureCard(
            title="開啟既有專案",
            subtitle="繼續先前的審查或重跑",
            description="選擇 .zgproj 資料夾內的 project.json 載入現有專案,"
                        "系統會依記錄的模式自動帶你進對應的畫面。",
            cta="選擇… →",
            accent=True,
        )

        self.card_dashcam.clicked.connect(self._on_dashcam)
        self.card_static.clicked.connect(self._on_static)
        self.card_open.clicked.connect(self._on_open)

        cards_row.addWidget(self.card_dashcam, stretch=1)
        cards_row.addWidget(self.card_static, stretch=1)
        cards_row.addWidget(self.card_open, stretch=1)
        outer.addLayout(cards_row, stretch=2)

        footer = QLabel(
            "首次執行若選「未禮讓行人 + Mask2Former」需下載 ~850 MB 模型;"
            "YOLO-seg 僅需 ~6 MB 權重(可於 resources/models/ 自行放置)。"
        )
        footer.setObjectName("Hint")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setWordWrap(True)
        outer.addSpacing(14)
        outer.addWidget(footer)
        outer.addStretch(1)

    # ---- public ---------------------------------------------------------

    def reset(self) -> None:
        """關閉專案後呼叫;目前無狀態可清。"""
        pass

    def prompt_open(self) -> None:
        """主選單「新專案…」會打到這裡;走 dashcam 流程。"""
        self._on_dashcam()

    # ---- card handlers --------------------------------------------------

    def _on_dashcam(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇行車記錄器影片",
            "",
            "影片 (*.mp4 *.mov *.mkv *.avi)",
        )
        if not path:
            return
        self._create_dashcam_project(Path(path))

    def _on_static(self) -> None:
        # M1:先跟使用者說一聲目前只做到建立專案,後續 milestone 再加 ROI / 分析
        notice = QMessageBox.question(
            self,
            "違停檢測 · 功能建置中",
            "違停檢測目前完成第一階段(建立專案 + mode 路由)。\n"
            "ROI 編輯器、分析、審查、匯出將於後續 milestone 逐步開放。\n\n"
            "仍要建立一個空的違停專案嗎?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if notice != QMessageBox.StandardButton.Yes:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇固定攝影機影片",
            "",
            "影片 (*.mp4 *.mov *.mkv *.avi)",
        )
        if not path:
            return
        self._create_static_project(Path(path))

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "開啟 ZebraGuard 專案",
            "",
            "ZebraGuard 專案 (project.json)",
        )
        if not path:
            return
        mw = self.window()
        if hasattr(mw, "_load_existing_project"):
            mw._load_existing_project(Path(path).parent)  # type: ignore[attr-defined]

    # ---- project creation ----------------------------------------------

    def _create_dashcam_project(self, video: Path) -> None:
        if not video.is_file():
            QMessageBox.warning(self, "檔案不存在", f"找不到:{video}")
            return

        wizard = NewProjectDialog(video, self)
        if wizard.exec() != NewProjectDialog.DialogCode.Accepted:
            return
        opts = wizard.result_options()
        if opts is None:
            return

        proj_path = self._prompt_project_location(video)
        if proj_path is None:
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

    def _create_static_project(self, video: Path) -> None:
        if not video.is_file():
            QMessageBox.warning(self, "檔案不存在", f"找不到:{video}")
            return

        proj_path = self._prompt_project_location(video, mode_hint="parking")
        if proj_path is None:
            return

        try:
            project = Project.create(proj_path, video, mode="static")
            project.close()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "建立專案失敗", str(exc))
            return

        self.static_project_created.emit(str(proj_path))

    def _prompt_project_location(
        self, video: Path, mode_hint: str = "yield"
    ) -> Path | None:
        default_name = (
            f"{video.stem}_{mode_hint}_{datetime.now():%Y%m%d_%H%M}.zgproj"
        )
        default_dir = video.parent / default_name
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "選擇專案存放位置",
            str(default_dir),
            "ZebraGuard 專案 (*.zgproj)",
        )
        if not chosen:
            return None
        proj_path = Path(chosen)
        if proj_path.suffix != ".zgproj":
            proj_path = proj_path.with_suffix(".zgproj")
        if proj_path.exists():
            QMessageBox.warning(
                self,
                "已存在",
                f"{proj_path.name} 已存在;請改名或刪除既有資料夾。",
            )
            return None
        return proj_path
