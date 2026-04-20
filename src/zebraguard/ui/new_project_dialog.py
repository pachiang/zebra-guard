"""新專案 wizard dialog:使用者選 mode + (dashcam 時) backend + weights。

配合 ImportView 的「選擇影片 → 建立 .zgproj」流程。選完後 ImportView 用這裡
回傳的結果呼叫 Project.create。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from zebraguard.utils.paths import resources_dir

# 預設的 Aleton/Zebra_crossing 權重路徑(6.5 MB YOLOv8n-seg,MIT)
# 使用者可透過檔案對話框改,或自行 fine-tune 後替換此檔
DEFAULT_YOLO_SEG_WEIGHTS = "zebra_yolov8n_seg.pt"


@dataclass
class NewProjectOptions:
    mode: str                     # "dashcam" | "static"
    crosswalk_backend: str        # "mask2former" | "yolo_seg"
    yolo_seg_weights: str         # "" if not yolo_seg


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("MetricLabel")
    return lbl


def _radio_row(title: str, hint: str) -> tuple[QRadioButton, QWidget]:
    """一個 radio + 說明文字的 row widget。回傳 (radio, container)。"""
    radio = QRadioButton(title)
    f = QFont()
    f.setPointSize(12)
    f.setWeight(QFont.Weight.Medium)
    radio.setFont(f)

    hint_lbl = QLabel(hint)
    hint_lbl.setObjectName("Hint")
    hint_lbl.setWordWrap(True)
    hint_lbl.setContentsMargins(26, 0, 0, 4)  # 縮進對齊 radio 文字

    wrap = QWidget()
    wrap.setProperty("radio-container", True)
    lay = QVBoxLayout(wrap)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    lay.addWidget(radio)
    lay.addWidget(hint_lbl)
    return radio, wrap


class NewProjectDialog(QDialog):
    def __init__(self, video_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("新專案 — 選擇分析模式")
        self.setModal(True)
        self.setMinimumSize(560, 620)
        self._video_path = video_path
        self._result: NewProjectOptions | None = None

        self._build()

    # ---- public ----------------------------------------------------------

    def result_options(self) -> NewProjectOptions | None:
        return self._result

    # ---- layout ----------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 20)
        outer.setSpacing(14)

        title = QLabel("分析模式")
        tf = QFont()
        tf.setPointSize(18)
        tf.setWeight(QFont.Weight.DemiBold)
        title.setFont(tf)

        sub = QLabel(f"影片:{self._video_path.name}")
        sub.setObjectName("Hint")

        outer.addWidget(title)
        outer.addWidget(sub)

        outer.addWidget(self._mode_card())
        outer.addWidget(self._backend_card())

        # Actions
        btns = QHBoxLayout()
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setProperty("ghost", True)
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn = QPushButton("繼續 →")
        self.ok_btn.setProperty("accent", True)
        self.ok_btn.setMinimumWidth(120)
        self.ok_btn.clicked.connect(self._on_ok)
        btns.addWidget(self.cancel_btn)
        btns.addStretch(1)
        btns.addWidget(self.ok_btn)
        outer.addStretch(1)
        outer.addLayout(btns)

        # 預設選擇
        self.radio_dashcam.setChecked(True)
        # 若偵測到預設 YOLO-seg 權重則直接填入 + 預選 YOLO-seg
        # (多數使用者從 zip release 拿到的情境)
        default_pt = resources_dir() / "models" / DEFAULT_YOLO_SEG_WEIGHTS
        if default_pt.is_file():
            self.weights_edit.setText(str(default_pt))
            self.radio_yolo_seg.setChecked(True)
        else:
            self.radio_mask2former.setChecked(True)
        self._apply_backend_ui()

    def _mode_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        card.setProperty("class", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(8)

        v.addWidget(_section_label("影片類型"))

        self.radio_dashcam, w1 = _radio_row(
            "行車記錄器(Dashcam)",
            "車上攝影機 — 斑馬線會由 AI 模型自動辨識,每隔幾幀更新一次。",
        )
        self.radio_static, w2 = _radio_row(
            "固定式攝影機 / 路口 CCTV(尚未支援)",
            "需要使用者手繪斑馬線 ROI 與可選的透視校正;本模式在 Phase 2 才會接線。",
        )
        self.radio_static.setEnabled(False)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.radio_dashcam)
        self.mode_group.addButton(self.radio_static)
        self.mode_group.buttonToggled.connect(self._apply_backend_ui)

        v.addWidget(w1)
        v.addWidget(w2)
        return card

    def _backend_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        card.setProperty("class", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(8)

        v.addWidget(_section_label("斑馬線辨識方式(僅 Dashcam 模式)"))

        self.radio_mask2former, w1 = _radio_row(
            "Mask2Former  —  高品質,需要 GPU",
            "Mapillary Vistas 預訓練的語意分割模型(~850 MB)。"
            "在 GPU 上 720p 影片約 10-15 fps;CPU 上極慢不建議。",
        )
        self.radio_yolo_seg, w2 = _radio_row(
            "YOLO-seg  —  CPU 可跑,品質依權重而定",
            "需要您提供一組訓練好的 YOLO-seg 斑馬線 .pt 權重。"
            "預設會偵測 resources/models/zebra_yolov8n_seg.pt "
            "(可從 HuggingFace Aleton/Zebra_crossing 下載,6.5 MB, MIT)。"
            "模型小、CPU 也跑得動;品質接近 Mask2Former 但夜間偶爾較糟。",
        )

        self.backend_group = QButtonGroup(self)
        self.backend_group.addButton(self.radio_mask2former)
        self.backend_group.addButton(self.radio_yolo_seg)
        self.backend_group.buttonToggled.connect(self._apply_backend_ui)

        v.addWidget(w1)
        v.addWidget(w2)

        # YOLO-seg 權重選擇
        weights_row = QHBoxLayout()
        self.weights_label = QLabel("YOLO-seg 權重 (.pt):")
        self.weights_label.setObjectName("Hint")
        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("例如 resources/models/crosswalk_yolo_seg.pt")
        self.weights_browse = QPushButton("選擇…")
        self.weights_browse.clicked.connect(self._browse_weights)
        weights_row.addWidget(self.weights_edit, stretch=1)
        weights_row.addWidget(self.weights_browse)
        v.addSpacing(4)
        v.addWidget(self.weights_label)
        v.addLayout(weights_row)

        return card

    # ---- slots -----------------------------------------------------------

    def _apply_backend_ui(self, *_args) -> None:  # noqa: ANN002
        dashcam = self.radio_dashcam.isChecked()
        self.radio_mask2former.setEnabled(dashcam)
        self.radio_yolo_seg.setEnabled(dashcam)
        use_yolo = dashcam and self.radio_yolo_seg.isChecked()
        self.weights_label.setEnabled(use_yolo)
        self.weights_edit.setEnabled(use_yolo)
        self.weights_browse.setEnabled(use_yolo)

    def _browse_weights(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇 YOLO-seg 權重",
            "",
            "PyTorch 權重 (*.pt)",
        )
        if path:
            self.weights_edit.setText(path)

    def _on_ok(self) -> None:
        if self.radio_static.isChecked():
            QMessageBox.information(
                self,
                "尚未支援",
                "Static mode 尚未實作;請選擇 Dashcam。",
            )
            return
        mode = "dashcam"  # Static 之後再開
        if self.radio_yolo_seg.isChecked():
            backend = "yolo_seg"
            weights = self.weights_edit.text().strip()
            if not weights:
                QMessageBox.warning(self, "缺少權重", "YOLO-seg 需要選擇一個 .pt 權重檔。")
                return
            if not Path(weights).is_file():
                QMessageBox.warning(self, "檔案不存在", f"找不到:{weights}")
                return
        else:
            backend = "mask2former"
            weights = ""

        self._result = NewProjectOptions(
            mode=mode,
            crosswalk_backend=backend,
            yolo_seg_weights=weights,
        )
        self.accept()
