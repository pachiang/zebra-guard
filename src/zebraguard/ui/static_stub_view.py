"""違停檢測 Static mode 的暫時佔位 view。

M1 只做到「能建立 mode=static 專案」+「載入時路由到這裡」。ROI 編輯、分析
pipeline、Review 都還沒做,這頁只顯示目前進度與下一步。M2-M6 會逐步將這裡
換成真正的流程(RoiEditorView → ProcessingView → ParkingReviewView)。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_MILESTONES = [
    ("M1", "功能入口 + 專案建立 (mode=static)", True),
    ("M2", "ROI 編輯器(合法停車區多邊形)", True),
    ("M3", "停留判定規則 + 單元測試", True),
    ("M4", "違停偵測 pipeline + Worker 分流", True),
    ("M5", "Review UI(候選清單 + 標籤下拉)", True),
    ("M6", "匯出過濾 + 違停檢舉書模板", False),
]


class StaticStubView(QWidget):
    """佔位頁 — 顯示目前 milestone 進度,提供關閉專案按鈕。"""

    request_close_project = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_path: Path | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 40, 40, 40)
        outer.setSpacing(18)
        outer.addStretch(1)

        title = QLabel("違停檢測 · 功能建置中")
        tf = QFont()
        tf.setPointSize(22)
        tf.setWeight(QFont.Weight.DemiBold)
        title.setFont(tf)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub = QLabel(
            "你建立的違停檢測專案已成功儲存。\n"
            "完整流程(ROI 編輯、分析、審查、匯出)將於後續 milestone 陸續開放。"
        )
        sub.setObjectName("Hint")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        sfont = QFont()
        sfont.setPointSize(12)
        sub.setFont(sfont)

        self.project_label = QLabel("—")
        self.project_label.setObjectName("Hint")
        self.project_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Milestone 清單卡
        card = QFrame()
        card.setObjectName("Card")
        card.setProperty("class", "card")
        card.setMaximumWidth(520)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(24, 18, 24, 18)
        cl.setSpacing(8)
        hdr = QLabel("Milestone 進度")
        hdr.setObjectName("MetricLabel")
        cl.addWidget(hdr)
        for code, label, done in _MILESTONES:
            row = QHBoxLayout()
            row.setSpacing(10)
            badge = QLabel("✓" if done else "○")
            badge.setFixedWidth(22)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(
                "color: #10b981; font-weight: 700;" if done
                else "color: #6b7280;"
            )
            code_lbl = QLabel(code)
            code_lbl.setFixedWidth(36)
            code_lbl.setStyleSheet(
                "color: #e8eaed; font-weight: 600;" if done
                else "color: #9aa1ad; font-weight: 500;"
            )
            text = QLabel(label)
            text.setStyleSheet(
                "color: #e8eaed;" if done else "color: #9aa1ad;"
            )
            row.addWidget(badge)
            row.addWidget(code_lbl)
            row.addWidget(text, stretch=1)
            cl.addLayout(row)

        card_wrap = QHBoxLayout()
        card_wrap.addStretch(1)
        card_wrap.addWidget(card)
        card_wrap.addStretch(1)

        # 關閉專案
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.close_btn = QPushButton("← 關閉專案回首頁")
        self.close_btn.setMinimumHeight(36)
        self.close_btn.setMinimumWidth(180)
        self.close_btn.clicked.connect(self.request_close_project.emit)
        btn_row.addWidget(self.close_btn)
        btn_row.addStretch(1)

        outer.addWidget(title)
        outer.addWidget(sub)
        outer.addWidget(self.project_label)
        outer.addSpacing(8)
        outer.addLayout(card_wrap)
        outer.addSpacing(8)
        outer.addLayout(btn_row)
        outer.addStretch(1)

    def load_project(self, project_path: Path) -> None:
        """MainWindow 路由過來時呼叫。目前只更新顯示用 label。"""
        self._project_path = project_path
        self.project_label.setText(f"專案:{project_path.name}")

    def shutdown(self) -> None:
        """佔位,對齊 Review / Processing 的 shutdown 介面。"""
        pass
