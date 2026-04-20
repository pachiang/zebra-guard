"""首次啟動免責聲明 dialog。

行為:
- 顯示 docs/disclaimer.md 的要點(非完整內容;完整文字由 help 選單開啟)
- 使用者必須捲到底才會亮起「我已閱讀並同意」按鈕
- 同意後寫入 settings.ini 的 disclaimer_version
- 版本升級時需再次彈出(比對設定版本 < 當前版本)
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from zebraguard.utils.paths import user_settings_file

DISCLAIMER_VERSION = 1

DISCLAIMER_TEXT = """\
本程式為輔助工具,從行車記錄器影片中識別「疑似」車輛未禮讓行人之片段。
請在使用前知悉下列事項:

一、關於判定
  · 本程式使用 AI 演算法進行偵測,結果為「候選」而非法律事實,可能誤判或漏判。
  · 判定邏輯以斑馬線 mask 的像素接近度近似「三枕木紋」距離,並非精確公尺換算。
  · 自車(ego vehicle)、騎士(rider)有自動過濾,但不保證 100% 準確。

二、使用者責任
  · 您取得影片之來源須符合個人資料保護法及相關法令。
  · 提報前請親自審閱每一則候選,確認違規事實、車牌、時間、地點均清晰正確。
  · 本程式之輸出不得用於騷擾、誹謗、勒索、商業蒐集他人資料等違法目的。
  · 您因使用本程式所生之任何民刑事責任,概由您自行承擔。

三、本程式不做的事
  · 不會向任何機關自動送件。
  · 不會將影片或分析結果上傳雲端。
  · 不代您判斷違規於法律上是否成立。
  · 不保證偵測準確率,亦不保證程式無瑕疵或不中斷。

四、責任限制
  · 於法律允許之最大範圍,本程式開發者不對因誤判、漏判、當機、資料損毀,或
    使用者違反本聲明所生之任何損害負責。本程式以「現狀」(as-is)提供,
    不負擔任何明示或默示之擔保。

五、管轄法律
  · 以中華民國法律為準據法。

按下下方「我已閱讀並同意」即表示您已完整閱讀並接受以上條款。
"""


class DisclaimerDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ZebraGuard — 使用前須知")
        self.setModal(True)
        self.setMinimumSize(640, 560)
        self._scrolled_to_bottom = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 28, 32, 24)
        outer.setSpacing(16)

        title = QLabel("使用前須知")
        title.setObjectName("SectionTitle")
        f = QFont()
        f.setPointSize(18)
        f.setWeight(QFont.Weight.DemiBold)
        title.setFont(f)

        sub = QLabel("本程式為輔助工具,使用前請閱讀以下聲明並捲動至底端。")
        sub.setObjectName("Hint")
        sub.setWordWrap(True)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QLabel(DISCLAIMER_TEXT)
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.setStyleSheet("QLabel { padding: 12px 16px; line-height: 1.6; }")
        inner = QWidget()
        inner.setObjectName("Card")
        inner.setProperty("class", "card")
        inner_layout = QVBoxLayout(inner)
        inner_layout.addWidget(body)
        self.scroll.setWidget(inner)

        self.hint = QLabel("請捲動至底端以啟用同意按鈕")
        self.hint.setObjectName("Hint")

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self.decline = QPushButton("取消並離開")
        self.decline.setProperty("ghost", True)
        self.accept_btn = QPushButton("我已閱讀並同意")
        self.accept_btn.setProperty("accent", True)
        self.accept_btn.setEnabled(False)
        self.accept_btn.setMinimumWidth(180)

        btn_row.addWidget(self.decline)
        btn_row.addStretch(1)
        btn_row.addWidget(self.accept_btn)

        outer.addWidget(title)
        outer.addWidget(sub)
        outer.addWidget(self.scroll, stretch=1)
        outer.addWidget(self.hint)
        outer.addLayout(btn_row)

        self.decline.clicked.connect(self.reject)
        self.accept_btn.clicked.connect(self._accept)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def _on_scroll(self, _value: int) -> None:
        sb = self.scroll.verticalScrollBar()
        # 留 2 px 容錯
        if sb.value() >= sb.maximum() - 2 and not self._scrolled_to_bottom:
            self._scrolled_to_bottom = True
            self.accept_btn.setEnabled(True)
            self.hint.setText("已閱讀完畢 — 可按「我已閱讀並同意」繼續")
            self.hint.setStyleSheet("color: #10b981;")

    def _accept(self) -> None:
        settings = QSettings(str(user_settings_file()), QSettings.Format.IniFormat)
        settings.setValue("disclaimer/version", DISCLAIMER_VERSION)
        settings.setValue("disclaimer/accepted_at",
                          __import__("datetime").datetime.now().isoformat())
        settings.sync()
        self.accept()


def user_has_accepted_latest() -> bool:
    settings = QSettings(str(user_settings_file()), QSettings.Format.IniFormat)
    version = settings.value("disclaimer/version", 0, type=int)
    return version >= DISCLAIMER_VERSION
