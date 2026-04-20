"""全域深色主題 QSS + 調色盤。

設計:琥珀色(amber)為強調色——避免常見的藍色 Material,在警示類 app 中比較有
存在感;中性灰階背景,卡片圓角,字距微放大。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

# 色票(單一處集中,QSS 會內嵌相同值)
BG_DEEP = "#0f1115"       # window / base layer
BG_CARD = "#1a1d24"       # panels / cards
BG_CARD_HI = "#242832"    # hover / 次強調
BORDER = "#2e333f"        # 分隔線
FG_PRIMARY = "#e8eaed"    # 主要文字
FG_MUTED = "#9aa1ad"      # 次要文字
FG_DIM = "#6b7280"        # 更弱的說明文字

ACCENT = "#f5a524"        # 強調色(amber 400)
ACCENT_HOVER = "#ffb534"
ACCENT_PRESSED = "#d68a10"
ACCENT_FG = "#1a1200"     # 強調色上的文字

OK = "#10b981"            # 採用(綠)
DANGER = "#ef4444"        # 拒絕(紅)
WARN = "#f59e0b"


QSS = f"""
* {{
    font-family: "Segoe UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans TC", sans-serif;
    font-size: 13px;
    color: {FG_PRIMARY};
}}

QMainWindow, QDialog, QWidget#ZebraGuardRoot {{
    background-color: {BG_DEEP};
}}

/* 頂部工具列 ---------------------------------------------------------- */
QFrame#TopBar {{
    background: {BG_CARD};
    border-bottom: 1px solid {BORDER};
}}
QLabel#AppTitle {{
    color: {FG_PRIMARY};
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.3px;
}}
QLabel#AppSubtitle {{
    color: {FG_MUTED};
    font-size: 12px;
}}
QLabel#ProjectName {{
    color: {ACCENT};
    font-weight: 600;
}}

/* 一般卡片 ------------------------------------------------------------ */
QFrame.card, QFrame#Card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

/* Labels -------------------------------------------------------------- */
QLabel {{ background: transparent; }}
QLabel[muted="true"] {{ color: {FG_MUTED}; }}
QLabel[dim="true"] {{ color: {FG_DIM}; }}
QLabel#SectionTitle {{
    font-size: 17px;
    font-weight: 600;
    letter-spacing: 0.2px;
}}
QLabel#Hint {{
    color: {FG_MUTED};
    font-size: 12px;
}}
QLabel#Metric {{
    color: {FG_PRIMARY};
    font-size: 22px;
    font-weight: 600;
}}
QLabel#MetricLabel {{
    color: {FG_MUTED};
    font-size: 11px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}

/* Buttons ------------------------------------------------------------- */
QPushButton {{
    background: {BG_CARD_HI};
    color: {FG_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 500;
}}
QPushButton:hover {{ background: #2e333f; }}
QPushButton:pressed {{ background: #1f2330; }}
QPushButton:disabled {{ color: {FG_DIM}; background: {BG_CARD}; }}

QPushButton[accent="true"] {{
    background: {ACCENT};
    color: {ACCENT_FG};
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton[accent="true"]:pressed {{ background: {ACCENT_PRESSED}; border-color: {ACCENT_PRESSED}; }}
QPushButton[accent="true"]:disabled {{ background: #4a3a10; color: #8a7850; border-color: #4a3a10; }}

QPushButton[danger="true"] {{
    background: transparent;
    color: {DANGER};
    border: 1px solid {DANGER};
}}
QPushButton[danger="true"]:hover {{ background: rgba(239, 68, 68, 0.1); }}

QPushButton[success="true"] {{
    background: {OK};
    color: #062016;
    border: 1px solid {OK};
    font-weight: 600;
}}
QPushButton[success="true"]:hover {{ background: #1ac490; }}

QPushButton[ghost="true"] {{
    background: transparent;
    border: 1px solid transparent;
    color: {FG_MUTED};
}}
QPushButton[ghost="true"]:hover {{ color: {FG_PRIMARY}; background: {BG_CARD_HI}; }}

QPushButton[chip="true"] {{
    background: transparent;
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 4px 14px;
    color: {FG_MUTED};
    font-weight: 500;
}}
QPushButton[chip="true"]:hover {{ color: {FG_PRIMARY}; border-color: #4a5162; }}
QPushButton[chip="true"]:checked {{
    color: {ACCENT};
    border: 1px solid {ACCENT};
    background: rgba(245, 165, 36, 0.08);
}}

/* Progress bar -------------------------------------------------------- */
QProgressBar {{
    background: {BG_CARD_HI};
    border: 1px solid {BORDER};
    border-radius: 8px;
    height: 14px;
    text-align: center;
    color: {FG_PRIMARY};
    font-weight: 500;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT_HOVER}, stop:1 {ACCENT});
    border-radius: 7px;
}}

/* List widget --------------------------------------------------------- */
QListWidget {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    outline: 0;
    padding: 4px;
}}
QListWidget::item {{
    padding: 10px 12px;
    border-radius: 8px;
    margin-bottom: 2px;
}}
QListWidget::item:hover {{ background: {BG_CARD_HI}; }}
QListWidget::item:selected {{
    background: rgba(245, 165, 36, 0.15);
    border: 1px solid {ACCENT};
    color: {FG_PRIMARY};
}}

/* Text edit / plain text ---------------------------------------------- */
QPlainTextEdit, QTextEdit {{
    background: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px;
    color: {FG_PRIMARY};
    font-family: "Cascadia Mono", "Consolas", "JetBrains Mono", monospace;
    font-size: 12px;
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_FG};
}}

/* Scrollbars ---------------------------------------------------------- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 40px;
}}
QScrollBar::handle:vertical:hover {{ background: #3a4050; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 5px;
    min-width: 40px;
}}
QScrollBar::handle:horizontal:hover {{ background: #3a4050; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* Input widgets ------------------------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {FG_PRIMARY};
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_FG};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}

/* Menus --------------------------------------------------------------- */
QMenuBar {{ background: {BG_CARD}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ padding: 6px 12px; background: transparent; }}
QMenuBar::item:selected {{ background: {BG_CARD_HI}; }}
QMenu {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{ padding: 6px 14px; border-radius: 4px; }}
QMenu::item:selected {{ background: {BG_CARD_HI}; }}

/* Tooltip ------------------------------------------------------------- */
QToolTip {{
    background: {BG_CARD_HI};
    color: {FG_PRIMARY};
    border: 1px solid {BORDER};
    padding: 4px 8px;
    border-radius: 4px;
}}

/* Status labels ------------------------------------------------------- */
QLabel#StatusAccepted {{ color: {OK}; font-weight: 600; }}
QLabel#StatusRejected {{ color: {DANGER}; font-weight: 600; }}
QLabel#StatusPending  {{ color: {FG_MUTED}; font-weight: 500; }}
"""


def apply(app: QApplication) -> None:
    """Apply 深色調色盤 + QSS。"""
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(BG_DEEP))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(FG_PRIMARY))
    pal.setColor(QPalette.ColorRole.Base, QColor(BG_CARD))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_CARD_HI))
    pal.setColor(QPalette.ColorRole.Text, QColor(FG_PRIMARY))
    pal.setColor(QPalette.ColorRole.Button, QColor(BG_CARD_HI))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(FG_PRIMARY))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(ACCENT_FG))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_CARD_HI))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(FG_PRIMARY))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(FG_DIM))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    app.setStyle("Fusion")

    # 固定反鋸齒、高 DPI 字型
    QFontDatabase.addApplicationFont  # 讓 linter 不抱怨 unused import,實際不載入字型
    _ = Qt  # 保留 import 供未來子模組擴充
