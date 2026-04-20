"""Review view — 審查候選事件。

Layout:
  ┌─────────────────────────────────────────────────────────────────┐
  │ 29 筆 · 採用 3 · 拒絕 2 · 未審 24    [✚ 新增 (N)]  [📤 匯出]      │ ← thin top row
  ├──────────┬──────────────────────────────────────────────────────┤
  │  事件    │                                                      │
  │  清單    │          VideoPlayer (large)                         │
  │ (thumb+  │                                                      │
  │  狀態    │                                                      │
  │  色條)   ├──────────────────────────────────────────────────────┤
  │          │  Timeline                                            │
  │          ├──────────────────────────────────────────────────────┤
  │          │  詳情 + 車牌 + 調整時段 + 採用/拒絕/刪除              │
  └──────────┴──────────────────────────────────────────────────────┘

Shortcuts:
  A 採用 / R 拒絕 / J/K 上下 / N 新增 / Del 刪除 / Space 播放 / E focus 車牌
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QByteArray, QPoint, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QPainter,
    QPixmap,
    QPolygon,
    QShortcut,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from zebraguard.core.project import Project
from zebraguard.ui.thumbnails import (
    THUMB_H,
    THUMB_W,
    ensure_event_thumbnails,
    thumbnail_path,
)
from zebraguard.ui.video_player import PlayerView

LOOP_PADDING_SEC = 3.0
MANUAL_EVENT_DEFAULT_SEC = 2.0
MANUAL_NOTE_MARKER = "manual"

# 狀態色
STATUS_COLOR = {
    "accepted": "#10b981",
    "rejected": "#ef4444",
    "pending": "#6b7280",
}
STATUS_LABEL = {"accepted": "已採用", "rejected": "已拒絕", "pending": "未審"}


def _fmt_time(sec: float) -> str:
    m = int(sec) // 60
    s = sec - 60 * m
    return f"{m:02d}:{s:05.2f}"


# ====================================================================
# Timeline
# ====================================================================


class EventTimeline(QWidget):
    clicked_at = Signal(float)

    _MIN_MARKER_PX = 8

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._duration: float = 0.0
        self._current: float = 0.0
        self._events: list[dict[str, Any]] = []
        self._selected_id: int | None = None
        self.setMouseTracking(True)

    def set_duration(self, duration_sec: float) -> None:
        self._duration = max(0.001, duration_sec)
        self.update()

    def set_events(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self.update()

    def set_current_time(self, t_sec: float) -> None:
        self._current = t_sec
        self.update()

    def set_selected(self, event_id: int | None) -> None:
        self._selected_id = event_id
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        x = event.position().x()
        w = max(1.0, float(self.width() - 20))
        ratio = max(0.0, min(1.0, (x - 10) / w))
        self.clicked_at.emit(ratio * self._duration)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        p.fillRect(0, 0, w, h, QColor("#15181f"))
        usable_w = w - 20
        y_bar = h - 22
        bar_h = 10

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#2e333f"))
        p.drawRoundedRect(10, y_bar, usable_w, bar_h, 4, 4)

        if self._duration <= 0 or usable_w <= 0:
            p.end()
            return

        selected_rect: tuple[int, int, int, int] | None = None
        for ev in self._events:
            start_r = max(0.0, ev["start_sec"] / self._duration)
            end_r = min(1.0, ev["end_sec"] / self._duration)
            x1 = 10 + int(start_r * usable_w)
            x2 = 10 + int(end_r * usable_w)
            natural_w = x2 - x1
            width = max(self._MIN_MARKER_PX, natural_w)
            draw_x = x1 - max(0, (width - natural_w) // 2)
            draw_x = max(10, min(10 + usable_w - width, draw_x))

            status = ev.get("user_status", "pending")
            manual = ev.get("_manual", False)
            if status == "accepted":
                color = QColor("#10b981")
            elif status == "rejected":
                color = QColor("#ef4444")
            else:
                color = QColor("#9aa1ad") if manual else QColor("#6b7280")

            radius = min(4, max(1, width // 4))
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(draw_x, y_bar - 3, width, bar_h + 6, radius, radius)

            if ev["id"] == self._selected_id:
                selected_rect = (draw_x, y_bar - 3, width, bar_h + 6)

        if selected_rect is not None:
            x, y, rw, rh = selected_rect
            p.setBrush(Qt.BrushStyle.NoBrush)
            # 加粗到 2.5 px + 擴大邊框,讓選取在 timeline 上更顯眼
            from PySide6.QtGui import QPen
            pen = QPen(QColor("#f5a524"))
            pen.setWidthF(2.5)
            p.setPen(pen)
            radius = min(5, max(2, rw // 3))
            p.drawRoundedRect(x - 2, y - 2, rw + 4, rh + 4, radius, radius)

        ratio = max(0.0, min(1.0, self._current / self._duration))
        xh = 10 + int(ratio * usable_w)
        p.setPen(QColor("#ffd374"))
        p.drawLine(xh, 6, xh, h - 6)

        p.setPen(QColor("#9aa1ad"))
        f = QFont()
        f.setPointSize(9)
        p.setFont(f)
        p.drawText(10, 16, _fmt_time(0.0))
        p.drawText(w - 60, 16, _fmt_time(self._duration))
        p.end()


# ====================================================================
# Event list row widget
# ====================================================================


# Feather / Lucide 的 "repeat" icon — 典型 media-player 循環播放符號
_LOOP_SVG = """\
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none'
     stroke='{color}' stroke-width='{width}' stroke-linecap='round'
     stroke-linejoin='round' opacity='{opacity}'>
  <polyline points='17 1 21 5 17 9'/>
  <path d='M3 11V9a4 4 0 0 1 4-4h14'/>
  <polyline points='7 23 3 19 7 15'/>
  <path d='M21 13v2a4 4 0 0 1-4 4H3'/>
</svg>"""


class _LoopIconButton(QPushButton):
    """循環播放 chip 按鈕:SVG repeat icon;開啟 amber、關閉 mute 灰。

    底框 / 背景由 theme.py 的 QPushButton[chip=true] 規則處理;
    圖示本身透過 QSvgRenderer 在 paintEvent 疊上去。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.setFixedSize(48, 30)
        self.setProperty("chip", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip("循環播放目前選取的事件時段 ±3 秒")

    def _build_svg(self) -> QByteArray:
        if self.isChecked():
            color, width, opacity = "#f5a524", "2.2", "1.0"
        else:
            color, width, opacity = "#9aa1ad", "1.7", "0.85"
        svg = _LOOP_SVG.format(color=color, width=width, opacity=opacity)
        return QByteArray(svg.encode("utf-8"))

    def paintEvent(self, event) -> None:  # noqa: N802
        # 先讓 QPushButton 畫 chip 的背景與邊框
        super().paintEvent(event)
        # 在上面疊 SVG icon
        renderer = QSvgRenderer(self._build_svg())
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        icon_size = 18
        cx = self.width() // 2
        cy = self.height() // 2
        renderer.render(
            p, QRectF(cx - icon_size / 2, cy - icon_size / 2, icon_size, icon_size)
        )
        p.end()


class _HoverScrollList(QListWidget):
    """滑鼠懸停時 scrollbar 才出現;離開後 250ms 再收起,避免 jitter。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(250)
        self._hide_timer.timeout.connect(self._hide_bar)

    def _hide_bar(self) -> None:
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hide_timer.stop()
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hide_timer.start()
        super().leaveEvent(event)


class EventRowWidget(QFrame):
    """左側事件清單的單行 — **只有縮圖**。

    縮圖上疊兩個小圓角矩形:
      · 左上:#001 序號(半透明黑底 + amber 字)
      · 右下:開始時間 mm:ss(半透明黑底 + 粗白字)

    狀態用 **左側 4 px 色條** 表達,選取用外圍 amber 框。
    """

    _BADGE_FILL = QColor(10, 12, 18, 205)
    _INDEX_TEXT = QColor("#ffd374")
    _TIME_TEXT = QColor("#ffffff")
    _MANUAL_DOT = QColor("#10b981")

    # widget 總尺寸:左色條 4 + 左 margin 6 + thumb W + 右 margin 6 + 外框 2
    ROW_W = 4 + 6 + THUMB_W + 6 + 2
    ROW_H = 6 + THUMB_H + 6 + 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("EventRow")
        self.setFixedSize(self.ROW_W, self.ROW_H)

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 6, 6, 6)  # 左邊 4px 是 border-left
        root.setSpacing(0)

        self.thumb = QLabel()
        self.thumb.setFixedSize(THUMB_W, THUMB_H)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet(
            "background: #0d0f14; border-radius: 4px; color: #6b7280;"
            "font-size: 14px; font-weight: 600;"
        )
        self.thumb.setText("…")

        root.addWidget(self.thumb)

        self._status = "pending"
        self._selected = False
        self._apply_style()

    def set_event(
        self,
        ev: dict[str, Any],
        thumb_path: Path | None,
    ) -> None:
        self._status = ev.get("user_status", "pending")

        base: QPixmap | None = None
        if thumb_path is not None and Path(thumb_path).is_file():
            loaded = QPixmap(str(thumb_path))
            if not loaded.isNull():
                base = loaded

        if base is None:
            base = QPixmap(THUMB_W, THUMB_H)
            base.fill(QColor("#0d0f14"))

        self.thumb.setPixmap(self._paint_overlay(base, ev))
        self.thumb.setText("")
        self._apply_style()

    def set_selected(self, selected: bool) -> None:
        if self._selected == selected:
            return
        self._selected = selected
        self._apply_style()

    def _paint_overlay(self, base: QPixmap, ev: dict[str, Any]) -> QPixmap:
        pm = base.copy()
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont()
        font.setFamilies(["Segoe UI", "Arial", "Helvetica"])

        # -- 左上角:#NNN pill ------------------------------------------------
        idx = ev.get("_display_index", 0)
        idx_text = f"#{idx:03d}"
        font.setPointSize(9)
        font.setWeight(QFont.Weight.Bold)
        p.setFont(font)
        fm = p.fontMetrics()
        pad_x, pad_y = 6, 2
        idx_w = fm.horizontalAdvance(idx_text) + pad_x * 2
        idx_h = fm.height() + pad_y * 2
        p.setBrush(self._BADGE_FILL)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(6, 6, idx_w, idx_h, 4, 4)
        p.setPen(self._INDEX_TEXT)
        p.drawText(6 + pad_x, 6 + pad_y + fm.ascent(), idx_text)

        # -- 右下角:開始時間 pill --------------------------------------------
        time_text = _fmt_time(ev["start_sec"]).split(".")[0]  # 捨去小數(mm:ss)
        font.setPointSize(10)
        font.setWeight(QFont.Weight.Bold)
        p.setFont(font)
        fm = p.fontMetrics()
        tpad_x, tpad_y = 7, 2
        t_w = fm.horizontalAdvance(time_text) + tpad_x * 2
        t_h = fm.height() + tpad_y * 2
        t_x = pm.width() - 6 - t_w
        t_y = pm.height() - 6 - t_h
        p.setBrush(self._BADGE_FILL)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(t_x, t_y, t_w, t_h, 4, 4)
        p.setPen(self._TIME_TEXT)
        p.drawText(t_x + tpad_x, t_y + tpad_y + fm.ascent(), time_text)

        # -- 手動事件:左下小綠圓點 ------------------------------------------
        if ev.get("_manual"):
            p.setBrush(self._MANUAL_DOT)
            p.setPen(Qt.PenStyle.NoPen)
            r = 5
            p.drawEllipse(QPoint(10 + r, pm.height() - 10 - r), r, r)

        p.end()
        return pm

    def _apply_style(self) -> None:
        color = STATUS_COLOR.get(self._status, STATUS_COLOR["pending"])
        if self._selected:
            bg = "#242832"
            sel_border = "#f5a524"
        else:
            bg = "#1a1d24"
            sel_border = "transparent"
        self.setStyleSheet(
            f"""
            QFrame#EventRow {{
                background: {bg};
                border-radius: 8px;
                border-left: 4px solid {color};
                border-top: 1.5px solid {sel_border};
                border-right: 1.5px solid {sel_border};
                border-bottom: 1.5px solid {sel_border};
            }}
            """
        )


# ====================================================================
# Main Review view
# ====================================================================


class ReviewView(QWidget):
    request_close_project = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_path: Path | None = None
        self._events: list[dict[str, Any]] = []
        self._current_id: int | None = None
        self._video_duration: float = 0.0
        self._thumbs: dict[int, Path] = {}

        self._build_ui()
        self._wire_shortcuts()

    # ---- layout -----------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 16)
        outer.setSpacing(10)

        outer.addLayout(self._header_row())

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左側清單
        list_wrap = QFrame()
        list_wrap.setObjectName("Card")
        list_wrap.setProperty("class", "card")
        # 寬度 = row widget 寬度(含色條 + 邊框)+ 左右 padding + scrollbar 預留空間
        fixed_w = EventRowWidget.ROW_W + 24
        list_wrap.setMinimumWidth(fixed_w)
        list_wrap.setMaximumWidth(fixed_w)
        ll = QVBoxLayout(list_wrap)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(6)

        self.event_list = _HoverScrollList()
        self.event_list.setStyleSheet(
            "QListWidget { background: transparent; border: none; padding: 0; }"
            "QListWidget::item { border: none; padding: 0; margin: 0 0 6px 0; }"
            "QListWidget::item:selected { background: transparent; }"
        )
        self.event_list.setSpacing(0)
        self.event_list.currentItemChanged.connect(self._on_list_item_changed)
        ll.addWidget(self.event_list, stretch=1)

        self.close_project_btn = QPushButton("← 關閉專案")
        self.close_project_btn.setMinimumHeight(34)
        self.close_project_btn.clicked.connect(self.request_close_project.emit)
        ll.addWidget(self.close_project_btn)

        splitter.addWidget(list_wrap)
        splitter.addWidget(self._right_side())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        outer.addWidget(splitter, stretch=1)

    def _header_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self.summary = QLabel("—")
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        # 足夠的高度放 22pt 數字
        self.summary.setMinimumHeight(42)

        row.addWidget(self.summary)
        row.addStretch(1)

        self.add_btn = QPushButton("新增事件")
        self.add_btn.setProperty("ghost", True)
        self.add_btn.setMinimumWidth(110)
        self.add_btn.clicked.connect(self._on_add_event)

        self.export_btn = QPushButton("匯出採用片段")
        self.export_btn.setProperty("accent", True)
        self.export_btn.setMinimumWidth(140)
        self.export_btn.clicked.connect(self._on_export_clicked)

        row.addWidget(self.add_btn)
        row.addWidget(self.export_btn)
        return row

    def _right_side(self) -> QWidget:
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)

        self.player = PlayerView()
        self.player.time_changed.connect(self._on_time_changed)
        rl.addWidget(self.player, stretch=3)

        # Timeline + loop chip 同一列
        tl_row = QHBoxLayout()
        tl_row.setContentsMargins(0, 0, 0, 0)
        tl_row.setSpacing(8)
        self.timeline = EventTimeline()
        self.timeline.clicked_at.connect(self.player.seek)
        tl_row.addWidget(self.timeline, stretch=1)

        self.loop_chip = _LoopIconButton()
        self.loop_chip.toggled.connect(self.player.set_loop_enabled)
        tl_row.addWidget(self.loop_chip)
        rl.addLayout(tl_row)

        rl.addWidget(self._build_detail_panel())
        return right

    def _build_detail_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        panel.setProperty("class", "card")
        v = QVBoxLayout(panel)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(10)

        # Row 1: 題目 + 狀態圓點 + 時間區間(單行)
        top = QHBoxLayout()
        top.setSpacing(10)

        self.detail_title = QLabel("請先選擇一則事件")
        df = QFont()
        df.setPointSize(14)
        df.setWeight(QFont.Weight.DemiBold)
        self.detail_title.setFont(df)

        # 狀態:彩色大圓點 + 小字標籤。HTML 讓一個 QLabel 內多顏色。
        self.status_line = QLabel("")
        self.status_line.setTextFormat(Qt.TextFormat.RichText)
        self.status_line.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        self._set_status_line("pending")

        self.detail_meta = QLabel("")
        self.detail_meta.setStyleSheet("color: #9aa1ad; font-size: 12px;")
        self.detail_meta.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
        )

        top.addWidget(self.detail_title)
        top.addSpacing(6)
        top.addWidget(self.status_line)
        top.addStretch(1)
        top.addWidget(self.detail_meta)

        # Row 2: 車牌輸入 + 調整時段 在同一列
        form = QHBoxLayout()
        form.setSpacing(12)

        plate_col = QVBoxLayout()
        plate_col.setSpacing(2)
        plate_lbl = QLabel("車牌")
        plate_lbl.setObjectName("MetricLabel")
        self.plate_edit = QLineEdit()
        self.plate_edit.setPlaceholderText("例:ABC-1234")
        self.plate_edit.setMaximumWidth(220)
        self.plate_edit.editingFinished.connect(self._on_plate_commit)
        plate_col.addWidget(plate_lbl)
        plate_col.addWidget(self.plate_edit)

        trim_col = QVBoxLayout()
        trim_col.setSpacing(2)
        trim_lbl = QLabel("調整時段 (秒)")
        trim_lbl.setObjectName("MetricLabel")
        trim_row = QHBoxLayout()
        trim_row.setSpacing(6)
        self.trim_start = QDoubleSpinBox()
        self.trim_start.setDecimals(2)
        self.trim_start.setMaximum(1e6)
        self.trim_start.setFixedWidth(96)
        self.trim_end = QDoubleSpinBox()
        self.trim_end.setDecimals(2)
        self.trim_end.setMaximum(1e6)
        self.trim_end.setFixedWidth(96)
        self.trim_apply = QPushButton("套用")
        self.trim_apply.setFixedWidth(64)
        self.trim_apply.clicked.connect(self._on_trim_applied)
        trim_row.addWidget(self.trim_start)
        trim_row.addWidget(QLabel("→"))
        trim_row.addWidget(self.trim_end)
        trim_row.addWidget(self.trim_apply)
        trim_col.addWidget(trim_lbl)
        trim_col.addLayout(trim_row)

        form.addLayout(plate_col)
        form.addSpacing(8)
        form.addLayout(trim_col)
        form.addStretch(1)

        # Row 3: actions
        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.delete_btn = QPushButton("刪除")
        self.delete_btn.setProperty("ghost", True)
        self.delete_btn.setMinimumWidth(90)
        self.delete_btn.clicked.connect(self._on_delete_event)

        self.reject_btn = QPushButton("拒絕")
        self.reject_btn.setProperty("danger", True)
        self.reject_btn.setMinimumWidth(110)
        self.reject_btn.clicked.connect(lambda: self._set_status("rejected"))

        self.accept_btn = QPushButton("採用")
        self.accept_btn.setProperty("success", True)
        self.accept_btn.setMinimumWidth(110)
        self.accept_btn.clicked.connect(lambda: self._set_status("accepted"))

        actions.addWidget(self.delete_btn)
        actions.addStretch(1)
        actions.addWidget(self.reject_btn)
        actions.addWidget(self.accept_btn)

        v.addLayout(top)
        v.addLayout(form)
        v.addLayout(actions)
        return panel

    def _wire_shortcuts(self) -> None:
        for key, handler in [
            ("A", lambda: self._set_status("accepted")),
            ("R", lambda: self._set_status("rejected")),
            ("J", lambda: self._move_selection(-1)),
            ("K", lambda: self._move_selection(+1)),
            ("N", self._on_add_event),
            ("Delete", self._on_delete_event),
            ("Space", self.player.toggle),
            ("E", self._focus_plate),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(handler)

    # ---- public -----------------------------------------------------

    def load_project(self, project_path: Path) -> None:
        self._project_path = project_path
        project = Project.load(project_path)
        try:
            events = project.load_events()
            video_path = Path(project.meta.video_path)
            self._video_duration = project.meta.video_duration_sec
            fps = project.meta.video_fps or 30.0
        finally:
            project.close()

        if not video_path.exists():
            QMessageBox.warning(
                self,
                "找不到影片",
                f"專案記錄的影片路徑不存在:\n{video_path}",
            )
            return

        self._events = []
        for i, e in enumerate(events):
            e["_display_index"] = i + 1
            e["_manual"] = (e.get("user_note") == MANUAL_NOTE_MARKER)
            self._events.append(e)

        self._thumbs = ensure_event_thumbnails(
            project_path, video_path, self._events, fps
        )

        self.player.open(video_path)
        self.timeline.set_duration(self._video_duration or 1.0)
        self.timeline.set_events(self._events)
        self._rebuild_list()
        self._update_summary()

        if self._events:
            self.event_list.setCurrentRow(0)
        else:
            self.detail_title.setText("此影片未偵測到候選事件")
            self.detail_meta.setText("按 N 或「✚ 新增事件」手動加入")
            self._set_status_line("pending")

    def shutdown(self) -> None:
        self.player.shutdown()

    # ---- list rendering --------------------------------------------

    def _rebuild_list(self) -> None:
        self.event_list.blockSignals(True)
        self.event_list.clear()
        for ev in self._events:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, ev["id"])
            widget = EventRowWidget()
            widget.set_event(ev, self._thumbs.get(ev["id"]))
            item.setSizeHint(QSize(EventRowWidget.ROW_W, EventRowWidget.ROW_H))
            self.event_list.addItem(item)
            self.event_list.setItemWidget(item, widget)
        self.event_list.blockSignals(False)

    def _refresh_row(self, ev: dict[str, Any]) -> None:
        for i in range(self.event_list.count()):
            item = self.event_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == ev["id"]:
                widget = self.event_list.itemWidget(item)
                if isinstance(widget, EventRowWidget):
                    widget.set_event(ev, self._thumbs.get(ev["id"]))
                    widget.set_selected(item.isSelected())
                break

    def _apply_selection_visuals(self, cur_item: QListWidgetItem | None = None) -> None:
        """選取 bug 修正:不靠 item.isSelected()(`currentItemChanged` 觸發時
        selection model 可能尚未同步),改用傳進來的 cur_item 直接比對。"""
        if cur_item is None:
            cur_item = self.event_list.currentItem()
        for i in range(self.event_list.count()):
            item = self.event_list.item(i)
            widget = self.event_list.itemWidget(item)
            if isinstance(widget, EventRowWidget):
                widget.set_selected(item is cur_item)

    # ---- slots -----------------------------------------------------

    def _on_list_item_changed(
        self,
        cur: QListWidgetItem | None,
        _prev: QListWidgetItem | None,
    ) -> None:
        self._apply_selection_visuals(cur)
        if cur is None:
            return
        ev_id = cur.data(Qt.ItemDataRole.UserRole)
        ev = self._find_event(ev_id)
        if ev is None:
            return
        self._current_id = ev["id"]
        self._show_event(ev)

    def _on_time_changed(self, t_sec: float) -> None:
        self.timeline.set_current_time(t_sec)

    def _show_event(self, ev: dict[str, Any]) -> None:
        idx = ev["_display_index"]
        manual_tag = "  ✚ 手動" if ev.get("_manual") else ""
        self.detail_title.setText(f"事件 #{idx:03d}{manual_tag}")

        # 只留時間區間,不顯示持續秒數 / 追蹤數
        self.detail_meta.setText(
            f"{_fmt_time(ev['start_sec'])}  –  {_fmt_time(ev['end_sec'])}"
        )

        self._set_status_line(ev.get("user_status", "pending"))

        self.plate_edit.blockSignals(True)
        self.plate_edit.setText(ev.get("license_plate") or "")
        self.plate_edit.blockSignals(False)

        self.trim_start.blockSignals(True)
        self.trim_end.blockSignals(True)
        self.trim_start.setValue(ev["start_sec"])
        self.trim_end.setValue(ev["end_sec"])
        self.trim_start.blockSignals(False)
        self.trim_end.blockSignals(False)

        loop_s = max(0.0, ev["start_sec"] - LOOP_PADDING_SEC)
        loop_e = min(self._video_duration or 1e9, ev["end_sec"] + LOOP_PADDING_SEC)
        self.player.set_loop(loop_s, loop_e)
        self.player.seek(loop_s)
        self.player.play()

        self.timeline.set_selected(ev["id"])

    def _set_status_line(self, status: str) -> None:
        color = STATUS_COLOR.get(status, STATUS_COLOR["pending"])
        label = STATUS_LABEL.get(status, "未審")
        # 大圓點 + 文字,彩色但不佔大空間。label 用 12pt,dot 16pt,對齊基線
        self.status_line.setText(
            f"<span style='color:{color}; font-size:17pt; line-height:1.0;'>●</span>"
            f"&nbsp;&nbsp;<span style='color:{color}; font-size:12pt; "
            f"font-weight:600;'>{label}</span>"
        )

    def _set_status(self, status: str) -> None:
        if self._current_id is None or self._project_path is None:
            return
        project = Project.load(self._project_path)
        try:
            project.update_event_status(self._current_id, status)
        finally:
            project.close()

        ev = self._find_event(self._current_id)
        if ev is not None:
            ev["user_status"] = status
            self._refresh_row(ev)
            self._show_event(ev)
            self.timeline.set_events(self._events)
        self._update_summary()
        self._move_selection(+1)

    def _on_plate_commit(self) -> None:
        if self._current_id is None or self._project_path is None:
            return
        plate = self.plate_edit.text().strip()
        ev = self._find_event(self._current_id)
        if ev is None:
            return
        current = (ev.get("license_plate") or "").strip()
        if plate == current:
            return
        project = Project.load(self._project_path)
        try:
            project.update_event_plate(self._current_id, plate or None)
        finally:
            project.close()
        ev["license_plate"] = plate or None
        self._refresh_row(ev)

    def _on_trim_applied(self) -> None:
        if self._current_id is None or self._project_path is None:
            return
        s = self.trim_start.value()
        e = self.trim_end.value()
        if e <= s:
            QMessageBox.warning(self, "時段無效", "結束秒數必須大於開始秒數。")
            return
        fps = self.player.fps() or 30.0
        project = Project.load(self._project_path)
        try:
            project.update_event_range(self._current_id, s, e, fps)
        finally:
            project.close()
        # trim 會改變事件中段時間 → 強制重新產該 id 的縮圖;
        # 也可能改變排序(start_sec 調動),所以整個清單都 reload 以重新編號
        self._reload_events_preserving_selection(
            self._current_id,
            force_thumb_ids={self._current_id},
        )

    def _on_add_event(self) -> None:
        if self._project_path is None:
            return
        fps = self.player.fps() or 30.0
        start = max(0.0, self.player.current_time())
        end = min(
            self._video_duration or (start + MANUAL_EVENT_DEFAULT_SEC),
            start + MANUAL_EVENT_DEFAULT_SEC,
        )
        if end <= start:
            QMessageBox.warning(self, "時間無效", "無法在影片結尾處新增事件。")
            return

        project = Project.load(self._project_path)
        try:
            new_id = project.insert_event(
                start_sec=start,
                end_sec=end,
                fps=fps,
                user_status="accepted",
                user_note=MANUAL_NOTE_MARKER,
            )
        finally:
            project.close()

        self._reload_events_preserving_selection(new_id)

    def _on_delete_event(self) -> None:
        if self._current_id is None or self._project_path is None:
            return
        ev = self._find_event(self._current_id)
        if ev is None:
            return
        confirm = QMessageBox.question(
            self,
            "刪除事件?",
            f"確定要刪除事件 #{ev['_display_index']:03d}"
            f"(t={_fmt_time(ev['start_sec'])})嗎?\n此操作無法復原。",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        project = Project.load(self._project_path)
        try:
            project.delete_event(self._current_id)
        finally:
            project.close()

        cur_row = self.event_list.currentRow()
        self._reload_events_preserving_selection(None)
        if self.event_list.count() > 0:
            new_row = min(cur_row, self.event_list.count() - 1)
            self.event_list.setCurrentRow(new_row)
        else:
            self._current_id = None
            self.detail_title.setText("已無事件")
            self.detail_meta.setText("")
            self._set_status_line("pending")
            self.player.pause()

    def _reload_events_preserving_selection(
        self,
        select_id: int | None,
        *,
        force_thumb_ids: set[int] | None = None,
    ) -> None:
        """從 DB 重新抓事件清單;按 start_sec 重新排序並重新編號 #NNN。

        `force_thumb_ids` 會強制重產那些事件的縮圖(例如 trim 後中段時間改變)。
        """
        if self._project_path is None:
            return
        project = Project.load(self._project_path)
        try:
            raw = project.load_events()  # SQL ORDER BY start_sec, id
            video_path = Path(project.meta.video_path)
            fps = project.meta.video_fps or 30.0
        finally:
            project.close()

        self._events = []
        for i, e in enumerate(raw):
            e["_display_index"] = i + 1  # 依 start_sec 排序後的新編號
            e["_manual"] = (e.get("user_note") == MANUAL_NOTE_MARKER)
            self._events.append(e)

        self._thumbs = ensure_event_thumbnails(
            self._project_path, video_path, self._events, fps,
            force_ids=force_thumb_ids,
        )

        self._rebuild_list()
        self.timeline.set_events(self._events)
        self._update_summary()

        if select_id is not None:
            for i, ev in enumerate(self._events):
                if ev["id"] == select_id:
                    self.event_list.setCurrentRow(i)
                    break

    # ---- helpers ---------------------------------------------------

    def _focus_plate(self) -> None:
        self.plate_edit.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.plate_edit.selectAll()

    def _move_selection(self, delta: int) -> None:
        cur = self.event_list.currentRow()
        new = max(0, min(self.event_list.count() - 1, cur + delta))
        if new != cur:
            self.event_list.setCurrentRow(new)

    def _find_event(self, event_id: int) -> dict[str, Any] | None:
        for e in self._events:
            if e["id"] == event_id:
                return e
        return None

    def _update_summary(self) -> None:
        total = len(self._events)
        accepted = sum(1 for e in self._events if e.get("user_status") == "accepted")
        rejected = sum(1 for e in self._events if e.get("user_status") == "rejected")
        pending = total - accepted - rejected

        def _stat(n: int, label: str, color: str) -> str:
            return (
                f"<span style='font-size:20pt; font-weight:700; color:{color};"
                f" letter-spacing:-0.5px;'>{n}</span>"
                f"<span style='font-size:10pt; color:#9aa1ad;'>"
                f"&nbsp;&nbsp;{label}</span>"
            )

        sep = (
            "<span style='color:#3a4050; font-size:14pt;'>"
            "&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;</span>"
        )
        self.summary.setText(sep.join([
            _stat(total, "事件", "#e8eaed"),
            _stat(accepted, "採用", "#10b981"),
            _stat(rejected, "拒絕", "#ef4444"),
            _stat(pending, "未審", "#9aa1ad"),
        ]))
        self.export_btn.setEnabled(accepted > 0)

    def _on_export_clicked(self) -> None:
        if self._project_path is None:
            return
        from zebraguard.ui.export_dialog import ExportDialog

        dlg = ExportDialog(self._project_path, self)
        dlg.exec()
