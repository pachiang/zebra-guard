"""Review view:逐一審查候選違規事件。

Layout:
  ┌──────────────────────────────────────────────────────────────┐
  │                       Top strip: title + 匯出                 │
  ├───────────────┬──────────────────────────────────────────────┤
  │   事件清單    │              VideoPlayer                      │
  │  (scrollable) │                                              │
  │               │              控制 / 進度                       │
  │               ├──────────────────────────────────────────────┤
  │               │   Timeline (所有事件 markers)                  │
  │               ├──────────────────────────────────────────────┤
  │               │   事件資訊 + Accept/Reject/Trim               │
  └───────────────┴──────────────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
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
from zebraguard.ui.video_player import PlayerView

LOOP_PADDING_SEC = 3.0  # ±3 秒 loop 範圍


class EventTimeline(QWidget):
    """水平 timeline,標註所有事件的位置。"""

    clicked_at = Signal(float)  # 點擊的時間 (秒)

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

        # 底色
        p.fillRect(0, 0, w, h, QColor("#15181f"))

        usable_w = w - 20
        y_bar = h - 22
        bar_h = 10

        # bar background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#2e333f"))
        p.drawRoundedRect(10, y_bar, usable_w, bar_h, 4, 4)

        if self._duration <= 0 or usable_w <= 0:
            p.end()
            return

        # event markers
        for ev in self._events:
            start_r = max(0.0, ev["start_sec"] / self._duration)
            end_r = min(1.0, ev["end_sec"] / self._duration)
            x1 = 10 + int(start_r * usable_w)
            x2 = 10 + int(end_r * usable_w)
            width = max(3, x2 - x1)
            status = ev.get("user_status", "pending")
            if status == "accepted":
                color = QColor("#10b981")
            elif status == "rejected":
                color = QColor("#ef4444")
            elif ev["id"] == self._selected_id:
                color = QColor("#f5a524")
            else:
                color = QColor("#6b7280")
            p.setBrush(color)
            p.drawRoundedRect(x1, y_bar - 2, width, bar_h + 4, 3, 3)

        # playhead
        ratio = max(0.0, min(1.0, self._current / self._duration))
        xh = 10 + int(ratio * usable_w)
        p.setPen(QColor("#ffd374"))
        p.drawLine(xh, 6, xh, h - 6)

        # time labels
        p.setPen(QColor("#9aa1ad"))
        f = QFont()
        f.setPointSize(9)
        p.setFont(f)
        p.drawText(10, 16, _fmt_time(0.0))
        p.drawText(w - 60, 16, _fmt_time(self._duration))
        p.end()


def _fmt_time(sec: float) -> str:
    m = int(sec) // 60
    s = sec - 60 * m
    return f"{m:02d}:{s:05.2f}"


def _status_label(status: str) -> str:
    return {"accepted": "已採用", "rejected": "已拒絕", "pending": "未審"}.get(status, status)


class EventListItem(QListWidgetItem):
    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__()
        self.event_id = event["id"]
        self.update_from_event(event)

    def update_from_event(self, event: dict[str, Any]) -> None:
        idx = event.get("_display_index", 0)
        dur = event["end_sec"] - event["start_sec"]
        mark = {"accepted": "✓", "rejected": "✗", "pending": "•"}[event.get("user_status", "pending")]
        text = (
            f"{mark}  #{idx:03d}    {_fmt_time(event['start_sec'])}\n"
            f"     持續 {dur:.1f}s    "
            f"車 {len(event.get('veh_track_ids', []))}  人 {len(event.get('ped_track_ids', []))}"
        )
        self.setText(text)


class ReviewView(QWidget):
    request_close_project = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_path: Path | None = None
        self._events: list[dict[str, Any]] = []
        self._current_id: int | None = None
        self._video_duration: float = 0.0

        self._build_ui()
        self._wire_shortcuts()

    # ---- layout -----------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        # Header
        header = QHBoxLayout()
        self.title = QLabel("審查候選事件")
        self.title.setObjectName("SectionTitle")
        tfont = QFont()
        tfont.setPointSize(18)
        tfont.setWeight(QFont.Weight.DemiBold)
        self.title.setFont(tfont)
        self.summary = QLabel("—")
        self.summary.setObjectName("Hint")
        header.addWidget(self.title)
        header.addSpacing(16)
        header.addWidget(self.summary)
        header.addStretch(1)
        self.export_btn = QPushButton("匯出採用片段…")
        self.export_btn.setProperty("accent", True)
        self.export_btn.setMinimumWidth(150)
        self.export_btn.clicked.connect(self._on_export_clicked)
        header.addWidget(self.export_btn)

        outer.addLayout(header)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: event list
        left = QFrame()
        left.setObjectName("Card")
        left.setProperty("class", "card")
        left.setMinimumWidth(280)
        left.setMaximumWidth(360)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(12, 12, 12, 12)
        ll.setSpacing(8)
        list_title = QLabel("事件清單")
        list_title.setObjectName("MetricLabel")
        self.event_list = QListWidget()
        self.event_list.setAlternatingRowColors(False)
        self.event_list.currentItemChanged.connect(self._on_list_item_changed)
        ll.addWidget(list_title)
        ll.addWidget(self.event_list, stretch=1)
        hint = QLabel("A 採用   R 拒絕   J/K 上下一則")
        hint.setObjectName("Hint")
        ll.addWidget(hint)
        splitter.addWidget(left)

        # Right: player + details
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)

        self.player = PlayerView()
        self.player.time_changed.connect(self._on_time_changed)
        rl.addWidget(self.player, stretch=2)

        self.timeline = EventTimeline()
        self.timeline.clicked_at.connect(self.player.seek)
        rl.addWidget(self.timeline)

        rl.addWidget(self._build_detail_panel())
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        outer.addWidget(splitter, stretch=1)

    def _build_detail_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        panel.setProperty("class", "card")
        panel.setMinimumHeight(140)
        grid = QVBoxLayout(panel)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(24)
        self.detail_title = QLabel("請先選擇一則事件")
        self.detail_title.setObjectName("SectionTitle")
        dfont = QFont()
        dfont.setPointSize(14)
        dfont.setWeight(QFont.Weight.DemiBold)
        self.detail_title.setFont(dfont)
        self.detail_status = QLabel("")
        self.detail_status.setObjectName("StatusPending")
        top.addWidget(self.detail_title)
        top.addWidget(self.detail_status)
        top.addStretch(1)

        details = QHBoxLayout()
        details.setSpacing(18)
        self.detail_time = QLabel("—")
        self.detail_dur = QLabel("—")
        self.detail_tracks = QLabel("—")
        for lbl, key in [
            (self.detail_time, "時間區間"),
            (self.detail_dur, "持續"),
            (self.detail_tracks, "追蹤數"),
        ]:
            col = QVBoxLayout()
            cap = QLabel(key)
            cap.setObjectName("MetricLabel")
            col.addWidget(cap)
            col.addWidget(lbl)
            container = QWidget()
            container.setLayout(col)
            details.addWidget(container)
        details.addStretch(1)

        # 調整時段
        trim = QHBoxLayout()
        trim.setSpacing(8)
        trim.addWidget(QLabel("調整時段(秒):"))
        self.trim_start = QDoubleSpinBox()
        self.trim_start.setDecimals(2)
        self.trim_start.setMaximum(1e6)
        self.trim_end = QDoubleSpinBox()
        self.trim_end.setDecimals(2)
        self.trim_end.setMaximum(1e6)
        trim.addWidget(self.trim_start)
        trim.addWidget(QLabel("→"))
        trim.addWidget(self.trim_end)
        self.trim_apply = QPushButton("套用")
        self.trim_apply.clicked.connect(self._on_trim_applied)
        trim.addWidget(self.trim_apply)
        trim.addStretch(1)

        # accept / reject
        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.reject_btn = QPushButton("✗  拒絕 (R)")
        self.reject_btn.setProperty("danger", True)
        self.reject_btn.setMinimumWidth(140)
        self.reject_btn.clicked.connect(lambda: self._set_status("rejected"))
        self.accept_btn = QPushButton("✓  採用 (A)")
        self.accept_btn.setProperty("success", True)
        self.accept_btn.setMinimumWidth(140)
        self.accept_btn.clicked.connect(lambda: self._set_status("accepted"))

        actions.addStretch(1)
        actions.addWidget(self.reject_btn)
        actions.addWidget(self.accept_btn)

        grid.addLayout(top)
        grid.addLayout(details)
        grid.addLayout(trim)
        grid.addLayout(actions)
        return panel

    def _wire_shortcuts(self) -> None:
        for key, handler in [
            ("A", lambda: self._set_status("accepted")),
            ("R", lambda: self._set_status("rejected")),
            ("J", lambda: self._move_selection(-1)),
            ("K", lambda: self._move_selection(+1)),
            ("Space", self.player.toggle),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(handler)

    # ---- public ----------------------------------------------------------

    def load_project(self, project_path: Path) -> None:
        self._project_path = project_path
        project = Project.load(project_path)
        try:
            events = project.load_events()
            video_path = Path(project.meta.video_path)
            self._video_duration = project.meta.video_duration_sec
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
            self._events.append(e)

        self.player.open(video_path)
        self.timeline.set_duration(self._video_duration or 1.0)
        self.timeline.set_events(self._events)
        self._rebuild_list()
        self._update_summary()

        if self._events:
            self.event_list.setCurrentRow(0)
        else:
            self.detail_title.setText("此影片未偵測到候選事件")
            self.detail_status.setText("")

    def shutdown(self) -> None:
        self.player.shutdown()

    # ---- slots -----------------------------------------------------------

    def _rebuild_list(self) -> None:
        self.event_list.blockSignals(True)
        self.event_list.clear()
        for e in self._events:
            self.event_list.addItem(EventListItem(e))
        self.event_list.blockSignals(False)

    def _on_list_item_changed(
        self,
        cur: QListWidgetItem | None,
        _prev: QListWidgetItem | None,
    ) -> None:
        if cur is None or not isinstance(cur, EventListItem):
            return
        ev = self._find_event(cur.event_id)
        if ev is None:
            return
        self._current_id = ev["id"]
        self._show_event(ev)

    def _on_time_changed(self, t_sec: float) -> None:
        self.timeline.set_current_time(t_sec)

    def _show_event(self, ev: dict[str, Any]) -> None:
        idx = ev["_display_index"]
        self.detail_title.setText(f"事件 #{idx:03d}")
        status = ev.get("user_status", "pending")
        obj_name = {
            "accepted": "StatusAccepted",
            "rejected": "StatusRejected",
            "pending": "StatusPending",
        }[status]
        self.detail_status.setObjectName(obj_name)
        self.detail_status.setText(_status_label(status))
        # QSS 需要 re-polish 才能反映 objectName 變更
        self.detail_status.style().unpolish(self.detail_status)
        self.detail_status.style().polish(self.detail_status)

        dur = ev["end_sec"] - ev["start_sec"]
        self.detail_time.setText(f"{_fmt_time(ev['start_sec'])} – {_fmt_time(ev['end_sec'])}")
        self.detail_dur.setText(f"{dur:.2f} 秒")
        self.detail_tracks.setText(
            f"行人 {len(ev.get('ped_track_ids', []))} · 車輛 {len(ev.get('veh_track_ids', []))}"
        )

        self.trim_start.blockSignals(True)
        self.trim_end.blockSignals(True)
        self.trim_start.setValue(ev["start_sec"])
        self.trim_end.setValue(ev["end_sec"])
        self.trim_start.blockSignals(False)
        self.trim_end.blockSignals(False)

        # 播放 ±3 秒 loop
        loop_s = max(0.0, ev["start_sec"] - LOOP_PADDING_SEC)
        loop_e = min(self._video_duration or 1e9, ev["end_sec"] + LOOP_PADDING_SEC)
        self.player.set_loop(loop_s, loop_e)
        self.player.seek(loop_s)
        self.player.play()

        self.timeline.set_selected(ev["id"])

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
            # 更新列表單元
            for i in range(self.event_list.count()):
                it = self.event_list.item(i)
                if isinstance(it, EventListItem) and it.event_id == self._current_id:
                    it.update_from_event(ev)
                    break
            self._show_event(ev)
            self.timeline.set_events(self._events)
        self._update_summary()
        # 自動跳下一則
        self._move_selection(+1)

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
        ev = self._find_event(self._current_id)
        if ev is not None:
            ev["start_sec"] = s
            ev["end_sec"] = e
            ev["start_frame"] = int(round(s * fps))
            ev["end_frame"] = int(round(e * fps))
            self._show_event(ev)
            self.timeline.set_events(self._events)

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
        self.summary.setText(
            f"共 {total} 筆  ·  採用 {accepted}  ·  拒絕 {rejected}  ·  未審 {pending}"
        )
        self.export_btn.setEnabled(accepted > 0)

    def _on_export_clicked(self) -> None:
        if self._project_path is None:
            return
        from zebraguard.ui.export_dialog import ExportDialog

        dlg = ExportDialog(self._project_path, self)
        dlg.exec()
