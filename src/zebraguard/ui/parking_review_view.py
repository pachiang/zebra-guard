"""違停檢測 · 審查頁。

與未禮讓 `ReviewView` 平行的另一條 Review 實作。複用的 widget 元件直接 import
自 `review_view`(縮圖列 / 時間軸 / loop icon / hover scroll list / kebab);
差別主要在 **detail panel**:用 6 類別下拉取代採用/拒絕按鈕。

UI 約定:
  · 左側清單:縮圖 + 左色條(未審灰 / 違停綠 / 非違規紅)
  · 右下 detail:事件資訊 + 車牌 + 調整時段 + 分類下拉
  · 頂部 summary:各類別計數
  · 快捷鍵 1-6 直接標分類;Del 刪事件;J/K 上下;Space 播放;N 不用(不支援手動加)
  · 匯出:只匯出 label ∈ violation 的事件(M6 實作)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from zebraguard.core.project import Project
from zebraguard.ui.review_view import (
    EventRowWidget,
    EventTimeline,
    _HoverScrollList,
    _KebabButton,
    _LoopIconButton,
    _fmt_time,
)
from zebraguard.ui.thumbnails import (
    ensure_event_thumbnails,
)
from zebraguard.ui.video_player import PlayerView

LOOP_PADDING_SEC = 3.0
# 手動新增事件的預設長度(秒) — 對齊 stopped_threshold_sec 預設 60s,
# 使用者手動加的事件也要達到 pipeline 視為「停著」的最低秒數
MANUAL_EVENT_DEFAULT_SEC = 60.0
MANUAL_NOTE_MARKER = "manual"

# (key, 中文標籤, 是否視為違規匯出, 快捷鍵)
PARKING_LABELS: list[tuple[str, str, bool, str]] = [
    ("parallel_park", "並排違停", True, "1"),
    ("intersection", "路口違停", True, "2"),
    ("other_violation", "其他違停", True, "3"),
    ("red_light", "等紅燈 / 車陣", False, "4"),
    ("legal_elsewhere", "合法(漏畫 ROI)", False, "5"),
    ("ignore", "忽略 / 誤偵測", False, "6"),
]

# 方便查表
_LABEL_META = {k: (zh, is_viol) for k, zh, is_viol, _ in PARKING_LABELS}
_VIOLATION_KEYS = {k for k, _, v, _ in PARKING_LABELS if v}


def _label_to_status(label: str | None) -> str:
    """將 label 映射到 EventRowWidget 能理解的「status」色碼:
    未標 → pending(灰);違規 → accepted(綠);非違規 → rejected(紅)。"""
    if label is None or label == "":
        return "pending"
    if label in _VIOLATION_KEYS:
        return "accepted"
    return "rejected"


class ParkingReviewView(QWidget):
    request_close_project = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_path: Path | None = None
        self._events: list[dict[str, Any]] = []
        self._current_id: int | None = None
        self._video_duration: float = 0.0
        self._thumbs: dict[int, Path] = {}
        self._video_fps: float = 30.0

        self._build_ui()
        self._wire_shortcuts()

    # ---- layout -----------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 16)
        outer.setSpacing(10)

        outer.addLayout(self._header_row())

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左側 list
        list_wrap = QFrame()
        list_wrap.setObjectName("Card")
        list_wrap.setProperty("class", "card")
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
        self.summary.setMinimumHeight(42)
        self.summary.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )

        row.addWidget(self.summary)
        row.addStretch(1)

        self.add_btn = QPushButton("新增事件")
        self.add_btn.setProperty("ghost", True)
        self.add_btn.setMinimumWidth(110)
        self.add_btn.clicked.connect(self._on_add_event)

        self.export_btn = QPushButton("匯出違停片段")
        self.export_btn.setProperty("accent", True)
        self.export_btn.setMinimumWidth(140)
        self.export_btn.clicked.connect(self._on_export_clicked)

        self.kebab_btn = _KebabButton()
        self.kebab_btn.clicked.connect(self._show_overflow_menu)

        row.addWidget(self.add_btn)
        row.addWidget(self.export_btn)
        row.addWidget(self.kebab_btn)
        return row

    def _right_side(self) -> QWidget:
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)

        self.player = PlayerView()
        self.player.time_changed.connect(self._on_time_changed)
        rl.addWidget(self.player, stretch=3)

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

        # Row 1: 標題 + 時段
        top = QHBoxLayout()
        top.setSpacing(10)
        self.detail_title = QLabel("請先選擇一則事件")
        df = QFont()
        df.setPointSize(14)
        df.setWeight(QFont.Weight.DemiBold)
        self.detail_title.setFont(df)

        self.detail_meta = QLabel("")
        self.detail_meta.setStyleSheet("color: #9aa1ad; font-size: 12px;")
        self.detail_meta.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
        )

        top.addWidget(self.detail_title)
        top.addStretch(1)
        top.addWidget(self.detail_meta)

        # Row 2: 車牌 + 調整時段
        form = QHBoxLayout()
        form.setSpacing(12)

        plate_col = QVBoxLayout()
        plate_col.setSpacing(2)
        plate_lbl = QLabel("車牌")
        plate_lbl.setObjectName("MetricLabel")
        self.plate_edit = QLineEdit()
        self.plate_edit.setPlaceholderText("例:ABC-1234")
        self.plate_edit.setMaximumWidth(200)
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
        self.trim_start.setFixedWidth(92)
        self.trim_end = QDoubleSpinBox()
        self.trim_end.setDecimals(2)
        self.trim_end.setMaximum(1e6)
        self.trim_end.setFixedWidth(92)
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

        # Row 3: 分類下拉 + 快捷鍵說明 + 刪除
        cat_row = QHBoxLayout()
        cat_row.setSpacing(10)

        cat_lbl = QLabel("分類")
        cat_lbl.setObjectName("MetricLabel")
        self.label_combo = QComboBox()
        self.label_combo.addItem("— 未審 —", userData=None)
        for key, zh, is_viol, short in PARKING_LABELS:
            mark = "✓" if is_viol else "—"
            self.label_combo.addItem(f"{short}  {mark}  {zh}", userData=key)
        self.label_combo.setMinimumWidth(200)
        self.label_combo.currentIndexChanged.connect(self._on_label_combo_changed)

        hint = QLabel("1-6 直接選  ·  N 新增  ·  Del 刪事件")
        hint.setObjectName("Hint")

        self.delete_btn = QPushButton("刪除")
        self.delete_btn.setProperty("ghost", True)
        self.delete_btn.setMinimumWidth(80)
        self.delete_btn.clicked.connect(self._on_delete_event)

        cat_row.addWidget(cat_lbl)
        cat_row.addWidget(self.label_combo)
        cat_row.addSpacing(8)
        cat_row.addWidget(hint)
        cat_row.addStretch(1)
        cat_row.addWidget(self.delete_btn)

        v.addLayout(top)
        v.addLayout(form)
        v.addLayout(cat_row)
        return panel

    def _wire_shortcuts(self) -> None:
        bindings: list[tuple[str, Any]] = [
            ("J", lambda: self._move_selection(-1)),
            ("K", lambda: self._move_selection(+1)),
            ("N", self._on_add_event),
            ("Delete", self._on_delete_event),
            ("Space", self.player.toggle),
        ]
        for _, zh, _, short in PARKING_LABELS:
            bindings.append((short, lambda k=_label_key_by_shortcut(short): self._apply_label(k)))

        for key, handler in bindings:
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
            self._video_fps = project.meta.video_fps or 30.0
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
            # label → status 映射讓 EventRowWidget 畫對色條
            e["user_status"] = _label_to_status(e.get("user_label"))
            e["_manual"] = (e.get("user_note") == MANUAL_NOTE_MARKER)
            self._events.append(e)

        self._thumbs = ensure_event_thumbnails(
            project_path, video_path, self._events, self._video_fps
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
            self.detail_meta.setText("")

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

    def _apply_selection_visuals(
        self, cur_item: QListWidgetItem | None = None
    ) -> None:
        if cur_item is None:
            cur_item = self.event_list.currentItem()
        for i in range(self.event_list.count()):
            item = self.event_list.item(i)
            widget = self.event_list.itemWidget(item)
            if isinstance(widget, EventRowWidget):
                widget.set_selected(item is cur_item)

    # ---- slots -----------------------------------------------------

    def _on_list_item_changed(
        self, cur: QListWidgetItem | None, _prev: QListWidgetItem | None
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
        self.detail_title.setText(f"事件 #{idx:03d}")

        dur = ev["end_sec"] - ev["start_sec"]
        # 停留秒數人類易讀
        if dur >= 60:
            m = int(dur) // 60
            s = int(dur) % 60
            dur_str = f"停留 {m}m{s:02d}s"
        else:
            dur_str = f"停留 {dur:.1f}s"
        self.detail_meta.setText(
            f"{_fmt_time(ev['start_sec'])}  –  {_fmt_time(ev['end_sec'])}  ·  {dur_str}"
        )

        # 車牌
        self.plate_edit.blockSignals(True)
        self.plate_edit.setText(ev.get("license_plate") or "")
        self.plate_edit.blockSignals(False)

        # Trim spinbox
        self.trim_start.blockSignals(True)
        self.trim_end.blockSignals(True)
        self.trim_start.setValue(ev["start_sec"])
        self.trim_end.setValue(ev["end_sec"])
        self.trim_start.blockSignals(False)
        self.trim_end.blockSignals(False)

        # Label combo
        label = ev.get("user_label")
        self.label_combo.blockSignals(True)
        target_idx = 0  # "未審"
        if label:
            for i in range(self.label_combo.count()):
                if self.label_combo.itemData(i) == label:
                    target_idx = i
                    break
        self.label_combo.setCurrentIndex(target_idx)
        self.label_combo.blockSignals(False)

        # 播放 ±3s loop
        loop_s = max(0.0, ev["start_sec"] - LOOP_PADDING_SEC)
        loop_e = min(self._video_duration or 1e9, ev["end_sec"] + LOOP_PADDING_SEC)
        self.player.set_loop(loop_s, loop_e)
        self.player.seek(loop_s)
        self.player.play()

        self.timeline.set_selected(ev["id"])

    def _on_label_combo_changed(self, _idx: int) -> None:
        label = self.label_combo.currentData()
        self._apply_label(label)

    def _apply_label(self, label: str | None) -> None:
        if self._current_id is None or self._project_path is None:
            return
        project = Project.load(self._project_path)
        try:
            project.update_event_label(self._current_id, label)
        finally:
            project.close()

        ev = self._find_event(self._current_id)
        if ev is not None:
            ev["user_label"] = label
            ev["user_status"] = _label_to_status(label)
            self._refresh_row(ev)
            self.timeline.set_events(self._events)
        self._update_summary()

        # 快捷鍵 1-6 直接標完,往下一則跳
        if label is not None:
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
        fps = self.player.fps() or self._video_fps or 30.0
        project = Project.load(self._project_path)
        try:
            project.update_event_range(self._current_id, s, e, fps)
        finally:
            project.close()
        # trim 改變中段時間 → 重產縮圖 + 重載列表
        self._reload_events_preserving_selection(
            self._current_id, force_thumb_ids={self._current_id}
        )

    def _on_add_event(self) -> None:
        """於 player 當前位置新增一筆候選事件(使用者看到 pipeline 漏抓的違停車時用)。

        預設長度 60 秒,對齊 stopped_threshold_sec。建後自動選取,使用者馬上就能
        按 1-6 分類或拖 trim 微調。
        """
        if self._project_path is None:
            return
        fps = self.player.fps() or self._video_fps or 30.0
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
                user_status="pending",
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
            self.player.pause()

    def _reload_events_preserving_selection(
        self,
        select_id: int | None,
        *,
        force_thumb_ids: set[int] | None = None,
    ) -> None:
        if self._project_path is None:
            return
        project = Project.load(self._project_path)
        try:
            raw = project.load_events()
            video_path = Path(project.meta.video_path)
            fps = project.meta.video_fps or 30.0
        finally:
            project.close()

        self._events = []
        for i, e in enumerate(raw):
            e["_display_index"] = i + 1
            e["user_status"] = _label_to_status(e.get("user_label"))
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
        counts = {k: 0 for k, _, _, _ in PARKING_LABELS}
        pending = 0
        for e in self._events:
            lbl = e.get("user_label")
            if lbl in counts:
                counts[lbl] += 1
            else:
                pending += 1
        violations = sum(counts[k] for k in _VIOLATION_KEYS)

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
            _stat(violations, "違規", "#10b981"),
            _stat(pending, "未審", "#9aa1ad"),
        ]))
        self.export_btn.setEnabled(violations > 0)

    def _show_overflow_menu(self) -> None:
        menu = QMenu(self)
        reveal = menu.addAction("開啟專案資料夾")
        pos = self.kebab_btn.mapToGlobal(self.kebab_btn.rect().bottomLeft())
        act = menu.exec(pos)
        if act is reveal:
            self._reveal_project()

    def _reveal_project(self) -> None:
        if self._project_path is None:
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._project_path)))

    def _on_export_clicked(self) -> None:
        # M6 會實作違停專屬匯出。先 placeholder 告知。
        QMessageBox.information(
            self,
            "匯出 · M6",
            "違停匯出功能在 M6 會開放(片段 + 關鍵幀 + 違停檢舉書模板)。\n\n"
            "目前已採用的分類:"
            + ", ".join(
                _LABEL_META[k][0] for k in _VIOLATION_KEYS
                if any(e.get("user_label") == k for e in self._events)
            ),
        )


def _label_key_by_shortcut(short: str) -> str | None:
    for key, _, _, s in PARKING_LABELS:
        if s == short:
            return key
    return None
